import logging

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import evaluate
import numpy as np

from core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def __init__(self, model,  dataset_name, args=None,):
        super().__init__(model, args)
        if dataset_name == "tinystories":
            self.tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rouge = evaluate.load('rouge')

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def get_model_scores(self):
        return self.scores

    @staticmethod
    def _attach_hooks(model, store, module_types=(nn.Conv2d, nn.Linear)):
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                def fwd_hook(mod, inp, out, name=name):
                    x = inp[0]
                    if not torch.is_tensor(x):
                        return
                    x = x.detach()
                    if isinstance(mod, nn.Conv2d):
                        patches = F.unfold(x, kernel_size=mod.kernel_size, dilation=mod.dilation, padding=mod.padding, stride=mod.stride)
                        a = patches.transpose(1, 2).reshape(-1, patches.shape[1])
                    else:
                        a = x.reshape(-1, x.shape[-1])
                    store.setdefault(name, {})["a"] = a.cpu()

                def bwd_hook(mod, grad_input, grad_output, name=name):
                    go = grad_output[0]
                    if not torch.is_tensor(go):
                        return
                    go = go.detach()
                    if isinstance(mod, nn.Conv2d):
                        g = go.permute(0, 2, 3, 1).reshape(-1, go.shape[1])
                    else:
                        g = go.reshape(-1, go.shape[-1])
                    store.setdefault(name, {})["g"] = g.cpu()

                handles.append(module.register_forward_hook(fwd_hook))
                if hasattr(module, "register_full_backward_hook"):
                    handles.append(module.register_full_backward_hook(bwd_hook))
                else:
                    handles.append(module.register_backward_hook(bwd_hook))
        return handles

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 :  training with mask 
        # mode 1 : training with mask 
        # mode 2 : training with mask, calculate mask
        # mode 3 : training with mask, calculate mask
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
            
        epoch_loss = []
        
        if mode in [2, 3]:
            local_epochs = args.adjustment_epochs if args.adjustment_epochs is not None else args.epochs
        else:
            local_epochs = args.epochs

        if mode in [2, 3]:
            A_epochs = local_epochs // 2 if args.A_epochs is None else args.A_epochs
            first_epochs = min(local_epochs, A_epochs)
        else:
            first_epochs = local_epochs

        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                #self.model.apply_mask_gradients()  # apply pruning mask
                
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        if mode in [2, 3]:
            model.zero_grad()
            store = {}
            handles = []

            if args.growth_data_mode == "score":
                handles = self._attach_hooks(model, store)
                gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
            
            elif args.growth_data_mode == "random":
                gradients = {name: torch.randn_like(param, device='cpu').clone() for name, param in model.named_parameters() if param.requires_grad}

            elif args.growth_data_mode == "single":
                x, labels = next(iter(train_data))
                x, labels = x[0].unsqueeze(0).repeat(2, 1, 1, 1).to(device), labels[0].unsqueeze(0).repeat(2).to(device)  # Duplicate the sample to create a pseudo-batch
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
                model.zero_grad()
            else:
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
                model.zero_grad()

            for h in handles:
                h.remove()

            score_prune = {}
            score_grow = {}
            weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

            for l, _ in model.named_modules():
                lg = f"{l}.weight"
                if l not in store or "a" not in store[l] or "g" not in store[l] or lg not in gradients or lg not in weights:
                    continue

                a = store[l]['a']
                g = store[l]['g'] # 对 pre-activation 的反向梯度
                A = (a.t() @ a) / max(a.shape[0], 1)
                G = (g.t() @ g) / max(g.shape[0], 1)

                adiag = torch.diagonal(A)
                gdiag = torch.diagonal(G)
                F_diag = (adiag[:, None] * gdiag[None, :]).t()

                eps = 1e-8
                score_prune[l] = -gradients[lg] * weights[lg] + 0.5 * (weights[lg] ** 2) * F_diag
                score_grow[l] = 0.5 * (gradients[lg] ** 2) / (F_diag + eps)

            # pruning and growing 第五步
            self.scores = {"prune": score_prune, "grow": score_grow}
            if args.growth_data_mode == "score":
                model.adjust_mask_dict(gradients, t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha, score=self.scores)
            else:
                model.adjust_mask_dict(gradients, t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha, score=None)
            model.apply_mask()

        # 第六步
        # for epoch in range(first_epochs, local_epochs):
        #     batch_loss = []
        #     for batch_idx, (x, labels) in enumerate(train_data):
        #         x, labels = x.to(device), labels.to(device)
        #         model.zero_grad()
        #         log_probs = model(x)
        #         loss = criterion(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
        #     logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        return model.mask_dict

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'Accuracy': 0,
            'Loss': 0,
            'test_total': 0
        }
        # predictions = []
        # references = []
        nlls = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                tokenized = self.tokenizer(batch['text'], padding=True, return_tensors='pt', max_length = 256, truncation = True)['input_ids'].to(device)
                labels = tokenized[..., 1:].cpu()
                logits, loss = model(tokenized, tokenized)
                pred_ids = torch.argmax(logits, dim=-1)[..., :-1].cpu()
                
                pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
                # pred_ids = np.where(labels != pad_token_id, pred_ids, pad_token_id)
                # preds = self.tokenizer.batch_decode( pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True )

                # predictions += preds
                # references += batch['text']

                # logging.info(tokenized[0])
                # logging.info(pred_ids[0])

                metrics['Loss'] += loss.item() * len(batch['text'])
                metrics['test_total'] += len(batch['text'])
                
                for i in range(len(labels)):
                    hit, total = 0, 0 
                    nlls.append([])
                    for j in range(len(labels[i])):
                        if labels[i][j] != pad_token_id:
                            poss = torch.nn.functional.softmax(logits[i][j])
                            logit = poss[labels[i][j]].item()
                            nlls[-1].append(logit)
                            
                            total += 1
                            if labels[i][j] == pred_ids[i][j]:
                                hit += 1

                    metrics['Accuracy'] += (hit / total)
                    ppl = np.exp(-np.sum(np.log(nlls[-1])) /len(nlls[-1]))

                    # # debug inf problem
                    # if np.isinf(ppl):
                    #     print(nlls[-1])

                    nlls[-1] = ppl
                

        # rouge_results = self.rouge.compute(references=references, predictions=predictions)
        # metrics.update(rouge_results)
        metrics["Perplexity"] = np.mean(nlls)
        metrics['Loss'] /= metrics['test_total']
        metrics['Accuracy'] /= metrics['test_total']
        
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
    
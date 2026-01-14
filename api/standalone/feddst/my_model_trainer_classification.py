import logging

import torch
from torch import nn
import torch.nn.functional as F
from ...pruning.init_scheme import f_decay

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def get_model_scores(self):
        return self.model.scores

    @staticmethod
    def _attach_hooks(model, store, module_types=(nn.Conv2d, nn.Linear)):
        handles = []
        for name, module in model.named_modules():
            if not isinstance(module, module_types):
                continue

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

                if not torch.is_tensor(out):
                    return

                def out_grad_hook(grad, name=name, mod=mod):
                    go = grad.detach()
                    if isinstance(mod, nn.Conv2d):
                        g = go.permute(0, 2, 3, 1).reshape(-1, go.shape[1])
                    else:
                        g = go.reshape(-1, go.shape[-1])
                    store.setdefault(name, {})["g"] = g.cpu()

                out.register_hook(out_grad_hook)

            handles.append(module.register_forward_hook(fwd_hook))

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
            A_epochs = max(1, A_epochs) # 预防 first_epochs 设为零
            first_epochs = min(local_epochs, A_epochs)
        else:
            first_epochs = local_epochs

        store = {}
        handles = []
        score_prune_sum = {}
        score_grow_sum = {}
        score_cnt = {}
        params = dict(model.named_parameters())
        
        if mode in [2, 3] and args.growth_data_mode == "score":
            handles = self._attach_hooks(model, store)

        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                if mode in [2, 3] and args.growth_data_mode == "score":
                    for l, _ in model.named_modules():
                        lg = f"{l[6:] if l.startswith('model.') else l}.weight" # params 需要的格式
                        # store = {model.layer: {'a': tensor, 'g': tensor}}
                        # print(list(params.keys())[:50])
                        if l not in store or "a" not in store[l] or "g" not in store[l] or lg not in params or params[lg].grad is None:
                            continue

                        grad = params[lg].grad.detach().cpu()
                        w = params[lg].data.detach().cpu()
                        eps = 1e-8

                        a = store[l]['a']
                        g = store[l]['g']

                        A = (a.t() @ a) * max(a.shape[0], 1)
                        G = (g.t() @ g) * max(g.shape[0], 1)

                        adiag = torch.diagonal(A)
                        gdiag = torch.diagonal(G)

                        expected_out = params[lg].data.size(0)
                        expected_in = params[lg].data.view(expected_out, -1).size(1)

                        if gdiag.numel() != expected_out:
                            if gdiag.numel() > expected_out:
                                if gdiag.numel() % expected_out == 0:
                                    gdiag = gdiag.reshape(expected_out, -1).mean(dim=1)
                                else:
                                    gdiag = gdiag[:expected_out]
                            else:
                                reps = (expected_out + gdiag.numel() - 1) // gdiag.numel()
                                gdiag = gdiag.repeat(reps)[:expected_out]

                        if adiag.numel() != expected_in:
                            if adiag.numel() > expected_in:
                                if adiag.numel() % expected_in == 0:
                                    adiag = adiag.reshape(expected_in, -1).mean(dim=1)
                                else:
                                    adiag = adiag[:expected_in]
                            else:
                                reps = (expected_in + adiag.numel() - 1) // adiag.numel()
                                adiag = adiag.repeat(reps)[:expected_in]

                        F_diag = (adiag[:, None] * gdiag[None, :]).t()
                        
                        w2 = w.view(w.size(0), -1)
                        g2 = grad.view(grad.size(0), -1)

                        sp2 = -g2 * w2 + 0.5 * (w2 ** 2) * F_diag
                        sg2 = 0.5 * (g2 ** 2) / (F_diag + eps)

                        sp = sp2.view_as(w)
                        sg = sg2.view_as(w)
                        
                        if l in score_prune_sum:
                            score_prune_sum[l] += sp
                            score_grow_sum[l] += sg
                            score_cnt[l] += 1
                        else:
                            score_prune_sum[l] = sp
                            score_grow_sum[l] = sg
                            score_cnt[l] = 1

                #self.model.apply_mask_gradients()  # apply pruning mask
                    
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        if mode in [2, 3]:
            if args.growth_data_mode == "score":
                gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
                model.zero_grad()
            
            elif args.growth_data_mode == "random":
                model.zero_grad()
                gradients = {name: torch.randn_like(param, device=device).clone() for name, param in model.named_parameters() if param.requires_grad}

            elif args.growth_data_mode == "single":
                model.zero_grad()
                x, labels = next(iter(train_data))
                x, labels = x[0].unsqueeze(0).repeat(2, 1, 1, 1).to(device), labels[0].unsqueeze(0).repeat(2).to(device)  # Duplicate the sample to create a pseudo-batch
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.requires_grad and param.grad is not None}
                model.zero_grad()
            else:
                model.zero_grad()
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.requires_grad and param.grad is not None}
                model.zero_grad()

            for h in handles:
                h.remove()

            score_prune = {}
            score_grow = {}
            if args.growth_data_mode == "score":
                score_prune = {l: score_prune_sum[l] / max(score_cnt[l], 1) for l in score_prune_sum}
                score_grow = {l: score_grow_sum[l] / max(score_cnt[l], 1) for l in score_grow_sum}

            # pruning and growing 第五步
            self.model.scores = {"prune": score_prune, "grow": score_grow}

            if args.growth_data_mode == "score":
                model.adjust_mask_dict(gradients, t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha, scores=self.model.scores)
            else:
                model.adjust_mask_dict(gradients, t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha, scores=None)
            model.apply_mask()
            
        logging.info("FURTHER TRAINING MODEL CLASSIFICATION")
        for epoch in range(first_epochs, local_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        logging.info(f"Model trainer finish training, score, adjust mask")
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

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['Accuracy'] += correct.item()
                metrics['Loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        
        metrics['Accuracy'] /= metrics['test_total'] 
        metrics['Loss'] /= metrics['test_total'] 
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
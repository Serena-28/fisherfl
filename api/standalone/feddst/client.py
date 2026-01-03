import logging
import copy
import torch
from torch import nn

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number
    
    @staticmethod
    def find_nn_module(obj):
        for attr in ("model", "net", "network", "module", "encoder", "backbone"):
            if hasattr(obj, attr):
                m = getattr(obj, attr)
                if isinstance(m, nn.Module):
                    return m

        for meth in ("get_model", "get_net", "get_network"):
            if hasattr(obj, meth) and callable(getattr(obj, meth)):
                m = getattr(obj, meth)()
                if isinstance(m, nn.Module):
                    return m

        if hasattr(obj, "__dict__"):
            for _, v in obj.__dict__.items():
                if isinstance(v, nn.Module):
                    return v

        return None

    @staticmethod
    def attach_hooks(model, store, module_types=(nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU)):
        handles = []

        for name, module in model.named_modules():
            if isinstance(module, module_types):
                def fwd_hook(mod, inp, out, name=name):
                    if name not in store:
                        store[name] = {}
                    x = inp[0]  # 该层输入激活
                    store[name]["a"] = x.detach().cpu() if torch.is_tensor(x) else x

                def bwd_hook(mod, grad_input, grad_output, name=name):
                    if name in store:
                        g = grad_output[0]  # 对 pre-activation 输出的梯度
                        store[name]["g"] = g.detach().cpu() if torch.is_tensor(g) else g

                handles.append(module.register_forward_hook(fwd_hook))
                if hasattr(module, "register_full_backward_hook"):
                    handles.append(module.register_full_backward_hook(bwd_hook))
                else:
                    handles.append(module.register_backward_hook(bwd_hook))

        return handles
    
    def train(self, w_global, mode, round_idx):
        #self.model_trainer.set_model_params(w_global)
        #self.model_trainer.train(self.local_training_data, self.device, self.args)
        #weights = self.model_trainer.get_model_params()
        #return weights
        local_w_global = copy.deepcopy(w_global)
        self.model_trainer.set_model_params(local_w_global)
        self.store = {}

        handles = []
        layers = self.find_nn_module(self.model_trainer)
        if layers is not None:
            handles = self.attach_hooks(layers, self.store)

        gradients = self.model_trainer.train(self.local_training_data, self.device, self.args, mode, round_idx)
        weights = self.model_trainer.get_model_params()

        for h in handles:
            h.remove()

        # print("grads type:", type(gradients))
        # print("weights type:", type(weights))

        # if isinstance(gradients, dict):
        #     print("grad keys sample:", list(gradients.keys())[:10])
        # if isinstance(weights, dict):
        #     print("weight keys sample:", list(weights.keys())[:10])

        self.score_prune = {}
        self.score_grow = {}

        for l, _ in list(layers.named_modules())[1:]: 
            lg = f"{l}.weight"
            if l not in self.store or lg not in gradients or lg not in weights: continue

            a = self.store[l]['a']
            g = self.store[l]['g'] # 对 pre-activation 的反向梯度
            A = (a.T @ a) / a.shape[0]
            G = (g.T @ g) / g.shape[0]

            # Fisher = torch.kron(A, G)
            gdiag = torch.diagonal(G)
            adiag = torch.diagonal(A)
            F_diag = (adiag[:, None] * gdiag[None, :]).T

            self.score_prune[l] = -gradients[lg] * weights[lg] + 0.5 * (weights[lg] ** 2) * F_diag
            self.score_grow[l] = 0.5 * (gradients[lg] ** 2) / (F_diag + 1e-8) # 1e-8 预防除于零
        
        return weights , gradients

        #logging.debug("Trained local weights: " + str(weights)) 
        # return weights,gradeints
    
    def get_gradients(self):
        return self.model_trainer.get_model_gradients()

    # def local_test(self, b_use_test_dataset):
    #     if b_use_test_dataset:
    #         test_data = self.local_test_data
    #     else:
    #         test_data = self.local_training_data
    #     metrics = self.model_trainer.test(test_data, self.device, self.args)
    #     return metrics

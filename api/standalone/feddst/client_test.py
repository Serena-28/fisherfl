import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import Client


# ---- 0) Minimal args container ----
class Args:
    def __init__(self, lr=0.05, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay


# ---- 1) Tiny model (keeps Fisher small) ----
class TinyMLP(nn.Module):
    def __init__(self, in_dim=10, hidden=5, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---- 2) Trainer matching Client expectations ----
class SimpleModelTrainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self._last_grads = None
        self._opt = None

    def set_model_params(self, w_global):
        self.model.load_state_dict(w_global, strict=True)

    def get_model_params(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def get_model_gradients(self):
        return self._last_grads

    def train(self, train_loader, device, args, mode=None, round_idx=None):
        self.model.to(device)
        self.model.train()

        if self._opt is None:
            self._opt = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # one minibatch is enough for sanity check
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            self._opt.zero_grad(set_to_none=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            self._last_grads = {
                name: (p.grad.detach().cpu().clone() if p.grad is not None else None)
                for name, p in self.model.named_parameters()
            }

            self._opt.step()
            break

        return self._last_grads


# ---- 3) Collector for expected KFAC ingredients (a_in and g_out) ----
class LinearKFACCollector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.a = {}        # layer -> (N, in)
        self.g = {}        # layer -> (N, out)
        self.handles = []

    def start(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._fwd(name)))
                if hasattr(module, "register_full_backward_hook"):
                    self.handles.append(module.register_full_backward_hook(self._bwd(name)))
                else:
                    self.handles.append(module.register_backward_hook(self._bwd(name)))

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _fwd(self, name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            self.a[name] = x.reshape(-1, x.shape[-1]).cpu()  # (N, in)
        return hook

    def _bwd(self, name):
        def hook(mod, grad_input, grad_output):
            go = grad_output[0].detach()
            self.g[name] = go.reshape(-1, go.shape[-1]).cpu()  # (N, out)
        return hook


# ---- Helpers ----
def stats(label, t):
    t = t.detach().cpu()
    print(f"{label:22s} shape={tuple(t.shape)}  mean={t.mean().item():+.3e}  "
          f"std={t.std().item():.3e}  min={t.min().item():+.3e}  max={t.max().item():+.3e}  "
          f"norm={t.norm().item():.3e}")

def relerr(a, b, eps=1e-12):
    a = a.detach().cpu()
    b = b.detach().cpu()
    return (a - b).norm().item() / (b.norm().item() + eps)

def kfac_diag_from_ag(a_in, g_out):
    N = a_in.shape[0]
    A = (a_in.t() @ a_in) / N           # (in,in)
    G = (g_out.t() @ g_out) / N         # (out,out)
    return torch.diagonal(G)[:, None] * torch.diagonal(A)[None, :]  # (out,in)

def get_score_dict(obj, attr_name):
    d = getattr(obj, attr_name, None)
    return d if isinstance(d, dict) else None

def pick_key(d, lname):
    # try common key conventions
    if d is None:
        return None
    if lname in d:
        return lname
    wkey = f"{lname}.weight"
    if wkey in d:
        return wkey
    mwkey = f"module.{wkey}"
    if mwkey in d:
        return mwkey
    return None


def main():
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=True)

    # data
    N = 32
    in_dim = 10
    num_classes = 3
    X = torch.randn(N, in_dim)
    y = torch.randint(0, num_classes, (N,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)

    # model / trainer / client
    device = torch.device("cpu")
    model = TinyMLP(in_dim=in_dim, hidden=5, out_dim=num_classes)
    trainer = SimpleModelTrainer(model)
    args = Args(lr=0.05)

    client = Client(
        client_idx=0,
        local_training_data=loader,
        local_test_data=None,
        local_sample_number=N,
        args=args,
        device=device,
        model_trainer=trainer,
    )

    w_global = trainer.get_model_params()

    # hooks to get expected KFAC ingredients
    collector = LinearKFACCollector(model)
    collector.start()

    print("=== Start train ===")
    out = client.train(w_global, mode="debug", round_idx=0)
    collector.stop()

    # support either (weights, grads) or (weights, grads, ...) returns
    if isinstance(out, tuple) and len(out) >= 2:
        weights, grads = out[0], out[1]
    else:
        raise RuntimeError("client.train did not return (weights, grads)")

    # actual scores exposed by your client.py
    score_prune_actual = get_score_dict(client, "score_prune")
    score_grow_actual  = get_score_dict(client, "score_grow")

    print("\n=== Client outputs (keys) ===")
    print("weights keys sample:", list(weights.keys())[:8])
    print("grads keys sample:  ", list(grads.keys())[:8])
    print("score_prune keys:   ", None if score_prune_actual is None else list(score_prune_actual.keys()))
    print("score_grow keys:    ", None if score_grow_actual is None else list(score_grow_actual.keys()))

    print("\n================= EXPECTED vs ACTUAL (per Linear layer) =================")
    eps = 1e-8
    linear_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    for lname in linear_layers:
        wkey = f"{lname}.weight"
        if wkey not in weights or wkey not in grads:
            print(f"\n[{lname}] missing {wkey} in weights/grads -> skip")
            continue
        if lname not in collector.a or lname not in collector.g:
            print(f"\n[{lname}] missing a/g in collector -> skip")
            continue

        # expected KFAC diagonal Fisher
        a_in = collector.a[lname]     # (N,in)
        g_out = collector.g[lname]    # (N,out)
        F_diag = kfac_diag_from_ag(a_in, g_out)  # (out,in)

        # expected scores using your formula (with parameter gradient dW)
        W  = weights[wkey]
        dW = grads[wkey]
        score_prune_expected = -dW * W + 0.5 * (W ** 2) * F_diag
        score_grow_expected  = 0.5 * (dW ** 2) / (F_diag + eps)

        # actual scores from client (try key conventions)
        kp = pick_key(score_prune_actual, lname) if score_prune_actual is not None else None
        kg = pick_key(score_grow_actual,  lname) if score_grow_actual  is not None else None

        print(f"\n--- Layer: {lname} ---")
        stats("F_diag (expected)", F_diag)
        stats("score_prune exp", score_prune_expected)
        stats("score_grow  exp", score_grow_expected)

        if kp is None:
            print("score_prune actual: (missing)")
        else:
            sp_a = score_prune_actual[kp]
            stats("score_prune act", sp_a)
            print(f"relerr(prune act, exp) = {relerr(sp_a, score_prune_expected):.6f}")

        if kg is None:
            print("score_grow actual:  (missing)")
        else:
            sg_a = score_grow_actual[kg]
            stats("score_grow  act", sg_a)
            print(f"relerr(grow  act, exp) = {relerr(sg_a, score_grow_expected):.6f}")

    print("\n================= DONE =================")


if __name__ == "__main__":
    main()

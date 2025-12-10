import torch
from torch import nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def _move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch  # leave non-tensors as-is

@torch.no_grad()
def _model_device(model: nn.Module):
    # best-effort: get any param/buffer device
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")

def find_unused_params(model, batch):
    # ensure grads enabled for this debug pass
    if not torch.is_grad_enabled():
        torch.set_grad_enabled(True)

    model.train()
    model.zero_grad(set_to_none=True)

    device = _model_device(model)
    batch = _move_to_device(batch, device)

    # Track which params actually receive a grad during backward
    touched = set()
    hooks = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            # default-arg trick binds current name
            hooks.append(p.register_hook(lambda g, n=name: touched.add(n)))

    # Forward/backward (avoid AMP for clarity)
    with torch.cuda.amp.autocast(enabled=False):
        out = model.training_step(batch, 0)

    # Lightning training_step may return a dict or a Tensor
    loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
    if not torch.is_tensor(loss):
        # If your training_step does manual optimization and returns None,
        # better call a dedicated loss function instead of training_step.
        for h in hooks: h.remove()
        raise RuntimeError("training_step didn't return a Tensor loss. "
                           "Call a pure loss function or make training_step return loss.")

    if loss.dim() != 0:
        loss = loss.mean()

    loss.backward()

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Params that require grad but never received one
    unused = [name for name, p in model.named_parameters()
              if p.requires_grad and name not in touched]

    return unused

def probe_grad_flow(module, x):
    assert torch.is_grad_enabled(), "Gradients must be enabled for probing."
    assert x.requires_grad, "Input x must require gradients for probing."

    y = module(x)
    y.retain_grad()                   # retain grad for output, else it may be freed after backward
    probe = (y ** 2).mean()
    probe.backward(retain_graph=True) # retain_graph in case caller wants to do more backward passes

    print("probe grad on x:", x.grad.norm().item())  # should be > 0
    for n, p in module.named_parameters():
        if p.grad is None:
            print("[probe] no grad:", n)


def check_grad_flow_from_loss_to_x(x, loss):
    """
    Check if gradients can flow from loss to input x.
    Args:
        x (torch.Tensor): Input tensor that requires gradients.
        loss (torch.Tensor): Scalar loss tensor.
    """
    assert torch.is_grad_enabled(), "Gradients must be enabled for probing."
    assert x.requires_grad, "Input x must require gradients for probing."
    
    g = torch.autograd.grad(loss, x, retain_graph=True, allow_unused=True)[0]
    print("grad wrt x is None?", g is None)


def log_top_grads(model, k=10):
    """
    This function logs the top k parameters with the highest gradient norms.
    """
    stats = []
    for n,p in model.named_parameters():
        if p.grad is None: continue
        stats.append((n, float(p.grad.norm().item())))
    stats.sort(key=lambda x: x[1], reverse=True)

    rank_zero_print("\n ======================== Top gradients ========================")
    for n,g in stats[:k]:
        rank_zero_print(f"[grad] {n:70s} {g:.3f}")


def log_top_update_strength(model, k=10, eps=1e-8):
    """
    Logs the top-k parameters ranked by effective update strength:
        ratio = ||grad|| / (||param|| + eps)
    
    NOTE: 
        - ratio < 1e-4          : too small, param may stagnate (negligible update)
        - ratio > 1.0           : too large, gradient explosion or wildly oversized step
        - ratio in [1e-3, 1e-1] : healthy update magnitudes
    """
    stats = []
    for n, p in model.named_parameters():
        if p.grad is None: 
            continue
        grad_norm = p.grad.norm().item()
        param_norm = p.norm().item() if p.norm().numel() > 0 else 0.0
        ratio = grad_norm / (param_norm + eps)
        stats.append((n, ratio, grad_norm, param_norm))

    stats.sort(key=lambda x: x[1], reverse=True)

    rank_zero_print("\n =================== Top update strengths ===================")
    for n, ratio, g_norm, p_norm in stats[:k]:
        rank_zero_print(
            f"[update_strength] {n:70s} "
            f"ratio={ratio:.4e} | g={g_norm:.3e} | p={p_norm:.3e}"
        )
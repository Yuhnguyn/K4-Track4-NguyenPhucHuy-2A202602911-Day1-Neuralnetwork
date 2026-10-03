"""optimizer.py — chọn bộ tối ưu và cắt gradient (dùng torch.optim.*, clip_grad_norm_).

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr*wd*w - lr*m_hat/(sqrt(v_hat)+eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer theo `name`."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        # weight_decay của Adam là L2 trộn vào gradient (khác AdamW)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học. Trả về None nếu name là None."""
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=kwargs.pop("max_lr"),
                                                   total_steps=total_steps, **kwargs)
    if name == "warmup_cosine":
        warmup = kwargs.pop("warmup_steps", max(1, total_steps // 20))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                                          total_iters=warmup),
                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=max(1, total_steps - warmup))],
            milestones=[warmup],
        )
    raise ValueError(f"scheduler không hỗ trợ: {name!r}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 TOÀN CỤC, trả về chuẩn gradient TRƯỚC KHI cắt.

    max_norm = None -> không cắt, chỉ ĐO chuẩn (clip_grad_norm_ với max_norm = inf).
    Giá trị trả về chính là `grad_norm` ghi lại mỗi bước, nên thấy được các "gai" gradient.
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(
        params, float("inf") if max_norm is None else float(max_norm)
    )
    if not math.isfinite(float(total_norm)):
        return float(total_norm)
    return float(total_norm)

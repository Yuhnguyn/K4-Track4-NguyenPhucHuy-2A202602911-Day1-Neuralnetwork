"""plots.py — ảnh biểu đồ là sản phẩm nộp: mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Khi notebook chạy trong code/, lưu vào "../figures/" (path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")            # không cần màn hình; notebook vẫn hiển thị được fig trả về
import matplotlib.pyplot as plt  # noqa: E402


def _cfg_label(cfg: dict) -> str:
    return (f"{cfg['optimizer']} lr={cfg['lr']} bs={cfg['batch']} "
            f"hidden={'-'.join(map(str, cfg['hidden']))} drop={cfg['dropout']} "
            f"init={cfg['init']} clip={cfg['clip_norm']} {cfg['precision']} loss={cfg['loss']} seed={cfg['seed']}")


def plot_run(result: dict, path: str):
    """Vẽ MỘT thí nghiệm thành PNG 3 ô: loss (train/val), acc + macro-F1, grad_norm (trước clip).

    Trả về fig để notebook hiển thị inline (bằng chứng nằm ngay trong output của notebook).
    """
    cfg, h = result["cfg"], result["history"]
    ep = h["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))

    ax[0].plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax[0].plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy / MSE", title="Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(ep, h["val_acc"], "o-", ms=3, color="tab:green", label="val accuracy")
    ax[1].plot(ep, h["val_macro_f1"], "s-", ms=3, color="tab:red", label="val macro-F1")
    ax[1].axhline(0.4876, ls=":", c="gray", lw=1, label="đoán lớp đa số (0.4876)")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val accuracy / macro-F1", ylim=(0, 1.02))
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(ep, h["grad_norm"], "d-", ms=3, color="tab:purple", label="‖g‖ toàn cục (trước clip)")
    if cfg.get("clip_norm") is not None:
        ax[2].axhline(float(cfg["clip_norm"]), ls="--", c="k", lw=1, label=f"c = {cfg['clip_norm']}")
    ax[2].set(xlabel="epoch", ylabel="grad norm (trung bình epoch)", title="Gradient norm")
    if all(v and v > 0 for v in h["grad_norm"]):
        ax[2].set_yscale("log")
    ax[2].legend()
    ax[2].grid(alpha=0.3, which="both")

    s = result["summary"]
    if s["best_epoch"]:
        for a in ax:
            a.axvline(s["best_epoch"], ls=":", c="k", alpha=0.5, lw=1)
        ax[0].plot([s["best_epoch"]], [s["best_val_loss"]], "k*", ms=14,
                   label=f"best epoch {s['best_epoch']}", zorder=5)
        ax[0].legend()

    fig.suptitle(f"{cfg['exp_id']}  |  {cfg['group']}  |  {_cfg_label(cfg)}\n"
                 f"step0_loss={s['step0_loss']:.4f}  best_val_loss={s['best_val_loss']:.4f}"
                 f" @ep{s['best_epoch']}  val_acc={s['val_acc']:.4f}  val_macroF1={s['val_macro_f1']:.4f}"
                 f"  {s['time_per_epoch_s']:.1f}s/epoch"
                 + ("  [DIVERGED]" if s["diverged"] else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=125, bbox_inches="tight")
    return fig


def plot_compare(results: list[dict], metric: str, path: str, title: str = ""):
    """Chồng một chỉ số (val_loss / val_macro_f1 / grad_norm / val_acc) của nhiều thí nghiệm.

    Dùng cho figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in results:
        h = r["history"]
        y = h[metric]
        if len(y) != len(h["epoch"]):                       # tránh lệch nếu diverged
            y = y[:len(h["epoch"])]
        ax.plot(h["epoch"][:len(y)], y, "o-", ms=3, label=r["cfg"]["exp_id"])
    ax.set(xlabel="epoch", ylabel=metric,
           title=title or f"So sánh {metric}")
    if metric in ("val_macro_f1", "val_acc"):
        ax.set_ylim(0, 1.02)
    if "loss" in metric:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=125, bbox_inches="tight")
    return fig

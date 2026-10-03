"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán, ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` chọn bằng VAL rồi điền vào (Part 2).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # chọn bằng val (0.01 / 0.03 / 0.1 đã thử ở Part 3, chủ đề hparam)
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)

EVAL_BATCH = 8192


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_type(device) -> str:
    d = str(device)
    return "cuda" if d.startswith("cuda") else ("mps" if d.startswith("mps") else "cpu")


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; cm: (7,7), hàng = nhãn thật, cột = dự đoán.

    F1_c = 2 P_c R_c / (P_c + R_c), bằng 0 nếu P_c + R_c = 0 (giống scripts/evaluate.py).
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = EVAL_BATCH) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits (chế độ eval, không dropout)."""
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i:i + batch_size])
        out.append(logits.argmax(dim=1))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.int64, device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = EVAL_BATCH) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() + no_grad (dropout TẮT).

    Dùng cho: train loss (toàn bộ train, đo ở eval mode -> cùng thang đo với val), val, và eval cuối.
    Loss gộp bằng reduction="sum" rồi chia đúng mẫu số (CE: N mẫu; MSE: N x 7 phần tử),
    nhờ vậy giá trị trả về đúng bằng loss trung bình của cả tập, không phụ thuộc kích thước lô.
    """
    model.eval()
    total, correct, n = 0.0, 0, X.shape[0]
    cm = np.zeros((7, 7), dtype=np.int64)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total += float(compute_loss(logits, yb, loss_name, reduction="sum"))
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum())
        np.add.at(cm, (yb.cpu().numpy(), pred.cpu().numpy()), 1)
    denom = max(n, 1) * (1 if loss_name == "ce" else 7)
    return dict(loss=total / denom, acc=correct / max(n, 1),
                macro_f1=macro_f1_from_confusion(cm))


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """\"ce\": cross-entropy nhận logit thô + nhãn int64.
       \"mse\": MSE giữa logit và one-hot của y — nn.MSELoss KHÔNG có hệ số 1/2 và lấy TRUNG BÌNH
              trên mọi phần tử (B x 7), tức đã chia thêm 7 so với trung bình theo mẫu.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        y1h = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
        return F.mse_loss(logits, y1h, reduction=reduction)
    raise ValueError(f"loss không hỗ trợ: {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt + best_state.

    TUYỆT ĐỐI không dùng X_eval trong hàm này (chỉ val), để không chọn cấu hình bằng eval.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    hidden = tuple(cfg["hidden"])
    device = data["X_tr"].device
    dev_type = _device_type(device)
    loss_name, precision = cfg["loss"], cfg.get("precision", "fp32")
    epochs = int(cfg["epochs"])
    assert hidden in EXPECTED_PARAMS, f"kiến trúc {hidden} không có trong EXPECTED_PARAMS"

    # ---- 0. dựng model / optimizer / scaler -------------------------------
    set_seed(int(cfg["seed"]))
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS[hidden], \
        f"số tham số sai: {n_params} (kỳ vọng {EXPECTED_PARAMS[hidden]})"
    opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=float(cfg["lr"]),
                          weight_decay=float(cfg["weight_decay"]),
                          momentum=float(cfg["momentum"]))
    scaler = None
    amp_dtype = None
    if precision != "fp32":
        assert dev_type == "cuda", f"precision={precision} chỉ hỗ trợ trên CUDA (hiện {dev_type})"
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        if precision == "fp16":
            scaler = torch.amp.GradScaler("cuda")
    if dev_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- 1. loss bước 0 (TRƯỚC mọi cập nhật), kỳ vọng ≈ ln 7 ---------------
    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name)["loss"]

    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg["seed"]))
    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc",
                            "val_macro_f1", "grad_norm", "epoch_time_s")}
    best = dict(val_loss=float("inf"), epoch=0, state=None)
    diverged = False

    for epoch in range(1, epochs + 1):
        if dev_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ---- 2a. một epoch SGD -------------------------------------------
        model.train()
        step_norms: list[float] = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]), gen):
            opt.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(dev_type, dtype=amp_dtype):     # chỉ bọc forward + loss
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)                # bắt buộc TRƯỚC khi clip
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                step_norms.append(gn)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                step_norms.append(gn)
                opt.step()

            if not torch.isfinite(loss):
                diverged = True
                print(f"  !! loss = {float(loss)} tại epoch {epoch} -> dừng sớm (diverged)")
                break

        # ---- 2b. cuối epoch: đo ở chế độ eval (dropout TẮT) --------------
        tr_m = evaluate(model, data["X_tr"], data["y_tr"], loss_name)
        va_m = evaluate(model, data["X_val"], data["y_val"], loss_name)
        if dev_type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_m["loss"])
        hist["val_loss"].append(va_m["loss"])
        hist["val_acc"].append(va_m["acc"])
        hist["val_macro_f1"].append(va_m["macro_f1"])
        hist["grad_norm"].append(float(np.mean(step_norms)) if step_norms else float("nan"))
        hist["epoch_time_s"].append(dt)

        if va_m["loss"] < best["val_loss"]:
            best = dict(val_loss=va_m["loss"], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  epoch {epoch:2d}/{epochs}  train {tr_m['loss']:.4f}  val {va_m['loss']:.4f}"
              f"  acc {va_m['acc']:.4f}  macroF1 {va_m['macro_f1']:.4f}"
              f"  grad_norm {hist['grad_norm'][-1]:.3f}  {dt:.1f}s")
        if diverged:
            break

    # ---- 3. tổng hợp summary tại best_epoch -----------------------------
    bi = best["epoch"] - 1 if best["epoch"] > 0 else len(hist["epoch"]) - 1
    peak_mem = (torch.cuda.max_memory_allocated() / 2 ** 20) if dev_type == "cuda" else 0.0
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best["val_loss"],
        best_epoch=best["epoch"],
        final_train_loss=hist["train_loss"][-1] if hist["train_loss"] else float("nan"),
        final_val_loss=hist["val_loss"][-1] if hist["val_loss"] else float("nan"),
        val_acc=hist["val_acc"][bi] if hist["val_acc"] else float("nan"),
        val_macro_f1=hist["val_macro_f1"][bi] if hist["val_macro_f1"] else float("nan"),
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])) if hist["epoch_time_s"] else float("nan"),
        peak_mem_MB=float(peak_mem),
        diverged=bool(diverged),
        n_params=int(n_params),
    )
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best["state"])


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi CSV `row_id,pred` cho scripts/evaluate.py (đủ 116 203 dòng, mỗi row_id một lần)."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds), "row_id và pred phải cùng độ dài"
    assert np.isfinite(preds).all() and preds.min() >= 0 and preds.max() <= 6, "pred phải trong 0..6"
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi CSV.

    Trả về mảng pred (int64) để notebook tự đối chiếu; ĐIỂM CHÍNH THỨC lấy từ scripts/evaluate.py.
    """
    device = data["X_tr"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])          # fp32, eval mode
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    m = evaluate(model, data["X_eval"], data["y_eval"], cfg["loss"])
    print(f"[{cfg['exp_id']}] best_epoch {result['summary']['best_epoch']} | ước lượng nội bộ trên eval: "
          f"acc {m['acc']:.4f}  macroF1 {m['macro_f1']:.4f}  (số chính thức lấy từ scripts/evaluate.py)")
    print(f"đã ghi {pred_path} ({len(preds)} dòng)")
    return preds.cpu().numpy()

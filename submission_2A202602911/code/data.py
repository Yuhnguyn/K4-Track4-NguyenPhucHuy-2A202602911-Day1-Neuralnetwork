"""data.py — nạp dữ liệu, tách validation, chuẩn hoá, đưa lên device.

Quy ước dữ liệu (README mục 2–3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10          # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54         # 10 số + 4 Wilderness_Area + 40 Soil_Type
N_TRAIN, N_EVAL = 464_809, 116_203          # kích thước cố định của repo
VAL_FRACTION_DEFAULT = 0.2


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz do `scripts/split_data.py` tạo ra.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    processed_dir = Path(processed_dir)
    tr = np.load(processed_dir / "train.npz")
    ev = np.load(processed_dir / "eval.npz")

    X_train_full, y_train_full = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    # ---- kiểm tra quy ước ở đầu file (shape/dtype) -------------------------
    for name, X, y in (("train", X_train_full, y_train_full), ("eval", X_eval, y_eval)):
        assert X.dtype == np.float32, f"{name}: X phải là float32, hiện là {X.dtype}"
        assert y.dtype == np.int64, f"{name}: y phải là int64, hiện là {y.dtype}"
        assert X.ndim == 2 and X.shape[1] == N_FEATURES, f"{name}: X phải có shape (N, {N_FEATURES})"
        assert y.shape == (X.shape[0],), f"{name}: y phải có shape (N,)"
        assert y.min() == 0 and y.max() == 6, f"{name}: nhãn phải nằm trong 0..6"
    assert X_train_full.shape[0] == N_TRAIN, f"train phải có {N_TRAIN} mẫu"
    assert X_eval.shape[0] == N_EVAL, f"eval phải có {N_EVAL} mẫu"
    assert np.isfinite(X_train_full).all() and np.isfinite(X_eval).all(), "dữ liệu có NaN/inf"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = VAL_FRACTION_DEFAULT, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho MỌI thí nghiệm -> phép so sánh mới công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed, shuffle=True
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))

    Không tính trên val/eval vì như vậy là rò rỉ thông tin: thống kê của tập đánh giá
    (được coi là dữ liệu chưa thấy) chảy vào quá trình tiền xử lý, làm điểm val/eval
    lạc quan giả tạo.
    """
    Xnum = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = Xnum.mean(axis=0)
    std = Xnum.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)        # cột hằng số (nếu có) -> tránh chia 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về BẢN SAO của X: 10 cột đầu -> (x - mean) / std, 44 cột nhị phân giữ nguyên."""
    Xo = X.copy()
    Xo[:, :N_NUMERIC] = (Xo[:, :N_NUMERIC] - mean) / std
    return Xo


def prepare_data(device: str, val_fraction: float = VAL_FRACTION_DEFAULT, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict: X_tr, y_tr, X_val, y_val, X_eval, y_eval (tensor trên device) và eval_row_id (numpy).
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)

    # 1) tách validation từ train; 2) chuẩn hoá CHỈ bằng thống kê của phần train còn lại
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # 3) lên device một lần; 4) in thông tin kiểm tra của Part 0
    to_t = lambda A, dt: torch.as_tensor(A, dtype=dt, device=device)
    data = dict(
        X_tr=to_t(X_tr, torch.float32), y_tr=to_t(y_tr, torch.int64),
        X_val=to_t(X_val, torch.float32), y_val=to_t(y_val, torch.int64),
        X_eval=to_t(X_eval, torch.float32), y_eval=to_t(y_eval, torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )

    print(f"device            : {device}")
    print(f"X_tr  {tuple(data['X_tr'].shape)}  X_val {tuple(data['X_val'].shape)}  "
          f"X_eval {tuple(data['X_eval'].shape)}")
    maj = int(torch.bincount(data["y_val"], minlength=7).argmax())
    acc_majority = float((data["y_val"] == maj).float().mean())
    print(f"lớp đa số trên val: lớp {maj} -> accuracy \"luôn đoán lớp đa số\" = {acc_majority:.4f}"
          f"  (mốc thấp nhất phải vượt)")
    print(f"10 cột số của X_tr: mean ≈ {data['X_tr'][:, :N_NUMERIC].mean(0).abs().max():.2e}, "
          f"std ≈ {data['X_tr'][:, :N_NUMERIC].std(0).mean():.4f}   (kỳ vọng 0 / 1)")
    print("tỉ lệ lớp val :", np.round(torch.bincount(data["y_val"], minlength=7).cpu().numpy()
                                      / len(data["y_val"]), 4))
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối có thể nhỏ hơn batch_size: ta GIỮ LẠI và dùng luôn (không bỏ), để mọi epoch
    đều dùng hết dữ liệu và số bước cập nhật xác định = ceil(N / batch_size).
    """
    N = X.shape[0]
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]

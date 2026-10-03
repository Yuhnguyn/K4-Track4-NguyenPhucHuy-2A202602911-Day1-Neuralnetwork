"""model.py — MLP cho bài toán 7 lớp, shape cố định (README mục 3, GUIDE "Quy định kiến trúc").

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}

INIT_CHOICES = ("zeros", "normal", "xavier", "he", "default")


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        if init not in INIT_CHOICES:
            raise ValueError(f"init phải thuộc {INIT_CHOICES}, nhận {init!r}")
        self.hidden = tuple(hidden)
        self.dropout_q = float(dropout)
        self.init_name = init

        layers: list[nn.Module] = []
        prev = in_features
        for h in self.hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:                     # chỉ sau ReLU của lớp ẩn
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))     # lớp ra: logit thô
        self.net = nn.Sequential(*layers)

        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear.

    Với 4 cách khởi tạo tường minh, bias luôn được đặt = 0 (như slide Chương 4).
    "default" = giữ nguyên khởi tạo mặc định của nn.Linear cho W lớp ẩn
    (uniform ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) — KHÔNG phải He).

    Lớp RA luôn dùng khởi tạo nhỏ N(0, 0.01) + bias 0 (GUIDE, mục Chẩn đoán) để logits ≈ 0 ở bước 0;
    nhờ vậy loss bước 0 ≈ ln 7 = 1,946 cho mọi cách khởi tạo và phép so sánh ở chủ đề `init` chỉ
    khác nhau ở các lớp ẩn. Mọi bias đều = 0. Số tham số không đổi (47 879).
    """
    if init not in INIT_CHOICES:
        raise ValueError(f"init phải thuộc {INIT_CHOICES}, nhận {init!r}")

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        if i == len(linears) - 1:
            # LỚP RA: khởi tạo nhỏ N(0, 0.01^2) + bias 0 để logits ≈ 0 ở bước 0 -> loss bước 0 ≈ ln 7.
            # Đây là khuyến nghị của GUIDE (mục "Chẩn đoán"): "Loss bước 0 cao hơn ln 7 nhiều -> ...
            # Giảm tỉ lệ W lớp cuối, đặt bias = 0". Áp dụng cho MỌI cách khởi tạo để phép so sánh
            # ở chủ đề `init` chỉ khác nhau ở các LỚP ẨN.
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)
            continue
        # LỚP ẨN: theo đúng cách khởi tạo được chọn
        if init == "default":
            pass                                  # giữ khởi tạo mặc định của nn.Linear (KHÔNG phải He)
        elif init == "zeros":
            nn.init.zeros_(m.weight)
        elif init == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.01)   # Var = 1e-4
        elif init == "xavier":
            # xavier_normal_: Var = 2 / (n_in + n_out)
            nn.init.xavier_normal_(m.weight)
        elif init == "he":
            # kaiming_normal_ với nonlinearity="relu": Var = 2 / n_in  -> đúng He
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt SAU MỖI ReLU của lớp ẩn (bước 0, một lô val).

    Dùng cho thí nghiệm khởi tạo: so với biểu đồ "30 lớp ReLU" của slide (Chương 4).
    """
    model.eval()
    h = x
    stds: list[float] = []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):          # đo sau ReLU, trước dropout
            stds.append(float(h.std().item()))
    return stds

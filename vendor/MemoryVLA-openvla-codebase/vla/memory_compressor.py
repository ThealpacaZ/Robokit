from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MemoryCompressor(nn.Module):
    """轻量级 Transformer 压缩器，将输入序列 S_compress 压缩为 target_len 个 key token。

    Asymmetric attention mask 规则：
      - S_compress token 之间互相 attend
      - S_target token 可以 attend 到所有 S_compress token
      - S_compress token 不能 attend 到 S_target token（单向信息流）
      - S_target token 之间互相 attend

    用途：
      1. value -> key：K = compressor(V)
      2. query -> query_key：Kq = compressor(Q)
    """

    def __init__(
        self,
        d_model: int,
        target_len: int,
        nhead: int = 4,
        num_layers: int = 2,
        d_key: int = 64,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.target_len = target_len
        self.pool = pool

        # 可学习目标 token
        self.target_tokens = nn.Parameter(torch.randn(1, target_len, d_model) * 0.02)

        # Transformer encoder（Pre-LN, batch_first）
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        # 投影到 key 空间
        self.key_proj = nn.Linear(d_model, d_key)

    def _build_mask(self, x_len: int, device: torch.device) -> Tensor:
        """构造 additive attention mask [x_len+target_len, x_len+target_len]。
        S_compress（前 x_len 行）不能 attend 到 S_target（后 target_len 列）。
        """
        total = x_len + self.target_len
        mask = torch.zeros(total, total, device=device)
        mask[:x_len, x_len:] = -1e4
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, x_len, d_model]
        returns: [B, d_key] if pool=True, else [B, target_len, d_key]
        """
        B, x_len, _ = x.shape
        target = self.target_tokens.expand(B, -1, -1).to(x.dtype)  # [B, target_len, d_model]
        seq = torch.cat([x, target], dim=1)                         # [B, x_len+target_len, d_model]

        attn_mask = self._build_mask(x_len, x.device).to(x.dtype)
        out = self.encoder(seq, mask=attn_mask)                     # [B, x_len+target_len, d_model]

        compressed = out[:, x_len:, :]                             # [B, target_len, d_model]
        keys = self.key_proj(compressed)                            # [B, target_len, d_key]

        if self.pool:
            return keys.mean(dim=1)                                 # [B, d_key]
        return keys

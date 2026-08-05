from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MemoryRetriever(nn.Module):
    """Global Workspace inspired memory bottleneck.

    Distills bank-retrieved sensory and cognitive features into compact memory tokens
    via self-attention, using the full VLM hidden states (visual + instruction tokens)
    as query context, then produces P_context for injection into DiT via
    the existing per_token cross-attention interface.

    Architecture:
        fh_proj = Linear(full_hidden)                    [B, num_patch+text_len, d_model]

        Sensory branch:
            seq = [16 learnable tokens; per_bank_tokens (T*N); fh_proj]
            out = TransformerEncoder(seq, src_key_padding_mask)
            sensory_mem = out[:, :16, :]                         [B, 16, d_model]

        Cognitive branch:
            seq = [8 learnable tokens; cog_bank_tokens (T); fh_proj]
            out = TransformerEncoder(seq, src_key_padding_mask)
            cognitive_mem = out[:, :8, :]                        [B, 8, d_model]

        P_context = Concat(sensory_mem, cognitive_mem)           [B, 24, d_model]
    """

    def __init__(
        self,
        vlm_hidden_size: int = 4096,
        per_token_size: int = 256,
        d_model: int = 256,
        num_sensory_tokens: int = 16,
        num_cognitive_tokens: int = 8,
        nhead: int = 8,
        num_sa_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_sensory_tokens = num_sensory_tokens
        self.num_cognitive_tokens = num_cognitive_tokens

        # Project full hidden states to d_model
        self.full_hidden_proj = nn.Linear(vlm_hidden_size, d_model)

        # Sensory branch input projection: per_token_size -> d_model
        self.per_proj = nn.Linear(per_token_size, d_model)

        # Cognitive branch input projection: vlm_hidden_size -> d_model
        self.cog_proj = nn.Linear(vlm_hidden_size, d_model)

        # Learnable memory tokens (batch dim = 1, expanded at runtime)
        self.sensory_mem_tokens = nn.Parameter(
            torch.randn(1, num_sensory_tokens, d_model) * 0.02
        )
        self.cognitive_mem_tokens = nn.Parameter(
            torch.randn(1, num_cognitive_tokens, d_model) * 0.02
        )

        # Sensory self-attention (Pre-LN, batch_first)
        self.per_sa = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_sa_layers,
        )

        # Cognitive self-attention (Pre-LN, batch_first)
        self.cog_sa = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_sa_layers,
        )

    def forward(
        self,
        full_hidden: Tensor,                         # [B, num_patch + text_len, vlm_hidden_size]
        per_bank_tokens: Tensor,                     # [B, L_per, per_token_size]  (padded)
        cog_bank_tokens: Tensor,                     # [B, L_cog, vlm_hidden_size] (padded)
        attention_mask: Tensor,                      # [B, text_len]  1=valid, 0=padding
        num_patch: int,
        per_bank_mask: Optional[Tensor] = None,      # [B, L_per]  True=padding
        cog_bank_mask: Optional[Tensor] = None,      # [B, L_cog]  True=padding
    ) -> Tensor:                                     # P_context [B, num_sensory + num_cognitive, d_model]
        B = full_hidden.shape[0]

        # ── A. Project full hidden states ────────────────────────────────────
        fh_proj = self.full_hidden_proj(full_hidden)                         # [B, num_patch+text_len, d_model]

        # Build padding mask for full_hidden: visual patches all valid, text follows attention_mask
        vis_valid = torch.zeros(B, num_patch, dtype=torch.bool, device=full_hidden.device)  # False=valid
        text_pad  = (attention_mask == 0)                                    # [B, text_len] True=padding
        fh_mask   = torch.cat([vis_valid, text_pad], dim=1)                  # [B, num_patch+text_len]

        # ── B. Sensory memory branch ─────────────────────────────────────────
        sensory_tok = self.sensory_mem_tokens.expand(B, -1, -1)              # [B, 16, d_model]
        per_proj    = self.per_proj(per_bank_tokens)                         # [B, L_per, d_model]
        per_seq     = torch.cat([sensory_tok, per_proj, fh_proj], dim=1)     # [B, 16+L_per+L_fh, d_model]

        if per_bank_mask is not None:
            valid_l      = torch.zeros(B, self.num_sensory_tokens, dtype=torch.bool, device=full_hidden.device)
            per_seq_mask = torch.cat([valid_l, per_bank_mask, fh_mask], dim=1)
        else:
            valid_l      = torch.zeros(B, self.num_sensory_tokens, dtype=torch.bool, device=full_hidden.device)
            per_seq_mask = torch.cat([valid_l, fh_mask], dim=1)

        per_out     = self.per_sa(per_seq, src_key_padding_mask=per_seq_mask)
        sensory_mem = per_out[:, :self.num_sensory_tokens, :]                # [B, 16, d_model]

        # ── C. Cognitive memory branch ───────────────────────────────────────
        cognitive_tok = self.cognitive_mem_tokens.expand(B, -1, -1)          # [B, 8, d_model]
        cog_proj      = self.cog_proj(cog_bank_tokens)                       # [B, L_cog, d_model]
        cog_seq       = torch.cat([cognitive_tok, cog_proj, fh_proj], dim=1) # [B, 8+L_cog+L_fh, d_model]

        if cog_bank_mask is not None:
            valid_l      = torch.zeros(B, self.num_cognitive_tokens, dtype=torch.bool, device=full_hidden.device)
            cog_seq_mask = torch.cat([valid_l, cog_bank_mask, fh_mask], dim=1)
        else:
            valid_l      = torch.zeros(B, self.num_cognitive_tokens, dtype=torch.bool, device=full_hidden.device)
            cog_seq_mask = torch.cat([valid_l, fh_mask], dim=1)

        cog_out       = self.cog_sa(cog_seq, src_key_padding_mask=cog_seq_mask)
        cognitive_mem = cog_out[:, :self.num_cognitive_tokens, :]            # [B, 8, d_model]

        # ── D. Concatenate P_context ─────────────────────────────────────────
        return torch.cat([sensory_mem, cognitive_mem], dim=1)                # [B, 24, d_model]

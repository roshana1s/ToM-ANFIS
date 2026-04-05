"""
ANFIS model definition — matches the checkpoint saved during training.

Checkpoint parameter keys:
    mf_layer.centers        – Gaussian MF centers   (in_features, num_mfs)
    mf_layer.sigmas         – Gaussian MF widths    (in_features, num_mfs)
    consequent_weights      – TSK linear params     (num_rules, in_features+1)

5-layer architecture:
  L1  Fuzzification  — Gaussian MFs per feature
  L2  Rule firing    — product T-norm (3^9 = 19,683 rules)
  L3  Normalisation  — w̄ₖ = wₖ / Σwₖ
  L4  TSK consequent — fₖ = pₖ₀ + Σᵢ pₖᵢ xᵢ
  L5  Defuzz+sigmoid — ŷ = σ(Σₖ w̄ₖ fₖ) ∈ [0,1]
"""

from __future__ import annotations

import itertools
from typing import List, Optional

import torch
import torch.nn as nn


class GaussianMF(nn.Module):
    """
    Gaussian membership function layer.

    Holds learnable centers and widths for all features.
    Produces µᵢⱼ(xᵢ) for each (feature i, MF j) pair.
    """

    def __init__(self, in_features: int, num_mfs: int) -> None:
        super().__init__()
        # Centers initialised uniformly over [0, 1]
        self.centers = nn.Parameter(
            torch.linspace(0.0, 1.0, num_mfs).unsqueeze(0).repeat(in_features, 1)
        )
        self.sigmas = nn.Parameter(torch.ones(in_features, num_mfs) * 0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F) → memberships: (B, F, M)"""
        return torch.exp(
            -0.5 * ((x.unsqueeze(-1) - self.centers) / (self.sigmas.abs() + 1e-8)) ** 2
        )


class ANFIS(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System with TSK consequents.

    Sub-module and parameter naming matches the saved checkpoint:
        mf_layer.centers, mf_layer.sigmas, consequent_weights
    """

    #: Human-readable label per MF index (Low=0, Medium=1, High=2)
    MF_LABELS = ["Low", "Medium", "High"]

    def __init__(
        self,
        in_features: int = 9,
        num_mfs: int = 3,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_mfs = num_mfs
        self.num_rules: int = num_mfs ** in_features
        self.feature_names: List[str] = feature_names or [
            f"x{i}" for i in range(in_features)
        ]

        # L1: Gaussian MF sub-module (keys: mf_layer.centers / mf_layer.sigmas)
        self.mf_layer = GaussianMF(in_features, num_mfs)

        # L4: TSK consequents (key: consequent_weights)
        self.consequent_weights = nn.Parameter(
            torch.zeros(self.num_rules, in_features + 1)
        )

        # Precomputed rule → MF-index mapping (buffer, not a learnable param)
        combos = list(itertools.product(range(num_mfs), repeat=in_features))
        self.register_buffer(
            "rule_mf_indices",
            torch.tensor(combos, dtype=torch.long),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ANFIS initialised  |  Rules: {self.num_rules:,}  |  Params: {n_params:,}")

    # ── Internal layers ───────────────────────────────────────────────────────

    def _fire_rules(self, mu: torch.Tensor) -> torch.Tensor:
        """Product T-norm: (B,F,M) → (B, num_rules)."""
        B, R, F = mu.shape[0], self.num_rules, self.in_features
        idx = self.rule_mf_indices.unsqueeze(0).expand(B, R, F)          # (B,R,F)
        gathered = mu.unsqueeze(1).expand(B, R, F, self.num_mfs).gather(
            dim=3, index=idx.unsqueeze(-1)
        ).squeeze(-1)                                                      # (B,R,F)
        return gathered.prod(dim=2)                                        # (B,R)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_features) → ŷ: (B,)"""
        mu = self.mf_layer(x)                                              # (B,F,M)
        w = self._fire_rules(mu)                                           # (B,R)
        w_bar = w / (w.sum(dim=1, keepdim=True) + 1e-10)                  # (B,R)
        x_aug = torch.cat([torch.ones(x.shape[0], 1, device=x.device), x], dim=1)
        f_k = (x_aug.unsqueeze(1) * self.consequent_weights.unsqueeze(0)).sum(dim=2)
        return torch.sigmoid((w_bar * f_k).sum(dim=1))

    # ── Explainability ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_explain(
        self,
        x: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict:
        """
        Run a forward pass and return the prediction together with the
        top-k fired fuzzy rules.

        Parameters
        ----------
        x : (1, in_features) or (in_features,) — a single scaled sample.
        top_k : int — maximum number of rules to return.
        threshold : float — minimum normalised firing strength to include.

        Returns
        -------
        dict with keys:
            predicted_crossing_probability : float
            rules : list[dict]  — each with rule_id, firing_strength,
                                   antecedents, consequent_value
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        mu = self.mf_layer(x)
        w = self._fire_rules(mu)
        w_bar = w / (w.sum(dim=1, keepdim=True) + 1e-10)

        x_aug = torch.cat([torch.ones(1, 1, device=x.device), x], dim=1)
        f_k = (x_aug.unsqueeze(1) * self.consequent_weights.unsqueeze(0)).sum(dim=2)

        y_hat = float(torch.sigmoid((w_bar * f_k).sum(dim=1)).item())

        w_flat = w_bar.squeeze(0)   # (R,)
        f_flat = f_k.squeeze(0)     # (R,)

        # Filter by threshold then keep top-k
        mask = w_flat >= threshold
        candidate_idx = mask.nonzero(as_tuple=True)[0]
        # sort by strength descending
        sorted_idx = candidate_idx[w_flat[candidate_idx].argsort(descending=True)]
        top_idx = sorted_idx[:top_k].tolist()

        rules = []
        for idx in top_idx:
            mf_idx = self.rule_mf_indices[idx].tolist()
            antecedents = [
                f"{name} is {self.MF_LABELS[j]}"
                for name, j in zip(self.feature_names, mf_idx)
            ]
            rules.append(
                {
                    "rule_id": idx,
                    "firing_strength": float(w_flat[idx]),
                    "antecedents": antecedents,
                    "consequent_value": float(f_flat[idx]),
                }
            )

        return {"predicted_crossing_probability": y_hat, "rules": rules}

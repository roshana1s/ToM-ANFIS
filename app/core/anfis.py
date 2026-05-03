import torch
import numpy as np
import torch.nn as nn
import itertools


class GaussianMF(nn.Module):
    """
    Layer 1 — Fuzzification via learnable Gaussian membership functions.

    Each input feature is mapped to M soft membership values representing
    the linguistic terms Low (j=0), Medium (j=1), High (j=2).
    Both the centres (cᵢⱼ) and widths (σᵢⱼ) are learnable parameters.
    """

    def __init__(self, num_mfs: int, in_features: int, feature_names=None):
        super().__init__()
        self.num_mfs = num_mfs
        self.in_features = in_features
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(in_features)]
        self.mf_names = ['Low', 'Medium', 'High'][:num_mfs]

        # Evenly-spaced initial centres across [0, 1]; sigma = 0.3
        centers_init = torch.linspace(0, 1, num_mfs).repeat(in_features, 1)
        self.centers = nn.Parameter(centers_init)              # shape (F, M)
        self.sigmas = nn.Parameter(torch.full((in_features, num_mfs), 0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x  (batch, F)
        Returns: µ (batch, F, M)  — membership values for every feature × MF
        """
        x = x.unsqueeze(2)                          # (B, F, 1)
        centers = self.centers.unsqueeze(0)               # (1, F, M)
        sigmas = torch.abs(self.sigmas.unsqueeze(0)) + 1e-6  # ensure positivity
        return torch.exp(-0.5 * ((x - centers) / sigmas) ** 2)


class ANFIS(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System for pedestrian crossing intention prediction.

    Implements the full 5-layer ANFIS architecture:
      L1 Fuzzification → L2 Rule Firing → L3 Normalisation
        → L4 TSK Consequent → L5 Defuzzification + Sigmoid

    Input  : 9 normalised ToM features  (B1-B4, G2, A1-A2, C1-C2)
    Output : crossing intention probability  ŷ ∈ [0, 1]
    Rules  : 3⁹ = 19,683
    """

    def __init__(self, in_features: int, num_mfs: int = 3, feature_names=None):
        super().__init__()
        self.in_features = in_features
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** in_features

        self.mf_layer = GaussianMF(num_mfs, in_features, feature_names)

        # Precompute all 3⁹ rule antecedent combinations (static, not a parameter)
        self.rule_indices = torch.tensor(
            list(itertools.product(range(num_mfs), repeat=in_features)),
            dtype=torch.long
        )  # shape (K, F)  where K = 19,683

        # TSK consequent: one first-order linear function per rule
        # Shape (K, F+1) — the +1 accounts for the bias term pₖ₀
        self.consequent_weights = nn.Parameter(
            torch.randn(self.num_rules, in_features + 1) * 0.5
        )

        total = sum(p.numel() for p in self.parameters())
        print(f'ANFIS initialised  |  Rules: {self.num_rules:,}  |  Params: {total:,}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        rules = self.rule_indices.to(device)          # (K, F)

        # L1 — Fuzzification
        mu = self.mf_layer(x)                          # (B, F, M)

        # L2 — Rule firing: product T-norm over all features
        w = torch.ones(B, self.num_rules, device=device)
        for i in range(self.in_features):
            w *= mu[:, i, :][:, rules[:, i]]           # broadcast MF selection

        # L3 — Normalisation
        w_bar = w / (w.sum(dim=1, keepdim=True) + 1e-10)

        # L4 — TSK consequents (linear per rule)
        x_aug = torch.cat([x, torch.ones(B, 1, device=device)], dim=1)  # (B, F+1)
        f_k = torch.matmul(x_aug, self.consequent_weights.T)           # (B, K)

        # L5 — Defuzzification + sigmoid
        return torch.sigmoid((w_bar * f_k).sum(dim=1, keepdim=True))    # (B, 1)

    # ── Explainability methods ────────────────────────────────────────────
    def global_rules(self, top_k: int = 10):
        """
        Rank all rules by their average |consequent weight| (a proxy for importance)
        and return the top-k as human-readable IF-THEN strings with TSK bodies.
        """
        importance  = self.consequent_weights.abs().mean(dim=1)          # (K,)
        top_idx = importance.argsort(descending=True)[:top_k]
        rules = []
        fn = self.mf_layer.feature_names
        mn = self.mf_layer.mf_names
        for idx in top_idx:
            r = idx.item()
            ante = ' AND '.join(f'{fn[i]} is {mn[self.rule_indices[r,i].item()]}'
                                 for i in range(self.in_features))
            w = self.consequent_weights[r].detach().cpu().numpy()
            body = f'{w[0]:.3f}' + ''.join(f' + {w[i+1]:.3f}·{fn[i]}'
                                             for i in range(self.in_features))
            rules.append((importance[idx].item(),
                          f'IF {ante}\n   THEN intention = {body}'))
        return rules

    def predict_explain(self, x_sample: np.ndarray, threshold: float = 0.0001):
        """
        Run inference on a single normalised sample and return the prediction
        together with all fired fuzzy rules as natural-language IF-THEN strings.

        Args:
            x_sample  : 1-D numpy array, length == in_features, already scaled
            threshold : Minimum normalised firing strength to include a rule

        Returns:
            dict:
                'probability'   : float ∈ [0, 1]
                'intention_level': 'LOW' | 'MEDIUM' | 'HIGH'
                'fired_rules'   : list of dicts sorted by firing_strength desc
        """
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.FloatTensor(x_sample).unsqueeze(0).to(device)
        fn = self.mf_layer.feature_names
        mn = self.mf_layer.mf_names

        with torch.no_grad():
            pred = self(x_t).item()
            mu = self.mf_layer(x_t)
            rules = self.rule_indices.to(device)

            w = torch.ones(1, self.num_rules, device=device)
            for i in range(self.in_features):
                w *= mu[:, i, :][:, rules[:, i]]
            w_bar = (w / (w.sum(dim=1, keepdim=True) + 1e-10)).squeeze(0)

            x_aug = torch.cat([x_t, torch.ones(1, 1, device=device)], dim=1)
            f_k = torch.matmul(x_aug, self.consequent_weights.T).squeeze(0)

        level = 'LOW' if pred < 0.33 else ('MEDIUM' if pred < 0.67 else 'HIGH')

        fired = []
        for k in range(self.num_rules):
            fs = w_bar[k].item()
            if fs > threshold:
                ante = ' AND '.join(f'{fn[i]} is {mn[rules[k,i].item()]}'
                                     for i in range(self.in_features))
                cv = torch.sigmoid(f_k[k]).item()
                clv = 'LOW' if cv < 0.33 else ('MEDIUM' if cv < 0.67 else 'HIGH')
                fired.append({
                    'rule_id': k, 'firing_strength': fs,
                    'rule': f'IF {ante} THEN crossing intention is {clv}',
                    'consequent_value': cv, 'consequent_linguistic': clv,
                })

        fired.sort(key=lambda r: r['firing_strength'], reverse=True)
        return {'probability': pred, 'intention_level': level, 'fired_rules': fired}
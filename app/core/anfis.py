import torch
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


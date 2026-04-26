"""
Prediction service.

Imports the loader singleton and calls loader.get_model() / loader.get_scaler().
Never touches file paths or torch directly.
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.core.loader import loader
from app.schemas.processed_features import ToMFeaturesInput
from app.schemas.output import FiredRule, PredictionResponse

import torch

_FEATURE_ORDER = [
    "B1_age_vulnerability_awareness",
    "B2_intersection_complexity",
    "B3_pedestrian_position_proximity",
    "B4_vehicle_density",
    "G2_urgency_rushing",
    "A1_vehicle_awareness",
    "A2_traffic_signal_awareness",
    "C1_relative_vehicle_speed",
    "C2_occlusion_factor",
]


def predict(features: ToMFeaturesInput) -> PredictionResponse:
    """
    Accept a ToMFeaturesInput of 9 named ToM feature values, apply the
    MinMaxScaler, run ANFIS inference via model.predict_explain(), and return a
    PredictionResponse containing the crossing probability, intention level,
    and the top TOP_K_RULES fired fuzzy rules.
    """
    model = loader.get_model()
    scaler = loader.get_scaler()
    device = next(model.parameters()).device

    # 1. Flatten nested features
    flat_data = {}
    flat_data.update(features.belief_state.model_dump())
    flat_data.update(features.goal_state.model_dump())
    flat_data.update(features.attention_state.model_dump())
    flat_data.update(features.context_environment.model_dump())

    # 2. Build ordered numpy array
    raw = np.array(
        [flat_data[name] for name in _FEATURE_ORDER], dtype=np.float32
    ).reshape(1, -1)

    # 2. Scale (MinMaxScaler fitted during training)
    scaled = scaler.transform(raw).astype(np.float32)

    # 3. Call predict_explain() on the ANFIS model
    x = torch.from_numpy(scaled).to(device)
    result = model.predict_explain(
        x,
        top_k=settings.TOP_K_RULES,
        threshold=settings.RULE_FIRING_THRESHOLD,
    )

    prob: float = float(np.clip(result["predicted_crossing_probability"], 0.0, 1.0))

    # 4. Derive human-readable labels
    if prob < 0.33:
        intention = "LOW"
    elif prob < 0.66:
        intention = "MEDIUM"
    else:
        intention = "HIGH"

    # 5. Build FiredRule list
    fired_rules = []
    for r in result["rules"]:
        antecedents_str = " AND ".join(r["antecedents"])
        rule_str = f"IF {antecedents_str} THEN crossing ≈ {r['consequent_value']:.4f}"
        cv = r["consequent_value"]
        if cv < 0.33:
            consequent_ling = "LOW"
        elif cv < 0.66:
            consequent_ling = "MEDIUM"
        else:
            consequent_ling = "HIGH"

        fired_rules.append(
            FiredRule(
                rule_id=r["rule_id"],
                firing_strength=r["firing_strength"],
                rule=rule_str,
                consequent_linguistic=consequent_ling,
            )
        )

    return PredictionResponse(
        predicted_crossing_probability=prob,
        intention_level=intention,
        top_fired_rules=fired_rules,
    )

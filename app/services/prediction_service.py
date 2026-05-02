import torch
import numpy as np

from app.config import settings
from app.core.loader import loader
from app.schemas.input import ToMFeaturesInput
from app.schemas.output import PredictionResponse, FiredRule


def predict(features: ToMFeaturesInput) -> PredictionResponse:
    """
    Run ANFIS inference on the provided ToM features.

    Args:
        features : ToMFeaturesInput with 9 normalised ToM feature values

    Returns:
        PredictionResponse with crossing probability, intention level, and top-k rules
    """

    # Extract feature vector in the expected order
    feature_vector = np.array([
        features.B1_age_vulnerability_awareness,
        features.B2_intersection_complexity,
        features.B3_pedestrian_position_proximity,
        features.B4_vehicle_density,
        features.G2_urgency_rushing,
        features.A1_vehicle_awareness,
        features.A2_traffic_signal_awareness,
        features.C1_relative_vehicle_speed,
        features.C2_occlusion_factor,
    ]).reshape(1, -1)

    # Get model and scaler from loader
    model = loader.get_model()
    scaler = loader.get_scaler()

    # Apply MinMaxScaler normalisation
    x_scaled = scaler.transform(feature_vector)

    # Convert to torch tensor and run inference with explanation
    x_tensor = torch.FloatTensor(x_scaled)
    result = model.predict_explain(x_tensor, threshold=settings.RULE_FIRING_THRESHOLD)

    # Extract results
    probability = result['probability']
    intention_level = result['intention_level']
    fired_rules = result['fired_rules']

    # Sort by firing strength and take top-k
    top_rules = fired_rules[:settings.TOP_K_RULES]

    # Convert to FiredRule schema objects
    fired_rule_objects = [
        FiredRule(
            rule_id=rule['rule_id'],
            firing_strength=rule['firing_strength'],
            rule=rule['rule'],
            consequent_linguistic=rule['consequent_linguistic'],
        )
        for rule in top_rules
    ]

    return PredictionResponse(
        predicted_crossing_probability=probability,
        intention_level=intention_level,
        top_fired_rules=fired_rule_objects,
    )

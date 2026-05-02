from pydantic import BaseModel
from typing import List


class FiredRule(BaseModel):
    """
    Represents a single fuzzy rule that fired during inference.
    """

    rule_id: int
    """
    Index of the rule in the ANFIS rule base (0 to num_rules-1).
    """

    firing_strength: float
    """
    Normalised firing strength of this rule, ∈ [0, 1].
    Rules with firing_strength < RULE_FIRING_THRESHOLD are filtered out.
    """

    rule: str
    """
    Human-readable IF-THEN rule description.
    Example: "IF B1 is LOW AND B2 is MEDIUM THEN crossing intention is HIGH"
    """

    consequent_linguistic: str
    """
    Linguistic label of the rule's consequent (output).
    One of: 'LOW', 'MEDIUM', 'HIGH'
    """


class PredictionResponse(BaseModel):
    """
    Response from either prediction endpoint.
    Contains the crossing probability, intention level, and explainable rules.
    """

    predicted_crossing_probability: float
    """
    Predicted crossing intention probability, ∈ [0, 1].
    Directly output from the ANFIS model's sigmoid activation.
    """

    intention_level: str
    """
    Categorical interpretation of crossing intention:
      - 'LOW'    : probability < 0.33
      - 'MEDIUM' : 0.33 ≤ probability < 0.67
      - 'HIGH'   : probability ≥ 0.67
    """

    top_fired_rules: List[FiredRule]
    """
    Top-K rules that fired (sorted by firing strength, descending).
    K is defined by TOP_K_RULES in config.py (default: 3).
    """

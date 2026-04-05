from pydantic import BaseModel
from typing import List


class FiredRule(BaseModel):
    rule_id: int
    firing_strength: float
    rule: str                   # full IF-THEN natural language string
    consequent_linguistic: str  # LOW / MEDIUM / HIGH


class PredictionResponse(BaseModel):
    predicted_crossing_probability: float
    intention_level: str        # LOW / MEDIUM / HIGH
    top_fired_rules: List[FiredRule]

from pydantic import BaseModel, Field


class BeliefStateFeatures(BaseModel):
    B1_age_vulnerability_awareness:   float = Field(..., ge=0.0, le=1.0)
    B2_intersection_complexity:       float = Field(..., ge=0.0, le=1.0)
    B3_pedestrian_position_proximity: float = Field(..., ge=0.0, le=1.0)
    B4_vehicle_density:               float = Field(..., ge=0.0, le=1.0)


class GoalStateFeatures(BaseModel):
    G2_urgency_rushing:               float = Field(..., ge=0.0, le=1.0)


class AttentionStateFeatures(BaseModel):
    A1_vehicle_awareness:             float = Field(..., ge=0.0, le=1.0)
    A2_traffic_signal_awareness:      float = Field(..., ge=-1.0, le=1.0)


class ContextEnvironmentFeatures(BaseModel):
    C1_relative_vehicle_speed:        float = Field(..., ge=0.0, le=1.0)
    C2_occlusion_factor:              float = Field(..., ge=0.0, le=1.0)


class ToMFeaturesInput(BaseModel):
    belief_state: BeliefStateFeatures
    goal_state: GoalStateFeatures
    attention_state: AttentionStateFeatures
    context_environment: ContextEnvironmentFeatures

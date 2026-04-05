from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ── TrafficObject ─────────────────────────────────────────────────────────────

class TrafficObject(BaseModel):
    obj_class: str                             # 'vehicle', 'crosswalk', 'traffic_light'
    obj_type: Optional[str] = None             # traffic_light only: 'pedestrian','regular','transit'
    frames: List[int]
    bbox: Optional[List[List[float]]] = None   # [[x1,y1,x2,y2],...] per frame
    state: Optional[List[int]] = None          # per frame: 0=undefined,1=red,2=yellow,3=green


# ── BehaviorAnnotation ────────────────────────────────────────────────────────

class BehaviorAnnotation(BaseModel):
    action: List[int]    # per frame: 0=standing, 1=walking
    gesture: List[int]   # per frame: 0=none,1=hand_ack,2=hand_yield,3=hand_row,4=nod
    look: List[int]      # per frame: 0=not looking at vehicle, 1=looking


# ── RawAnnotationInput ────────────────────────────────────────────────────────

class RawAnnotationInput(BaseModel):
    age: int                           # 0=child,1=young,2=adult,3=senior
    intersection: int                  # 0=midblock,1=T,2=T-left,3=T-right,4=four-way
    frames: List[int]
    bbox: List[List[float]]            # [[x1,y1,x2,y2],...] per frame
    occlusion: List[int]               # per frame: 0=visible,1=partial,2=full
    behavior: BehaviorAnnotation
    traffic_objects: List[TrafficObject]
    obd_speed: Dict[int, float]        # frame_id -> speed in m/s


# ── ToMFeaturesInput ──────────────────────────────────────────────────────────

class ToMFeaturesInput(BaseModel):
    B1_age_vulnerability_awareness:   float = Field(..., ge=0.0, le=1.0)
    B2_intersection_complexity:       float = Field(..., ge=0.0, le=1.0)
    B3_pedestrian_position_proximity: float = Field(..., ge=0.0, le=1.0)
    B4_vehicle_density:               float = Field(..., ge=0.0, le=1.0)
    G2_urgency_rushing:               float = Field(..., ge=0.0, le=1.0)
    A1_vehicle_awareness:             float = Field(..., ge=0.0, le=1.0)
    A2_traffic_signal_awareness:      float = Field(..., ge=-1.0, le=1.0)
    C1_relative_vehicle_speed:        float = Field(..., ge=0.0, le=1.0)
    C2_occlusion_factor:              float = Field(..., ge=0.0, le=1.0)

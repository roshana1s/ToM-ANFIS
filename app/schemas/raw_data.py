from pydantic import BaseModel
from typing import List, Optional, Dict


class TrafficObject(BaseModel):
    obj_class: str                             # 'vehicle', 'crosswalk', 'traffic_light'
    obj_type: Optional[str] = None             # traffic_light only: 'pedestrian','regular','transit'
    frames: List[int]
    bbox: Optional[List[List[float]]] = None   # [[x1,y1,x2,y2],...] per frame
    state: Optional[List[int]] = None          # per frame: 0=undefined,1=red,2=yellow,3=green


class BehaviorAnnotation(BaseModel):
    action: List[int]    # per frame: 0=standing, 1=walking
    gesture: List[int]   # per frame: 0=none,1=hand_ack,2=hand_yield,3=hand_row,4=nod
    look: List[int]      # per frame: 0=not looking at vehicle, 1=looking


class PedestrianSchema(BaseModel):
    age: int                           # 0=child,1=young,2=adult,3=senior
    bbox: List[List[float]]            # [[x1,y1,x2,y2],...] per frame
    occlusion: List[int]               # per frame: 0=visible,1=partial,2=full
    behavior: BehaviorAnnotation


class EnvironmentSchema(BaseModel):
    intersection: int                  # 0=midblock,1=T,2=T-left,3=T-right,4=four-way
    traffic_objects: List[TrafficObject]


class VehicleSchema(BaseModel):
    obd_speed: Dict[int, float]        # frame_id -> speed in m/s


class RawAnnotationInput(BaseModel):
    frames: List[int]
    pedestrian: PedestrianSchema
    environment: EnvironmentSchema
    vehicle: VehicleSchema

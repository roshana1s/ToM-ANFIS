from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ── Raw Annotation Input Schemas ──────────────────────────────────────


class TrafficObject(BaseModel):
    """
    Represents a single traffic object (vehicle, crosswalk, traffic light)
    annotated in the PIE dataset.
    """

    obj_class: str
    """
    Type of traffic object: 'vehicle', 'crosswalk', 'traffic_light'
    """

    obj_type: Optional[str] = None
    """
    Subtype (for traffic lights only): 'pedestrian', 'regular', 'transit'
    """

    frames: List[int]
    """
    List of frame IDs where this object appears.
    """

    bbox: Optional[List[List[float]]] = None
    """
    Bounding boxes [x1, y1, x2, y2] for each frame in self.frames.
    If None, no spatial annotation is provided.
    """

    state: Optional[List[int]] = None
    """
    State per frame (for traffic lights):
      0 = undefined
      1 = red
      2 = yellow
      3 = green
    """


class BehaviorAnnotation(BaseModel):
    """
    Represents the pedestrian's behaviour annotations per frame.
    """

    action: List[int]
    """
    Action per frame:
      0 = standing
      1 = walking
    """

    gesture: List[int]
    """
    Hand gesture per frame:
      0 = none
      1 = hand_ack (hand acknowledgement)
      2 = hand_yield (hand yielding to vehicle)
      3 = hand_row (hand requesting right-of-way)
      4 = nod (nodding gesture)
    """

    look: List[int]
    """
    Gaze direction per frame:
      0 = not looking at vehicle
      1 = looking at vehicle
    """


class RawAnnotationInput(BaseModel):
    """
    Raw pedestrian annotations from the PIE dataset.
    Contains complete multimodal observations for feature extraction.
    """

    age: int
    """
    Pedestrian age category:
      0 = child
      1 = young adult
      2 = middle-aged adult
      3 = senior
    """

    intersection: int
    """
    Intersection type:
      0 = midblock (not at intersection)
      1 = T-intersection
      2 = T-intersection with left turn
      3 = T-intersection with right turn
      4 = four-way intersection
    """

    frames: List[int]
    """
    List of frame IDs in the observation window.
    """

    bbox: List[List[float]]
    """
    Pedestrian bounding boxes [[x1,y1,x2,y2], ...] per frame.
    """

    occlusion: List[int]
    """
    Occlusion level per frame:
      0 = fully visible
      1 = partially occluded
      2 = fully occluded
    """

    behavior: BehaviorAnnotation
    """
    Behaviour annotations (action, gesture, look) per frame.
    """

    traffic_objects: List[TrafficObject]
    """
    List of traffic objects (vehicles, crosswalks, lights) in the scene.
    """

    obd_speed: Dict[int, float]
    """
    Ego-vehicle speed (m/s) per frame ID.
    Mapping from frame_id → speed.
    """

    # Optional runtime inference controls
    prediction_horizon: Optional[int] = None
    """
    Frames before the crossing event to predict for. If omitted, the
    service will use the configured default (see app.config). Semantics:
      - `0` means the prediction frame is the last frame in `frames` (predict now).
      - `30` means predict the state 30 frames before the crossing event (matches training).
    """

    observation_window: Optional[int] = None
    """
    Number of frames to aggregate temporal features over. If omitted,
    the service will use the configured default (`OBSERVATION_WINDOW`) to
    ensure runtime matches training (typically 15).
    """


# ── Direct ToM Features Input ─────────────────────────────────────────


class ToMFeaturesInput(BaseModel):
    """
    Pre-extracted 9-dimensional Theory of Mind feature vector.
    Each feature is normalised to [0, 1] (or [-1, 1] for A2).

    This schema is used by the /predict/from-tom-features endpoint,
    allowing clients to skip feature extraction and provide features directly.
    """

    B1_age_vulnerability_awareness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Belief: Age-based vulnerability awareness ∈ [0, 1]",
    )

    B2_intersection_complexity: float = Field(
        ..., ge=0.0, le=1.0, description="Belief: Intersection complexity ∈ [0, 1]"
    )

    B3_pedestrian_position_proximity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Belief: Pedestrian proximity to crosswalk ∈ [0, 1]",
    )

    B4_vehicle_density: float = Field(
        ..., ge=0.0, le=1.0, description="Belief: Density of vehicles in scene ∈ [0, 1]"
    )

    G2_urgency_rushing: float = Field(
        ..., ge=0.0, le=1.0, description="Goal: Urgency / rushing behaviour ∈ [0, 1]"
    )

    A1_vehicle_awareness: float = Field(
        ..., ge=0.0, le=1.0, description="Attention: Awareness of vehicles ∈ [0, 1]"
    )

    A2_traffic_signal_awareness: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Attention: Awareness of traffic signals ∈ [-1, 1]",
    )

    C1_relative_vehicle_speed: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context: Relative speed of approaching vehicle ∈ [0, 1]",
    )

    C2_occlusion_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context: Occlusion / visibility factor ∈ [0, 1]",
    )

from typing import Dict

from app.core.extractor import ToMFeatureExtractor
from app.schemas.input import RawAnnotationInput
from app.config import settings


def extract_features(payload: RawAnnotationInput) -> Dict[str, float]:
    """Convert raw PIE annotations into ToM features compatible with the
    `ToMFeatureExtractor` core API.

    This builds a minimal in-memory `db` structure expected by the extractor
    and returns the 9 input features with the exact key names used by the
    extractor and `ToMFeaturesInput` schema.
    """

    # Build pedestrian annotation as expected by the extractor
    pedestrian_dict = {
        "attributes": {
            "age": payload.age,
            "intersection": payload.intersection,
        },
        "frames": payload.frames,
        "bbox": payload.bbox,
        "occlusion": payload.occlusion,
        "behavior": {
            "action": payload.behavior.action,
            "gesture": payload.behavior.gesture,
            "look": payload.behavior.look,
        },
    }

    # Convert list of TrafficObject to the extractor's traffic_annotations mapping
    traffic_annotations = {
        f"traffic_obj_{i}": {
            "obj_class": obj.obj_class,
            "obj_type": obj.obj_type,
            "frames": obj.frames,
            "bbox": obj.bbox or [],
            "state": obj.state or [],
        }
        for i, obj in enumerate(payload.traffic_objects)
    }

    # Convert obd_speed mapping to the extractor's vehicle_annotations format
    # expected by _C1 (frame_id -> {'OBD_speed': value})
    vehicle_annotations = {
        int(fid): {"OBD_speed": float(speed)}
        for fid, speed in (payload.obd_speed or {}).items()
    }

    # Minimal db structure matching core/extractor expectations
    stub_db = {
        "set_stub": {
            "video_stub": {
                "ped_annotations": {"pedestrian_1": pedestrian_dict},
                "traffic_annotations": traffic_annotations,
                "vehicle_annotations": vehicle_annotations,
            }
        }
    }

    # Determine observation window: prefer explicit payload value, else use
    # the configured default to ensure runtime matches training.
    obs_win = (
        payload.observation_window
        if payload.observation_window is not None
        else settings.OBSERVATION_WINDOW
    )

    # If client supplied an observation_window, require it match the configured value
    if (
        payload.observation_window is not None
        and payload.observation_window != settings.OBSERVATION_WINDOW
    ):
        raise ValueError(
            f"observation_window must equal configured value {settings.OBSERVATION_WINDOW}"
        )

    # Determine prediction horizon: use client value when provided, otherwise use default
    pred_horizon = (
        int(payload.prediction_horizon)
        if getattr(payload, "prediction_horizon", None) is not None
        else settings.DEFAULT_PREDICTION_HORIZON
    )
    if pred_horizon < 0:
        raise ValueError("prediction_horizon must be non-negative")

    # Ensure the provided frames cover the full horizon + observation window (so extraction is not truncated)
    if len(payload.frames) < (obs_win + pred_horizon):
        raise ValueError(
            f"payload.frames must contain at least observation_window + prediction_horizon ({obs_win + pred_horizon}) frames"
        )

    extractor = ToMFeatureExtractor(db=stub_db, observation_window=obs_win)

    features_dict = extractor.extract_for_pedestrian(
        "set_stub", "video_stub", "pedestrian_1", prediction_horizon=pred_horizon
    )

    if not features_dict:
        raise ValueError("Feature extraction failed for provided annotations")

    # Map and return the nine ToM input features using the extractor's key names
    return {
        "B1_age_vulnerability_awareness": features_dict.get(
            "B1_age_vulnerability_awareness", 0.0
        ),
        "B2_intersection_complexity": features_dict.get(
            "B2_intersection_complexity", 0.0
        ),
        "B3_pedestrian_position_proximity": features_dict.get(
            "B3_pedestrian_position_proximity", 0.0
        ),
        "B4_vehicle_density": features_dict.get("B4_vehicle_density", 0.0),
        "G2_urgency_rushing": features_dict.get("G2_urgency_rushing", 0.0),
        "A1_vehicle_awareness": features_dict.get("A1_vehicle_awareness", 0.0),
        "A2_traffic_signal_awareness": features_dict.get(
            "A2_traffic_signal_awareness", 0.0
        ),
        "C1_relative_vehicle_speed": features_dict.get(
            "C1_relative_vehicle_speed", 0.0
        ),
        "C2_occlusion_factor": features_dict.get("C2_occlusion_factor", 0.0),
    }

from typing import Dict

from app.core.extractor import ToMFeatureExtractor
from app.schemas.input import RawAnnotationInput
from app.schemas.output import PredictionResponse


def extract_features(payload: RawAnnotationInput) -> Dict[str, float]:
    """
    Convert raw PIE annotations into ToM features.

    Args:
        payload : RawAnnotationInput with complete multimodal pedestrian data

    Returns:
        Dictionary with 9 ToM feature values: {
            'B1_age_vulnerability_awareness': float,
            'B2_intersection_complexity': float,
            'B3_pedestrian_position_proximity': float,
            'B4_vehicle_density': float,
            'G2_urgency_rushing': float,
            'A1_vehicle_awareness': float,
            'A2_traffic_signal_awareness': float,
            'C1_relative_vehicle_speed': float,
            'C2_occlusion_factor': float,
        }
    """

    # Reconstruct the single-pedestrian dict structure expected by ToMFeatureExtractor
    pedestrian_dict = {
        'attributes': {
            'age': payload.age,
            'intersection': payload.intersection,
        },
        'frames': payload.frames,
        'bbox': payload.bbox,
        'occlusion': payload.occlusion,
        'behavior': {
            'action': payload.behavior.action,
            'gesture': payload.behavior.gesture,
            'look': payload.behavior.look,
        },
        'traffic_objects': {
            f'traffic_obj_{i}': {
                'obj_class': obj.obj_class,
                'obj_type': obj.obj_type,
                'frames': obj.frames,
                'bbox': obj.bbox,
                'state': obj.state,
            }
            for i, obj in enumerate(payload.traffic_objects)
        },
        'vehicle_speed': payload.obd_speed,
    }

    # Use a minimal stub db structure just for this extraction
    stub_db = {
        'set_stub': {
            'video_stub': {
                'ped_annotations': {
                    'pedestrian_1': pedestrian_dict
                }
            }
        }
    }

    # Instantiate the feature extractor
    extractor = ToMFeatureExtractor(db=stub_db, observation_window=len(payload.frames))

    # Extract features for the single pedestrian
    features_dict = extractor.extract_for_pedestrian(
        set_id='set_stub',
        video_id='video_stub',
        pedestrian_id='pedestrian_1',
        prediction_frame=max(payload.frames)  # Use last frame as prediction point
    )

    # Return only the 9 input features (skip metadata and G1 target)
    return {
        'B1_age_vulnerability_awareness': features_dict.get('B1', 0.0),
        'B2_intersection_complexity': features_dict.get('B2', 0.0),
        'B3_pedestrian_position_proximity': features_dict.get('B3', 0.0),
        'B4_vehicle_density': features_dict.get('B4', 0.0),
        'G2_urgency_rushing': features_dict.get('G2', 0.0),
        'A1_vehicle_awareness': features_dict.get('A1', 0.0),
        'A2_traffic_signal_awareness': features_dict.get('A2', 0.0),
        'C1_relative_vehicle_speed': features_dict.get('C1', 0.0),
        'C2_occlusion_factor': features_dict.get('C2', 0.0),
    }

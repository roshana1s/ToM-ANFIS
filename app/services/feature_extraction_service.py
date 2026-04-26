"""
Feature extraction service.

Wraps ToMFeatureExtractor for single-pedestrian use.
Converts a RawAnnotationInput payload into the minimal dict structure
expected by ToMFeatureExtractor.extract_for_pedestrian(), then returns
the resulting 9 named ToM feature values as a dict.
"""

from __future__ import annotations

from typing import Dict, List

from app.core.extractor import ToMFeatureExtractor
from app.schemas.raw_data import RawAnnotationInput, TrafficObject

_extractor = ToMFeatureExtractor()


def extract_features(payload: RawAnnotationInput) -> Dict[str, float]:
    """
    Accept a RawAnnotationInput payload and reconstruct the minimal
    single-pedestrian dict structure expected by ToMFeatureExtractor.
    Call extract_for_pedestrian() and return the resulting 9 named
    ToM feature values as a dict ready to pass into predict().
    """
    ped = _build_ped_dict(payload)
    return _extractor.extract_for_pedestrian(ped)


# ── Private builder 

def _build_ped_dict(payload: RawAnnotationInput) -> dict:
    """
    Convert RawAnnotationInput into the internal dict format consumed by
    ToMFeatureExtractor.extract_for_pedestrian().
    """
    frame_ids = payload.frames

    # ── OBD speed per frame 
    obd_speed: List[float] = [
        payload.vehicle.obd_speed.get(fid, 0.0) for fid in frame_ids
    ]

    # ── Crosswalk bboxes per frame 
    crosswalk_objects = [
        obj for obj in payload.environment.traffic_objects if obj.obj_class == "crosswalk"
    ]
    crosswalk_bboxes: List[List[List[float]]] = _per_frame_bboxes(
        frame_ids, crosswalk_objects
    )

    # ── Vehicle count per frame 
    vehicle_objects = [
        obj for obj in payload.environment.traffic_objects if obj.obj_class == "vehicle"
    ]
    vehicle_counts: List[int] = _per_frame_counts(frame_ids, vehicle_objects)

    # ── Traffic-light signals 
    tl_objects = [
        obj for obj in payload.environment.traffic_objects if obj.obj_class == "traffic_light"
    ]
    traffic_lights: List[dict] = _traffic_light_summary(frame_ids, tl_objects)

    return {
        "age": payload.pedestrian.age,
        "intersection": payload.environment.intersection,
        "bbox": payload.pedestrian.bbox,
        "occlusion": payload.pedestrian.occlusion,
        "action": payload.pedestrian.behavior.action,
        "gesture": payload.pedestrian.behavior.gesture,
        "look": payload.pedestrian.behavior.look,
        "obd_speed": obd_speed,
        "crosswalk_bboxes": crosswalk_bboxes,
        "vehicle_counts": vehicle_counts,
        "traffic_lights": traffic_lights,
    }


def _per_frame_bboxes(
    frame_ids: List[int],
    objects: List[TrafficObject],
) -> List[List[List[float]]]:
    """
    For each frame in frame_ids return the list of bboxes from matching objects.
    Returns a list-of-lists (outer = frames, inner = bboxes in that frame).
    """
    # Build frame → list[bbox] mapping
    frame_map: Dict[int, List[List[float]]] = {fid: [] for fid in frame_ids}
    for obj in objects:
        if obj.bbox is None:
            continue
        for i, fid in enumerate(obj.frames):
            if fid in frame_map and i < len(obj.bbox):
                frame_map[fid].append(obj.bbox[i])

    return [frame_map[fid] for fid in frame_ids]


def _per_frame_counts(
    frame_ids: List[int],
    objects: List[TrafficObject],
) -> List[int]:
    """Count how many objects appear in each frame."""
    frame_map: Dict[int, int] = {fid: 0 for fid in frame_ids}
    for obj in objects:
        for fid in obj.frames:
            if fid in frame_map:
                frame_map[fid] += 1
    return [frame_map[fid] for fid in frame_ids]


def _traffic_light_summary(
    frame_ids: List[int],
    tl_objects: List[TrafficObject],
) -> List[dict]:
    """
    Return one summary dict per traffic-light object (across all frames):
        type  : str   (obj_type from annotation)
        state : int   (most common non-zero state across observed frames)
    """
    summaries = []
    for obj in tl_objects:
        states = obj.state or []
        # Pick most common non-zero state; fall back to 0
        non_zero = [s for s in states if s and s != 0]
        state = max(set(non_zero), key=non_zero.count) if non_zero else 0
        summaries.append(
            {"type": obj.obj_type or "", "state": state}
        )
    return summaries

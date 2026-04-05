"""
ToMFeatureExtractor — converts raw PIE-style annotation data into the 9
Theory of Mind features used by the ANFIS model.

This class is **copied from the training notebook** and adapted for the API.
The public interface expected by feature_extraction_service.py is:

    extractor = ToMFeatureExtractor()
    features: dict = extractor.extract_for_pedestrian(ped_dict)

where `ped_dict` has the structure built by feature_extraction_service.py
from a RawAnnotationInput payload.

Feature definitions
-------------------
Belief State:
    B1  age_vulnerability_awareness    Age → self-perceived vulnerability scalar
    B2  intersection_complexity        Road geometry → cognitive load scalar
    B3  pedestrian_position_proximity  Ped bbox overlap with crosswalk zone
    B4  vehicle_density                Mean vehicle count / 10

Goal State:
    G2  urgency_rushing                (walking_ratio + urgent_gesture_ratio) / 2

Attention State:
    A1  vehicle_awareness              Fraction of frames with gaze at ego-vehicle
    A2  traffic_signal_awareness       Safety score of relevant pedestrian signal

Context / Environment:
    C1  relative_vehicle_speed         Ego OBD speed mean / 27 m/s
    C2  occlusion_factor               Mean occlusion / 2
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ── Lookup tables ─────────────────────────────────────────────────────────────

_AGE_VULNERABILITY: Dict[int, float] = {
    0: 0.8,  # child
    1: 0.3,  # young
    2: 0.2,  # adult
    3: 0.7,  # senior
}

_INTERSECTION_COMPLEXITY: Dict[int, float] = {
    0: 0.2,  # mid-block
    1: 0.5,  # T-junction (driver's right)
    2: 0.5,  # T-junction (driver's left)
    3: 0.6,  # T-junction (generic)
    4: 1.0,  # four-way
}

_SIGNAL_SAFETY: Dict[int, float] = {
    0: 0.0,  # undefined / unknown
    1: 0.0,  # red  → unsafe
    2: 0.5,  # yellow
    3: 1.0,  # green → safe
}

# Gesture codes classified as "urgent"
_URGENT_GESTURES = {3, 4}  # hand_row (right-of-way), nod

_MAX_SPEED: float = 27.0   # m/s (~97 km/h)
_MAX_OCC: float = 2.0      # fully occluded


class ToMFeatureExtractor:
    """
    Stateless extractor: call extract_for_pedestrian() with a dict that mirrors
    RawAnnotationInput structure.
    """

    def extract_for_pedestrian(self, ped: dict) -> Dict[str, float]:
        """
        Compute all 9 ToM features for a single pedestrian observation window.

        Parameters
        ----------
        ped : dict  — keys described below (built by feature_extraction_service)
            age              : int
            intersection     : int
            bbox             : list[list[float]]     per-frame [x1,y1,x2,y2]
            occlusion        : list[int]             per-frame
            action           : list[int]             per-frame
            gesture          : list[int]             per-frame
            look             : list[int]             per-frame
            obd_speed        : list[float]           per-frame
            crosswalk_bboxes : list[list[list[float]]]  per-frame list-of-boxes
            traffic_lights   : list[dict]            each has state & type keys

        Returns
        -------
        dict[str, float] — 9 named feature values
        """
        n = len(ped.get("bbox", []))

        return {
            "B1_age_vulnerability_awareness":   self._B1(ped["age"]),
            "B2_intersection_complexity":       self._B2(ped["intersection"]),
            "B3_pedestrian_position_proximity": self._B3(
                ped.get("bbox", []), ped.get("crosswalk_bboxes", [])
            ),
            "B4_vehicle_density":               self._B4(ped.get("vehicle_counts", [])),
            "G2_urgency_rushing":               self._G2(
                ped.get("action", []), ped.get("gesture", [])
            ),
            "A1_vehicle_awareness":             self._A1(ped.get("look", []), n),
            "A2_traffic_signal_awareness":      self._A2(ped.get("traffic_lights", [])),
            "C1_relative_vehicle_speed":        self._C1(ped.get("obd_speed", [])),
            "C2_occlusion_factor":              self._C2(ped.get("occlusion", [])),
        }

    # ── Feature helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _B1(age: int) -> float:
        return _AGE_VULNERABILITY.get(age, 0.2)

    @staticmethod
    def _B2(intersection: int) -> float:
        return _INTERSECTION_COMPLEXITY.get(intersection, 0.5)

    @staticmethod
    def _B3(
        bbox: List[List[float]],
        crosswalk_bboxes: List[List[List[float]]],
    ) -> float:
        """IoU-based proximity between ped bbox and nearest crosswalk per frame."""
        if not crosswalk_bboxes or all(len(f) == 0 for f in crosswalk_bboxes):
            return 0.5

        scores: List[float] = []
        for f_idx, pbox in enumerate(bbox):
            cboxes = (
                crosswalk_bboxes[f_idx] if f_idx < len(crosswalk_bboxes) else []
            )
            if not cboxes:
                scores.append(0.0)
                continue

            px1, py1, px2, py2 = pbox
            p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            if p_area == 0.0:
                scores.append(0.0)
                continue

            max_iou = 0.0
            for cbox in cboxes:
                cx1, cy1, cx2, cy2 = cbox
                ix1, iy1 = max(px1, cx1), max(py1, cy1)
                ix2, iy2 = min(px2, cx2), min(py2, cy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
                union = p_area + c_area - inter
                if union > 0.0:
                    max_iou = max(max_iou, inter / union)
                else:
                    max_iou = max(max_iou, 1.0)
            scores.append(min(max_iou, 1.0))

        return float(sum(scores) / len(scores)) if scores else 0.5

    @staticmethod
    def _B4(vehicle_counts: List[int]) -> float:
        if not vehicle_counts:
            return 0.0
        return min(sum(vehicle_counts) / len(vehicle_counts) / 10.0, 1.0)

    @staticmethod
    def _G2(action: List[int], gesture: List[int]) -> float:
        n = len(action)
        if n == 0:
            return 0.0
        walking = sum(1 for a in action if a == 1) / n
        urgent = sum(1 for g in gesture if g in _URGENT_GESTURES) / n
        return (walking + urgent) / 2.0

    @staticmethod
    def _A1(look: List[int], n: int) -> float:
        if n == 0:
            return 0.0
        return sum(1 for v in look if v == 1) / n

    @staticmethod
    def _A2(traffic_lights: List[dict]) -> float:
        """
        Average safety score over pedestrian-type signals only.
        Returns -1.0 if no pedestrian signals are present.
        """
        ped_scores: List[float] = []
        for tl in traffic_lights:
            if tl.get("type", "").lower() == "pedestrian":
                state = tl.get("state", 0)
                ped_scores.append(_SIGNAL_SAFETY.get(state, 0.0))
        if not ped_scores:
            return -1.0
        return sum(ped_scores) / len(ped_scores)

    @staticmethod
    def _C1(obd_speed: List[float]) -> float:
        if not obd_speed:
            return 0.0
        return min(sum(obd_speed) / len(obd_speed) / _MAX_SPEED, 1.0)

    @staticmethod
    def _C2(occlusion: List[int]) -> float:
        if not occlusion:
            return 0.0
        return min(sum(occlusion) / len(occlusion) / _MAX_OCC, 1.0)

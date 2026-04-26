from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import joblib
import torch

from app.config import settings
from app.core.anfis import ANFIS

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

_FEATURE_NAMES = [
    "B1_age_vulnerability_awareness",
    "B2_intersection_complexity",
    "B3_pedestrian_position_proximity",
    "B4_vehicle_density",
    "G2_urgency_rushing",
    "A1_vehicle_awareness",
    "A2_traffic_signal_awareness",
    "C1_relative_vehicle_speed",
    "C2_occlusion_factor",
]


class ModelLoader:
    """
    Responsible for loading and holding the ANFIS model and MinMaxScaler as
    singleton instances. Instantiated once at application startup via the
    FastAPI lifespan and shared across the entire app lifetime.
    """

    def __init__(self) -> None:
        self._model: Optional[ANFIS] = None
        self._scaler: Optional["MinMaxScaler"] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """
        Load the ANFIS model weights and MinMaxScaler from the paths defined
        in config. Instantiates the ANFIS class from core/anfis.py, applies
        the saved state dict, and loads the scaler. Raises a clear error on
        startup if either artifact file is missing or corrupt.
        """
        self._load_scaler()
        self._load_model()
        logger.info(
            "ToM-ANFIS model and scaler loaded successfully on %s.", self._device
        )

    def get_model(self) -> ANFIS:
        """
        Return the loaded ANFIS model instance ready for inference.
        Raises RuntimeError if called before load() has been invoked.
        """
        if self._model is None:
            raise RuntimeError(
                "ANFIS model not loaded. Call loader.load() at application startup."
            )
        return self._model

    def get_scaler(self) -> "MinMaxScaler":
        """
        Return the loaded MinMaxScaler instance.
        Raises RuntimeError if called before load() has been invoked.
        """
        if self._scaler is None:
            raise RuntimeError(
                "Scaler not loaded. Call loader.load() at application startup."
            )
        return self._scaler



    def _load_scaler(self) -> None:
        path: Path = settings.SCALER_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found: {path}\n"
                "Place 'scaler_optimized.save' in app/artifacts/."
            )
        logger.info("Loading scaler from %s …", path)
        self._scaler = joblib.load(path)

    def _load_model(self) -> None:
        path: Path = settings.MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                "Place 'anfis_model_optimized.pth' in app/artifacts/."
            )
        logger.info("Loading ANFIS model from %s …", path)

        model = ANFIS(in_features=9, num_mfs=3, feature_names=_FEATURE_NAMES)

        checkpoint = torch.load(path, map_location=self._device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        # strict=False: rule_mf_indices is a buffer recomputed by __init__
        # and may not be present in older checkpoints.
        model.load_state_dict(state_dict, strict=False)
        model.to(self._device)
        model.eval()

        self._model = model


# Module-level singleton — import this everywhere in the application.
loader = ModelLoader()

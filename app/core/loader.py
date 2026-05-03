import torch
import joblib
from typing import Optional

from app.config import settings
from app.core.anfis import ANFIS


class ModelLoader:
    """
    Singleton that holds the loaded ANFIS model and MinMaxScaler.
    Instantiated once at app startup via the FastAPI lifespan.
    """

    def __init__(self):
        self._model: Optional[ANFIS] = None
        self._scaler: Optional[object] = None

    def load(self):
        """
        Load the ANFIS model weights and MinMaxScaler from the paths
        defined in config. Raises clear errors on startup if artifacts
        are missing or corrupt.
        """
        try:
            # Load the ANFIS model architecture and weights
            checkpoint = torch.load(settings.MODEL_PATH, map_location='cpu')

            # Instantiate the ANFIS model with 9 inputs (3 membership functions each = 3^9 = 19,683 rules)
            self._model = ANFIS(in_features=9, num_mfs=3)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()  # Set to evaluation mode

            print(f"Model loaded from {settings.MODEL_PATH}")

        except FileNotFoundError:
            raise RuntimeError(
                f"Model file not found: {settings.MODEL_PATH}\n"
                "Ensure anfis_model_optimized.pth exists in app/artifacts/"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {settings.MODEL_PATH}: {str(e)}"
            )

        try:
            # Load the MinMaxScaler used during training
            self._scaler = joblib.load(settings.SCALER_PATH)
            print(f"Scaler loaded from {settings.SCALER_PATH}")

        except FileNotFoundError:
            raise RuntimeError(
                f"Scaler file not found: {settings.SCALER_PATH}\n"
                "Ensure scaler_optimized.save exists in app/artifacts/"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load scaler from {settings.SCALER_PATH}: {str(e)}"
            )

    def get_model(self) -> ANFIS:
        """
        Return the loaded ANFIS model instance.
        Raises RuntimeError if load() has not been called yet.
        """
        if self._model is None:
            raise RuntimeError(
                "Model has not been loaded. Call load() during app startup."
            )
        return self._model

    def get_scaler(self) -> object:
        """
        Return the loaded MinMaxScaler instance.
        Raises RuntimeError if load() has not been called yet.
        """
        if self._scaler is None:
            raise RuntimeError(
                "Scaler has not been loaded. Call load() during app startup."
            )
        return self._scaler


# Module-level singleton instance
loader = ModelLoader()

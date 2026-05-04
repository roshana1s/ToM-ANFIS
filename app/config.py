from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    MODEL_PATH: str = str(
        Path(__file__).parent / "artifacts" / "anfis_model_optimized.pth"
    )
    SCALER_PATH: str = str(
        Path(__file__).parent / "artifacts" / "scaler_optimized.save"
    )
    RULE_FIRING_THRESHOLD: float = 0.0001
    TOP_K_RULES: int = 3
    # Feature extraction / inference defaults
    OBSERVATION_WINDOW: int = 15
    DEFAULT_PREDICTION_HORIZON: int = 30


# Instantiate settings singleton
settings = Settings()

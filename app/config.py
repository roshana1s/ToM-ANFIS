from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration using Pydantic BaseSettings.
    Paths are relative to the app/ directory.
    """

    # ── Model & Scaler Artifacts ──────────────────────────────────────
    MODEL_PATH: str = str(Path(__file__).parent / "artifacts" / "anfis_model_optimized.pth")
    SCALER_PATH: str = str(Path(__file__).parent / "artifacts" / "scaler_optimized.save")

    # ── Inference Parameters ──────────────────────────────────────────
    RULE_FIRING_THRESHOLD: float = 0.0001
    """
    Minimum normalised firing strength (0.0 - 1.0) for a fuzzy rule
    to be included in the explainable output.
    """

    TOP_K_RULES: int = 3
    """
    Number of top-firing rules to include in each prediction response.
    """

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instantiate settings singleton
settings = Settings()

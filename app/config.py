"""
Application settings for the ToM-ANFIS API.

Pure settings only — no instances, no imports from core/.
Reads from environment variables if set, otherwise uses sensible defaults.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


# Resolve the artifacts directory relative to this file so the server can be
# started from any working directory.
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class Settings(BaseSettings):
    # ── Artefact paths ─────────────────────────────────────────────────────
    MODEL_PATH: Path = _ARTIFACTS_DIR / "anfis_model_optimized.pth"
    SCALER_PATH: Path = _ARTIFACTS_DIR / "scaler_optimized.save"

    # ── Inference settings ──────────────────────────────────────────────────
    # Minimum normalised firing strength for a rule to be included in output
    RULE_FIRING_THRESHOLD: float = 0.0001
    # Number of top rules to return in the response
    TOP_K_RULES: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

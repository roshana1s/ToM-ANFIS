from fastapi import APIRouter, HTTPException
import logging

from app.schemas.raw_data import RawAnnotationInput
from app.schemas.processed_features import ToMFeaturesInput
from app.schemas.output import PredictionResponse
from app.services.feature_extraction_service import extract_features
from app.services.prediction_service import predict

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/from-annotations", response_model=PredictionResponse)
def predict_from_annotations(payload: RawAnnotationInput) -> PredictionResponse:
    """
    Predict pedestrian crossing intention from raw PIE annotation data.

    Flow: RawAnnotationInput → feature_extraction_service → prediction_service
    """
    try:
        features_dict = extract_features(payload)
        features = ToMFeaturesInput(
            belief_state=features_dict,
            goal_state=features_dict,
            attention_state=features_dict,
            context_environment=features_dict
        )
        return predict(features)
    except Exception as e:
        logger.error(f"Error in predict_from_annotations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-features", response_model=PredictionResponse)
def predict_from_features(payload: ToMFeaturesInput) -> PredictionResponse:
    """
    Predict pedestrian crossing intention from pre-computed ToM features.

    Flow: ToMFeaturesInput → prediction_service (no extraction needed)
    """
    try:
        return predict(payload)
    except Exception as e:
        logger.error(f"Error in predict_from_features: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
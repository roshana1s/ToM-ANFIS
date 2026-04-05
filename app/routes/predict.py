from fastapi import APIRouter

from app.schemas.input import RawAnnotationInput, ToMFeaturesInput
from app.schemas.output import PredictionResponse
from app.services.feature_extraction_service import extract_features
from app.services.prediction_service import predict

router = APIRouter()


@router.post("/from-annotations", response_model=PredictionResponse)
def predict_from_annotations(payload: RawAnnotationInput) -> PredictionResponse:
    """
    Predict pedestrian crossing intention from raw PIE annotation data.

    Flow: RawAnnotationInput → feature_extraction_service → prediction_service
    """
    features_dict = extract_features(payload)
    features = ToMFeaturesInput(**features_dict)
    return predict(features)


@router.post("/from-features", response_model=PredictionResponse)
def predict_from_features(payload: ToMFeaturesInput) -> PredictionResponse:
    """
    Predict pedestrian crossing intention from pre-computed ToM features.

    Flow: ToMFeaturesInput → prediction_service (no extraction needed)
    """
    return predict(payload)
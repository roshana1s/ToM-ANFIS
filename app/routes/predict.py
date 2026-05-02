"""
Prediction routes for the ToM-ANFIS API.

Two endpoints:
  1. POST /from-annotations - accepts raw PIE annotations
  2. POST /from-features - accepts pre-extracted ToM features
"""

from fastapi import APIRouter, HTTPException

from app.schemas.input import RawAnnotationInput, ToMFeaturesInput
from app.schemas.output import PredictionResponse
from app.services.prediction_service import predict as run_prediction
from app.services.feature_extraction_service import extract_features


router = APIRouter()


@router.post(
    "/from-annotations",
    response_model=PredictionResponse,
    summary="Predict from raw PIE annotations",
    description=(
        "Accept raw pedestrian annotations from the PIE dataset, "
        "extract ToM features, and return crossing intention prediction."
    ),
)
async def predict_from_annotations(payload: RawAnnotationInput) -> PredictionResponse:
    """
    Extract ToM features from raw PIE annotations and run ANFIS inference.

    Args:
        payload : RawAnnotationInput with complete multimodal pedestrian data

    Returns:
        PredictionResponse with crossing probability, intention level, and top-k rules

    Raises:
        HTTPException (422) if annotation payload is invalid
        HTTPException (500) if feature extraction or inference fails
    """
    try:
        # Extract ToM features from raw annotations
        features_dict = extract_features(payload)

        # Convert dict to ToMFeaturesInput schema
        features = ToMFeaturesInput(**features_dict)

        # Run prediction
        return run_prediction(features)

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid annotation payload: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/from-features",
    response_model=PredictionResponse,
    summary="Predict from pre-extracted ToM features",
    description=(
        "Accept 9 pre-normalised Theory of Mind feature values "
        "and return crossing intention prediction."
    ),
)
async def predict_from_features(payload: ToMFeaturesInput) -> PredictionResponse:
    """
    Run ANFIS inference directly on pre-extracted ToM features.

    Args:
        payload : ToMFeaturesInput with 9 normalised feature values

    Returns:
        PredictionResponse with crossing probability, intention level, and top-k rules

    Raises:
        HTTPException (422) if feature values are invalid
        HTTPException (500) if inference fails
    """
    try:
        return run_prediction(payload)

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

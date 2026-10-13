import logging
from typing import List
from fastapi import FastAPI, HTTPException, status
from src.api.schemas import PredictionRequest, PredictionResponse, DelaySeverityCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight-delay-api")

app = FastAPI(
    title="Flight Delay Severity Prediction API",
    description="Inference API exposing /classify endpoint for batch predictions of flight delay severity.",
    version="0.1.0",
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to verify service readiness."""
    return {"status": "ok", "service": "flight-delay-api"}


@app.post(
    "/classify",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict arrival delay severity for a batch of flight records",
)
def classify_flights(payload: PredictionRequest) -> PredictionResponse:
    """
    Receives an array of raw flight records, preprocesses features,
    and returns predicted severity classes preserving instance order.
    Valid categories: 'on_time', 'moderate_delay', 'severe_delay'.
    """
    if not payload.instances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The instances list cannot be empty.",
        )

    predictions: List[DelaySeverityCategory] = []

    try:
        # TODO: Load trained model artifacts and execute pipeline inference
        # In this baseline skeleton, we provide a placeholder default prediction
        for _ in payload.instances:
            predictions.append("on_time")

        return PredictionResponse(classifications=predictions)
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import PredictionRequest, PredictionResponse
from src.data.contract import records_to_frame
from src.data.preprocessor import DEFAULT_CONFIG_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight-delay-api")


def serving_settings() -> dict:
    """Model location and identity. The environment overrides the configuration."""
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["serving"]
    return {
        "path": Path(os.environ.get("MODEL_PATH") or config["model_path"]),
        "name": os.environ.get("MODEL_NAME") or config["model_name"],
        "version": os.environ.get("MODEL_VERSION") or "unknown",
    }


def load_model(path: Path):
    """Load the pipeline, or stop the service with a clear error."""
    if not path.is_file():
        raise RuntimeError(f"Model file not found: {path}. Set MODEL_PATH.")
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Could not load model from {path}: {error}") from error


@asynccontextmanager
async def lifespan(service: FastAPI):
    settings = serving_settings()
    service.state.model = load_model(settings["path"])
    service.state.model_name = settings["name"]
    service.state.model_version = settings["version"]
    logger.info("Loaded model %s %s", settings["name"], settings["version"])
    yield


app = FastAPI(
    title="Flight Delay Severity Prediction API",
    description="Inference API exposing /classify endpoint for batch predictions of flight delay severity.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    """Every invalid request is a 400 naming the field, without echoing its data."""
    detail = [
        {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": detail}
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Report that the service is ready and which model it loaded."""
    return {
        "status": "ok",
        "service": "flight-delay-api",
        "model": {
            "name": app.state.model_name,
            "version": app.state.model_version,
        },
    }


@app.post(
    "/classify",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict arrival delay severity for a batch of flight records",
)
def classify_flights(payload: PredictionRequest) -> PredictionResponse:
    """
    Receives an array of raw flight records and returns one predicted severity
    class per record, preserving instance order. The loaded pipeline applies the
    same feature function as training, so there is no train/serve skew.
    Valid categories: 'on_time', 'moderate_delay', 'severe_delay'.
    """
    try:
        predictions = app.state.model.predict(records_to_frame(payload.instances))
        return PredictionResponse(classifications=predictions.tolist())
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed.",
        )

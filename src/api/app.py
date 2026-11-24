import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.inference import classify
from src.api.schemas import (
    LabelsDelivery,
    LabelsReceipt,
    PredictionRequest,
    PredictionResponse,
)
from src.data.preprocessor import CLASSES, DEFAULT_CONFIG_PATH
from src.monitoring.traffic import TrafficLog, normalize_mode, utc_now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight-delay-api")


def serving_settings() -> dict:
    """Model location and identity. The environment overrides the configuration."""
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["serving"]
    if config["fallback_class"] not in CLASSES:
        raise RuntimeError(f"serving.fallback_class must be one of {CLASSES}")
    return {
        "path": Path(os.environ.get("MODEL_PATH") or config["model_path"]),
        "name": os.environ.get("MODEL_NAME") or config["model_name"],
        "version": os.environ.get("MODEL_VERSION") or "unknown",
        "fallback_class": config["fallback_class"],
        "traffic_dir": Path(os.environ.get("TRAFFIC_DIR") or config["traffic_dir"]),
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
    service.state.fallback_class = settings["fallback_class"]
    service.state.traffic = TrafficLog(settings["traffic_dir"])
    logger.info("Loaded model %s %s", settings["name"], settings["version"])
    yield


app = FastAPI(
    title="Flight Delay Severity Prediction API",
    description=(
        "Batch predictions of flight delay severity (/classify) and delayed true "
        "outcomes from the evaluation service (/labels)."
    ),
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


def store_predictions(entry: dict) -> None:
    """Runs after the response is sent; a storage failure never fails /classify."""
    try:
        app.state.traffic.append("predictions", entry)
    except Exception:
        logger.exception(
            "Could not store predictions of request %s", entry["request_id"]
        )


@app.post(
    "/classify",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict arrival delay severity for a batch of flight records",
)
def classify_flights(
    payload: PredictionRequest, background: BackgroundTasks
) -> PredictionResponse:
    """
    Returns one class per instance, in input order: 'on_time', 'moderate_delay'
    or 'severe_delay'. The loaded pipeline applies the same feature function as
    training, so there is no train/serve skew. An instance that cannot be
    classified gets the configured fallback class instead of failing the batch.
    """
    started = time.perf_counter()
    received_at = utc_now()
    mode = normalize_mode(payload.mode)
    result = classify(app.state.model, payload.instances, app.state.fallback_class)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    logger.info(
        "classify request_id=%s mode=%s instances=%d fallback=%d latency_ms=%s",
        payload.request_id,
        mode,
        len(payload.instances),
        result.fallback,
        latency_ms,
    )
    if result.discarded_fields:
        logger.warning(
            "Discarded invalid values (field: instances): %s",
            dict(result.discarded_fields),
        )

    entry = {
        "received_at": received_at,
        "request_id": payload.request_id,
        "mode": mode,
        "model": {"name": app.state.model_name, "version": app.state.model_version},
        "latency_ms": latency_ms,
        "fallback": result.fallback,
        "discarded_fields": dict(result.discarded_fields),
        "classifications": result.classifications,
    }
    # The sandbox sends the same January flights every 5 minutes: inputs are
    # kept only where they matter for drift.
    if mode == "live":
        entry["instances"] = payload.instances
    background.add_task(store_predictions, entry)
    return PredictionResponse(classifications=result.classifications)


@app.post(
    "/labels",
    response_model=LabelsReceipt,
    status_code=status.HTTP_200_OK,
    summary="Receive the true outcomes of a scored batch",
)
def receive_labels(payload: LabelsDelivery) -> LabelsReceipt:
    """
    Stores the delivery as received, by mode, for monitoring and retraining.
    Answers 500 when it cannot be stored, so a lost delivery is visible.
    """
    entry = {
        "received_at": utc_now(),
        **payload.model_dump(),
        "mode": normalize_mode(payload.mode),
    }
    try:
        app.state.traffic.append("labels", entry)
    except Exception:
        logger.exception("Could not store labels of request %s", payload.request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store labels.",
        )
    logger.info("labels request_id=%s mode=%s", payload.request_id, entry["mode"])
    return LabelsReceipt(request_id=payload.request_id)

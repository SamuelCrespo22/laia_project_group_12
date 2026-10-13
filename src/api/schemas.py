from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

DelaySeverityCategory = Literal["on_time", "moderate_delay", "severe_delay"]


class PredictionRequest(BaseModel):
    """
    Schema for batch prediction input according to BTS flight record specifications.
    Accepts raw flight attributes available before scheduled departure.
    """
    instances: List[Dict[str, Any]] = Field(
        ...,
        description="Array of flight records with raw attributes available prior to departure.",
        examples=[
            [
                {
                    "FL_DATE": "2025-05-15",
                    "OP_CARRIER": "AA",
                    "ORIGIN": "JFK",
                    "DEST": "LAX",
                    "CRS_DEP_TIME": 900,
                    "CRS_ARR_TIME": 1215,
                    "CRS_ELAPSED_TIME": 375,
                    "DISTANCE": 2475
                }
            ]
        ]
    )


class PredictionResponse(BaseModel):
    """
    Schema for batch prediction output.
    Must return categories preserving input order: on_time, moderate_delay, severe_delay.
    """
    classifications: List[DelaySeverityCategory] = Field(
        ...,
        description="Array of predicted delay severity categories corresponding to input instances.",
        examples=[["moderate_delay", "on_time"]]
    )


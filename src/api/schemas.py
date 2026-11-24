from typing import Annotated, Any, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

DelaySeverityCategory = Literal["on_time", "moderate_delay", "severe_delay"]

# The evaluation service may send ids as numbers; a type mismatch here must
# not cost the batch.
Text = Annotated[
    Optional[str], BeforeValidator(lambda value: None if value is None else str(value))
]

EXAMPLE_FLIGHT = {
    "FL_DATE": "2024-02-01",
    "OP_UNIQUE_CARRIER": "AA",
    "ORIGIN": "JFK",
    "DEST": "LAX",
    "CRS_DEP_TIME": 900,
    "CRS_ARR_TIME": 1215,
    "CRS_ELAPSED_TIME": 375,
    "DISTANCE": 2475,
}


class PredictionRequest(BaseModel):
    """
    Batch of flights to classify. Each instance is validated on its own against
    the data contract (src/data/contract.py), so an invalid instance never
    rejects the batch: see src/api/inference.py.
    """

    request_id: Text = Field(None, description="Identifier echoed in /labels.")
    mode: Text = Field(
        None,
        description="'live' or 'sandbox'. Anything else is stored as 'unknown'.",
    )
    instances: List[Any] = Field(
        ...,
        min_length=1,
        description="Flight records with raw attributes available prior to departure.",
        examples=[[EXAMPLE_FLIGHT]],
    )


class PredictionResponse(BaseModel):
    """One class per instance, in input order."""

    classifications: List[DelaySeverityCategory] = Field(
        ...,
        description="Predicted delay severity of each instance.",
        examples=[["moderate_delay", "on_time"]],
    )


class LabelsDelivery(BaseModel):
    """
    True outcomes of a scored batch. The format belongs to the evaluation
    service, so every field is optional and unknown fields are kept as received.
    """

    model_config = ConfigDict(extra="allow")

    request_id: Text = None
    mode: Text = None
    labels: Any = Field(None, description="True class of each instance.")
    outcomes: Any = Field(
        None,
        description="Actual times, delays, taxi times and causes of delay.",
    )


class LabelsReceipt(BaseModel):
    status: Literal["received"] = "received"
    request_id: Optional[str] = None

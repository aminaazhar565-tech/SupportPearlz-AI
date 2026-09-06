from enum import Enum
from pydantic import BaseModel, Field

class Confidence(str, Enum):
    HIGH = "high"
    PARTIAL = "partial"
    NONE = "none"

class SupportResponse(BaseModel):
    answer: str = Field(description="Customer-facing answer.")
    sources: list[str] = Field(default_factory=list)
    confidence: Confidence
    answered: bool

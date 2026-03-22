from pydantic import BaseModel, Field

from core.config import SUPPORTED_LANGUAGES


class ReviewRequest(BaseModel):
    code: str
    language: str = Field(default="python", description="programming language")
    focus: str = Field(default="general", description="review focus area")
    max_tokens: int = Field(default=512, ge=64, le=1024)


class ReviewResponse(BaseModel):
    review: str
    language: str
    focus: str
    inference_ms: float
    model: str
    device: str

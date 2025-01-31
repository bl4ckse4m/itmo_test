from typing import List

from pydantic import BaseModel, HttpUrl


class PredictionRequest(BaseModel):
    id: int
    query: str


class PredictionResponse(BaseModel):
    id: int
    answer: int | None
    reasoning: str
    sources: List[HttpUrl]

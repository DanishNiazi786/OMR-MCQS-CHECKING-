from pydantic import BaseModel, validator
from typing import List
from datetime import datetime

class SolutionItem(BaseModel):
    question: int
    answer: str  # Must be A, B, C, D, or E

    @validator("answer")
    def validate_answer(cls, value):
        """Ensure answer is one of A, B, C, D, or E."""
        valid_answers = ["A", "B", "C", "D", "E"]
        if value not in valid_answers:
            raise ValueError(f"Answer must be one of {valid_answers}, got {value}")
        return value

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SolutionCreate(BaseModel):
    examId: str
    solutions: List[SolutionItem]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SolutionResponse(BaseModel):
    examId: str
    solutions: List[SolutionItem]
    uploadedAt: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
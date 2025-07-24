from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ProcessingMetadata(BaseModel):
    confidence: int
    processingTime: float
    imageQuality: Optional[str] = None
    bubbleDetection: Optional[Dict[str, Any]] = None

class ResponseCreate(BaseModel):
    examId: str
    studentId: str
    responses: List[str]
    score: int
    accuracy: float
    correctAnswers: int
    incorrectAnswers: int
    blankAnswers: int
    multipleMarks: int
    processingMetadata: ProcessingMetadata

class ResponseResponse(BaseModel):
    examId: str
    studentId: str
    responses: List[str]
    score: int
    accuracy: float
    correctAnswers: int
    incorrectAnswers: int
    blankAnswers: int
    multipleMarks: int
    processingMetadata: ProcessingMetadata
    processedAt: datetime
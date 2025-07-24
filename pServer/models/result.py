from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ResultCreate(BaseModel):
    examId: str
    studentId: str
    studentName: str
    examName: str
    responses: List[str]
    score: int
    totalMarks: int
    percentage: float
    passFailStatus: str
    correctAnswers: int
    incorrectAnswers: int
    blankAnswers: int
    multipleMarks: int
    sponsorDS: Optional[str] = None  # Added optional field
    course: Optional[str] = None     # Added optional field
    wing: Optional[str] = None       # Added optional field
    module: Optional[str] = None     # Added optional field

class ResultResponse(BaseModel):
    examId: str
    studentId: str
    studentName: str
    examName: str
    responses: List[str]
    score: int
    totalMarks: int
    percentage: float
    passFailStatus: str
    correctAnswers: int
    incorrectAnswers: int
    blankAnswers: int
    multipleMarks: int
    sponsorDS: Optional[str] = None  # Added optional field
    course: Optional[str] = None     # Added optional field
    wing: Optional[str] = None       # Added optional field
    module: Optional[str] = None     # Added optional field
    processedAt: datetime
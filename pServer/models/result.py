from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ResultCreate(BaseModel):
    examId: str
    studentId: str
    studentName: str
    rank: Optional[str] = None
    lockerNumber: Optional[str] = None
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
    sponsorDS: Optional[str] = None
    course: Optional[str] = None
    wing: Optional[str] = None
    module: Optional[str] = None

class ResultResponse(BaseModel):
    examId: str
    studentId: str
    studentName: str
    rank: Optional[str] = None
    lockerNumber: Optional[str] = None
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
    sponsorDS: Optional[str] = None
    course: Optional[str] = None
    wing: Optional[str] = None
    module: Optional[str] = None
    processedAt: datetime
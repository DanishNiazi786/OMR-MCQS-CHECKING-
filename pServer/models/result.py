from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class StudentInfo(BaseModel):
    name: Optional[str] = None
    lockerNumber: Optional[str] = None
    rank: Optional[str] = None
    ocrConfidence: Optional[float] = None
    rawOcrText: Optional[str] = None
    ocrAvailable: Optional[bool] = None

class ResultCreate(BaseModel):
    examId: str
    studentId: str
    studentName: Optional[str] = None
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
    studentInfo: Optional[StudentInfo] = None  # Added studentInfo field

class ResultResponse(BaseModel):
    examId: str
    studentId: str
    studentName: Optional[str] = None
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
    studentInfo: Optional[StudentInfo] = None  # Added studentInfo field
    processedAt: datetime
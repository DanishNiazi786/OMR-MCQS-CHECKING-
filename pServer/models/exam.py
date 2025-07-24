from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ExamCreate(BaseModel):
    examId: Optional[str] = None
    name: str
    wing: str
    course: str
    module: str
    sponsorDS: str
    dateTime: str
    time: str
    numQuestions: int
    marksPerMcq: int = 1
    passingPercentage: int = 60
    instructions: Optional[str] = "Fill bubbles neatly with a black/blue pen, mark only one option per question—any extra, unclear, or incorrect marking will be considered wrong."
    settings: Optional[Dict[str, Any]] = {}
    createdAt: Optional[datetime] = None

class ExamUpdate(BaseModel):
    name: Optional[str] = None
    wing: Optional[str] = None
    course: Optional[str] = None
    module: Optional[str] = None
    sponsorDS: Optional[str] = None
    dateTime: Optional[str] = None
    time: Optional[str] = None
    numQuestions: Optional[int] = None
    marksPerMcq: Optional[int] = None
    passingPercentage: Optional[int] = None
    instructions: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

class ExamResponse(BaseModel):
    examId: str
    name: str
    wing: str
    course: str
    module: str
    sponsorDS: str
    dateTime: str
    time: str
    numQuestions: int
    marksPerMcq: int
    passingPercentage: int
    instructions: str
    settings: Dict[str, Any]
    createdAt: datetime
    studentsUploaded: Optional[bool] = False
    solutionUploaded: Optional[bool] = False
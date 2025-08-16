from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class StudentCreate(BaseModel):
    examId: str
    name: str
    lockerNumber: str
    rank: str
    copyNumber: str

class StudentResponse(BaseModel):
    examId: str
    name: str
    lockerNumber: str
    rank: str
    copyNumber: str
    createdAt: datetime
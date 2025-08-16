from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class ReportCreate(BaseModel):
    examId: str
    reportType: str  # "Excel" or "PDF"
    data: Dict[str, Any]
    generatedBy: str

class ReportResponse(BaseModel):
    examId: str
    reportType: str
    data: Dict[str, Any]
    generatedBy: str
    generatedAt: datetime
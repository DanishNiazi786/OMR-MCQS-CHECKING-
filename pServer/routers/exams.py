from fastapi import APIRouter, HTTPException, Depends
from typing import List
from models.exam import ExamCreate, ExamUpdate, ExamResponse
from datetime import datetime
import uuid

router = APIRouter()

def get_database():
    from main import app
    return app.state.database

@router.post("/", response_model=dict)
async def create_exam(exam: ExamCreate, db=Depends(get_database)):
    try:
        # Generate examId if not provided
        if not exam.examId:
            exam.examId = str(uuid.uuid4())
        
        # Prepare exam data
        exam_data = exam.dict()
        exam_data['createdAt'] = exam_data.get('createdAt') or datetime.utcnow()
        
        # Insert into database
        result = await db.exams.insert_one(exam_data)
        
        return {
            "examId": exam.examId,
            "message": "Exam created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create exam")

@router.get("/", response_model=List[dict])
async def get_all_exams(db=Depends(get_database)):
    try:
        cursor = db.exams.find().sort("createdAt", -1)
        exams = await cursor.to_list(length=None)
        
        # Convert ObjectId to string
        for exam in exams:
            exam['_id'] = str(exam['_id'])
        
        return exams
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch exams")

@router.get("/{exam_id}", response_model=dict)
async def get_exam(exam_id: str, db=Depends(get_database)):
    try:
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam['_id'] = str(exam['_id'])
        return exam
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch exam")

@router.put("/{exam_id}", response_model=dict)
async def update_exam(exam_id: str, exam_update: ExamUpdate, db=Depends(get_database)):
    try:
        update_data = {k: v for k, v in exam_update.dict().items() if v is not None}
        
        result = await db.exams.update_one(
            {"examId": exam_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        updated_exam = await db.exams.find_one({"examId": exam_id})
        updated_exam['_id'] = str(updated_exam['_id'])
        
        return updated_exam
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update exam")

@router.delete("/{exam_id}")
async def delete_exam(exam_id: str, db=Depends(get_database)):
    try:
        result = await db.exams.delete_one({"examId": exam_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        return {"message": "Exam deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete exam")
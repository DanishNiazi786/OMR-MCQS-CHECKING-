# C:\Users\Administrator\Downloads\Updated Omr\pServer\routers\students.py
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List
import pandas as pd
import io
from models.student import StudentCreate, StudentResponse
from datetime import datetime

router = APIRouter()

def get_database():
    from main import app
    return app.state.database

@router.post("/{exam_id}/upload")
async def upload_students(
    exam_id: str,
    file: UploadFile = File(...),
    db=Depends(get_database)
):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be Excel format")
        
        # Verify exam exists
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        # Read Excel file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Excel file is empty")
        
        # Validate required columns
        required_columns = ['Name', 'Locker number', 'Rank']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Clear existing students for this exam
        await db.students.delete_many({"examId": exam_id})
        
        # Process and save students
        students = []
        for i, row in df.iterrows():
            if pd.isna(row['Name']) or pd.isna(row['Locker number']) or pd.isna(row['Rank']):
                continue  # Skip empty rows
            
            student_data = {
                "examId": exam_id,
                "name": str(row['Name']).strip(),
                "lockerNumber": str(row['Locker number']).strip(),
                "rank": str(row['Rank']).strip(),
                "copyNumber": str(i + 1).zfill(3),
                "createdAt": datetime.utcnow()
            }
            students.append(student_data)
        
        if not students:
            raise HTTPException(status_code=400, detail="No valid student data found")
        
        await db.students.insert_many(students)
        
        # Update exam to mark students as uploaded
        await db.exams.update_one(
            {"examId": exam_id},
            {"$set": {"studentsUploaded": True}}
        )
        
        return {
            "message": "Students uploaded successfully",
            "count": len(students),
            "students": [
                {
                    "name": s["name"],
                    "lockerNumber": s["lockerNumber"],
                    "rank": s["rank"],
                    "copyNumber": s["copyNumber"]
                }
                for s in students
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to upload students")

@router.get("/{exam_id}", response_model=List[dict])
async def get_students(exam_id: str, db=Depends(get_database)):
    try:
        cursor = db.students.find({"examId": exam_id}).sort("copyNumber", 1)
        students = await cursor.to_list(length=None)
        
        # Convert ObjectId to string
        for student in students:
            student['_id'] = str(student['_id'])
        
        return students
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch students")
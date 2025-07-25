from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PIL import Image
from datetime import datetime
from .result import ResultCreate, ResultResponse
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

def get_database():
    from main import app
    return app.state.database

class BatchPDFRequest(BaseModel):
    examId: str
    examName: str
    results: List[dict]

@router.post("/save")
async def save_result(result: ResultCreate, db=Depends(get_database)):
    try:
        logger.info(f"Received result for saving: {result.dict()}")
        result_data = result.dict()
        result_data["processedAt"] = datetime.utcnow()

        existing_result = await db.results.find_one({
            "examId": result.examId,
            "studentId": result.studentId
        })

        if existing_result:
            await db.results.update_one(
                {"examId": result.examId, "studentId": result.studentId},
                {"$set": result_data}
            )
            logger.info(f"Updated existing result for student {result.studentId}")
        else:
            await db.results.insert_one(result_data)
            logger.info(f"Inserted new result for student {result.studentId}")

        return {"message": "Result saved successfully"}
    except ValueError as ve:
        logger.error(f"Validation error saving result: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        logger.error(f"Failed to save result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save result: {str(e)}")

@router.post("/publish")
async def publish_results(publish_data: dict, db=Depends(get_database)):
    try:
        logger.info(f"Received publish request: {json.dumps(publish_data, default=str)}")
        exam_id = publish_data.get("examId")
        exam_name = publish_data.get("examName")
        results = publish_data.get("results", [])
        
        logger.info(f"Publishing {len(results)} results for exam {exam_id}")

        if not exam_id or not results:
            logger.error("Missing examId or results in publish request")
            raise HTTPException(status_code=400, detail="examId and results are required")

        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            logger.error(f"Exam {exam_id} not found")
            raise HTTPException(status_code=404, detail="Exam not found")

        for result in results:
            result_data = {
                "examId": exam_id,
                "examName": exam_name,
                "studentId": result.get("studentId"),
                "studentName": result.get("studentName"),
                "rank": result.get("rank"),
                "lockerNumber": result.get("lockerNumber"),
                "score": result.get("score"),
                "totalMarks": result.get("totalMarks"),
                "percentage": result.get("percentage"),
                "passFailStatus": result.get("passFailStatus"),
                "correctAnswers": result.get("correctAnswers"),
                "incorrectAnswers": result.get("incorrectAnswers"),
                "blankAnswers": result.get("blankAnswers"),
                "multipleMarks": result.get("multipleMarks"),
                "responses": result.get("responses"),
                "sponsorDS": result.get("sponsorDS"),
                "course": result.get("course"),
                "wing": result.get("wing"),
                "module": result.get("module"),
                "publishedAt": datetime.utcnow()
            }

            existing_result = await db.results.find_one({
                "examId": exam_id,
                "studentId": result.get("studentId")
            })

            if existing_result:
                await db.results.update_one(
                    {"examId": exam_id, "studentId": result.get("studentId")},
                    {"$set": result_data}
                )
                logger.info(f"Updated published result for student {result.get('studentId')}")
            else:
                await db.results.insert_one(result_data)
                logger.info(f"Inserted published result for student {result.get('studentId')}")

        return {"message": f"Successfully published {len(results)} results for exam {exam_id}"}
    except Exception as e:
        logger.error(f"Failed to publish results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to publish results: {str(e)}")

@router.get("/all", response_model=List[dict])
async def get_all_results(db=Depends(get_database)):
    try:
        logger.info("Fetching all results")
        cursor = db.results.find().sort("processedAt", -1)
        results = await cursor.to_list(length=None)
        
        for result in results:
            result['_id'] = str(result['_id'])
        
        logger.info(f"Retrieved {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Failed to fetch results: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch results")

@router.post("/download-batch-pdf")
async def download_batch_pdf(request_data: BatchPDFRequest, images: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received batch PDF request: {request_data.dict()}")
        logger.info(f"Received {len(images)} images")
        exam_id = request_data.examId
        exam_name = request_data.examName
        results = request_data.results
        
        logger.info(f"Generating batch PDF for exam {exam_id} with {len(results)} results")

        if not exam_id or not exam_name or not results:
            logger.error("Missing examId, examName, or results in request")
            raise HTTPException(status_code=400, detail="examId, examName, and results are required")

        if len(images) != len(results):
            logger.error(f"Number of images ({len(images)}) does not match number of results ({len(results)})")
            raise HTTPException(status_code=400, detail=f"Number of images ({len(images)}) must match number of results ({len(results)})")

        required_fields = ["studentId", "studentName", "score", "totalMarks", "passFailStatus"]
        for result in results:
            missing_fields = [field for field in required_fields if field not in result or result[field] is None]
            if missing_fields:
                logger.error(f"Result missing required fields: {missing_fields}")
                raise HTTPException(status_code=400, detail=f"Result missing required fields: {missing_fields}")

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title page
        p.setFont("Helvetica-Bold", 20)
        p.drawCentredString(width/2, height-50, "OMR Checked Answer Sheets")
        p.setFont("Helvetica", 14)
        p.drawCentredString(width/2, height-80, f"Exam: {exam_name}")
        p.drawCentredString(width/2, height-100, f"Generated: {datetime.now().strftime('%Y-%m-%d')}")

        # Process each image and result
        for i, (image, result) in enumerate(zip(images, results)):
            if i > 0:
                p.showPage()

            # Read image data
            image_data = await image.read()
            img = Image.open(io.BytesIO(image_data))
            
            # Resize image to fit page while maintaining aspect ratio
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            target_width = width - 100
            target_height = target_width * aspect
            
            if target_height > height - 150:
                target_height = height - 150
                target_width = target_height / aspect
            
            # Draw image
            p.drawImage(ImageReader(io.BytesIO(image_data)), 50, 100, target_width, target_height)

            # Header
            p.setFont("Helvetica-Bold", 16)
            p.drawCentredString(width/2, height-50, f"Answer Sheet - {result['studentName']}")
            p.setFont("Helvetica", 12)
            p.drawCentredString(width/2, height-70, f"Student ID: {result['studentId']}")
            p.drawCentredString(width/2, height-90, f"Rank: {result.get('rank', 'N/A')}")
            p.drawCentredString(width/2, height-110, f"Locker: {result.get('lockerNumber', 'N/A')}")

            # Footer with result
            p.setFont("Helvetica-Bold", 12)
            p.setFillColorRGB(1, 0, 0)  # Red color
            result_text = f"Result: {result['score']}/{result['totalMarks']} - {result['passFailStatus']}"
            p.drawCentredString(width/2, 50, result_text)
            p.setFillColorRGB(0, 0, 0)  # Reset to black

        p.save()
        buffer.seek(0)

        logger.info(f"Generated batch PDF for exam {exam_id}")
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={exam_name}_Checked_Sheets.pdf"}
        )

    except Exception as e:
        logger.error(f"Failed to generate PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@router.post("/download-all-pdf")
async def download_all_pdf(request_data: dict):
    try:
        logger.info(f"Received summary PDF request: {json.dumps(request_data, default=str)}")
        results = request_data.get("results", [])
        filters = request_data.get("filters", {})
        
        logger.info(f"Generating summary PDF with {len(results)} results")

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        p.setFont("Helvetica-Bold", 20)
        p.drawCentredString(width/2, height-50, "OMR Results Report")
        p.setFont("Helvetica", 14)
        p.drawCentredString(width/2, height-80, f"Generated: {datetime.now().strftime('%Y-%m-%d')}")

        total_students = len(results)
        passed_students = len([r for r in results if r.get('passFailStatus') == 'Pass'])
        failed_students = total_students - passed_students
        avg_percentage = sum(r.get('percentage', 0) for r in results) / total_students if total_students > 0 else 0

        y_pos = height - 150
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Summary Statistics")
        p.setFont("Helvetica", 12)
        y_pos -= 20
        p.drawString(50, y_pos, f"Total Students: {total_students}")
        y_pos -= 15
        p.drawString(50, y_pos, f"Passed: {passed_students} ({(passed_students/total_students*100):.1f}%)")
        y_pos -= 15
        p.drawString(50, y_pos, f"Failed: {failed_students} ({(failed_students/total_students*100):.1f}%)")
        y_pos -= 15
        p.drawString(50, y_pos, f"Average Score: {avg_percentage:.1f}%")

        y_pos -= 40
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Individual Results")
        
        y_pos -= 30
        p.setFont("Helvetica", 8)
        p.drawString(50, y_pos, "Student Name")
        p.drawString(150, y_pos, "ID")
        p.drawString(200, y_pos, "Exam")
        p.drawString(280, y_pos, "Score")
        p.drawString(320, y_pos, "%")
        p.drawString(350, y_pos, "Result")

        y_pos -= 20
        for result in results[:40]:
            if y_pos < 100:
                p.showPage()
                y_pos = height - 50

            p.drawString(50, y_pos, str(result.get('studentName', ''))[:15])
            p.drawString(150, y_pos, str(result.get('studentId', '')))
            p.drawString(200, y_pos, str(result.get('examName', ''))[:12])
            p.drawString(280, y_pos, f"{result.get('score', 0)}/{result.get('totalMarks', 0)}")
            p.drawString(320, y_pos, f"{result.get('percentage', 0):.1f}%")
            
            if result.get('passFailStatus') == 'Pass':
                p.setFillColorRGB(0, 0.5, 0)
            else:
                p.setFillColorRGB(1, 0, 0)
            p.drawString(350, y_pos, str(result.get('passFailStatus', '')))
            p.setFillColorRGB(0, 0, 0)
            
            y_pos -= 15

        p.save()
        buffer.seek(0)

        logger.info("Generated summary PDF")
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=OMR_Results_Report.pdf"}
        )

    except Exception as e:
        logger.error(f"Failed to generate summary PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

@router.get("/exam/{exam_id}")
async def get_exam_results(
    exam_id: str,
    page: int = 1,
    limit: int = 20,
    sort_by: str = "processedAt",
    order: str = "desc",
    db=Depends(get_database)
):
    try:
        logger.info(f"Fetching results for exam {exam_id}")
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            logger.error(f"Exam {exam_id} not found")
            raise HTTPException(status_code=404, detail="Exam not found")

        sort_order = -1 if order == "desc" else 1
        skip = (page - 1) * limit

        cursor = db.results.find({"examId": exam_id}).sort(sort_by, sort_order).skip(skip).limit(limit)
        responses = await cursor.to_list(length=None)

        total = await db.results.count_documents({"examId": exam_id})

        for response in responses:
            response['_id'] = str(response['_id'])

        stats = await calculate_exam_stats(exam_id, db)

        logger.info(f"Retrieved {len(responses)} results for exam {exam_id}")
        return {
            "responses": responses,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": (total + limit - 1) // limit
            },
            "statistics": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch results for exam {exam_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch results")

async def calculate_exam_stats(exam_id: str, db):
    try:
        logger.info(f"Calculating stats for exam {exam_id}")
        cursor = db.results.find({"examId": exam_id})
        responses = await cursor.to_list(length=None)
        
        exam = await db.exams.find_one({"examId": exam_id})

        if not responses:
            logger.info(f"No results found for exam {exam_id}")
            return {
                "totalStudents": 0,
                "averageScore": 0,
                "highestScore": 0,
                "lowestScore": 0,
                "passingRate": 0,
                "scoreDistribution": [],
                "questionAnalysis": []
            }

        scores = [r["score"] for r in responses]
        total_students = len(responses)
        average_score = sum(scores) / total_students if total_students > 0 else 0
        highest_score = max(scores) if scores else 0
        lowest_score = min(scores) if scores else 0
        
        passing_score = exam.get("settings", {}).get("passingScore", 60)
        passing_count = len([s for s in scores if (s / exam["numQuestions"]) * 100 >= passing_score])
        passing_rate = (passing_count / total_students) * 100 if total_students > 0 else 0

        ranges = [
            {"min": 0, "max": 20, "label": "0-20%"},
            {"min": 21, "max": 40, "label": "21-40%"},
            {"min": 41, "max": 60, "label": "41-60%"},
            {"min": 61, "max": 80, "label": "61-80%"},
            {"min": 81, "max": 100, "label": "81-100%"}
        ]

        score_distribution = []
        for range_item in ranges:
            count = len([s for s in scores if range_item["min"] <= (s / exam["numQuestions"]) * 100 <= range_item["max"]])
            score_distribution.append({
                "range": range_item["label"],
                "count": count,
                "percentage": (count / total_students) * 100 if total_students > 0 else 0
            })

        logger.info(f"Calculated stats for exam {exam_id}: {total_students} students, {passing_rate:.1f}% passing")
        return {
            "totalStudents": total_students,
            "averageScore": round(average_score, 2),
            "highestScore": highest_score,
            "lowestScore": lowest_score,
            "passingRate": round(passing_rate, 2),
            "scoreDistribution": score_distribution,
            "questionAnalysis": []
        }
    except Exception as e:
        logger.error(f"Failed to calculate stats for exam {exam_id}: {str(e)}")
        raise
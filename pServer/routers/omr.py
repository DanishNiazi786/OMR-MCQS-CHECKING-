from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import io
import zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import logging

router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database():
    from main import app
    return app.state.database

@router.get("/{exam_id}/sheets")
async def generate_omr_sheets(exam_id: str, db=Depends(get_database)):
    try:
        # Verify exam exists
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")

        # Get students
        cursor = db.students.find({"examId": exam_id}).sort("copyNumber", 1)
        students = await cursor.to_list(length=None)
        
        if not students:
            raise HTTPException(status_code=400, detail="No students found for this exam")

        sheets = []
        for student in students:
            sheet = generate_omr_sheet(exam, student)
            sheets.append(sheet)

        return {
            "examName": exam["name"],
            "totalSheets": len(sheets),
            "sheets": [
                {
                    "studentName": sheet["studentName"],
                    "copyNumber": sheet["copyNumber"],
                    "previewData": sheet["previewData"]
                }
                for sheet in sheets
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating OMR sheets for exam {exam_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate OMR sheets: {str(e)}")

@router.get("/{exam_id}/download")
async def download_omr_sheets(exam_id: str, db=Depends(get_database)):
    try:
        # Verify exam exists
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")

        # Get students
        cursor = db.students.find({"examId": exam_id}).sort("copyNumber", 1)
        students = await cursor.to_list(length=None)
        
        if not students:
            raise HTTPException(status_code=400, detail="No students found for this exam")

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for student in students:
                try:
                    pdf_buffer = generate_omr_pdf(exam, student)
                    # Sanitize student name for filename
                    sanitized_name = ''.join(c for c in student['name'] if c.isalnum() or c in (' ', '_')).replace(' ', '_')
                    filename = f"{student['copyNumber']}_{sanitized_name}_OMR.pdf"
                    zip_file.writestr(filename, pdf_buffer.getvalue())
                except Exception as e:
                    logger.error(f"Error generating PDF for student {student.get('name', 'unknown')}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to generate PDF for student {student.get('name', 'unknown')}: {str(e)}")

        zip_buffer.seek(0)

        # Sanitize exam name for ZIP filename
        sanitized_exam_name = ''.join(c for c in exam['name'] if c.isalnum() or c in (' ', '_')).replace(' ', '_')
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={sanitized_exam_name}_OMR_Sheets.zip"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_omr_sheets for exam {exam_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download OMR sheets: {str(e)}")

def generate_omr_sheet(exam, student):
    try:
        total_marks = exam["numQuestions"] * exam["marksPerMcq"]
        
        return {
            "studentName": student["name"],
            "copyNumber": student["copyNumber"],
            "previewData": {
                "header": {
                    "dateTime": exam["dateTime"],
                    "time": exam["time"],
                    "examSecret": "EXAM SECRET",
                    "copyNumber": student["copyNumber"]
                },
                "body": {
                    "centerLine": "SA & MW",
                    "examDetails": {
                        "wing": exam["wing"],
                        "course": exam["course"],
                        "module": exam["module"],
                        "sponsorDS": exam["sponsorDS"],
                        "numMcqs": exam["numQuestions"],
                        "marksPerMcq": exam["marksPerMcq"],
                        "totalMarks": total_marks,
                        "passingPercentage": exam["passingPercentage"]
                    },
                    "studentInfo": {
                        "lockerNumber": student["lockerNumber"],
                        "rank": student["name"],
                        "name": student["name"]
                    },
                    "instructions": exam.get("instructions", "Fill bubbles neatly with a black/blue pen, mark only one option per question—any extra, unclear, or incorrect marking will be considered wrong.")
                },
                "mcqSection": {
                    "questions": [
                        {
                            "number": i + 1,
                            "options": ["A", "B", "C", "D", "E"]
                        }
                        for i in range(exam["numQuestions"])
                    ]
                },
                "footer": {
                    "studentSignature": "",
                    "result": "",
                    "invigilatorSignature": ""
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating OMR sheet for student {student.get('name', 'unknown')}: {str(e)}")
        raise ValueError(f"Failed to generate OMR sheet: {str(e)}")

def generate_omr_pdf(exam, student):
    try:
        # Validate inputs
        required_exam_fields = ["dateTime", "time", "wing", "course", "module", "sponsorDS", "numQuestions", "marksPerMcq", "passingPercentage"]
        required_student_fields = ["name", "copyNumber", "lockerNumber", "rank"]
        if not all(key in exam for key in required_exam_fields):
            raise ValueError(f"Missing required exam fields: {', '.join(k for k in required_exam_fields if k not in exam)}")
        if not all(key in student for key in required_student_fields):
            raise ValueError(f"Missing required student fields: {', '.join(k for k in required_student_fields if k not in student)}")

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        margin = 30
        
        # Header
        p.setFont("Helvetica", 12)
        p.drawString(margin, height - 50, exam["dateTime"])
        p.setFont("Helvetica-Bold", 16)
        p.drawCentredString(width/2, height - 60, "EXAM SECRET")
        p.setFont("Helvetica", 12)
        p.drawString(margin, height - 80, exam["time"])
        p.drawRightString(width - margin, height - 50, f"Copy #: {student['copyNumber']}")

        # Body - Center line
        y_pos = height - 110
        p.setFont("Helvetica-Bold", 14)
        p.drawCentredString(width/2, y_pos, "SA & MW")

        # Wing below SA & MW
        y_pos -= 20
        p.setFont("Helvetica-Bold", 12)
        p.drawCentredString(width/2, y_pos, exam["wing"])

        # Exam details
        y_pos -= 30
        p.setFont("Helvetica-Bold", 10)
        p.drawString(margin, y_pos, f"COURSE: {exam['course']}")
        p.drawString(width/2 + 20, y_pos, f"MCQs: {exam['numQuestions']}")
        y_pos -= 15
        p.drawString(margin, y_pos, f"MODULE: {exam['module']}")
        p.drawString(width/2 + 20, y_pos, f"Total Marks: {exam['numQuestions'] * exam['marksPerMcq']}")
        y_pos -= 15
        p.drawString(margin, y_pos, f"Sponsor DS: {exam['sponsorDS']}")
        p.drawString(margin + 120, y_pos, "Instructor Initial: ______")
        y_pos -= 15
        p.drawString(width/2 + 20, y_pos, f"Passing Percentage: {exam['passingPercentage']}%")

        # Student info and instructions boxes
        y_pos -= 20
        box_width = (width - 3 * margin) / 2
        box_height = 80

        # Student info box
        p.rect(margin, y_pos - box_height, box_width, box_height)
        p.setFont("Helvetica-Bold", 10)
        p.drawString(margin + 10, y_pos - 20, "STUDENT INFORMATION")
        p.setFont("Helvetica", 9)
        p.drawString(margin + 10, y_pos - 35, f"Locker Number: {student['lockerNumber']}")
        p.drawString(margin + 10, y_pos - 50, f"Rank: {student['rank']}")
        p.drawString(margin + 10, y_pos - 65, f"Name: {student['name']}")

        # Instructions box
        instructions_x = margin + box_width + margin
        p.rect(instructions_x, y_pos - box_height, box_width, box_height)
        p.setFont("Helvetica-Bold", 10)
        p.drawString(instructions_x + 10, y_pos - 20, "INSTRUCTIONS")
        p.setFont("Helvetica", 8)
        instructions = exam.get("instructions", "Please fill the bubbles completely with a dark pencil. Mark only one answer per question.")
        
        # Split instructions into lines
        lines = []
        words = instructions.split()
        current_line = ""
        for word in words:
            if len(current_line + word) < 35:  # Approximate character limit per line
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        line_y = y_pos - 35
        for line in lines[:3]:  # Limit to 3 lines
            p.drawString(instructions_x + 10, line_y, line)
            line_y -= 12

        # MCQ bubbles section
        y_pos -= box_height + 30
        p.setFont("Helvetica-Bold", 12)
        p.drawCentredString(width/2, y_pos, "ANSWER SHEET")

        # Bubble sheet
        y_pos -= 30
        num_questions = min(exam["numQuestions"], 200)
        num_columns = 4
        max_questions_per_column = (num_questions + num_columns - 1) // num_columns
        column_width = (width - 2 * margin) / num_columns
        
        bubble_size = 8
        bubble_spacing = 4
        row_height = 15
        
        for col in range(num_columns):
            for row in range(max_questions_per_column):
                question_index = col * max_questions_per_column + row
                if question_index >= num_questions:
                    break
                
                question_x = margin + col * column_width
                question_y = y_pos - row * row_height
                
                if question_y > 100:  # Ensure we don't go below footer area
                    # Question number
                    p.setFont("Helvetica-Bold", 8)
                    p.drawString(question_x, question_y, f"{question_index + 1}")
                    
                    # Bubbles
                    option_start_x = question_x + 25
                    for j in range(5):  # A, B, C, D, E
                        bubble_x = option_start_x + j * (bubble_size + bubble_spacing)
                        p.circle(bubble_x + bubble_size/2, question_y + bubble_size/2, bubble_size/2, fill=0)
                        p.setFont("Helvetica", 6)
                        p.drawCentredString(bubble_x + bubble_size/2, question_y + bubble_size/2 - 2, ["A", "B", "C", "D", "E"][j])

        # Footer
        footer_y = 50
        p.setFont("Helvetica", 8)
        p.drawString(margin, footer_y, "Student Signature: ________________")
        p.drawCentredString(width/2, footer_y, "Result: ________________")
        p.drawRightString(width - margin, footer_y, "Invigilator Signature: ________________")

        p.save()
        buffer.seek(0)
        return buffer

    except Exception as e:
        logger.error(f"Failed to generate OMR PDF for student {student.get('name', 'unknown')}: {str(e)}")
        raise ValueError(f"Failed to generate OMR PDF: {str(e)}")
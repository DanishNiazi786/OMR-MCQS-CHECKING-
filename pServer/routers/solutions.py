from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from models.solution import SolutionCreate, SolutionResponse, SolutionItem
from datetime import datetime
import pdfplumber
from typing import List
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def get_database():
    from main import app
    return app.state.database

@router.post("/{exam_id}/upload")
async def upload_solution(
    exam_id: str,
    file: UploadFile = File(...),
    db=Depends(get_database)
):
    try:
        # Verify exam exists
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        logger.info(f"Processing solution for exam {exam_id} with {exam['numQuestions']} questions")
        
        # Process PDF to extract solutions
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        valid_solutions = []
        with pdfplumber.open(file.file) as pdf:
            all_chars = []
            # Collect all characters from all pages
            for page in pdf.pages:
                chars = page.chars
                all_chars.extend(chars)
            
            # Log font information for debugging
            font_info = set((c.get("fontname", "Unknown"), c.get("size", 0)) for c in all_chars)
            logger.info(f"Font info: {font_info}")
            
            # Group characters by approximate line
            lines = []
            current_line = []
            last_y0 = None
            y0_threshold = 5  # Flexible threshold for line grouping
            
            sorted_chars = sorted(all_chars, key=lambda c: (c["page_number"], -c["y0"], c["x0"]))
            
            for char in sorted_chars:
                if "text" not in char:
                    continue
                if last_y0 is None or abs(char["y0"] - last_y0) < y0_threshold:
                    current_line.append(char)
                else:
                    if current_line:
                        line_text = "".join(c["text"] for c in sorted(current_line, key=lambda c: c["x0"]))
                        avg_font_size = sum(c.get("size", 0) for c in current_line) / len(current_line) if current_line else 0
                        lines.append((line_text.strip(), current_line, avg_font_size))
                    current_line = [char]
                last_y0 = char["y0"]
            
            if current_line:
                line_text = "".join(c["text"] for c in sorted(current_line, key=lambda c: c["x0"]))
                avg_font_size = sum(c.get("size", 0) for c in current_line) / len(current_line) if current_line else 0
                lines.append((line_text.strip(), current_line, avg_font_size))
            
            logger.info(f"Extracted lines: {[line[0] for line in lines]}")
            
            # Flexible regex patterns
            question_pattern = r"^\s*(\d+)\s*(?:[\.\)]|\s*Question\b.*)?$"
            option_pattern = r"^\s*([a-eA-E])[\.\)]\s*.*$"
            
            current_question = None
            question_options = {}
            
            for line_text, line_chars, avg_font_size in lines:
                # Match question number
                question_match = re.match(question_pattern, line_text, re.IGNORECASE)
                if question_match:
                    current_question = int(question_match.group(1))
                    if current_question not in question_options:
                        question_options[current_question] = []
                    logger.info(f"Detected question: {current_question}")
                    continue
                
                # Match option letter
                option_match = re.match(option_pattern, line_text, re.IGNORECASE)
                if option_match and current_question:
                    option_letter = option_match.group(1).upper()
                    question_options[current_question].append({
                        "option": option_letter,
                        "text": line_text,
                        "font_size": avg_font_size
                    })
                    logger.info(f"Question {current_question}, Option {option_letter}: '{line_text}' - Avg Font Size: {avg_font_size}")
                else:
                    logger.info(f"Unmatched line: '{line_text}'")
            
            # Determine correct answer by comparing font sizes
            for question, options in question_options.items():
                if not options:
                    continue
                # Find option with the largest font size
                max_font_size = max(opt["font_size"] for opt in options)
                # Allow some tolerance for floating-point comparison
                correct_option = next(
                    (opt for opt in options if abs(opt["font_size"] - max_font_size) < 0.1 and opt["option"] in ['A', 'B', 'C', 'D', 'E']),
                    None
                )
                if correct_option:
                    valid_solutions.append({
                        "question": question,
                        "answer": correct_option["option"]
                    })
            
            # Remove duplicates and sort by question number
            seen_questions = set()
            valid_solutions = [
                sol for sol in sorted(valid_solutions, key=lambda x: x["question"])
                if not (sol["question"] in seen_questions or seen_questions.add(sol["question"]))
            ]
        
        logger.info(f"Extracted {len(valid_solutions)} solutions: {valid_solutions}")
        
        # Validate number of answers matches number of MCQs
        if len(valid_solutions) != exam['numQuestions']:
            raise HTTPException(
                status_code=400,
                detail=f"Error: Expected {exam['numQuestions']} answers, but extracted {len(valid_solutions)}. Ensure each question (1 to {exam['numQuestions']}) is followed by options (a. to e.), with the correct answer having a larger font size. Extracted lines: {[line[0] for line in lines]}"
            )
        
        # Validate solutions format
        for sol in valid_solutions:
            if not (isinstance(sol, dict) and 
                    'question' in sol and 
                    isinstance(sol['question'], int) and 
                    'answer' in sol and 
                    sol['answer'] in ['A', 'B', 'C', 'D', 'E']):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid solution format for question {sol.get('question', 'unknown')}. Answer must be A, B, C, D, or E."
                )
        
        # Remove existing solution if any
        await db.solutions.delete_one({"examId": exam_id})
        
        # Save new solution
        solution_data = {
            "examId": exam_id,
            "solutions": valid_solutions,
            "uploadedAt": datetime.utcnow()
        }
        
        await db.solutions.insert_one(solution_data)
        
        # Update exam to mark solution as uploaded and store answer key
        await db.exams.update_one(
            {"examId": exam_id},
            {"$set": {"solutionUploaded": True, "answerKey": [sol['answer'] for sol in valid_solutions]}}
        )
        
        return {
            "message": "Solution uploaded successfully",
            "solutionCount": len(valid_solutions)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload solution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload solution: {str(e)}")

@router.post("/{exam_id}/manual")
async def save_manual_solution(
    exam_id: str,
    solution_data: dict,
    db=Depends(get_database)
):
    try:
        # Verify exam exists
        exam = await db.exams.find_one({"examId": exam_id})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        solutions = solution_data.get("solutions", [])
        
        # Validate solutions
        if len(solutions) != exam['numQuestions']:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {exam['numQuestions']} solutions, got {len(solutions)}"
            )
        
        # Validate each solution
        for sol in solutions:
            if not isinstance(sol, dict) or 'question' not in sol or 'answer' not in sol:
                raise HTTPException(status_code=400, detail="Invalid solution format")
            if sol['answer'] not in ['A', 'B', 'C', 'D', 'E']:
                raise HTTPException(status_code=400, detail=f"Invalid answer '{sol['answer']}' for question {sol['question']}")
        
        # Remove existing solution if any
        await db.solutions.delete_one({"examId": exam_id})
        
        # Save new solution
        solution_doc = {
            "examId": exam_id,
            "solutions": solutions,
            "uploadedAt": datetime.utcnow()
        }
        
        await db.solutions.insert_one(solution_doc)
        
        # Update exam to mark solution as uploaded
        await db.exams.update_one(
            {"examId": exam_id},
            {"$set": {"solutionUploaded": True}}
        )
        
        return {
            "message": "Manual solution saved successfully",
            "solutionCount": len(solutions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save manual solution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save manual solution: {str(e)}")
@router.get("/{exam_id}", response_model=dict)
async def get_solution(exam_id: str, db=Depends(get_database)):
    try:
        solution = await db.solutions.find_one({"examId": exam_id})
        if not solution:
            raise HTTPException(status_code=404, detail="Solution not found")
        
        solution['_id'] = str(solution['_id'])
        return solution
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch solution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch solution: {str(e)}")
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from datetime import datetime
import cv2
import numpy as np
import logging
import io
import tempfile
import os
from typing import List, Tuple, Dict, Optional
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

def get_database():
    from main import app
    return app.state.database

# OMR Processing Configuration
NUM_CHOICES = 5
NUM_COLUMNS = 4
MIN_BUBBLE_AREA = 80
MAX_BUBBLE_AREA = 1600
ASPECT_RATIO_RANGE = (0.5, 2.0)
CIRCULARITY_THRESHOLD = 0.8
FILL_THRESHOLD = 0.60
PARTIAL_FILL_THRESHOLD = 0.25
ROW_TOLERANCE = 50
COLUMN_TOLERANCE = 150
MIN_CONTOUR_POINTS = 5
GAUSSIAN_BLUR_SIZE = 5
MORPH_KERNEL_SIZE = 3

class BubbleState:
    BLANK = "BLANK"
    FILLED = "FILLED" 
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"

def preprocess_scanned_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Preprocessing scanned image")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    blurred = cv2.GaussianBlur(denoised, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    _, thresh_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, thresh_manual = cv2.threshold(
        blurred, 180, 255, cv2.THRESH_BINARY_INV
    )
    
    thresh_combined = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
    thresh_combined = cv2.bitwise_or(thresh_combined, thresh_manual)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    return gray, thresh_combined, blurred

def detect_bubbles_scanned(thresh: np.ndarray, gray: np.ndarray) -> List[Tuple[np.ndarray, int, int, int, int]]:
    logger.info("Detecting bubble contours in scanned image")
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    
    img_height, img_width = thresh.shape
    relative_min_area = (img_width * img_height) * 0.00005
    relative_max_area = (img_width * img_height) * 0.002
    
    min_area = max(MIN_BUBBLE_AREA, relative_min_area)
    max_area = min(MAX_BUBBLE_AREA, relative_max_area)
    
    logger.info(f"Using area range: {min_area:.0f} - {max_area:.0f}")
    
    header_boundary = int(img_height * 0.4)
    footer_boundary = int(img_height * 0.95)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            if y < header_boundary or y > footer_boundary:
                continue
                
            aspect_ratio = w / float(h)
            
            if ASPECT_RATIO_RANGE[0] < aspect_ratio < ASPECT_RATIO_RANGE[1]:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > CIRCULARITY_THRESHOLD:
                        if is_valid_bubble_scanned(contour, gray[y:y+h, x:x+w]):
                            if not is_text_or_line(contour, x, y, w, h, gray):
                                bubble_contours.append((contour, x, y, w, h))
    
    logger.info(f"Found {len(bubble_contours)} potential bubble contours")
    return bubble_contours

def is_text_or_line(contour: np.ndarray, x: int, y: int, w: int, h: int, gray: np.ndarray) -> bool:
    if w < 10 or h < 10 or w > h * 3 or h > w * 3:
        return True
    
    region = gray[y:y+h, x:x+w]
    if region.size == 0:
        return True
    
    edges = cv2.Canny(region, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    if edge_ratio > 0.3:
        return True
    
    img_height = gray.shape[0]
    relative_y = y / img_height
    
    if relative_y < 0.35 or relative_y > 0.95:
        return True
    
    return False

def is_valid_bubble_scanned(contour: np.ndarray, region: np.ndarray) -> bool:
    if len(contour) < MIN_CONTOUR_POINTS:
        return False
    
    try:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
            contour_area = cv2.contourArea(contour)
            
            if ellipse_area > 0:
                area_ratio = contour_area / ellipse_area
                if 0.4 < area_ratio < 1.6:
                    return True
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area > 0:
            solidity = contour_area / hull_area
            return solidity > 0.6
            
    except Exception as e:
        logger.debug(f"Validation error: {e}")
        return False
    
    return False

def group_bubbles_scanned(bubble_contours: List[Tuple], num_questions: int) -> List[List[Dict]]:
    logger.info(f"Grouping bubbles for {num_questions} questions")
    
    if not bubble_contours:
        return []
    
    questions_per_column = num_questions // NUM_COLUMNS
    if num_questions % NUM_COLUMNS != 0:
        questions_per_column += 1
    
    logger.info(f"Expected {questions_per_column} questions per column")
    
    sorted_by_x = sorted(bubble_contours, key=lambda b: b[1])
    
    columns = []
    for bubble in sorted_by_x:
        x, y, w, h = bubble[1:5]
        center_x = x + w // 2
        
        best_match = None
        min_distance = float('inf')
        
        for i, column in enumerate(columns):
            distance = abs(center_x - column["center_x"])
            if distance < COLUMN_TOLERANCE and distance < min_distance:
                min_distance = distance
                best_match = i
        
        if best_match is not None:
            columns[best_match]["bubbles"].append(bubble)
            all_centers = [b[1] + b[3]//2 for b in columns[best_match]["bubbles"]]
            columns[best_match]["center_x"] = np.mean(all_centers)
        else:
            columns.append({
                "center_x": center_x,
                "bubbles": [bubble]
            })
    
    columns = sorted(columns, key=lambda c: len(c["bubbles"]), reverse=True)[:NUM_COLUMNS]
    columns = sorted(columns, key=lambda c: c["center_x"])
    
    logger.info(f"Found {len(columns)} columns with bubbles: {[len(c['bubbles']) for c in columns]}")
    
    all_rows = []
    for col_idx, column in enumerate(columns):
        logger.info(f"Processing column {col_idx + 1} with {len(column['bubbles'])} bubbles")
        
        sorted_bubbles = sorted(column["bubbles"], key=lambda b: b[2])
        
        rows = []
        for bubble in sorted_bubbles:
            x, y, w, h = bubble[1:5]
            center_y = y + h // 2
            
            best_row = None
            min_distance = float('inf')
            adaptive_tolerance = ROW_TOLERANCE * (1 + 0.1 * len(rows) / questions_per_column)
            
            for i, row in enumerate(rows):
                distance = abs(center_y - row["center_y"])
                if distance < adaptive_tolerance and distance < min_distance:
                    min_distance = distance
                    best_row = i
            
            if best_row is not None:
                rows[best_row]["bubbles"].append(bubble)
                all_y = [b[2] + b[4]//2 for b in rows[best_row]["bubbles"]]
                rows[best_row]["center_y"] = np.mean(all_y)
            else:
                rows.append({
                    "center_y": center_y,
                    "bubbles": [bubble],
                    "column": col_idx
                })
        
        rows = sorted(rows, key=lambda r: r["center_y"])
        
        current_column_questions = questions_per_column
        if col_idx == NUM_COLUMNS - 1:
            remaining_questions = num_questions - (col_idx * questions_per_column)
            current_column_questions = max(1, remaining_questions)
        
        if len(rows) > current_column_questions:
            rows = sorted(rows, key=lambda r: len(r["bubbles"]), reverse=True)[:current_column_questions]
            rows = sorted(rows, key=lambda r: r["center_y"])
        
        while len(rows) < current_column_questions:
            if rows:
                last_y = rows[-1]["center_y"]
                avg_row_height = (rows[-1]["center_y"] - rows[0]["center_y"]) / max(1, len(rows) - 1) if len(rows) > 1 else 30
                estimated_y = last_y + avg_row_height
            else:
                estimated_y = 100
            
            rows.append({
                "center_y": estimated_y,
                "bubbles": [],
                "column": col_idx
            })
            logger.warning(f"Added placeholder for missing row in column {col_idx + 1}")
        
        for row_idx, row in enumerate(rows):
            row["bubbles"].sort(key=lambda b: b[1])
            
            if len(row["bubbles"]) != NUM_CHOICES:
                logger.warning(f"Column {col_idx + 1}, Row {row_idx + 1} has {len(row['bubbles'])} bubbles, expected {NUM_CHOICES}")
                
                if len(row["bubbles"]) > NUM_CHOICES:
                    row["bubbles"] = filter_overlapping_bubbles(row["bubbles"])
        
        all_rows.extend(rows)
    
    all_rows = sorted(all_rows, key=lambda r: (r["column"], r["center_y"]))
    
    logger.info(f"Total rows found: {len(all_rows)}")
    return all_rows

def filter_overlapping_bubbles(bubbles: List[Tuple]) -> List[Tuple]:
    if len(bubbles) <= NUM_CHOICES:
        return bubbles
    
    filtered = []
    bubbles_with_scores = []
    
    for bubble in bubbles:
        contour, x, y, w, h = bubble
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            aspect_ratio = w / float(h)
            score = circularity * (1.0 / (abs(aspect_ratio - 1.0) + 0.1))
        else:
            score = 0
            
        bubbles_with_scores.append((bubble, score))
    
    bubbles_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    for bubble, score in bubbles_with_scores:
        if len(filtered) >= NUM_CHOICES:
            break
            
        x, y, w, h = bubble[1:5]
        is_overlapping = False
        
        for existing_bubble in filtered:
            ex_x, ex_y, ex_w, ex_h = existing_bubble[1:5]
            
            overlap_x = max(0, min(x + w, ex_x + ex_w) - max(x, ex_x))
            overlap_y = max(0, min(y + h, ex_y + ex_h) - max(y, ex_y))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > 0.3 * min(w * h, ex_w * ex_h):
                is_overlapping = True
                break
        
        if not is_overlapping:
            filtered.append(bubble)
    
    return filtered[:NUM_CHOICES]

def analyze_bubble_fill_scanned(gray: np.ndarray, contour: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[str, float]:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    bubble_region = gray[y:y+h, x:x+w]
    mask_region = mask[y:y+h, x:x+w]
    
    if bubble_region.size == 0 or mask_region.size == 0:
        return BubbleState.INVALID, 0.0
    
    bubble_pixels = bubble_region[mask_region > 0]
    
    if len(bubble_pixels) == 0:
        return BubbleState.INVALID, 0.0
    
    fill_ratio = calculate_fill_ratio_scanned(bubble_region, mask_region, bubble_pixels)
    
    if not is_valid_fill_pattern_scanned(bubble_region, mask_region):
        return BubbleState.INVALID, fill_ratio
    
    if fill_ratio > FILL_THRESHOLD:
        return BubbleState.FILLED, fill_ratio
    elif fill_ratio > PARTIAL_FILL_THRESHOLD:
        return BubbleState.PARTIAL, fill_ratio
    else:
        return BubbleState.BLANK, fill_ratio

def calculate_fill_ratio_scanned(bubble_region: np.ndarray, mask_region: np.ndarray, bubble_pixels: np.ndarray) -> float:
    if len(bubble_pixels) == 0:
        return 0.0
    
    mean_intensity = np.mean(bubble_pixels)
    background_intensity = np.mean(bubble_region[mask_region == 0]) if np.sum(mask_region == 0) > 0 else 255
    
    intensity_ratio = 1.0 - (mean_intensity / max(background_intensity, 1))
    intensity_ratio = max(0, min(1, intensity_ratio))
    
    threshold = max(np.mean(bubble_pixels) - np.std(bubble_pixels), 
                   np.percentile(bubble_pixels, 25))
    dark_pixels = np.sum(bubble_pixels < threshold)
    threshold_ratio = dark_pixels / len(bubble_pixels)
    
    try:
        masked_region = bubble_region.copy()
        masked_region[mask_region == 0] = 255
        _, otsu_thresh = cv2.threshold(masked_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_dark = np.sum((otsu_thresh < 127) & (mask_region > 0))
        otsu_ratio = otsu_dark / np.sum(mask_region > 0) if np.sum(mask_region > 0) > 0 else 0
    except:
        otsu_ratio = 0
    
    final_ratio = (0.4 * intensity_ratio + 0.3 * threshold_ratio + 0.3 * otsu_ratio)
    return max(0, min(1, final_ratio))

def is_valid_fill_pattern_scanned(bubble_region: np.ndarray, mask_region: np.ndarray) -> bool:
    if bubble_region.size == 0:
        return False
    
    edges = cv2.Canny(bubble_region, 30, 100)
    edge_pixels = edges[mask_region > 0]
    
    if len(edge_pixels) == 0:
        return True
    
    edge_ratio = np.sum(edge_pixels > 0) / len(edge_pixels)
    
    if edge_ratio > 0.4:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=5, minLineLength=3, maxLineGap=2)
        if lines is not None and len(lines) > 3:
            return False
    
    return True

def detect_marked_bubbles_scanned(gray: np.ndarray, rows: List[Dict], num_questions: int) -> List[Optional[int]]:
    logger.info(f"Detecting marked bubbles for {num_questions} questions")
    marked_answers = []
    
    for i, row in enumerate(rows):
        if i >= num_questions:
            break
            
        question_num = i + 1
        
        if len(row["bubbles"]) == 0:
            logger.warning(f"Question {question_num}: No bubbles found")
            marked_answers.append(None)
            continue
        
        bubble_states = []
        fill_ratios = []
        
        for j, (contour, x, y, w, h) in enumerate(row["bubbles"]):
            if j >= NUM_CHOICES:
                break
                
            state, ratio = analyze_bubble_fill_scanned(gray, contour, x, y, w, h)
            bubble_states.append(state)
            fill_ratios.append(ratio)
            
            choice_letter = chr(65 + j) if j < NUM_CHOICES else f"Extra{j}"
            logger.debug(f"Q{question_num} {choice_letter}: {state} (ratio: {ratio:.3f})")
        
        while len(bubble_states) < NUM_CHOICES:
            bubble_states.append(BubbleState.BLANK)
            fill_ratios.append(0.0)
        
        filled_indices = [j for j, state in enumerate(bubble_states) if state == BubbleState.FILLED]
        partial_indices = [j for j, state in enumerate(bubble_states) if state == BubbleState.PARTIAL]
        invalid_indices = [j for j, state in enumerate(bubble_states) if state == BubbleState.INVALID]
        
        if len(filled_indices) == 1:
            marked_answers.append(filled_indices[0])
            logger.info(f"Q{question_num}: Answer {chr(65 + filled_indices[0])}")
        elif len(filled_indices) > 1:
            logger.info(f"Q{question_num}: Multiple answers marked - Invalid")
            marked_answers.append(-2)
        elif len(filled_indices) == 0 and len(partial_indices) == 1:
            logger.info(f"Q{question_num}: Partial fill detected at {chr(65 + partial_indices[0])}")
            marked_answers.append(partial_indices[0])
        elif len(partial_indices) > 1:
            logger.info(f"Q{question_num}: Multiple partial fills - Invalid")
            marked_answers.append(-3)
        else:
            logger.info(f"Q{question_num}: No marks detected - Blank")
            marked_answers.append(None)
    
    while len(marked_answers) < num_questions:
        marked_answers.append(None)
    
    return marked_answers[:num_questions]

def score_answers_with_key(marked_answers: List[Optional[int]], answer_key: List[str], num_questions: int) -> Dict:
    logger.info(f"Scoring {len(marked_answers)} answers against answer key")
    
    score = 0
    attempted = 0
    multiple_marks = 0
    partial_marks = 0
    wrong_answers = 0
    blank_answers = 0
    invalid_answers = 0
    
    detailed_responses = []
    
    for i in range(num_questions):
        marked = marked_answers[i] if i < len(marked_answers) else None
        correct_letter = answer_key[i] if i < len(answer_key) else 'A'
        correct_index = ord(correct_letter) - ord('A')
        
        response_data = {
            "question": i + 1,
            "marked": chr(65 + marked) if marked is not None and marked >= 0 else None,
            "correct": correct_letter,
            "is_correct": False,
            "status": "blank"
        }
        
        if marked == correct_index:
            score += 1
            attempted += 1
            response_data["is_correct"] = True
            response_data["status"] = "correct"
        elif marked == -2:
            multiple_marks += 1
            attempted += 1
            response_data["status"] = "multiple"
            invalid_answers += 1
        elif marked == -3:
            partial_marks += 1
            attempted += 1
            response_data["status"] = "partial"
            invalid_answers += 1
        elif marked is None:
            blank_answers += 1
            response_data["status"] = "blank"
        elif marked >= 0:
            wrong_answers += 1
            attempted += 1
            response_data["status"] = "wrong"
        else:
            invalid_answers += 1
            attempted += 1
            response_data["status"] = "invalid"
        
        detailed_responses.append(response_data)
    
    accuracy = (score / attempted * 100) if attempted > 0 else 0
    confidence = max(50, min(95, 70 + (accuracy - 50) * 0.3))
    invalid_penalty = (multiple_marks + partial_marks) / num_questions * 20
    confidence = max(30, confidence - invalid_penalty)
    
    result = {
        "score": score,
        "total_questions": num_questions,
        "attempted": attempted,
        "correct_answers": score,
        "incorrect_answers": wrong_answers,
        "blank_answers": blank_answers,
        "multiple_marks": multiple_marks,
        "partial_marks": partial_marks,
        "invalid_answers": invalid_answers,
        "accuracy": accuracy,
        "responses": [resp["marked"] for resp in detailed_responses],
        "detailed_responses": detailed_responses,
        "processing_metadata": {
            "confidence": confidence,
            "bubbles_detected": True,
            "image_quality": "good"
        }
    }
    
    logger.info(f"Scoring complete: {score}/{num_questions} ({accuracy:.1f}%)")
    return result

def process_omr_image(image_data: bytes, answer_key: List[str], num_questions: int, student_id: str) -> Dict:
    try:
        logger.info(f"Processing OMR image for student {student_id} with {num_questions} questions")
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        logger.info(f"Image loaded - Dimensions: {img.shape[1]}x{img.shape[0]}")
        
        gray, thresh, blurred = preprocess_scanned_image(img)
        
        bubble_contours = detect_bubbles_scanned(thresh, gray)
        
        if not bubble_contours:
            raise ValueError("No bubble contours detected. Check image quality and parameters.")
        
        logger.info(f"Detected {len(bubble_contours)} bubble contours")
        
        rows = group_bubbles_scanned(bubble_contours, num_questions)
        
        if len(rows) == 0:
            raise ValueError("No structured rows detected. Check bubble grouping parameters.")
        
        logger.info(f"Organized {len(rows)} question rows")
        
        marked_answers = detect_marked_bubbles_scanned(gray, rows, num_questions)
        
        result = score_answers_with_key(marked_answers, answer_key, num_questions)
        
        result["studentId"] = student_id
        # Mock student credentials (replace with actual OCR extraction)
        result["studentName"] = f"Student {student_id.split('_')[1]}"
        result["rank"] = "Cadet"
        result["lockerNumber"] = f"L{student_id.split('_')[1].zfill(3)}"
        
        logger.info(f"OMR processing completed for student {student_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing OMR image: {str(e)}")
        raise ValueError(f"Failed to process OMR image: {str(e)}")

@router.post("/process")
async def process_answer_sheet(
    image: UploadFile = File(...),
    examId: str = Form(...),
    studentId: str = Form(...),
    db=Depends(get_database)
):
    try:
        logger.info(f"Processing answer sheet for exam {examId}, student {studentId}")
        
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        exam = await db.exams.find_one({"examId": examId})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        solution = await db.solutions.find_one({"examId": examId})
        if not solution:
            raise HTTPException(status_code=404, detail="Answer key not found for this exam")
        
        image_data = await image.read()
        
        answer_key = []
        for sol in sorted(solution["solutions"], key=lambda x: x["question"]):
            answer_key.append(sol["answer"])
        
        if len(answer_key) != exam["numQuestions"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Answer key length ({len(answer_key)}) doesn't match exam questions ({exam['numQuestions']})"
            )
        
        processing_result = process_omr_image(
            image_data=image_data,
            answer_key=answer_key,
            num_questions=exam["numQuestions"],
            student_id=studentId
        )
        
        total_marks = exam["numQuestions"] * exam["marksPerMcq"]
        score = processing_result["score"] * exam["marksPerMcq"]
        percentage = (score / total_marks * 100) if total_marks > 0 else 0
        pass_fail_status = "Pass" if percentage >= exam["passingPercentage"] else "Fail"
        
        response_data = {
            "success": True,
            "examId": examId,
            "studentId": studentId,
            "response": {
                "studentId": studentId,
                "studentName": processing_result["studentName"],
                "rank": processing_result["rank"],
                "lockerNumber": processing_result["lockerNumber"],
                "score": score,
                "totalMarks": total_marks,
                "percentage": percentage,
                "accuracy": processing_result["accuracy"],
                "responses": processing_result["responses"],
                "correctAnswers": processing_result["correct_answers"],
                "incorrectAnswers": processing_result["incorrect_answers"],
                "blankAnswers": processing_result["blank_answers"],
                "multipleMarks": processing_result["multiple_marks"],
                "invalidAnswers": processing_result["invalid_answers"],
                "passFailStatus": pass_fail_status,
                "processingMetadata": processing_result["processing_metadata"]
            }
        }
        
        logger.info(f"Processed answer sheet for student {studentId}: {score}/{total_marks}")
        return response_data
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process answer sheet: {str(e)}")

@router.post("/batch-process")
async def batch_process_answer_sheets(
    images: List[UploadFile] = File(...),
    examId: str = Form(...),
    db=Depends(get_database)
):
    try:
        logger.info(f"Batch processing {len(images)} answer sheets for exam {examId}")
        
        exam = await db.exams.find_one({"examId": examId})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        solution = await db.solutions.find_one({"examId": examId})
        if not solution:
            raise HTTPException(status_code=404, detail="Answer key not found for this exam")
        
        answer_key = []
        for sol in sorted(solution["solutions"], key=lambda x: x["question"]):
            answer_key.append(sol["answer"])
        
        if len(answer_key) != exam["numQuestions"]:
            raise HTTPException(
                status_code=400,
                detail=f"Answer key length ({len(answer_key)}) doesn't match exam questions ({exam['numQuestions']})"
            )
        
        results = []
        for index, image in enumerate(images):
            if not image.content_type.startswith('image/'):
                logger.warning(f"Skipping file {image.filename}: Not an image")
                continue
                
            student_id = f"STUDENT_{(index + 1):03d}"
            logger.info(f"Processing sheet {index + 1}/{len(images)} for student {student_id}")
            
            try:
                image_data = await image.read()
                
                processing_result = process_omr_image(
                    image_data=image_data,
                    answer_key=answer_key,
                    num_questions=exam["numQuestions"],
                    student_id=student_id
                )
                
                total_marks = exam["numQuestions"] * exam["marksPerMcq"]
                score = processing_result["score"] * exam["marksPerMcq"]
                percentage = (score / total_marks * 100) if total_marks > 0 else 0
                pass_fail_status = "Pass" if percentage >= exam["passingPercentage"] else "Fail"
                
                results.append({
                    "studentId": student_id,
                    "studentName": processing_result["studentName"],
                    "rank": processing_result["rank"],
                    "lockerNumber": processing_result["lockerNumber"],
                    "score": score,
                    "totalMarks": total_marks,
                    "percentage": percentage,
                    "accuracy": processing_result["accuracy"],
                    "responses": processing_result["responses"],
                    "correctAnswers": processing_result["correct_answers"],
                    "incorrectAnswers": processing_result["incorrect_answers"],
                    "blankAnswers": processing_result["blank_answers"],
                    "multipleMarks": processing_result["multiple_marks"],
                    "invalidAnswers": processing_result["invalid_answers"],
                    "passFailStatus": pass_fail_status,
                    "processingMetadata": processing_result["processing_metadata"]
                })
                
                logger.info(f"Processed sheet {index + 1}: {score}/{total_marks}")
                
            except Exception as e:
                logger.error(f"Failed to process sheet {index + 1}: {str(e)}")
                results.append({
                    "studentId": student_id,
                    "studentName": f"Student {(index + 1):03d}",
                    "rank": "Unknown",
                    "lockerNumber": "Unknown",
                    "error": f"Processing failed: {str(e)}"
                })
        
        logger.info(f"Batch processing completed: {len(results)} sheets processed")
        return {
            "success": True,
            "examId": examId,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")
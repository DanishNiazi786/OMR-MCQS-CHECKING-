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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

def get_database():
    from main import app
    return app.state.database

# OMR Processing Configuration
NUM_CHOICES = 5  # A, B, C, D, E
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
    """Enum-like class for bubble states"""
    BLANK = "BLANK"
    FILLED = "FILLED" 
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"

def preprocess_scanned_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced preprocessing specifically for scanned images."""
    logger.info("Preprocessing scanned image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for scanned images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Remove noise common in scanned images
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(denoised, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    
    # Multiple thresholding approaches for scanned images
    # 1. Adaptive threshold
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    # 2. Otsu's threshold
    _, thresh_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # 3. Manual threshold for scanned images (often works better)
    _, thresh_manual = cv2.threshold(
        blurred, 180, 255, cv2.THRESH_BINARY_INV
    )
    
    # Combine thresholds
    thresh_combined = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
    thresh_combined = cv2.bitwise_or(thresh_combined, thresh_manual)
    
    # Morphological operations to clean up scanned artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Close small gaps
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small noise
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Additional cleaning for scanned images
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    return gray, thresh_combined, blurred

def detect_bubbles_scanned(thresh: np.ndarray, gray: np.ndarray) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """Enhanced bubble detection for scanned images with better filtering."""
    logger.info("Detecting bubble contours in scanned image")
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    
    # Calculate image dimensions for relative sizing
    img_height, img_width = thresh.shape
    relative_min_area = (img_width * img_height) * 0.00005  # 0.005% of image
    relative_max_area = (img_width * img_height) * 0.002    # 0.2% of image
    
    min_area = max(MIN_BUBBLE_AREA, relative_min_area)
    max_area = min(MAX_BUBBLE_AREA, relative_max_area)
    
    logger.info(f"Using area range: {min_area:.0f} - {max_area:.0f}")
    
    # Define the main answer sheet region (exclude header area)
    header_boundary = int(img_height * 0.4)  # Skip top 40% to avoid header detection
    footer_boundary = int(img_height * 0.95)  # Skip bottom 5% to avoid footer
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out contours in header/footer areas
            if y < header_boundary or y > footer_boundary:
                continue
                
            aspect_ratio = w / float(h)
            
            # Check aspect ratio
            if ASPECT_RATIO_RANGE[0] < aspect_ratio < ASPECT_RATIO_RANGE[1]:
                # Calculate circularity (more lenient for scanned images)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > CIRCULARITY_THRESHOLD:
                        # Additional validation for scanned images
                        if is_valid_bubble_scanned(contour, gray[y:y+h, x:x+w]):
                            # Extra check: ensure it's not text or lines
                            if not is_text_or_line(contour, x, y, w, h, gray):
                                bubble_contours.append((contour, x, y, w, h))
    
    logger.info(f"Found {len(bubble_contours)} potential bubble contours")
    return bubble_contours

def is_text_or_line(contour: np.ndarray, x: int, y: int, w: int, h: int, gray: np.ndarray) -> bool:
    """Check if contour is likely text or line rather than a bubble."""
    # Skip very small or very elongated shapes
    if w < 10 or h < 10 or w > h * 3 or h > w * 3:
        return True
    
    # Extract region
    region = gray[y:y+h, x:x+w]
    if region.size == 0:
        return True
    
    # Check for high edge density (typical of text)
    edges = cv2.Canny(region, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    if edge_ratio > 0.3:  # Too many edges, likely text
        return True
    
    # Check vertical position - if it's in typical text areas
    img_height = gray.shape[0]
    relative_y = y / img_height
    
    # Skip areas where student info or headers typically appear
    if relative_y < 0.35 or relative_y > 0.95:
        return True
    
    return False

def is_valid_bubble_scanned(contour: np.ndarray, region: np.ndarray) -> bool:
    """Validate bubble shape for scanned images."""
    if len(contour) < MIN_CONTOUR_POINTS:
        return False
    
    try:
        # Check if contour can fit an ellipse
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
            contour_area = cv2.contourArea(contour)
            
            if ellipse_area > 0:
                area_ratio = contour_area / ellipse_area
                # More lenient for scanned images
                if 0.4 < area_ratio < 1.6:
                    return True
        
        # Alternative validation using convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area > 0:
            solidity = contour_area / hull_area
            return solidity > 0.6  # Bubbles should be reasonably solid
            
    except Exception as e:
        logger.debug(f"Validation error: {e}")
        return False
    
    return False

def group_bubbles_scanned(bubble_contours: List[Tuple], num_questions: int) -> List[List[Dict]]:
    """Group bubbles for scanned images with dynamic question count."""
    logger.info(f"Grouping bubbles for {num_questions} questions")
    
    if not bubble_contours:
        return []
    
    # Calculate questions per column dynamically
    questions_per_column = num_questions // NUM_COLUMNS
    if num_questions % NUM_COLUMNS != 0:
        questions_per_column += 1
    
    logger.info(f"Expected {questions_per_column} questions per column")
    
    # Sort by x-coordinate for column grouping
    sorted_by_x = sorted(bubble_contours, key=lambda b: b[1])
    
    # Group into columns using clustering approach
    columns = []
    for bubble in sorted_by_x:
        x, y, w, h = bubble[1:5]
        center_x = x + w // 2
        
        # Find closest existing column or create new one
        best_match = None
        min_distance = float('inf')
        
        for i, column in enumerate(columns):
            distance = abs(center_x - column["center_x"])
            if distance < COLUMN_TOLERANCE and distance < min_distance:
                min_distance = distance
                best_match = i
        
        if best_match is not None:
            columns[best_match]["bubbles"].append(bubble)
            # Update center as weighted average
            all_centers = [b[1] + b[3]//2 for b in columns[best_match]["bubbles"]]
            columns[best_match]["center_x"] = np.mean(all_centers)
        else:
            columns.append({
                "center_x": center_x,
                "bubbles": [bubble]
            })
    
    # Sort columns by center_x and take the 4 most populated
    columns = sorted(columns, key=lambda c: len(c["bubbles"]), reverse=True)[:NUM_COLUMNS]
    columns = sorted(columns, key=lambda c: c["center_x"])
    
    logger.info(f"Found {len(columns)} columns with bubbles: {[len(c['bubbles']) for c in columns]}")
    
    # Group bubbles within each column into rows with improved detection
    all_rows = []
    for col_idx, column in enumerate(columns):
        logger.info(f"Processing column {col_idx + 1} with {len(column['bubbles'])} bubbles")
        
        # Sort bubbles by y-coordinate
        sorted_bubbles = sorted(column["bubbles"], key=lambda b: b[2])
        
        # Improved row grouping with adaptive tolerance
        rows = []
        for bubble in sorted_bubbles:
            x, y, w, h = bubble[1:5]
            center_y = y + h // 2
            
            # Find closest existing row with adaptive tolerance
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
                # Update center
                all_y = [b[2] + b[4]//2 for b in rows[best_row]["bubbles"]]
                rows[best_row]["center_y"] = np.mean(all_y)
            else:
                rows.append({
                    "center_y": center_y,
                    "bubbles": [bubble],
                    "column": col_idx
                })
        
        # Sort rows by y-coordinate and take expected number of questions per column
        rows = sorted(rows, key=lambda r: r["center_y"])
        
        # Adjust for the current column's expected question count
        current_column_questions = questions_per_column
        if col_idx == NUM_COLUMNS - 1:  # Last column might have fewer questions
            remaining_questions = num_questions - (col_idx * questions_per_column)
            current_column_questions = max(1, remaining_questions)
        
        # If we have more rows than expected, keep the most populated ones
        if len(rows) > current_column_questions:
            rows = sorted(rows, key=lambda r: len(r["bubbles"]), reverse=True)[:current_column_questions]
            rows = sorted(rows, key=lambda r: r["center_y"])
        
        # If we have fewer rows, add empty placeholders
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
        
        # Sort bubbles within each row by x-coordinate and validate
        for row_idx, row in enumerate(rows):
            row["bubbles"].sort(key=lambda b: b[1])
            
            # Handle rows with incorrect number of bubbles
            if len(row["bubbles"]) != NUM_CHOICES:
                logger.warning(f"Column {col_idx + 1}, Row {row_idx + 1} has {len(row['bubbles'])} bubbles, expected {NUM_CHOICES}")
                
                # Try to fix by removing duplicates or adding placeholders
                if len(row["bubbles"]) > NUM_CHOICES:
                    row["bubbles"] = filter_overlapping_bubbles(row["bubbles"])
        
        all_rows.extend(rows)
    
    # Sort all rows by column and y-coordinate
    all_rows = sorted(all_rows, key=lambda r: (r["column"], r["center_y"]))
    
    logger.info(f"Total rows found: {len(all_rows)}")
    return all_rows

def filter_overlapping_bubbles(bubbles: List[Tuple]) -> List[Tuple]:
    """Remove overlapping bubbles, keeping the most circular ones."""
    if len(bubbles) <= NUM_CHOICES:
        return bubbles
    
    filtered = []
    bubbles_with_scores = []
    
    # Calculate quality score for each bubble
    for bubble in bubbles:
        contour, x, y, w, h = bubble
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            aspect_ratio = w / float(h)
            # Prefer circular bubbles with good aspect ratio
            score = circularity * (1.0 / (abs(aspect_ratio - 1.0) + 0.1))
        else:
            score = 0
            
        bubbles_with_scores.append((bubble, score))
    
    # Sort by score (best first)
    bubbles_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select non-overlapping bubbles
    for bubble, score in bubbles_with_scores:
        if len(filtered) >= NUM_CHOICES:
            break
            
        x, y, w, h = bubble[1:5]
        is_overlapping = False
        
        for existing_bubble in filtered:
            ex_x, ex_y, ex_w, ex_h = existing_bubble[1:5]
            
            # Check for overlap
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
    """Analyze bubble fill state for scanned images."""
    # Create mask for the bubble
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Extract bubble region
    bubble_region = gray[y:y+h, x:x+w]
    mask_region = mask[y:y+h, x:x+w]
    
    if bubble_region.size == 0 or mask_region.size == 0:
        return BubbleState.INVALID, 0.0
    
    # Get pixels inside the bubble
    bubble_pixels = bubble_region[mask_region > 0]
    
    if len(bubble_pixels) == 0:
        return BubbleState.INVALID, 0.0
    
    # Calculate fill ratio using multiple methods
    fill_ratio = calculate_fill_ratio_scanned(bubble_region, mask_region, bubble_pixels)
    
    # Validate fill pattern (check for invalid marks like ticks, crosses)
    if not is_valid_fill_pattern_scanned(bubble_region, mask_region):
        return BubbleState.INVALID, fill_ratio
    
    # Determine state based on fill ratio
    if fill_ratio > FILL_THRESHOLD:
        return BubbleState.FILLED, fill_ratio
    elif fill_ratio > PARTIAL_FILL_THRESHOLD:
        return BubbleState.PARTIAL, fill_ratio
    else:
        return BubbleState.BLANK, fill_ratio

def calculate_fill_ratio_scanned(bubble_region: np.ndarray, mask_region: np.ndarray, bubble_pixels: np.ndarray) -> float:
    """Calculate fill ratio optimized for scanned images."""
    if len(bubble_pixels) == 0:
        return 0.0
    
    # Method 1: Intensity-based (works well for pencil marks)
    mean_intensity = np.mean(bubble_pixels)
    background_intensity = np.mean(bubble_region[mask_region == 0]) if np.sum(mask_region == 0) > 0 else 255
    
    # Normalize intensity difference
    intensity_ratio = 1.0 - (mean_intensity / max(background_intensity, 1))
    intensity_ratio = max(0, min(1, intensity_ratio))
    
    # Method 2: Threshold-based counting
    threshold = max(np.mean(bubble_pixels) - np.std(bubble_pixels), 
                   np.percentile(bubble_pixels, 25))
    dark_pixels = np.sum(bubble_pixels < threshold)
    threshold_ratio = dark_pixels / len(bubble_pixels)
    
    # Method 3: Otsu's threshold on bubble region
    try:
        masked_region = bubble_region.copy()
        masked_region[mask_region == 0] = 255
        _, otsu_thresh = cv2.threshold(masked_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_dark = np.sum((otsu_thresh < 127) & (mask_region > 0))
        otsu_ratio = otsu_dark / np.sum(mask_region > 0) if np.sum(mask_region > 0) > 0 else 0
    except:
        otsu_ratio = 0
    
    # Combine methods with weights
    final_ratio = (0.4 * intensity_ratio + 0.3 * threshold_ratio + 0.3 * otsu_ratio)
    return max(0, min(1, final_ratio))

def is_valid_fill_pattern_scanned(bubble_region: np.ndarray, mask_region: np.ndarray) -> bool:
    """Check for valid fill patterns in scanned images."""
    if bubble_region.size == 0:
        return False
    
    # Apply edge detection to find line patterns
    edges = cv2.Canny(bubble_region, 30, 100)
    edge_pixels = edges[mask_region > 0]
    
    if len(edge_pixels) == 0:
        return True
    
    edge_ratio = np.sum(edge_pixels > 0) / len(edge_pixels)
    
    # If too many edge pixels, might be a tick or cross
    if edge_ratio > 0.4:
        # Check for line patterns
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=5, minLineLength=3, maxLineGap=2)
        if lines is not None and len(lines) > 3:
            return False  # Too many lines, likely invalid
    
    return True

def detect_marked_bubbles_scanned(gray: np.ndarray, rows: List[Dict], num_questions: int) -> List[Optional[int]]:
    """Detect marked bubbles optimized for scanned images."""
    logger.info(f"Detecting marked bubbles for {num_questions} questions")
    marked_answers = []
    
    for i, row in enumerate(rows):
        if i >= num_questions:  # Don't process more than expected questions
            break
            
        question_num = i + 1
        
        if len(row["bubbles"]) == 0:
            logger.warning(f"Question {question_num}: No bubbles found")
            marked_answers.append(None)
            continue
        
        bubble_states = []
        fill_ratios = []
        
        # Analyze each bubble
        for j, (contour, x, y, w, h) in enumerate(row["bubbles"]):
            if j >= NUM_CHOICES:  # Skip extra bubbles
                break
                
            state, ratio = analyze_bubble_fill_scanned(gray, contour, x, y, w, h)
            bubble_states.append(state)
            fill_ratios.append(ratio)
            
            choice_letter = chr(65 + j) if j < NUM_CHOICES else f"Extra{j}"
            logger.debug(f"Q{question_num} {choice_letter}: {state} (ratio: {ratio:.3f})")
        
        # Pad with blanks if not enough bubbles
        while len(bubble_states) < NUM_CHOICES:
            bubble_states.append(BubbleState.BLANK)
            fill_ratios.append(0.0)
        
        # Determine answer
        filled_indices = [j for j, state in enumerate(bubble_states) if state == BubbleState.FILLED]
        partial_indices = [j for j, state in enumerate(bubble_states) if state == BubbleState.PARTIAL]
        invalid_indices = [j for j, state in enumerate(bubble_states) if state == BubbleState.INVALID]
        
        if len(filled_indices) == 1:
            # Single clear answer
            marked_answers.append(filled_indices[0])
            logger.info(f"Q{question_num}: Answer {chr(65 + filled_indices[0])}")
        elif len(filled_indices) > 1:
            # Multiple filled bubbles
            logger.info(f"Q{question_num}: Multiple answers marked - Invalid")
            marked_answers.append(-2)
        elif len(filled_indices) == 0 and len(partial_indices) == 1:
            # Single partial fill (might be a light mark)
            logger.info(f"Q{question_num}: Partial fill detected at {chr(65 + partial_indices[0])}")
            marked_answers.append(partial_indices[0])  # Accept partial as answer
        elif len(partial_indices) > 1:
            # Multiple partial fills
            logger.info(f"Q{question_num}: Multiple partial fills - Invalid")
            marked_answers.append(-3)
        else:
            # No marks detected
            logger.info(f"Q{question_num}: No marks detected - Blank")
            marked_answers.append(None)
    
    # Ensure we have exactly num_questions answers
    while len(marked_answers) < num_questions:
        marked_answers.append(None)
    
    return marked_answers[:num_questions]

def score_answers_with_key(marked_answers: List[Optional[int]], answer_key: List[str], num_questions: int) -> Dict:
    """Score answers against the answer key and provide detailed analysis."""
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
        correct_index = ord(correct_letter) - ord('A')  # Convert A,B,C,D,E to 0,1,2,3,4
        
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
    
    # Calculate accuracy and confidence
    accuracy = (score / attempted * 100) if attempted > 0 else 0
    confidence = max(50, min(95, 70 + (accuracy - 50) * 0.3))  # Base confidence calculation
    
    # Adjust confidence based on invalid/multiple marks
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
    """Main function to process OMR image and return results."""
    try:
        logger.info(f"Processing OMR image for student {student_id} with {num_questions} questions")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        logger.info(f"Image loaded - Dimensions: {img.shape[1]}x{img.shape[0]}")
        
        # Preprocess the scanned image
        gray, thresh, blurred = preprocess_scanned_image(img)
        
        # Detect bubbles
        bubble_contours = detect_bubbles_scanned(thresh, gray)
        
        if not bubble_contours:
            raise ValueError("No bubble contours detected. Check image quality and parameters.")
        
        logger.info(f"Detected {len(bubble_contours)} bubble contours")
        
        # Group bubbles into structured format
        rows = group_bubbles_scanned(bubble_contours, num_questions)
        
        if len(rows) == 0:
            raise ValueError("No structured rows detected. Check bubble grouping parameters.")
        
        logger.info(f"Organized {len(rows)} question rows")
        
        # Detect marked answers
        marked_answers = detect_marked_bubbles_scanned(gray, rows, num_questions)
        
        # Score the answers
        result = score_answers_with_key(marked_answers, answer_key, num_questions)
        
        # Add student ID to result
        result["studentId"] = student_id
        
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
    """Process a single answer sheet image."""
    try:
        logger.info(f"Processing answer sheet for exam {examId}, student {studentId}")
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Get exam details
        exam = await db.exams.find_one({"examId": examId})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        # Check if solution/answer key exists
        solution = await db.solutions.find_one({"examId": examId})
        if not solution:
            raise HTTPException(status_code=404, detail="Answer key not found for this exam")
        
        # Read image data
        image_data = await image.read()
        
        # Extract answer key from solution
        answer_key = []
        for sol in sorted(solution["solutions"], key=lambda x: x["question"]):
            answer_key.append(sol["answer"])
        
        # Ensure answer key matches number of questions
        if len(answer_key) != exam["numQuestions"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Answer key length ({len(answer_key)}) doesn't match exam questions ({exam['numQuestions']})"
            )
        
        # Process the OMR image
        processing_result = process_omr_image(
            image_data=image_data,
            answer_key=answer_key,
            num_questions=exam["numQuestions"],
            student_id=studentId
        )
        
        # Calculate additional metrics
        total_marks = exam["numQuestions"] * exam["marksPerMcq"]
        score = processing_result["score"] * exam["marksPerMcq"]
        percentage = (score / total_marks * 100) if total_marks > 0 else 0
        
        # Prepare response
        response_data = {
            "success": True,
            "examId": examId,
            "studentId": studentId,
            "response": {
                "studentId": studentId,
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
                "processingMetadata": processing_result["processing_metadata"],
                "detailedResponses": processing_result["detailed_responses"]
            },
            "processingTime": datetime.utcnow().isoformat(),
            "imageProcessed": True
        }
        
        logger.info(f"Answer sheet processed successfully: {score}/{total_marks} ({percentage:.1f}%)")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process answer sheet: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/batch-process")
async def batch_process_answer_sheets(
    images: List[UploadFile] = File(...),
    examId: str = Form(...),
    db=Depends(get_database)
):
    """Process multiple answer sheets in batch."""
    try:
        logger.info(f"Batch processing {len(images)} answer sheets for exam {examId}")
        
        # Get exam details
        exam = await db.exams.find_one({"examId": examId})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        # Check if solution/answer key exists
        solution = await db.solutions.find_one({"examId": examId})
        if not solution:
            raise HTTPException(status_code=404, detail="Answer key not found for this exam")
        
        # Extract answer key
        answer_key = []
        for sol in sorted(solution["solutions"], key=lambda x: x["question"]):
            answer_key.append(sol["answer"])
        
        results = []
        for i, image in enumerate(images):
            try:
                # Generate student ID for batch processing
                student_id = f"STUDENT_{str(i+1).zfill(3)}"
                
                # Validate image
                if not image.content_type.startswith('image/'):
                    logger.warning(f"Skipping non-image file: {image.filename}")
                    continue
                
                # Read image data
                image_data = await image.read()
                
                # Process the OMR image
                processing_result = process_omr_image(
                    image_data=image_data,
                    answer_key=answer_key,
                    num_questions=exam["numQuestions"],
                    student_id=student_id
                )
                
                # Calculate metrics
                total_marks = exam["numQuestions"] * exam["marksPerMcq"]
                score = processing_result["score"] * exam["marksPerMcq"]
                percentage = (score / total_marks * 100) if total_marks > 0 else 0
                
                result_data = {
                    "studentId": student_id,
                    "filename": image.filename,
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
                    "processingMetadata": processing_result["processing_metadata"],
                    "success": True
                }
                
                results.append(result_data)
                logger.info(f"Processed {student_id}: {score}/{total_marks} ({percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {str(e)}")
                results.append({
                    "studentId": f"STUDENT_{str(i+1).zfill(3)}",
                    "filename": image.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "examId": examId,
            "totalImages": len(images),
            "processedSuccessfully": len([r for r in results if r.get("success", False)]),
            "results": results,
            "processingTime": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.get("/test-image-processing")
async def test_image_processing():
    """Test endpoint to verify image processing capabilities."""
    try:
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        return {
            "success": True,
            "message": "Image processing libraries are working correctly",
            "opencv_version": cv2.__version__,
            "test_image_shape": test_image.shape,
            "grayscale_shape": gray.shape
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Image processing test failed: {str(e)}"
        }

@router.post("/validate-image")
async def validate_image_for_omr(
    image: UploadFile = File(...)
):
    """Validate if an image is suitable for OMR processing."""
    try:
        # Read image data
        image_data = await image.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "valid": False,
                "message": "Could not decode image file",
                "recommendations": ["Ensure the file is a valid image format (JPG, PNG, etc.)"]
            }
        
        height, width = img.shape[:2]
        recommendations = []
        warnings = []
        
        # Check resolution
        if width < 800 or height < 600:
            warnings.append(f"Low resolution ({width}x{height}). Recommend at least 800x600 for better accuracy")
        
        # Check brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:
            warnings.append("Image appears too dark - may affect bubble detection")
            recommendations.append("Increase brightness or improve lighting when scanning")
        elif mean_brightness > 200:
            warnings.append("Image appears too bright - may affect bubble detection") 
            recommendations.append("Reduce brightness or adjust scanner settings")
        
        # Test bubble detection
        try:
            _, thresh, _ = preprocess_scanned_image(img)
            bubble_contours = detect_bubbles_scanned(thresh, gray)
            bubbles_found = len(bubble_contours)
            
            if bubbles_found < 20:
                warnings.append(f"Only {bubbles_found} potential bubbles detected - may be insufficient")
                recommendations.append("Ensure the OMR sheet is clearly visible and properly aligned")
            
        except Exception as e:
            warnings.append(f"Bubble detection test failed: {str(e)}")
            recommendations.append("Check image quality and ensure it contains an OMR sheet")
        
        is_valid = len([w for w in warnings if "failed" in w.lower()]) == 0
        
        return {
            "valid": is_valid,
            "image_info": {
                "width": width,
                "height": height,
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
                "mean_brightness": round(mean_brightness, 2),
                "file_size": len(image_data)
            },
            "bubbles_detected": bubbles_found if 'bubbles_found' in locals() else 0,
            "warnings": warnings,
            "recommendations": recommendations,
            "message": "Image validation completed"
        }
        
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        return {
            "valid": False,
            "message": f"Validation failed: {str(e)}",
            "recommendations": ["Ensure the file is a valid image and try again"]
        }

@router.get("/processing-config")
async def get_processing_config():
    """Get current OMR processing configuration."""
    return {
        "bubble_detection": {
            "min_area": MIN_BUBBLE_AREA,
            "max_area": MAX_BUBBLE_AREA,
            "aspect_ratio_range": ASPECT_RATIO_RANGE,
            "circularity_threshold": CIRCULARITY_THRESHOLD
        },
        "fill_detection": {
            "fill_threshold": FILL_THRESHOLD,
            "partial_fill_threshold": PARTIAL_FILL_THRESHOLD
        },
        "layout": {
            "num_choices": NUM_CHOICES,
            "num_columns": NUM_COLUMNS,
            "row_tolerance": ROW_TOLERANCE,
            "column_tolerance": COLUMN_TOLERANCE
        },
        "preprocessing": {
            "gaussian_blur_size": GAUSSIAN_BLUR_SIZE,
            "morph_kernel_size": MORPH_KERNEL_SIZE
        }
    }
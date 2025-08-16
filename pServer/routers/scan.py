from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from datetime import datetime
import cv2
import numpy as np
import logging
import io
import tempfile
import os
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import re
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import json

# OCR functionality
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    print("Warning: pytesseract not installed. Student information extraction will be skipped.")
    print("To enable OCR: pip install pytesseract pillow")
    print("Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
    OCR_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

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

# Student Information Extraction Configuration
STUDENT_INFO_BOX_REGION = (0.05, 0.05, 0.5, 0.35)  # (x_start, y_start, width, height) as ratios
OCR_CONFIG = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.\n '

# Result overlay configuration
RESULT_FOOTER_REGION = (0.3, 0.87, 0.4, 0.15)  # Increased y_start to 0.90, height to 0.15  # (x_start, y_start, width, height) as ratios for footer area

class BubbleState:
    """Enum-like class for bubble states"""
    BLANK = "BLANK"
    FILLED = "FILLED" 
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"

class StudentInfo:
    """Class to hold student information"""
    def __init__(self):
        self.locker_number = ""
        self.name = ""
        self.rank = ""
        self.raw_text = ""
        self.confidence = 0.0

def get_database():
    from main import app
    return app.state.database

def extract_student_information(img: np.ndarray) -> StudentInfo:
    """Extract student information from the top-left box using OCR."""
    logger.info("Extracting student information using OCR")
    
    student_info = StudentInfo()
    
    if not OCR_AVAILABLE:
        logger.warning("OCR not available - skipping student information extraction")
        return student_info
    
    try:
        img_height, img_width = img.shape[:2]
        x_start = int(img_width * STUDENT_INFO_BOX_REGION[0])
        y_start = int(img_height * STUDENT_INFO_BOX_REGION[1])
        width = int(img_width * STUDENT_INFO_BOX_REGION[2])
        height = int(img_height * STUDENT_INFO_BOX_REGION[3])
        
        info_region = img[y_start:y_start+height, x_start:x_start+width]
        
        if info_region.size == 0:
            logger.warning("Student info region is empty")
            return student_info
        
        processed_region = preprocess_for_ocr(info_region)
        pil_image = Image.fromarray(processed_region)
        extracted_text = pytesseract.image_to_string(pil_image, config=OCR_CONFIG)
        student_info.raw_text = extracted_text.strip()
        
        try:
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            student_info.confidence = np.mean(confidences) if confidences else 0.0
        except:
            student_info.confidence = 0.0
        
        student_info = parse_student_info(student_info)
        
        logger.info(f"OCR extracted text: {student_info.raw_text}")
        logger.info(f"OCR confidence: {student_info.confidence:.1f}%")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            debug_path = temp_file.name
            cv2.imwrite(debug_path, processed_region)
            logger.info(f"Student info region saved to: {debug_path}")
        
    except Exception as e:
        logger.error(f"Error extracting student information: {e}")
        student_info.raw_text = f"OCR Error: {str(e)}"
    
    return student_info

def preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """Preprocess the student info region for better OCR accuracy."""
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    scale_factor = 2
    height, width = denoised.shape
    resized = cv2.resize(denoised, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def parse_student_info(student_info: StudentInfo) -> StudentInfo:
    """Parse the raw OCR text to extract structured student information."""
    text = student_info.raw_text.lower()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    locker_patterns = [
        r'locker\s*number[:\s]*(\d+)',
        r'locker[:\s]*(\d+)',
        r'id[:\s]*(\d+)',
        r'number[:\s]*(\d+)'
    ]
    
    rank_patterns = [
        r'rank[:\s]*([a-zA-Z\s]+)',
        r'position[:\s]*([a-zA-Z\s]+)',
        r'designation[:\s]*([a-zA-Z\s]+)'
    ]
    
    name_patterns = [
        r'name[:\s]*([a-zA-Z\s]+)',
        r'student[:\s]*([a-zA-Z\s]+)'
    ]
    
    for pattern in locker_patterns:
        match = re.search(pattern, text)
        if match:
            student_info.locker_number = match.group(1).strip()
            break
    
    for pattern in rank_patterns:
        match = re.search(pattern, text)
        if match:
            rank_text = match.group(1).strip()
            # Clean up rank by removing any following name parts
            rank_words = rank_text.split()
            if rank_words:
                rank = rank_words[0].title()
                # Check if it's a valid rank prefix
                valid_ranks = ['Lieutenant', 'Captain', 'Major', 'Colonel', 'Sergeant', 'Private']
                if rank in valid_ranks:
                    student_info.rank = rank
            break
    
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            name_text = match.group(1).strip()
            # Preserve spaces between first and last names
            name_words = [word for word in name_text.split() if word]
            student_info.name = ' '.join(word.title() for word in name_words if word)
            break
    
    # Fallback parsing if patterns don't match
    if not any([student_info.locker_number, student_info.rank, student_info.name]):
        for line in lines:
            line_clean = line.lower().strip()
            if re.search(r'\d{4,}', line) and not student_info.locker_number:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    student_info.locker_number = numbers[0]
            rank_keywords = ['lieutenant', 'captain', 'major', 'colonel', 'sergeant', 'private']
            for keyword in rank_keywords:
                if keyword in line_clean and not student_info.rank:
                    student_info.rank = keyword.title()
                    break
            if (re.match(r'^[a-zA-Z\s]+$', line_clean) and 
                len(line_clean) > 3 and 
                not any(keyword in line_clean for keyword in ['locker', 'rank', 'name', 'number']) and
                not student_info.name):
                name_words = [word for word in line_clean.split() if word]
                student_info.name = ' '.join(word.title() for word in name_words if word)
    
    student_info.locker_number = re.sub(r'[^\d]', '', student_info.locker_number)
    student_info.name = re.sub(r'[^\w\s]', '', student_info.name).strip()
    student_info.rank = re.sub(r'[^\w\s]', '', student_info.rank).strip()
    
    return student_info

def preprocess_scanned_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced preprocessing specifically for scanned images."""
    logger.info("Preprocessing scanned image")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    blurred = cv2.GaussianBlur(denoised, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_manual = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    
    thresh_combined = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
    thresh_combined = cv2.bitwise_or(thresh_combined, thresh_manual)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    return gray, thresh_combined, blurred

def detect_bubbles_scanned(thresh: np.ndarray, gray: np.ndarray) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """Enhanced bubble detection for scanned images with better filtering."""
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
    """Check if contour is likely text or line rather than a bubble."""
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
    """Validate bubble shape for scanned images."""
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
    """Group bubbles for scanned images with dynamic question count."""
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
    """Remove overlapping bubbles, keeping the most circular ones."""
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
    """Analyze bubble fill state for scanned images."""
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
    """Calculate fill ratio optimized for scanned images."""
    if len(bubble_pixels) == 0:
        return 0.0
    mean_intensity = np.mean(bubble_pixels)
    background_intensity = np.mean(bubble_region[mask_region == 0]) if np.sum(mask_region == 0) > 0 else 255
    intensity_ratio = 1.0 - (mean_intensity / max(background_intensity, 1))
    intensity_ratio = max(0, min(1, intensity_ratio))
    threshold = max(np.mean(bubble_pixels) - np.std(bubble_pixels), np.percentile(bubble_pixels, 25))
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
    """Check for valid fill patterns in scanned images."""
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
    """Detect marked bubbles optimized for scanned images."""
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

def overlay_result_on_sheet(img: np.ndarray, score: int, total_marks: int, pass_fail_status: str, student_name: str = "") -> np.ndarray:
    """Overlay result information on the answer sheet in the footer placeholder area without a text box."""
    logger.info(f"Overlaying result on sheet: {score}/{total_marks} - {pass_fail_status}")
    
    # Convert to PIL Image for text rendering
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    img_height, img_width = img.shape[:2]
    
    # Calculate footer region coordinates
    x_start = int(img_width * RESULT_FOOTER_REGION[0])
    y_start = int(img_height * RESULT_FOOTER_REGION[1])
    region_width = int(img_width * RESULT_FOOTER_REGION[2])
    region_height = int(img_height * RESULT_FOOTER_REGION[3])
    
    # Calculate font sizes based on image dimensions
    base_font_size = max(20, int(img_width * 0.02))
    marks_font_size = base_font_size
    status_font_size = int(base_font_size * 1.2)
    
    try:
        # Try to use a better font if available
        try:
            marks_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", marks_font_size)
            status_font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", status_font_size)
        except:
            try:
                marks_font = ImageFont.truetype("arial.ttf", marks_font_size)
                status_font = ImageFont.truetype("arialbd.ttf", status_font_size)
            except:
                marks_font = ImageFont.load_default()
                status_font = ImageFont.load_default()
    except:
        marks_font = ImageFont.load_default()
        status_font = ImageFont.load_default()
    
    # Format result text
    marks_text = f"{score}/{total_marks}"
    status_text = pass_fail_status.upper()
    
    # Get text dimensions for centering
    marks_bbox = draw.textbbox((0, 0), marks_text, font=marks_font)
    status_bbox = draw.textbbox((0, 0), status_text, font=status_font)
    
    marks_width = marks_bbox[2] - marks_bbox[0]
    marks_height = marks_bbox[3] - marks_bbox[1]
    status_width = status_bbox[2] - status_bbox[0]
    status_height = status_bbox[3] - status_bbox[1]
    
    # Calculate positions for centered text
    marks_x = x_start + (region_width - marks_width) // 2
    marks_y = y_start + region_height // 4
    
    status_x = x_start + (region_width - status_width) // 2
    status_y = marks_y + marks_height + 10
    
    # Draw the marks text in black
    draw.text((marks_x, marks_y), marks_text, fill='black', font=marks_font)
    
    # Draw the pass/fail status in red
    draw.text((status_x, status_y), status_text, fill='red', font=status_font)
    
    # Convert back to OpenCV format
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    logger.info("Result overlay completed successfully")
    return result_img

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

def process_omr_image(image_data: bytes, answer_key: List[str], num_questions: int, student_id: str, marks_per_mcq: int = 1, passing_percentage: float = 50.0) -> Dict:
    """Main function to process OMR image and return results including student information."""
    try:
        logger.info(f"Processing OMR image for student {student_id} with {num_questions} questions")
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        logger.info(f"Image loaded - Dimensions: {img.shape[1]}x{img.shape[0]}")
        
        # Extract student information
        student_info = extract_student_information(img)
        
        # Process the image for bubble detection
        gray, thresh, blurred = preprocess_scanned_image(img)
        bubble_contours = detect_bubbles_scanned(thresh, gray)
        if not bubble_contours:
            raise ValueError("No bubble contours detected. Check image quality and parameters.")
        logger.info(f"Detected {len(bubble_contours)} bubble contours")
        
        # Group bubbles and detect marked answers
        rows = group_bubbles_scanned(bubble_contours, num_questions)
        if len(rows) == 0:
            raise ValueError("No structured rows detected. Check bubble grouping parameters.")
        logger.info(f"Organized {len(rows)} question rows")
        
        marked_answers = detect_marked_bubbles_scanned(gray, rows, num_questions)
        result = score_answers_with_key(marked_answers, answer_key, num_questions)
        
        # Calculate final scores and pass/fail status
        total_marks = num_questions * marks_per_mcq
        obtained_marks = result["score"] * marks_per_mcq
        percentage = (obtained_marks / total_marks * 100) if total_marks > 0 else 0
        pass_fail_status = "PASS" if percentage >= passing_percentage else "FAIL"
        
        # Format student name properly
        student_name = student_info.name
        if student_name:
            if not student_name.count(' ') and len(student_name) > 3:
                # Handle camelCase like "faizanbasheer" or "FaizanBasheer"
                student_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', student_name)
            student_name = ' '.join(word.capitalize() for word in student_name.split())
        
        # Overlay result on the original image
        processed_img = overlay_result_on_sheet(img, obtained_marks, total_marks, pass_fail_status, student_name)
        
        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare result data
        result["studentId"] = student_id
        result["obtainedMarks"] = obtained_marks
        result["totalMarks"] = total_marks
        result["percentage"] = percentage
        result["passFailStatus"] = pass_fail_status
        result["processedImage"] = processed_img_base64
        result["studentInfo"] = {
            "name": student_name,
            "lockerNumber": student_info.locker_number,
            "rank": student_info.rank,
            "ocrConfidence": student_info.confidence,
            "rawOcrText": student_info.raw_text,
            "ocrAvailable": OCR_AVAILABLE
        }
        
        logger.info(f"OMR processing completed for student {student_id}: {obtained_marks}/{total_marks} ({percentage:.1f}%) - {pass_fail_status}")
        return result
    except Exception as e:
        logger.error(f"Error processing OMR image: {str(e)}")
        raise ValueError(f"Failed to process OMR image: {str(e)}")

def generate_batch_results_pdf(processed_results: List[Dict], exam_name: str) -> bytes:
    """Generate a PDF containing only processed answer sheets with results overlaid."""
    logger.info(f"Generating batch results PDF for {len(processed_results)} sheets")
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    
    # Use A4 page size
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Add each processed sheet
    for i, result in enumerate(processed_results):
        if result.get("processedImage"):
            try:
                # Decode base64 image
                img_data = base64.b64decode(result["processedImage"])
                img_buffer = io.BytesIO(img_data)
                
                # Add processed image
                # Scale image to fit page while maintaining aspect ratio
                max_width = 500
                max_height = 600
                
                img = RLImage(img_buffer, width=max_width, height=max_height)
                img.hAlign = 'CENTER'
                story.append(img)
                
                # Add page break except for last sheet
                if i < len(processed_results) - 1:
                    from reportlab.platypus import PageBreak
                    story.append(PageBreak())
                    
            except Exception as e:
                logger.error(f"Error adding sheet {i+1} to PDF: {str(e)}")
                # Add error message instead of image
                error_text = f"<b>Sheet {i+1}: Error loading processed image</b>"
                error_para = Paragraph(error_text, styles['Normal'])
                story.append(error_para)
                if i < len(processed_results) - 1:
                    from reportlab.platypus import PageBreak
                    story.append(PageBreak())
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    
    logger.info(f"Batch results PDF generated successfully ({len(pdf_bytes)} bytes)")
    return pdf_bytes

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
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        exam = await db.exams.find_one({"examId": examId})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        solution = await db.solutions.find_one({"examId": examId})
        if not solution:
            raise HTTPException(status_code=404, detail="Answer key not found for this exam")
        
        image_data = await image.read()
        answer_key = [sol["answer"] for sol in sorted(solution["solutions"], key=lambda x: x["question"])]
        
        if len(answer_key) != exam["numQuestions"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Answer key length ({len(answer_key)}) doesn't match exam questions ({exam['numQuestions']})"
            )
        
        processing_result = process_omr_image(
            image_data=image_data,
            answer_key=answer_key,
            num_questions=exam["numQuestions"],
            student_id=studentId,
            marks_per_mcq=exam["marksPerMcq"],
            passing_percentage=exam["passingPercentage"]
        )
        
        response_data = {
            "success": True,
            "examId": examId,
            "studentId": studentId,
            "response": {
                "studentId": studentId,
                "score": processing_result["obtainedMarks"],
                "totalMarks": processing_result["totalMarks"],
                "percentage": processing_result["percentage"],
                "accuracy": processing_result["accuracy"],
                "responses": processing_result["responses"],
                "correctAnswers": processing_result["correct_answers"],
                "incorrectAnswers": processing_result["incorrect_answers"],
                "blankAnswers": processing_result["blank_answers"],
                "multipleMarks": processing_result["multiple_marks"],
                "invalidAnswers": processing_result["invalid_answers"],
                "processingMetadata": processing_result["processing_metadata"],
                "detailedResponses": processing_result["detailed_responses"],
                "studentInfo": processing_result["studentInfo"],
                "processedImage": processing_result["processedImage"],
                "passFailStatus": processing_result["passFailStatus"]
            },
            "processingTime": datetime.utcnow().isoformat(),
            "imageProcessed": True
        }
        
        logger.info(f"Answer sheet processed successfully: {processing_result['obtainedMarks']}/{processing_result['totalMarks']} ({processing_result['percentage']:.1f}%)")
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
        
        exam = await db.exams.find_one({"examId": examId})
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        solution = await db.solutions.find_one({"examId": examId})
        if not solution:
            raise HTTPException(status_code=404, detail="Answer key not found for this exam")
        
        answer_key = [sol["answer"] for sol in sorted(solution["solutions"], key=lambda x: x["question"])]
        results = []
        processed_images_data = []
        
        for i, image in enumerate(images):
            try:
                student_id = f"STUDENT_{str(i+1).zfill(3)}"
                
                if not image.content_type.startswith('image/'):
                    logger.warning(f"Skipping non-image file: {image.filename}")
                    continue
                
                image_data = await image.read()
                
                processing_result = process_omr_image(
                    image_data=image_data,
                    answer_key=answer_key,
                    num_questions=exam["numQuestions"],
                    student_id=student_id,
                    marks_per_mcq=exam["marksPerMcq"],
                    passing_percentage=exam["passingPercentage"]
                )
                
                result_data = {
                    "studentId": student_id,
                    "filename": image.filename,
                    "score": processing_result["obtainedMarks"],
                    "totalMarks": processing_result["totalMarks"],
                    "percentage": processing_result["percentage"],
                    "accuracy": processing_result["accuracy"],
                    "responses": processing_result["responses"],
                    "correctAnswers": processing_result["correct_answers"],
                    "incorrectAnswers": processing_result["incorrect_answers"],
                    "blankAnswers": processing_result["blank_answers"],
                    "multipleMarks": processing_result["multiple_marks"],
                    "invalidAnswers": processing_result["invalid_answers"],
                    "processingMetadata": processing_result["processing_metadata"],
                    "studentInfo": processing_result["studentInfo"],
                    "processedImage": processing_result["processedImage"],
                    "passFailStatus": processing_result["passFailStatus"],
                    "success": True
                }
                results.append(result_data)
                
                # Store processed image data for PDF generation
                processed_images_data.append({
                    "studentId": student_id,
                    "studentInfo": processing_result["studentInfo"],
                    "obtainedMarks": processing_result["obtainedMarks"],
                    "totalMarks": processing_result["totalMarks"],
                    "percentage": processing_result["percentage"],
                    "passFailStatus": processing_result["passFailStatus"],
                    "processedImage": processing_result["processedImage"],
                    "filename": image.filename
                })
                
                logger.info(f"Processed {student_id}: {processing_result['obtainedMarks']}/{processing_result['totalMarks']} ({processing_result['percentage']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1} ({image.filename}): {str(e)}")
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
            "processedImages": processed_images_data,
            "processingTime": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.post("/download-batch-results-pdf")
async def download_batch_results_pdf(request_data: dict):
    """Generate and return a PDF containing all processed answer sheets with results."""
    try:
        exam_id = request_data.get("examId")
        exam_name = request_data.get("examName", "Exam Results")
        processed_images = request_data.get("processedImages", [])
        
        if not processed_images:
            raise HTTPException(status_code=400, detail="No processed images data provided")
        
        logger.info(f"Generating batch results PDF for {len(processed_images)} processed sheets")
        
        # Generate PDF with all processed sheets
        pdf_bytes = generate_batch_results_pdf(processed_images, exam_name)
        
        # Return PDF as response
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={exam_name.replace(' ', '_')}_Processed_Sheets_{datetime.now().strftime('%Y%m%d')}.pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@router.get("/test-image-processing")
async def test_image_processing():
    """Test endpoint to verify image processing capabilities."""
    try:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        return {
            "success": True,
            "message": "Image processing libraries are working correctly",
            "opencv_version": cv2.__version__,
            "test_image_shape": test_image.shape,
            "grayscale_shape": gray.shape,
            "ocr_available": OCR_AVAILABLE
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Image processing test failed: {str(e)}"
        }

@router.get("/validate-image")
async def validate_image_for_omr(
    image: UploadFile = File(...)
):
    """Validate if an image is suitable for OMR processing."""
    try:
        image_data = await image.read()
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
        
        if width < 800 or height < 600:
            warnings.append(f"Low resolution ({width}x{height}). Recommend at least 800x600 for better accuracy")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:
            warnings.append("Image appears too dark - may affect bubble detection")
            recommendations.append("Increase brightness or improve lighting when scanning")
        elif mean_brightness > 200:
            warnings.append("Image appears too bright - may affect bubble detection") 
            recommendations.append("Reduce brightness or adjust scanner settings")
        
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
        
        ocr_status = "OCR not available"
        if OCR_AVAILABLE:
            try:
                student_info = extract_student_information(img)
                if student_info.name or student_info.locker_number or student_info.rank:
                    ocr_status = f"OCR successful - detected info (confidence: {student_info.confidence:.1f}%)"
                else:
                    ocr_status = "OCR performed but no student information detected"
                    warnings.append("No student information detected in OCR region")
                    recommendations.append("Ensure student information is clearly written in the designated area")
            except Exception as e:
                ocr_status = f"OCR failed: {str(e)}"
                warnings.append(f"OCR processing failed: {str(e)}")
                recommendations.append("Ensure Tesseract OCR is properly installed and configured")
        
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
            "ocr_status": ocr_status,
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
    config = {
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
        },
        "ocr": {
            "available": OCR_AVAILABLE,
            "student_info_box_region": STUDENT_INFO_BOX_REGION,
            "ocr_config": OCR_CONFIG
        },
        "result_overlay": {
            "footer_region": RESULT_FOOTER_REGION
        }
    }
    return config
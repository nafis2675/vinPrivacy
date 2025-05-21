import os
import re
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid requiring a GUI
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
# Import draw_ocr if available, otherwise define our own version
try:
    from paddleocr import draw_ocr
    USING_BUILTIN_DRAW_OCR = True
except ImportError:
    USING_BUILTIN_DRAW_OCR = False
    # We'll define a basic alternative to draw_ocr function later
from PIL import Image
import logging
import base64
from io import BytesIO
from manufacturer_codes import MANUFACTURER_CODES

# Import manufacturer codes for VIN detection
try:
    from manufacturer_codes import MANUFACTURER_CODES
    USING_MANUFACTURER_CODES = True
    print(f"Loaded {len(MANUFACTURER_CODES)} manufacturer codes for VIN detection")
except ImportError:
    MANUFACTURER_CODES = {}
    USING_MANUFACTURER_CODES = False
    print("Warning: manufacturer_codes.py not found. Manufacturer code detection disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload and output directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Define our own draw_ocr function if paddleocr's version is not available
if not USING_BUILTIN_DRAW_OCR:
    def draw_ocr(image, boxes, txts, scores=None, font_path=None):
        """
        Alternative implementation of draw_ocr function when built-in is not available
        Args:
            image (numpy.ndarray): Image array
            boxes (list): List of text box coordinates
            txts (list): List of text strings
            scores (list, optional): List of confidence scores
            font_path (str, optional): Path to font for text rendering
        Returns:
            numpy.ndarray: Image with drawn OCR results
        """
        img = image.copy()
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            score = scores[idx] if scores else 1.0
            
            # Use a different color for each text block (cycle through colors)
            color = (0, 0, 255) if idx % 3 == 0 else (0, 255, 0) if idx % 3 == 1 else (255, 0, 0)
            
            # Convert box points to numpy array
            pts = np.array(box, dtype=np.int32)
            
            # Draw box around text
            cv2.polylines(img, [pts], True, color, 2)
            
            # Add text for reference
            x, y = pts[0]
            text_to_show = f"{txt}"
            if scores:
                text_to_show += f" ({score:.2f})"
                
            cv2.putText(img, text_to_show, (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
        return img

# Initialize PaddleOCR engine
ocr_engine = None

def init_ocr_engine(use_gpu=False):
    """Initialize PaddleOCR engine with multi-language support"""
    global ocr_engine
    try:
        # Use the multi-language model with both Japanese and English support
        ocr_engine = PaddleOCR(
            use_angle_cls=True, 
            lang='japan',  # 'japan' includes both Japanese and English
            det_model_dir=None,  # Use default detection model
            rec_model_dir=None,  # Use default recognition model
            use_gpu=use_gpu, 
            show_log=False
        )
        logger.info(f"PaddleOCR engine initialized with Japanese+English, GPU: {use_gpu}")
        return True
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {e}")
        # Fallback to English-only if Japanese model fails
        try:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)
            logger.info(f"Fallback: PaddleOCR engine initialized with English-only, GPU: {use_gpu}")
            return True
        except Exception as e2:
            logger.error(f"Error initializing fallback OCR: {e2}")
            return False

def get_system_font_path():
    """Get a suitable font path for the current system"""
    # Default PaddleOCR font path on Linux/Colab
    default_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    
    # Check if the font exists
    if os.path.exists(default_path):
        return default_path
    
    # Windows fonts
    windows_fonts = [
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/SIMLI.TTF'  # SimLi
    ]
    
    for font in windows_fonts:
        if os.path.exists(font):
            return font
    
    # Couldn't find a suitable font, log warning
    logger.warning("Could not find a suitable font file for text rendering.")
    return None

# Configuration for identifier detection
KEYWORDS = [
    # English keywords
    'vin', 'chassis', 'frame no', 'frameno', 'frame', 'serial', 'vehicle id', 
    'identifica', 'chassisno', 'chassis no',
    
    # Japanese keywords 
    '車台番号', '車体番号', 'シャーシ', 'シャシー', 'フレーム番号',
    '車 台 番 号', '車 体 番 号', 'シャーシ 番号', 'シャシー 番号',
    'フレーム 番号', '識別番号', '車両識別番号'
]
# Words that should be excluded when they appear alone after a keyword
EXCLUDED_WORDS = ['no', 'number', '#']
# Standard 17-character VIN pattern
VIN_PATTERN = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b', re.IGNORECASE)
# More flexible pattern for chassis numbers (8-20 characters)
CHASSIS_PATTERN = re.compile(r'\b[A-Z0-9]{8,20}\b', re.IGNORECASE)
IDENTIFIER_PART_PATTERN = re.compile(r'^(?=.*[A-Z0-9])[A-Z0-9-]+$', re.IGNORECASE)
CONTINUATION_PATTERN = re.compile(r'^[A-Z0-9-]+$', re.IGNORECASE)
STRIP_CHARS = ' .:-'

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def merge_boxes(box1, box2):
    """Merge two bounding boxes"""
    points1 = np.array(box1, dtype=np.int32)
    points2 = np.array(box2, dtype=np.int32)
    all_points = np.vstack((points1, points2))
    x, y, w, h = cv2.boundingRect(all_points)
    return [x, y, w, h]

def find_manufacturer_code(text):
    """Find manufacturer code in text and return the code and manufacturer name if found"""
    if not USING_MANUFACTURER_CODES:
        return None, None
        
    # Sort codes by length (longest first) to ensure we match the most specific code
    sorted_codes = sorted(MANUFACTURER_CODES.keys(), key=len, reverse=True)
    
    # Check if text starts with any manufacturer code
    for code in sorted_codes:
        if text.upper().startswith(code):
            return code, MANUFACTURER_CODES[code]
    
    return None, None

def identify_manufacturer_code(text):
    """
    Check if any manufacturer code is present at the start of the text.
    Returns a tuple of (code, manufacturer, start_index) if found, otherwise None.
    """
    text = text.upper()  # Convert to uppercase for matching
    
    # Sort manufacturer codes by length (descending) to match longer codes first
    sorted_codes = sorted(MANUFACTURER_CODES.keys(), key=len, reverse=True)
    
    for code in sorted_codes:
        if text.startswith(code):
            return (code, MANUFACTURER_CODES[code], 0)
        
        # Also check if the code appears after some text (with a space or separator)
        # This helps when text like "CHASSIS NO: WDD1234567" is detected
        for separator in [' ', ':', '-', '.', '#']:
            pattern = f"{separator}{code}"
            if pattern in text:
                index = text.find(pattern) + len(separator)
                return (code, MANUFACTURER_CODES[code], index)
    
    return None

def process_image(img_path, obscure_part='Last', obscure_method='Blur', 
                 blur_amount=51, mosaic_block_size=10, debug_mode=False, use_manufacturer_codes=True):
    """Process image to find and obscure identifiers"""
    # Check if OCR engine is initialized
    global ocr_engine
    if ocr_engine is None:
        success = init_ocr_engine()
        if not success:
            return None, "Failed to initialize OCR engine. Please check your PaddleOCR installation."
    
    # Ensure OCR engine is not None before proceeding
    if ocr_engine is None:
        return None, "OCR engine could not be initialized. Please check your PaddleOCR installation."
    
    # Perform OCR
    try:
        result = ocr_engine.ocr(img_path, cls=True)
    except Exception as e:
        logger.exception("Error during OCR processing")
        return None, f"OCR processing error: {str(e)}"
    
    # Handle potential nesting in the result
    if result and isinstance(result[0], list) and len(result) == 1:
        ocr_results = result[0]
    else:
        ocr_results = result
      # Load the image for processing
    image = cv2.imread(img_path)
    if image is None:
        return None, "Could not load image file"
    
    output_image = image.copy()
    
    # If no text detected
    if not ocr_results:
        return output_image, "No text detected in image"
    
    # Extract detected text and boxes
    found_ids = []
    processed_indices = set()
    texts_and_boxes = [(line[1][0], line[0]) for line in ocr_results]
    num_blocks = len(texts_and_boxes)
    
    # Process all text blocks to find identifiers
    detection_results = []
    
    # If debug mode is enabled, create a debug image with all text detections
    if debug_mode:
        debug_image = image.copy()
        # Draw all detected text blocks with different colors for clarity
        boxes = [line[0] for line in ocr_results]
        texts = [line[1][0] for line in ocr_results]
        scores = [line[1][1] for line in ocr_results]
          # Default font path or try to find one available on the system
        font_path = get_system_font_path()
        try:
            # Use the draw_ocr function (either built-in or our alternative)
            debug_image_with_boxes = draw_ocr(debug_image, boxes, texts, scores, font_path=font_path)
            debug_image = debug_image_with_boxes
        except Exception as e:
            logger.error(f"Error using draw_ocr: {e}. Using OpenCV fallback.")
            # Fallback to basic OpenCV drawing if draw_ocr fails
            for i, (text, box) in enumerate(zip(texts, boxes)):
                pts = np.array(box, dtype=np.int32)
                # Use a different color for each text block (cycle through colors)
                color = (0, 0, 255) if i % 3 == 0 else (0, 255, 0) if i % 3 == 1 else (255, 0, 0)
                cv2.polylines(debug_image, [pts], True, color, 2)                # Add text index for reference
                x, y = pts[0]
                cv2.putText(debug_image, f"{i+1}: {text[:10]}...", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
    for i in range(num_blocks):
        if i in processed_indices:
            continue
        
        text, box = texts_and_boxes[i]
        text_lower = text.lower()
        match_found = False
        
        # 1. Check for manufacturer code at the beginning of text (most reliable method)
        if use_manufacturer_codes and USING_MANUFACTURER_CODES:
            mfr_code, mfr_name = find_manufacturer_code(text)
            if mfr_code:
                detection_results.append(f"Found manufacturer code {mfr_code} ({mfr_name}): {text}")
                pts = np.array(box, dtype=np.int32)
                rect = cv2.boundingRect(pts)
                # Store both the manufacturer code and the full ID for later use in obscuring
                found_ids.append({
                    'full_text': text, 
                    'id_part': text, 
                    'box_rect': list(rect), 
                    'method': f'Manufacturer Code ({mfr_name})',
                    'mfr_code': mfr_code,
                    'mfr_name': mfr_name
                })
                processed_indices.add(i)
                match_found = True
                continue
        
        # 2. Check for standard 17-digit VIN pattern
        vin_match = VIN_PATTERN.search(text)
        if vin_match:
            matched_id_string = vin_match.group(0)
            # Check if the VIN starts with a known manufacturer code
            mfr_code, mfr_name = find_manufacturer_code(matched_id_string)
            detection_method = f'Pattern (VIN)'
            if mfr_code:
                detection_method = f'Pattern (VIN) - {mfr_name}'
                
            detection_results.append(f"Found VIN: {matched_id_string} - {detection_method}")
            pts = np.array(box, dtype=np.int32)
            rect = cv2.boundingRect(pts)
            found_ids.append({
                'full_text': text, 
                'id_part': matched_id_string, 
                'box_rect': list(rect), 
                'method': detection_method,
                'mfr_code': mfr_code,
                'mfr_name': mfr_name if mfr_code else None
            })
            processed_indices.add(i)
            match_found = True
            continue
          # Check if block starts with a keyword
        if not match_found:
            for keyword in KEYWORDS:
                if text_lower.startswith(keyword):
                    # Found a keyword prefix
                    potential_id_part1 = text[len(keyword):].strip(STRIP_CHARS)
                    
                    # Check if the rest is just an excluded word (like "no" in "chassis no")
                    is_excluded_word = False
                    for excluded in EXCLUDED_WORDS:
                        if potential_id_part1.lower() == excluded:
                            is_excluded_word = True
                            detection_results.append(f"Skipping '{text}' - '{potential_id_part1}' is an excluded word")
                            break
                    
                    # If this is just "chassis no" or similar (excluded), check the next text block directly
                    if is_excluded_word and i + 1 < num_blocks and (i + 1) not in processed_indices:
                        next_text, next_box = texts_and_boxes[i+1]
                        
                        # Check if the next block starts with a manufacturer code
                        mfr_code = None
                        mfr_name = None
                        if use_manufacturer_codes and USING_MANUFACTURER_CODES:
                            mfr_code, mfr_name = find_manufacturer_code(next_text)
                        
                        is_continuation = CONTINUATION_PATTERN.match(next_text)
                        
                        # If we found a manufacturer code or it looks like a continuation
                        if mfr_code or is_continuation:
                            detection_method = "After Keyword + Excluded Word"
                            if mfr_code:
                                detection_method = f"After Keyword + Excluded Word - {mfr_name} ({mfr_code})"
                                
                            detection_results.append(f"Found {keyword} {potential_id_part1} → {next_text} - {detection_method}")
                            pts = np.array(next_box, dtype=np.int32)
                            rect = cv2.boundingRect(pts)
                            found_ids.append({
                                'full_text': next_text, 
                                'id_part': next_text,
                                'box_rect': list(rect), 
                                'method': detection_method,
                                'mfr_code': mfr_code,
                                'mfr_name': mfr_name
                            })
                            processed_indices.add(i)
                            processed_indices.add(i+1)
                            match_found = True
                            break
                            
                    # Skip checking this part if we already found a match or if it's an excluded word
                    if match_found or is_excluded_word:
                        continue
                    
                    # Check if the rest of this block looks like an identifier PART
                    is_part1_valid = IDENTIFIER_PART_PATTERN.match(potential_id_part1)
                    
                    # Check if it contains a manufacturer code
                    mfr_code = None
                    mfr_name = None
                    if use_manufacturer_codes and USING_MANUFACTURER_CODES and potential_id_part1:
                        mfr_code, mfr_name = find_manufacturer_code(potential_id_part1)
                    
                    # Check for continuation in next block
                    if i + 1 < num_blocks and (i + 1) not in processed_indices:
                        next_text, next_box = texts_and_boxes[i+1]
                        is_continuation = CONTINUATION_PATTERN.match(next_text)
                        
                        if is_continuation and (is_part1_valid or not potential_id_part1 or mfr_code):
                            combined_id = (potential_id_part1 + next_text).strip()
                            merged_rect = merge_boxes(box, next_box)
                            full_combined_text = f"{text} | {next_text}"
                            
                            detection_method = "Keyword Prefix + Continuation"
                            if mfr_code:
                                detection_method = f"Keyword Prefix + Manufacturer Code ({mfr_name})"
                                
                            detection_results.append(f"Found {keyword}: {combined_id} - {detection_method}")
                            found_ids.append({
                                'full_text': full_combined_text, 
                                'id_part': combined_id,
                                'box_rect': merged_rect, 
                                'method': detection_method,
                                'mfr_code': mfr_code,
                                'mfr_name': mfr_name
                            })
                            processed_indices.add(i)
                            processed_indices.add(i+1)
                            match_found = True
                            break
                    
                    # If no continuation, check if part1 is valid or has a manufacturer code
                    if not match_found and (is_part1_valid or mfr_code):
                        detection_method = "Keyword Prefix Only"
                        if mfr_code:
                            detection_method = f"Keyword Prefix + Manufacturer Code ({mfr_name})"
                            
                        detection_results.append(f"Found {keyword}: {potential_id_part1} - {detection_method}")
                        pts = np.array(box, dtype=np.int32)
                        rect = cv2.boundingRect(pts)
                        found_ids.append({
                            'full_text': text, 
                            'id_part': potential_id_part1,
                            'box_rect': list(rect), 
                            'method': detection_method,
                            'mfr_code': mfr_code,
                            'mfr_name': mfr_name
                        })
                        processed_indices.add(i)
                        match_found = True
                        break
                
                if match_found:
                    break
      # Apply obscuring to identified regions
    if not found_ids:
        if debug_mode:
            # If no IDs found but debug mode is on, return the debug image
            return debug_image, "Debug mode: No identifiers found to obscure, showing all detections"
        return output_image, "No identifiers found to obscure"
    
    # If debug mode is enabled, highlight the detected identifiers in a different color
    if debug_mode:
        for id_info in found_ids:
            x, y, w, h = id_info['box_rect']
            id_part = id_info['id_part']
            method = id_info['method']
            # Draw a green rectangle around detected identifiers
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add identifier information
            cv2.putText(debug_image, f"ID: {id_part[:10]}... ({method})", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Return the debug image without obscuring
        return debug_image, ["Debug mode: Showing all detections", 
                            f"Found {len(found_ids)} potential identifier(s)"] + detection_results
    
    # Process each identified region
    for id_info in found_ids:
        x, y, w, h = id_info['box_rect']
        id_part = id_info['id_part']
        method = id_info['method']
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(output_image.shape[1], x + w), min(output_image.shape[0], y + h)
        
        final_w = x2 - x1
        final_h = y2 - y1
        
        if final_w <= 0 or final_h <= 0:
            continue
        
        # Calculate ROI to obscure based on user choice
        obscure_x1, obscure_y1 = x1, y1
        obscure_x2, obscure_y2 = x2, y2
        
        mfr_code = id_info.get('mfr_code')
        mfr_name = id_info.get('mfr_name')
        
        # If it's a manufacturer code and obscure_part is 'AfterPrefix'
        if obscure_part == "AfterPrefix" and mfr_code:
            # Calculate the width of the manufacturer code (approximately)
            mfr_code_width = int(final_w * (len(mfr_code) / len(id_part)))
            # Obscure everything after the manufacturer code
            obscure_x1 = x1 + mfr_code_width
        elif obscure_part == "OnlyPrefix" and mfr_code:
            # Calculate the width of the manufacturer code (approximately)
            mfr_code_width = int(final_w * (len(mfr_code) / len(id_part)))
            # Obscure only the manufacturer code
            obscure_x2 = x1 + mfr_code_width
        elif obscure_part == "First":
            obscure_x2 = x1 + final_w // 3
        elif obscure_part == "Middle":
            obscure_x1 = x1 + final_w // 3
            obscure_x2 = x1 + (2 * final_w) // 3
        elif obscure_part == "Last":
            obscure_x1 = x1 + (2 * final_w) // 3
        # else: "Full" uses the default x1, y1, x2, y2
        
        # Ensure region has size (int conversion)
        obscure_x1, obscure_y1 = int(obscure_x1), int(obscure_y1)
        obscure_x2, obscure_y2 = int(obscure_x2), int(obscure_y2)
        
        if obscure_x1 >= obscure_x2 or obscure_y1 >= obscure_y2:
            obscure_x1, obscure_y1, obscure_x2, obscure_y2 = x1, y1, x2, y2
        
        # Extract ROI and apply obscuring
        roi = output_image[obscure_y1:obscure_y2, obscure_x1:obscure_x2]
        
        if roi.size == 0:
            continue
        
        try:
            if obscure_method == 'Blur':
                ksize = blur_amount if blur_amount % 2 != 0 else blur_amount + 1
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                output_image[obscure_y1:obscure_y2, obscure_x1:obscure_x2] = blurred_roi
            elif obscure_method == 'Mosaic':
                roi_h, roi_w = roi.shape[:2]
                temp_block_size_w = max(2, min(mosaic_block_size, roi_w // 2 if roi_w > 1 else 2))
                temp_block_size_h = max(2, min(mosaic_block_size, roi_h // 2 if roi_h > 1 else 2))
                target_w = max(1, roi_w // temp_block_size_w)
                target_h = max(1, roi_h // temp_block_size_h)
                small_roi = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                mosaic_roi = cv2.resize(small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                output_image[obscure_y1:obscure_y2, obscure_x1:obscure_x2] = mosaic_roi
        except Exception as e:
            logger.error(f"Error applying {obscure_method} to ROI: {e}")
    
    return output_image, detection_results

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Process uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400
    
    # Get form parameters
    obscure_part = request.form.get('obscure_part', 'Last')
    obscure_method = request.form.get('obscure_method', 'Blur')
    blur_amount = int(request.form.get('blur_amount', 51))
    mosaic_block_size = int(request.form.get('mosaic_block_size', 10))
    debug_mode = 'debug_mode' in request.form
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
          # Process the image
        output_image, results = process_image(
            file_path, 
            obscure_part=obscure_part,
            obscure_method=obscure_method,
            blur_amount=blur_amount,
            mosaic_block_size=mosaic_block_size,
            debug_mode=debug_mode
        )
        
        if output_image is None:
            return jsonify({'error': results}), 500
        
        # Save the output image
        output_filename = f"obscured_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, output_image)
        
        # Create image preview as base64
        _, buffer = cv2.imencode('.jpg', output_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'results': results if isinstance(results, list) else [results],
            'original_image': filename,
            'processed_image': output_filename,
            'image_preview': f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        logger.exception("Error processing image")
        return jsonify({'error': str(e)}), 500

@app.route('/process_batch', methods=['POST'])
def process_batch():
    """Process multiple uploaded images"""
    if 'files[]' not in request.files:
        # Fallback to check for the key 'file' which might be used for multiple files
        files = request.files.getlist('file')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files found in request'}), 400
    else:
        files = request.files.getlist('files[]')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
    
    # Get form parameters (same for all images in batch)
    obscure_part = request.form.get('obscure_part', 'Last')
    obscure_method = request.form.get('obscure_method', 'Blur')
    blur_amount = int(request.form.get('blur_amount', 51))
    mosaic_block_size = int(request.form.get('mosaic_block_size', 10))
    debug_mode = 'debug_mode' in request.form
    
    batch_results = []
    
    try:
        for file in files:
            if not file or not allowed_file(file.filename):
                batch_results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': 'Invalid file type. Please upload JPG, JPEG, or PNG'
                })
                continue
                
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image
            output_image, results = process_image(
                file_path, 
                obscure_part=obscure_part,
                obscure_method=obscure_method,
                blur_amount=blur_amount,
                mosaic_block_size=mosaic_block_size,
                debug_mode=debug_mode
            )
            
            if output_image is None:
                batch_results.append({
                    'filename': filename,
                    'status': 'error',
                    'message': results
                })
                continue
            
            # Save the output image
            output_filename = f"obscured_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_image)
            
            # Create image preview as base64
            _, buffer = cv2.imencode('.jpg', output_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            batch_results.append({
                'filename': filename,
                'status': 'success',
                'results': results if isinstance(results, list) else [results],
                'original_image': filename,
                'processed_image': output_filename,
                'image_preview': f"data:image/jpeg;base64,{img_base64}"
            })
        
        # If no images were processed, return error
        if len(batch_results) == 0:
            return jsonify({'error': 'No valid images were processed'}), 400
            
        return jsonify({
            'success': True,
            'message': f'Processed {len(batch_results)} images',
            'batch_results': batch_results
        })
        
    except Exception as e:
        logger.exception("Error processing batch")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize OCR engine at startup with multi-language support
    success = init_ocr_engine(use_gpu=False)
    if not success:
        print("WARNING: Failed to initialize OCR engine at startup.")
        print("The application will try to initialize OCR engine when processing the first image.")
    else:
        print("OCR engine initialized successfully.")
    app.run(debug=True)

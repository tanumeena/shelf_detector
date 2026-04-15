"""
Shelf Product Identifier - Main Module
=====================================

This module implements a complete shelf product identification system that:
1. Detects products in a shelf image using YOLO object detection
2. Compares them with a query product image using deep learning features
3. Highlights matching products and provides detailed analysis

Key Components:
- YOLO object detection for product localization
- ResNet18 feature extraction for similarity matching
- Intelligent row/column assignment for grid organization
- Comprehensive visualization and reporting
"""

import os
import glob
import cv2
import numpy as np
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse
import datetime
from collections import defaultdict
import re

# Configuration constants
DATA_PATH = 'input'
OUTPUT_PATH = 'output'


def parse_yolo_labels(labels_dir, img_shape):
    """
    Parse YOLO label files to extract bounding box coordinates.
    
    YOLO saves labels in format: <class> <x_center> <y_center> <width> <height>
    This function converts normalized coordinates to pixel coordinates.
    
    Args:
        labels_dir (str): Directory containing YOLO label files
        img_shape (tuple): Original image dimensions (width, height)
    
    Returns:
        list: List of bounding boxes as (x1, y1, x2, y2) tuples
    """
    bboxes = []
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
    print(f"Found {len(label_files)} label files in {labels_dir}")
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                continue
            lines = content.split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    # Parse YOLO format: class x_center y_center width height
                    _, cx, cy, w, h = map(float, parts[:5])
                    img_w, img_h = img_shape
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((cx - w/2) * img_w)  # Left edge
                    y1 = int((cy - h/2) * img_h)  # Top edge
                    x2 = int((cx + w/2) * img_w)  # Right edge
                    y2 = int((cy + h/2) * img_h)  # Bottom edge
                    bboxes.append((x1, y1, x2, y2))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue
    
    print(f"Successfully parsed {len(bboxes)} bounding boxes")
    return bboxes


def get_row_assignments(bboxes, n_rows):
    """
    Assign products to rows using histogram binning based on Y-coordinates.
    
    This function uses a simple histogram approach to group products into rows
    based on their vertical positions. This is a fallback method when more
    sophisticated row detection is not available.
    
    Args:
        bboxes (list): List of bounding boxes
        n_rows (int): Number of rows to assign
    
    Returns:
        list: Row assignments for each bounding box
    """
    if not bboxes or n_rows < 1:
        return []
    
    # Calculate Y-centers of all bounding boxes
    y_centers = np.array([(y1 + y2) / 2 for (_, y1, _, y2) in bboxes])
    
    # Use histogram binning for row assignment
    min_y, max_y = y_centers.min(), y_centers.max()
    bins = np.linspace(min_y, max_y, n_rows + 1)
    row_labels = np.digitize(y_centers, bins) - 1  # bins are 1-indexed
    
    # Clamp to valid range
    row_labels = np.clip(row_labels, 0, n_rows - 1)
    
    # Debug print
    print("\nRow assignment debug table:")
    for i, (yc, row) in enumerate(zip(y_centers, row_labels)):
        print(f"Product {i+1}: y-center={yc:.1f}, assigned row={row+1}")
    
    return list(row_labels)


def query_image_in_shelf(query_img_path, shelf_img_path, threshold=0.85):
    """
    Main function to analyze a shelf image and find products similar to a query image.
    
    This function implements the complete pipeline:
    1. YOLO object detection to find products in shelf
    2. Feature extraction using ResNet18 for similarity comparison
    3. Row/column assignment for grid organization
    4. Visualization and detailed reporting
    
    Args:
        query_img_path (str): Path to the query product image
        shelf_img_path (str): Path to the shelf image to analyze
        threshold (float): Cosine similarity threshold for matching (0.0-1.0)
    
    Returns:
        tuple: (predictions_path, products_per_row, matches_per_row, result_image_path)
    """
    print(f"Analyzing shelf image: {shelf_img_path} with query: {query_img_path}")
    
    # ============================================================================
    # STEP 1: SETUP AND DIRECTORY CREATION
    # ============================================================================
    
    # Create output directory structure
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Create timestamped run directory for this analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_PATH, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    # ============================================================================
    # STEP 2: YOLO OBJECT DETECTION
    # ============================================================================
    
    from ultralytics import YOLO
    MODEL_PATH = 'models/best.pt'
    model = YOLO(MODEL_PATH)
    shelf_stem = Path(shelf_img_path).stem
    
    print("Detecting objects with YOLO...")
    results = model.predict(
        source=shelf_img_path,
        save=True,          # Save detection results
        save_crop=True,     # Save individual product crops
        save_txt=True,      # Save label files with coordinates
        conf=0.5,          # Confidence threshold for detection
        project=run_dir,    # Output directory
        name=shelf_stem,    # Subdirectory name
        verbose=False
    )
    
    # ============================================================================
    # STEP 3: LOCATE YOLO OUTPUT FILES
    # ============================================================================
    
    # Find the YOLO output directory within the run directory
    # YOLO creates subdirectories with the image name
    possible_dirs = []
    for i in range(10):
        test_dir = os.path.join(run_dir, f"{shelf_stem}{i if i > 0 else ''}")
        if os.path.exists(test_dir):
            possible_dirs.append(test_dir)
    
    if not possible_dirs:
        print(f"Error: No output directory found for {shelf_stem}")
        return None
    
    latest_dir = sorted(possible_dirs)[-1]
    crops_dir = f"{latest_dir}/crops/object"
    
    if not os.path.exists(crops_dir):
        print(f"Error: Crops directory not found at {crops_dir}")
        return None
    
    print(f"Using crops directory: {crops_dir}")
    crop_files = sorted(glob.glob(f"{crops_dir}/*.jpg"))
    print(f"Detected {len(crop_files)} products in the shelf image.")
    
    # Get original image dimensions for coordinate conversion
    shelf_img = Image.open(shelf_img_path)
    img_w, img_h = shelf_img.size
    shelf_img.close()
    
    # Parse bounding boxes from YOLO label files (fallback method)
    labels_dir = os.path.join(latest_dir, 'labels')
    bboxes = parse_yolo_labels(labels_dir, (img_w, img_h))
    print(f"Parsed {len(bboxes)} bounding boxes from labels")
    
    # ============================================================================
    # STEP 4: EXTRACT BOUNDING BOXES FROM YOLO RESULTS
    # ============================================================================
    
    print("Using YOLO results for proper alignment...")
    
    # Extract bounding boxes directly from YOLO results for perfect alignment
    aligned_bboxes = []
    if results and len(results) > 0:
        result = results[0]  # First result
        if result.boxes is not None:
            boxes = result.boxes
            if len(boxes.xyxy) > 0:
                # Get confidence scores and sort by them to match crop order
                confidences = boxes.conf.cpu().numpy()
                xyxy_boxes = boxes.xyxy.cpu().numpy()
                
                # Sort by confidence (descending) to match crop order
                # Higher confidence detections are typically saved first
                sorted_indices = np.argsort(confidences)[::-1]
                
                for idx in sorted_indices:
                    x1, y1, x2, y2 = xyxy_boxes[idx]
                    aligned_bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                
                print(f"Extracted {len(aligned_bboxes)} bounding boxes from YOLO results")
                print(f"Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
            else:
                print("No boxes found in YOLO results")
        else:
            print("No boxes in YOLO results")
    else:
        print("No YOLO results available")
    
    # ============================================================================
    # STEP 5: ALIGN BOUNDING BOXES WITH CROP FILES
    # ============================================================================
    
    # Use aligned bboxes if available, otherwise fall back to parsed labels
    if len(aligned_bboxes) > 0 and len(aligned_bboxes) == len(crop_files):
        print(f"Using aligned bounding boxes from YOLO results")
        bboxes = aligned_bboxes
    elif len(bboxes) > 0:
        print(f"Using parsed bounding boxes from labels")
        # Ensure we have the same number of bboxes as crop files
        if len(bboxes) != len(crop_files):
            print(f"Warning: Bounding box count ({len(bboxes)}) doesn't match crop count ({len(crop_files)})")
            # Pad or truncate bboxes to match crop count
            if len(bboxes) < len(crop_files):
                bboxes.extend([None] * (len(crop_files) - len(bboxes)))
            else:
                bboxes = bboxes[:len(crop_files)]
    else:
        print(f"No bounding boxes found, will use fallback assignment")
        bboxes = [None] * len(crop_files)
    
    # ============================================================================
    # STEP 6: FEATURE EXTRACTION FROM CROPS
    # ============================================================================
    
    # Initialize the ResNet18 feature extractor
    img2vec = Img2VecResnet18()
    crop_vecs = []
    
    # Extract features from each crop file
    for crop_file in crop_files:
        I = Image.open(crop_file)
        vec = img2vec.getVec(I)  # Get 512-dimensional feature vector
        I.close()
        crop_vecs.append(vec)
    
    # Extract features from query image
    Iq = Image.open(query_img_path)
    query_vec = img2vec.getVec(Iq)
    Iq.close()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # ============================================================================
    # STEP 7: ROBUST CROP-TO-BBOX MATCHING (COORDINATE-BASED)
    # ============================================================================
    
    def parse_crop_coords(filename):
        """
        Parse coordinates from crop filename if available.
        
        Some YOLO implementations save crops with coordinates in filename:
        format: ..._x1_y1_x2_y2.jpg
        
        Args:
            filename (str): Crop filename
            
        Returns:
            tuple or None: (x1, y1, x2, y2) coordinates or None if not found
        """
        # Try to extract coordinates from filename: ..._x1_y1_x2_y2.jpg
        match = re.search(r'_(\d+)_(\d+)_(\d+)_(\d+)\.jpg$', filename)
        if match:
            return tuple(map(int, match.groups()))
        return None

    # Parse coordinates from all crop filenames
    crop_coords = [parse_crop_coords(f) for f in crop_files]
    
    # If coordinates found, use them for precise matching
    if not all(crop_coords):
        print("Warning: Could not parse all crop coordinates. Falling back to order-based matching.")
        crop_to_bbox = list(range(len(crop_files)))
    else:
        # For each crop, find the bbox with the closest coordinates
        crop_to_bbox = []
        used_bbox = set()
        for coords in crop_coords:
            min_dist = float('inf')
            min_idx = -1
            for i, bbox in enumerate(bboxes):
                if i in used_bbox or bbox is None:
                    continue
                # Calculate Manhattan distance between coordinates
                dist = sum(abs(a - b) for a, b in zip(coords, bbox))
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            crop_to_bbox.append(min_idx)
            used_bbox.add(min_idx)
        print(f"Crop-to-bbox mapping: {crop_to_bbox[:10]} ... (showing first 10)")

    # Reorder bboxes to match crops
    bboxes_aligned = [bboxes[i] if i >= 0 and i < len(bboxes) else None for i in crop_to_bbox]
    bboxes = bboxes_aligned
    
    # ============================================================================
    # STEP 8: YOLO DETECTION-BASED CROPPING (IMPROVED ALIGNMENT)
    # ============================================================================
    
    # Get bounding boxes directly from YOLO results (already aligned)
    yolo_boxes = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            yolo_boxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist()
    
    if not yolo_boxes:
        print("No bounding boxes found in YOLO results.")
        return None
    
    bboxes = [tuple(box) for box in yolo_boxes]

    # Crop the shelf image for each bounding box
    shelf_img_cv = cv2.imread(shelf_img_path)
    crop_imgs = []
    for (x1, y1, x2, y2) in bboxes:
        crop = shelf_img_cv[y1:y2, x1:x2]
        crop_imgs.append(crop)

    # Convert crops to PIL Images for feature extraction
    crop_pil_imgs = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) 
                     if crop is not None and crop.size > 0 else None 
                     for crop in crop_imgs]

    # Extract features from the cropped images
    img2vec = Img2VecResnet18()
    crop_vecs = []
    for pil_img in crop_pil_imgs:
        if pil_img is not None:
            vec = img2vec.getVec(pil_img)
            crop_vecs.append(vec)
        else:
            crop_vecs.append(np.zeros(512))  # fallback for empty crop

    # Extract features from query image
    Iq = Image.open(query_img_path)
    query_vec = img2vec.getVec(Iq)
    Iq.close()
    
    # ============================================================================
    # STEP 9: SIMILARITY CALCULATION
    # ============================================================================
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Normalize vectors for numerical stability and calculate similarities
    try:
        # Normalize query vector
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        # Normalize all crop vectors
        crop_vecs_norm = np.array([vec / (np.linalg.norm(vec) + 1e-8) for vec in crop_vecs])
        # Calculate cosine similarities
        sims = cosine_similarity([query_vec_norm], crop_vecs_norm)[0]
    except Exception as e:
        print(f"Warning: Error in similarity calculation: {e}")
        # Fallback to manual cosine similarity calculation
        sims = np.array([np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8) 
                        for vec in crop_vecs])
    
    # ============================================================================
    # STEP 10: MATCH IDENTIFICATION AND DEBUGGING
    # ============================================================================
    
    # Show top similarities for debugging
    top_indices = np.argsort(sims)[::-1][:10]
    print(f"\nTop 10 similarities:")
    for i, idx in enumerate(top_indices):
        filename = os.path.basename(crop_files[idx]) if idx < len(crop_files) else f"crop_{idx}"
        bbox_info = f"bbox:{bboxes[idx]}" if idx < len(bboxes) and bboxes[idx] else "no_bbox"
        print(f"  {i+1}. {filename}: {sims[idx]:.3f} ({bbox_info})")
    
    # Find products that match the threshold
    match_idxs = [i for i, s in enumerate(sims) if s >= threshold]
    print(f"\nMatches found with threshold {threshold}: {len(match_idxs)}")
    print(f"Match indices: {match_idxs}")
    
    # Debug: Show first few crop files and their corresponding bboxes
    print(f"\nDebug - First 5 crops and bboxes:")
    for i in range(min(5, len(crop_files))):
        filename = os.path.basename(crop_files[i])
        bbox_info = f"bbox:{bboxes[i]}" if i < len(bboxes) and bboxes[i] else "no_bbox"
        print(f"  Crop {i+1}: {filename} -> {bbox_info}")
    
    # ============================================================================
    # STEP 11: ROW AND COLUMN ASSIGNMENT
    # ============================================================================
    
    # Assign rows by sorting bboxes top-to-bottom, then columns left-to-right within each row
    # This creates a grid-like organization of products
    assigned = [False] * len(bboxes)
    row_assignments = [-1] * len(bboxes)
    col_assignments = [-1] * len(bboxes)
    row_to_indices = defaultdict(list)
    
    # Sort bounding boxes by Y-coordinate (top-to-bottom)
    bboxes_with_idx = [(i, b) for i, b in enumerate(bboxes) if b is not None]
    bboxes_with_idx.sort(key=lambda x: x[1][1])  # sort by y1 (top)
    
    row_num = 0
    while any(not assigned[i] for i, _ in bboxes_with_idx):
        # Find all unassigned boxes
        unassigned = [(i, b) for i, b in bboxes_with_idx if not assigned[i]]
        if not unassigned:
            break
        
        # Start new row with the topmost unassigned box
        _, first_box = unassigned[0]
        y_center_first = (first_box[1] + first_box[3]) / 2
        
        # Calculate vertical threshold based on median box height
        heights = [b[3] - b[1] for _, b in unassigned]
        median_height = sorted(heights)[len(heights)//2] if heights else 1
        vertical_thresh = median_height * 0.5
        
        # Group boxes that are in the same row (similar Y-coordinates)
        row_indices = []
        for i, b in unassigned:
            y_center = (b[1] + b[3]) / 2
            if abs(y_center - y_center_first) <= vertical_thresh:
                row_indices.append(i)
        
        # Sort row boxes left-to-right by X-coordinate
        row_indices = sorted(row_indices, key=lambda i: (bboxes[i][0] + bboxes[i][2]) / 2)
        
        # Assign row and column numbers
        for pos_in_row, idx in enumerate(row_indices):
            assigned[idx] = True
            row_assignments[idx] = row_num
            col_assignments[idx] = pos_in_row
            row_to_indices[row_num].append(idx)
        
        row_num += 1
    
    n_rows = row_num
    print(f"Total rows detected: {n_rows}")
    
    # ============================================================================
    # STEP 12: RESULT IMAGE GENERATION
    # ============================================================================
    
    def create_result_image_with_grid(shelf_img_path, bboxes, match_idxs, sims, run_dir, row_assignments, col_assignments):
        """
        Create a result image with highlighted matches and grid labels.
        
        This function generates a visual representation of the analysis results:
        - Green boxes (thick): Matching products with similarity scores
        - Red boxes (thin): Non-matching products
        - Yellow labels: Grid coordinates (R1C1, R2C5, etc.)
        - Legend and summary statistics
        
        Args:
            shelf_img_path (str): Path to original shelf image
            bboxes (list): List of bounding boxes
            match_idxs (list): Indices of matching products
            sims (array): Similarity scores for all products
            run_dir (str): Output directory
            row_assignments (list): Row assignments for each product
            col_assignments (list): Column assignments for each product
            
        Returns:
            str: Path to saved result image
        """
        # Load the original shelf image
        shelf_img = cv2.imread(shelf_img_path)
        if shelf_img is None:
            print(f"Error: Could not load shelf image {shelf_img_path}")
            return None
        
        # Create a copy for drawing
        result_img = shelf_img.copy()
        
        # Draw bounding boxes for all detected products
        for i, bbox in enumerate(bboxes):
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox
            similarity = sims[i] if i < len(sims) else 0
            row = row_assignments[i] + 1 if row_assignments[i] >= 0 else -1
            col = col_assignments[i] + 1 if col_assignments[i] >= 0 else -1
            label = f"R{row}C{col}"
            
            # Determine color and thickness based on match status
            if i in match_idxs:
                # Green for matches (thick border)
                color = (0, 255, 0)  # BGR format
                thickness = 3
                # Add similarity score above the box
                cv2.putText(result_img, f"{similarity:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Red for non-matches (thin border)
                color = (0, 0, 255)  # BGR format
                thickness = 1
            
            # Draw the bounding box rectangle
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            
            # Add grid coordinate label below the box
            cv2.putText(result_img, label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add legend to explain the color coding
        legend_y = 30
        cv2.putText(result_img, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(result_img, (10, legend_y + 10), (30, legend_y + 30), 
                     (0, 255, 0), 2)
        cv2.putText(result_img, "Match", (35, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(result_img, (10, legend_y + 40), (30, legend_y + 60), 
                     (0, 0, 255), 1)
        cv2.putText(result_img, "No Match", (35, legend_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add summary statistics at the bottom
        summary_text = f"Matches: {len(match_idxs)}/{len(bboxes)} (threshold: {threshold})"
        cv2.putText(result_img, summary_text, (10, result_img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save the result image
        result_path = os.path.join(run_dir, "result.jpg")
        cv2.imwrite(result_path, result_img)
        print(f"Result image saved to: {result_path}")
        return result_path
    
    # Generate the result image
    result_path = create_result_image_with_grid(shelf_img_path, bboxes, match_idxs, sims, 
                                              run_dir, row_assignments, col_assignments)

    # ============================================================================
    # STEP 13: OUTPUT FILE GENERATION
    # ============================================================================
    
    # Prepare row/position-based identifiers for all products
    idx_to_rowpos = {}
    for row, indices in row_to_indices.items():
        for pos_in_row, idx in enumerate(indices):
            idx_to_rowpos[idx] = (row + 1, pos_in_row + 1)  # 1-based indexing

    # Prepare output rows for sorting and reporting
    output_rows = []
    for idx in range(len(bboxes)):
        row_num = row_assignments[idx] + 1 if row_assignments[idx] >= 0 else -1
        col_num = col_assignments[idx] + 1 if col_assignments[idx] >= 0 else -1
        identifier = f"R{row_num}C{col_num}"
        match = 'YES' if idx in match_idxs else 'NO'
        sim = round(sims[idx], 3)
        output_rows.append((match == 'YES', f"{identifier},{match},{row_num},{col_num},{sim}"))

    # Sort: matches first, then by similarity descending
    output_rows.sort(key=lambda x: (not x[0], -float(x[1].split(',')[-1])))

    # Save predictions to CSV file
    pred_path = os.path.join(run_dir, "predictions.txt")
    with open(pred_path, "w") as f:
        f.write("identifier,match,row,column,similarity\n")
        for _, line in output_rows:
            f.write(line + "\n")

    # ============================================================================
    # STEP 14: DETAILED ANALYSIS GENERATION
    # ============================================================================
    
    # Create detailed row-by-row analysis with individual product details
    row_analysis = []
    for row, indices in row_to_indices.items():
        row_details = []
        match_count = 0
        for pos_in_row, idx in enumerate(indices):
            row_num = row_assignments[idx] + 1 if row_assignments[idx] >= 0 else -1
            col_num = col_assignments[idx] + 1 if col_assignments[idx] >= 0 else -1
            identifier = f"R{row_num}C{col_num}"
            similarity = round(sims[idx], 3)
            match_status = 'YES' if idx in match_idxs else 'NO'
            if match_status == 'YES':
                match_count += 1
            row_details.append(f"    {identifier}, {similarity}, {match_status}")
        
        total_count = len(indices)
        match_percentage = (match_count / total_count * 100) if total_count > 0 else 0
        row_header = f"Row {row+1}: ({total_count} products, {match_count} matches, {match_percentage:.1f}% match rate)"
        row_analysis.append((row_header, row_details))

    # Save detailed row analysis
    detailed_path = os.path.join(run_dir, "detailed_analysis.txt")
    with open(detailed_path, "w") as f:
        f.write("Detailed Row-by-Row Analysis\n")
        f.write("==========================\n\n")
        for row_header, row_details in row_analysis:
            f.write(f"{row_header}\n")
            for detail in row_details:
                f.write(f"{detail}\n")
            f.write("\n")
        f.write(f"Overall Statistics:\n")
        f.write(f"Total products: {len(bboxes)}\n")
        f.write(f"Total matches: {len(match_idxs)}\n")
        f.write(f"Overall match rate: {(len(match_idxs) / len(bboxes) * 100):.1f}%\n")
    
    # ============================================================================
    # STEP 15: STATISTICS CALCULATION AND REPORTING
    # ============================================================================
    
    # Compute products per row and matches per row
    products_per_row = [len(indices) for row, indices in sorted(row_to_indices.items())]
    matches_per_row = [sum(1 for idx in indices if idx in match_idxs) 
                      for row, indices in sorted(row_to_indices.items())]
    rows_with_matches = [row+1 for row, indices in row_to_indices.items() 
                        if any(idx in match_idxs for idx in indices)]

    print(f"Products per row: {products_per_row}")
    print(f"Total matches: {len(match_idxs)}")
    
    # Print detailed row statistics
    print(f"\n=== Detailed Row Statistics ===")
    for row, indices in sorted(row_to_indices.items()):
        row_num = row + 1
        total_products = len(indices)
        matches_in_row = sum(1 for idx in indices if idx in match_idxs)
        other_products = total_products - matches_in_row
        match_percentage = (matches_in_row / total_products * 100) if total_products > 0 else 0
        
        print(f"Row {row_num}:")
        print(f"  - Total products: {total_products}")
        print(f"  - Matches: {matches_in_row}")
        print(f"  - Other products: {other_products}")
        print(f"  - Match percentage: {match_percentage:.1f}%")
        
        # Show which products in this row are matches
        if matches_in_row > 0:
            match_products = [f"R{row_num}C{col_assignments[idx]+1}" 
                            for idx in indices if idx in match_idxs]
            print(f"  - Matching products: {', '.join(match_products)}")
        print()

    # ============================================================================
    # STEP 16: SUMMARY REPORT GENERATION
    # ============================================================================
    
    # Create a comprehensive summary report
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Shelf Analysis Summary\n")
        f.write(f"====================\n")
        f.write(f"Query Image: {query_img_path}\n")
        f.write(f"Shelf Image: {shelf_img_path}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Results:\n")
        f.write(f"- Total products detected: {len(bboxes)}\n")
        f.write(f"- Total matches: {len(match_idxs)}\n")
        f.write(f"- Overall match rate: {(len(match_idxs) / len(bboxes) * 100):.1f}%\n")
        f.write(f"\nRow-by-Row Statistics:\n")
        f.write(f"====================\n")
        for row, indices in sorted(row_to_indices.items()):
            row_num = row + 1
            total_products = len(indices)
            matches_in_row = sum(1 for idx in indices if idx in match_idxs)
            other_products = total_products - matches_in_row
            match_percentage = (matches_in_row / total_products * 100) if total_products > 0 else 0
            
            f.write(f"Row {row_num}:\n")
            f.write(f"  - Total products: {total_products}\n")
            f.write(f"  - Matches: {matches_in_row}\n")
            f.write(f"  - Other products: {other_products}\n")
            f.write(f"  - Match percentage: {match_percentage:.1f}%\n")
            
            # Show which products in this row are matches
            if matches_in_row > 0:
                match_products = [f"R{row_num}C{col_assignments[idx]+1}" 
                                for idx in indices if idx in match_idxs]
                f.write(f"  - Matching products: {', '.join(match_products)}\n")
            f.write(f"\n")
    
    return pred_path, products_per_row, matches_per_row, result_path


def main():
    """
    Main entry point for the shelf product identifier.
    
    Parses command line arguments and runs the analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Find and analyze similar products in a shelf image using a query image')
    parser.add_argument('--query', required=True, help='Query product image file path')
    parser.add_argument('--shelf', required=True, help='Shelf image file path')
    parser.add_argument('--threshold', type=float, default=0.80, help='Cosine similarity threshold for match (0.0-1.0)')
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.query):
        print(f"Error: Query file {args.query} not found")
        return
    if not os.path.exists(args.shelf):
        print(f"Error: Shelf file {args.shelf} not found")
        return
    
    # Run the analysis
    query_image_in_shelf(args.query, args.shelf, args.threshold)


if __name__ == "__main__":
    main() 
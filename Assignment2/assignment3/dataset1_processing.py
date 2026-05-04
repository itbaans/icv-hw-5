import os
import glob
import cv2
import csv
import pickle
import numpy as np
import sys

# Allow importing baseline_code from the parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from baseline_code.preprocessing import load_config, apply_grayscale, apply_noise_reduction
from baseline_code.evaluate import generate_performance_summary
from edge_detection import apply_canny, apply_sobel, extract_contours_and_features, count_objects_from_contours

def get_ground_truth(filename):
    basename = os.path.basename(filename)
    try:
        return int(os.path.splitext(basename)[0])
    except ValueError:
        return None

def process_dataset1():
    data_dir = os.path.join(PARENT_DIR, "data")
    output_dir = os.path.join(SCRIPT_DIR, "edge_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = load_config(os.path.join(PARENT_DIR, "baseline_code/config.yaml"))
    
    # We will downsample high-res images to speed up and improve edge reliability
    RESIZE_FACTOR = 0.5
    
    # Output structures
    canny_dir = os.path.join(output_dir, "dataset1_canny_edges")
    sobel_dir = os.path.join(output_dir, "dataset1_sobel_edges")
    contours_vis_dir = os.path.join(output_dir, "dataset1_contours_vis")
    os.makedirs(canny_dir, exist_ok=True)
    os.makedirs(sobel_dir, exist_ok=True)
    os.makedirs(contours_vis_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    
    canny_records = []
    sobel_records = []
    all_contours = {}
    
    # Area thresholds from config, need to be scaled down by RESIZE_FACTOR**2
    orig_min_area = cfg.get("blob_detection", {}).get("min_area", 8000)
    orig_max_area = cfg.get("blob_detection", {}).get("max_area", 200000)
    scaled_min_area = orig_min_area * (RESIZE_FACTOR ** 2)
    scaled_max_area = orig_max_area * (RESIZE_FACTOR ** 2)
    
    print(f"Target count thresholds -> Min Area: {scaled_min_area}, Max Area: {scaled_max_area}")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        gt = get_ground_truth(img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # 1. Resize/Downsample
        h, w = image.shape[:2]
        new_w, new_h = int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 2. Preprocess (same as Assignment 1/2)
        gray = apply_grayscale(image_resized)
        smoothed = apply_noise_reduction(gray, cfg, method="gaussian")
        
        # 3. Apply Edge Detection
        canny_edges = apply_canny(smoothed, low_thresh=20, high_thresh=60) # Increased threshold to reduce noise
        _, _, sobel_edges = apply_sobel(smoothed, ksize=3)
        
        # Threshold sobel edges to get binary map for contours
        _, sobel_binary = cv2.threshold(sobel_edges, 40, 255, cv2.THRESH_BINARY)
        
        # 4. Apply Morphology on Edges Before Contour Extraction
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        canny_dilated = cv2.dilate(canny_edges, kernel_dilate, iterations=1)
        sobel_closed = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel_close)
        
        # 5. Extract Contours & features
        canny_features = extract_contours_and_features(canny_dilated, close_kernel_size=0)
        sobel_features = extract_contours_and_features(sobel_closed, close_kernel_size=0)
        
        # 5. Count seeds
        canny_count = count_objects_from_contours(canny_features, min_area=scaled_min_area, max_area=scaled_max_area, min_circ=0.2, circ=0.08)
        sobel_count = count_objects_from_contours(sobel_features, min_area=scaled_min_area, max_area=scaled_max_area, min_circ=0.2, circ=0.03)
        
        canny_records.append({"filename": img_name, "prediction": canny_count, "ground_truth": gt})
        sobel_records.append({"filename": img_name, "prediction": sobel_count, "ground_truth": gt})
        
        # Save edge maps
        cv2.imwrite(os.path.join(canny_dir, img_name), canny_edges)
        cv2.imwrite(os.path.join(sobel_dir, img_name), sobel_edges)
        
        # Draw contours on the original (resized) image for visualization
        vis_image = image_resized.copy()
        canny_contours_to_draw = [f['contour'] for f in canny_features]
        # Draw Canny contours in Green
        cv2.drawContours(vis_image, canny_contours_to_draw, -1, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(os.path.join(contours_vis_dir, img_name), vis_image)
        
        # Save contour data (exclude the heavy numpy contour arrays)
        all_contours[img_name] = {
            "canny_features": [{"area": f["area"], "perimeter": f["perimeter"], "circularity": f["circularity"]} for f in canny_features],
            "sobel_features": [{"area": f["area"], "perimeter": f["perimeter"], "circularity": f["circularity"]} for f in sobel_features]
        }
        
        print(f"Processed {img_name}: GT={gt}, Canny={canny_count}, Sobel={sobel_count}")
        
    # Evaluate and save summary
    generate_performance_summary(canny_records, output_path=os.path.join(output_dir, "dataset1_canny_summary.json"))
    generate_performance_summary(sobel_records, output_path=os.path.join(output_dir, "dataset1_sobel_summary.json"))
    
    # Save output CSVs
    with open(os.path.join(output_dir, "dataset1_edge_counts.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ground_truth", "canny_pred", "sobel_pred"])
        writer.writeheader()
        for c_rec, s_rec in zip(canny_records, sobel_records):
            writer.writerow({
                "filename": c_rec["filename"],
                "ground_truth": c_rec["ground_truth"],
                "canny_pred": c_rec["prediction"],
                "sobel_pred": s_rec["prediction"]
            })
            
    # Save contour features
    with open(os.path.join(output_dir, "dataset1_contours.pkl"), "wb") as f:
        pickle.dump(all_contours, f)
        
if __name__ == "__main__":
    process_dataset1()

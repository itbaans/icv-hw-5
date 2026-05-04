import os
import glob
import cv2
import csv
import pickle
import numpy as np
import random
import sys
from scipy.io import loadmat

# Allow importing modules from the parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from baseline_code.preprocessing import load_config, apply_grayscale, apply_noise_reduction
from edge_detection import apply_canny, apply_sobel, extract_contours_and_features, count_objects_from_contours

def load_bsds_gt(mat_path):
    """
    Load BSDS500 ground truth .mat file and combine annotators.
    """
    if not os.path.exists(mat_path):
        return None
    
    data = loadmat(mat_path)
    gt_cells = data['groundTruth']
    num_annotators = gt_cells.shape[1]
    
    combined_gt = None
    for i in range(num_annotators):
        # The structure is: gt_cells[0, i]['Boundaries'][0, 0]
        gt_bound = gt_cells[0, i][0, 0]['Boundaries']
        if combined_gt is None:
            combined_gt = gt_bound.copy()
        else:
            combined_gt = np.logical_or(combined_gt, gt_bound)
            
    return combined_gt.astype(np.uint8) * 255

def calculate_metrics(pred, gt):
    """
    Calculate precision, recall, f1, and IoU.
    pred and gt should be binary masks (0 or 255).
    """
    if gt is None:
        return {}
    
    # Ensure binary
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    
    tp = np.sum(np.logical_and(p == 1, g == 1))
    fp = np.sum(np.logical_and(p == 1, g == 0))
    fn = np.sum(np.logical_and(p == 0, g == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou)
    }

def process_dataset2():
    data_dir = os.path.join(PARENT_DIR, "BSDS500/data/images/train")
    gt_dir = os.path.join(PARENT_DIR, "BSDS500/data/groundTruth/train")
    output_dir = os.path.join(SCRIPT_DIR, "edge_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = load_config(os.path.join(PARENT_DIR, "baseline_code/config.yaml"))
    
    canny_dir = os.path.join(output_dir, "dataset2_canny_edges")
    sobel_dir = os.path.join(output_dir, "dataset2_sobel_edges")
    os.makedirs(canny_dir, exist_ok=True)
    os.makedirs(sobel_dir, exist_ok=True)
    
    # Get all training images
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    image_paths = sorted(image_paths)
    
    # Select a subset of 50 images
    random.seed(42)
    selected_paths = random.sample(image_paths, min(50, len(image_paths)))
    
    canny_records = []
    sobel_records = []
    all_contours = {}
    
    # Min area for BSDS500, no circularity filter since it's generic scenes
    min_area = 50
    max_area = float('inf')
    
    for img_path in selected_paths:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # 1. Preprocess (no resize needed for BSDS500, they are typical 481x321)
        gray = apply_grayscale(image)
        smoothed = apply_noise_reduction(gray, cfg, method="gaussian")
        
        # 2. Edge Detection
        canny_edges = apply_canny(smoothed, low_thresh=50, high_thresh=150)
        _, _, sobel_edges = apply_sobel(smoothed, ksize=3)
        _, sobel_binary = cv2.threshold(sobel_edges, 40, 255, cv2.THRESH_BINARY)
        
        # 3. Extract Contours
        canny_features = extract_contours_and_features(canny_edges, close_kernel_size=3)
        sobel_features = extract_contours_and_features(sobel_binary, close_kernel_size=3)
        
        # 4. Count Objects / Boundary Components (Apply new custom circularities)
        canny_count = count_objects_from_contours(canny_features, min_area=min_area, max_area=max_area, min_circ=0.08)
        sobel_count = count_objects_from_contours(sobel_features, min_area=min_area, max_area=max_area, min_circ=0.03)
        
        # 5. Evaluate against Ground Truth
        gt_path = os.path.join(gt_dir, img_name.replace(".jpg", ".mat"))
        gt_mask = load_bsds_gt(gt_path)
        
        canny_metrics = calculate_metrics(canny_edges, gt_mask)
        sobel_metrics = calculate_metrics(sobel_binary, gt_mask)
        
        canny_records.append({
            "filename": img_name, 
            "prediction": canny_count, 
            "metrics": canny_metrics
        })
        sobel_records.append({
            "filename": img_name, 
            "prediction": sobel_count, 
            "metrics": sobel_metrics
        })
        
        cv2.imwrite(os.path.join(canny_dir, img_name), canny_edges)
        cv2.imwrite(os.path.join(sobel_dir, img_name), sobel_edges)
        
        all_contours[img_name] = {
            "canny_features": [{"area": f["area"], "perimeter": f["perimeter"], "circularity": f["circularity"]} for f in canny_features],
            "sobel_features": [{"area": f["area"], "perimeter": f["perimeter"], "circularity": f["circularity"]} for f in sobel_features]
        }
        print(f"Processed BSDS500 {img_name}: Canny Objects={canny_count}, Sobel Objects={sobel_count}")

    # Output CSV
    with open(os.path.join(output_dir, "dataset2_edge_metrics.csv"), "w", newline='') as f:
        fieldnames = ["filename", "canny_count", "canny_precision", "canny_recall", "canny_f1", "canny_iou",
                      "sobel_count", "sobel_precision", "sobel_recall", "sobel_f1", "sobel_iou"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c_rec, s_rec in zip(canny_records, sobel_records):
            writer.writerow({
                "filename": c_rec["filename"],
                "canny_count": c_rec["prediction"],
                "canny_precision": c_rec["metrics"].get("precision", 0),
                "canny_recall": c_rec["metrics"].get("recall", 0),
                "canny_f1": c_rec["metrics"].get("f1", 0),
                "canny_iou": c_rec["metrics"].get("iou", 0),
                "sobel_count": s_rec["prediction"],
                "sobel_precision": s_rec["metrics"].get("precision", 0),
                "sobel_recall": s_rec["metrics"].get("recall", 0),
                "sobel_f1": s_rec["metrics"].get("f1", 0),
                "sobel_iou": s_rec["metrics"].get("iou", 0)
            })
            
    with open(os.path.join(output_dir, "dataset2_contours.pkl"), "wb") as f:
        pickle.dump(all_contours, f)

if __name__ == "__main__":
    process_dataset2()

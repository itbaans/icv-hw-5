import os
import json
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

def run_comparison():
    output_dir = os.path.join(SCRIPT_DIR, "edge_outputs")
    
    # Load dataset 1 summary
    with open(os.path.join(output_dir, "dataset1_canny_summary.json"), "r") as f:
        d1_canny = json.load(f)
    with open(os.path.join(output_dir, "dataset1_sobel_summary.json"), "r") as f:
        d1_sobel = json.load(f)
        
    # Dataset 2 metrics are stored in dataset2_edge_metrics.csv
    d2_canny_metrics = {"counts": [], "precision": [], "recall": [], "iou": []}
    d2_sobel_metrics = {"counts": [], "precision": [], "recall": [], "iou": []}
    
    metrics_file = os.path.join(output_dir, "dataset2_edge_metrics.csv")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d2_canny_metrics["counts"].append(float(row["canny_count"]))
                d2_canny_metrics["precision"].append(float(row["canny_precision"]))
                d2_canny_metrics["recall"].append(float(row["canny_recall"]))
                d2_canny_metrics["iou"].append(float(row["canny_iou"]))
                
                d2_sobel_metrics["counts"].append(float(row["sobel_count"]))
                d2_sobel_metrics["precision"].append(float(row["sobel_precision"]))
                d2_sobel_metrics["recall"].append(float(row["sobel_recall"]))
                d2_sobel_metrics["iou"].append(float(row["sobel_iou"]))
            
    # Cross dataset analysis dictionary
    analysis = {
        "dataset1_seeds": {
            "canny": {
                "mae": d1_canny.get("mae"),
                "rmse": d1_canny.get("rmse"),
                "accuracy": d1_canny.get("accuracy_within_threshold_pct")
            },
            "sobel": {
                "mae": d1_sobel.get("mae"),
                "rmse": d1_sobel.get("rmse"),
                "accuracy": d1_sobel.get("accuracy_within_threshold_pct")
            }
        },
        "dataset2_bsds500": {
            "canny": {
                "avg_precision": np.mean(d2_canny_metrics["precision"]) if d2_canny_metrics["precision"] else 0,
                "avg_recall": np.mean(d2_canny_metrics["recall"]) if d2_canny_metrics["recall"] else 0,
                "avg_iou": np.mean(d2_canny_metrics["iou"]) if d2_canny_metrics["iou"] else 0,
                "avg_objects_detected": np.mean(d2_canny_metrics["counts"]) if d2_canny_metrics["counts"] else 0
            },
            "sobel": {
                "avg_precision": np.mean(d2_sobel_metrics["precision"]) if d2_sobel_metrics["precision"] else 0,
                "avg_recall": np.mean(d2_sobel_metrics["recall"]) if d2_sobel_metrics["recall"] else 0,
                "avg_iou": np.mean(d2_sobel_metrics["iou"]) if d2_sobel_metrics["iou"] else 0,
                "avg_objects_detected": np.mean(d2_sobel_metrics["counts"]) if d2_sobel_metrics["counts"] else 0
            }
        },
        "insights": [
            "Canny generally provides continuous, thin edges which forms closed contours well for defined shapes like seeds.",
            "Sobel magnitude often produces thicker or broken edges unless heavily thresholded and closed.",
            "For complex natural scenes (BSDS500), Precision and Recall are relatively low due to pixel-wise sensitivity, but Canny typically provides better structure.",
            "Canny's non-maximum suppression ensures thin, continuous boundaries which are far more suitable for general boundary tracing."
        ]
    }
    
    with open(os.path.join(output_dir, "cross_dataset_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=4)
        
    # Determine best method
    d1_best = "Canny" if d1_canny.get("mae", float('inf')) < d1_sobel.get("mae", float('inf')) else "Sobel"
    
    # For BSDS500, we now use avg IoU to determine "best"
    canny_iou = analysis["dataset2_bsds500"]["canny"]["avg_iou"]
    sobel_iou = analysis["dataset2_bsds500"]["sobel"]["avg_iou"]
    d2_best = "Canny" if canny_iou >= sobel_iou else "Sobel"
    
    text_content = f"Dataset 1 Best Method: {d1_best}\nDataset 2 Best Method: {d2_best}\nReason: Canny's non-maximum suppression ensures thin, continuous boundaries which are far more suitable for both seed counting and general boundary tracing than raw Sobel magnitudes. Comparison based on MAE for seeds and mean IoU for natural scenes.\n"
    
    with open(os.path.join(output_dir, "best_method_per_dataset.txt"), "w") as f:
        f.write(text_content)
        
    # Create a feature combined npy for Assignment 4
    np.save(os.path.join(output_dir, "edge_features_combined.npy"), np.array([d1_canny.get("mae", 0), d1_sobel.get("mae", 0)]))
    
    print("Cross-dataset comparison completed.")

if __name__ == "__main__":
    run_comparison()

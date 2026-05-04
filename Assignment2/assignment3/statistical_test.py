import json
import csv
import scipy.stats as stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_errors(csv_path, pred_col, gt_col="count"):
    """Reads a CSV and computes the absolute errors between prediction and ground truth"""
    errors = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = int(row.get(gt_col, row.get("ground_truth")))
            pred = int(row[pred_col])
            errors.append(abs(gt - pred))
    return sorted(errors)

def main():
    print("=== Statistical Significance Testing (Dataset 1 Absolute Errors) ===")
    
    # 1. Load Assignment 2 Errors
    # Wait, assignment 2 generated `baseline_counts.csv` which has columns: filename, count, predicted, error
    a2_csv = os.path.join(SCRIPT_DIR, "output", "metrics_assignment2", "baseline_counts.csv")
    a2_errors = []
    with open(a2_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = float(row["prediction"])
            gt = float(row["ground_truth"])
            a2_errors.append(abs(pred - gt))
            
    # 2. Load Assignment 3 Errors (Canny and Sobel)
    a3_csv = os.path.join(SCRIPT_DIR, "edge_outputs", "dataset1_edge_counts.csv")
    canny_errors = get_errors(a3_csv, "canny_pred", "ground_truth")
    sobel_errors = get_errors(a3_csv, "sobel_pred", "ground_truth")
    
    # Ensure they are the same length for paired testing
    print(f"Sample Sizes -> Baseline: {len(a2_errors)}, Canny: {len(canny_errors)}, Sobel: {len(sobel_errors)}")
    
    # We will use the Wilcoxon signed-rank test instead of Paired T-Test
    # because absolute error distributions are generally not normally distributed (often right-skewed)
    
    # Baseline vs Canny
    stat1, p1 = stats.wilcoxon(a2_errors, canny_errors)
    print(f"\n1. Baseline vs Canny -> Wilcoxon Stat: {stat1}, p-value: {p1}")
    if p1 < 0.05:
        print("   Result: STATISTICALLY SIGNIFICANT difference between Baseline and Canny.")
    else:
        print("   Result: NO significant difference between Baseline and Canny.")
        
    # Baseline vs Sobel
    stat2, p2 = stats.wilcoxon(a2_errors, sobel_errors)
    print(f"\n2. Baseline vs Sobel -> Wilcoxon Stat: {stat2}, p-value: {p2}")
    if p2 < 0.05:
        print("   Result: STATISTICALLY SIGNIFICANT difference between Baseline and Sobel.")
    else:
        print("   Result: NO significant difference between Baseline and Sobel.")
        
    # Canny vs Sobel
    stat3, p3 = stats.wilcoxon(canny_errors, sobel_errors)
    print(f"\n3. Canny vs Sobel -> Wilcoxon Stat: {stat3}, p-value: {p3}")
    if p3 < 0.05:
        print("   Result: STATISTICALLY SIGNIFICANT difference between Canny and Sobel.")
    else:
        print("   Result: NO significant difference between Canny and Sobel.")

if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

BSDS_DIR = os.path.join(PARENT_DIR, "BSDS500", "data")
IMAGES_DIR = os.path.join(BSDS_DIR, "images", "train")
GT_DIR = os.path.join(BSDS_DIR, "groundTruth", "train")
EDGE_DIR = os.path.join(SCRIPT_DIR, "edge_outputs")

OUTPUT_DIR = os.path.join(EDGE_DIR, "dataset2_comparisons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_IDS = ["41004", "135069", "105053", "76002"]

def load_bsds_gt(mat_path):
    if not os.path.exists(mat_path):
        return None
    data = loadmat(mat_path)
    gt_cells = data['groundTruth']
    num_annotators = gt_cells.shape[1]
    combined_gt = None
    for i in range(num_annotators):
        gt_bound = gt_cells[0, i][0, 0]['Boundaries']
        if combined_gt is None:
            combined_gt = gt_bound.copy()
        else:
            combined_gt = np.logical_or(combined_gt, gt_bound)
    return combined_gt.astype(np.uint8) * 255

def create_dataset2_montage(image_id, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"BSDS500 Comparison - Image {image_id}", fontsize=20)
    
    # 1. Original
    img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image", fontsize=15)
    
    # 2. Ground Truth
    gt_path = os.path.join(GT_DIR, f"{image_id}.mat")
    gt = load_bsds_gt(gt_path)
    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title("Ground Truth (Boundaries)", fontsize=15)
    
    # 3. Canny
    canny_path = os.path.join(EDGE_DIR, "dataset2_canny_edges", f"{image_id}.jpg")
    canny = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
    axes[0, 2].imshow(canny, cmap='gray')
    axes[0, 2].set_title("Canny Edges", fontsize=15)
    
    # 4. Sobel
    sobel_path = os.path.join(EDGE_DIR, "dataset2_sobel_edges", f"{image_id}.jpg")
    sobel = cv2.imread(sobel_path, cv2.IMREAD_GRAYSCALE)
    axes[1, 0].imshow(sobel, cmap='gray')
    axes[1, 0].set_title("Sobel Edges", fontsize=15)
    
    # 5. Contour (Recalculate or use some placeholder if not saved individually)
    # Since we didn't save contour vis for dataset2 individually, we can generate it
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = img.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
    axes[1, 1].imshow(contour_vis)
    axes[1, 1].set_title("Canny Contours", fontsize=15)
    
    # 6. Metrics
    metrics_text = (
        f"Canny Metrics:\n"
        f"  Precision: {metrics['canny_precision']:.4f}\n"
        f"  Recall:    {metrics['canny_recall']:.4f}\n"
        f"  IoU:       {metrics['canny_iou']:.4f}\n\n"
        f"Sobel Metrics:\n"
        f"  Precision: {metrics['sobel_precision']:.4f}\n"
        f"  Recall:    {metrics['sobel_recall']:.4f}\n"
        f"  IoU:       {metrics['sobel_iou']:.4f}"
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14, family='monospace', va='center')
    axes[1, 2].axis('off')
    
    for ax in axes.flatten():
        if ax != axes[1, 2]:
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, f"dataset2_comp_{image_id}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # Load metrics
    all_metrics = {}
    with open(os.path.join(EDGE_DIR, "dataset2_edge_metrics.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['filename'].split('.')[0]
            if img_id in SAMPLE_IDS:
                all_metrics[img_id] = {
                    "canny_precision": float(row['canny_precision']),
                    "canny_recall": float(row['canny_recall']),
                    "canny_iou": float(row['canny_iou']),
                    "sobel_precision": float(row['sobel_precision']),
                    "sobel_recall": float(row['sobel_recall']),
                    "sobel_iou": float(row['sobel_iou'])
                }
                
    for mid in SAMPLE_IDS:
        if mid in all_metrics:
            create_dataset2_montage(mid, all_metrics[mid])

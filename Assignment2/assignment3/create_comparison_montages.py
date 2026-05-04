import os
import cv2
import matplotlib.pyplot as plt

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

SEG_DIR = os.path.join(PARENT_DIR, "output", "segmentation")
EDGE_DIR = os.path.join(SCRIPT_DIR, "edge_outputs")

# Source directories for images
DIRS = {
    "K-Means": os.path.join(SEG_DIR, "clustering", "kmeans"),
    "DBSCAN": os.path.join(SEG_DIR, "clustering", "dbscan"),
    "Final Detection": os.path.join(SEG_DIR, "final_detections"),
    "Canny Edges": os.path.join(EDGE_DIR, "dataset1_canny_edges"),
    "Sobel Edges": os.path.join(EDGE_DIR, "dataset1_sobel_edges"),
    "Contours Vis": os.path.join(EDGE_DIR, "dataset1_contours_vis")
}

OUTPUT_DIR = os.path.join(EDGE_DIR, "comparison_montages")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_IDS = ["1", "5", "10", "25", "50", "100"]

def create_montage(image_id):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Segmentation vs. Edge Detection Comparison - Image {image_id}", fontsize=20)
    
    methods = [
        ("K-Means", "png"),
        ("DBSCAN", "png"),
        ("Final Detection", "png"),
        ("Canny Edges", "jpg"),
        ("Sobel Edges", "jpg"),
        ("Contours Vis", "jpg")
    ]
    
    for i, (method_name, ext) in enumerate(methods):
        row = i // 3
        col = i % 3
        
        path = os.path.join(DIRS[method_name], f"{image_id}.{ext}")
        
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img)
            axes[row, col].set_title(method_name, fontsize=15)
        else:
            axes[row, col].text(0.5, 0.5, f"Missing:\n{method_name}", 
                                ha='center', va='center', fontsize=12)
            axes[row, col].set_title(f"{method_name} (NOT FOUND)", fontsize=15)
        
        axes[row, col].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, f"comparison_{image_id}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    for img_id in IMAGE_IDS:
        create_montage(img_id)

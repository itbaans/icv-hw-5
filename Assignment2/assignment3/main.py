import sys
import os
import subprocess
import time
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Add the parent directory to sys.path so we can import baseline_code if needed
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import the new Assignment 3 processing scripts directly
from dataset1_processing import process_dataset1
from dataset2_processing import process_dataset2
from cross_dataset_comparison import run_comparison

def main():
    print("="*60)
    print(" AGRITECH SEEDCOUNTER - ASSIGNMENT 3 MASTER PIPELINE ")
    print("="*60)
    
    print("\n[1/4] Running Assignment 2 Pipeline (Baseline & KMeans)...")
    print("      (Resolution is toned down by 0.5x in the core config natively)")
    start_t = time.time()
    
    # Run the original main.py from root
    # We use subprocess to avoid argparse or state conflicts
    root_main = os.path.join(PARENT_DIR, "main.py")
    try:
        # We run it with --save-for=none to avoid generating hundreds of segmentation images 
        # as requested previously by the user (metric focus)
        result = subprocess.run([sys.executable, root_main, "--save-for=none"], check=True)
        print(f"Assignment 2 pipeline completed in {time.time() - start_t:.1f}s.")
        
        # Copy the metrics over to the assignment 3 folder 
        src_metrics = os.path.join(PARENT_DIR, "output", "metrics")
        dst_metrics = os.path.join(SCRIPT_DIR, "output", "metrics_assignment2")
        if os.path.exists(src_metrics):
            if os.path.exists(dst_metrics):
                shutil.rmtree(dst_metrics)
            shutil.copytree(src_metrics, dst_metrics)
            print(f"Metrics copied to {dst_metrics}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running Assignment 2 pipeline: {e}")
        return

    print("\n[2/4] Running Dataset 1 (Seeds) Feature Extraction (Canny & Sobel)...")
    start_t = time.time()
    process_dataset1()
    print(f"Dataset 1 processing completed in {time.time() - start_t:.1f}s.")
    
    print("\n[3/4] Running Dataset 2 (BSDS500) Feature Extraction (Canny & Sobel)...")
    start_t = time.time()
    process_dataset2()
    print(f"Dataset 2 processing completed in {time.time() - start_t:.1f}s.")
    
    print("\n[4/4] Running Cross-Dataset Comparison & Analysis...")
    start_t = time.time()
    run_comparison()
    print(f"Comparison completed in {time.time() - start_t:.1f}s.")
    
    print("\n" + "="*60)
    print(" ALL PIPELINES COMPLETED SUCCESSFULLY ")
    print("="*60)

if __name__ == "__main__":
    main()

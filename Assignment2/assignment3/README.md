# AgriTech SeedCounter - Assignment 3

This directory contains the code and report for Assignment 3 of the AgriTech SeedCounter project. The focus of this assignment is on feature extraction (edge detection) and cross-dataset evaluation.

## Project Structure

- `main.py`: The master pipeline script that orchestrates the execution of all components.
- `dataset1_processing.py`: Script for processing Seed Dataset 1 using Canny and Sobel edge detection.
- `dataset2_processing.py`: Script for processing BSDS500 Dataset 2.
- `edge_detection.py`: Core utilities for image processing and edge detection filters.
- `statistical_test.py`: Statistical analysis of the results.
- `cross_dataset_comparison.py`: Generates comparison reports and visualizations across different datasets.
- `report/`: Contains the LaTeX source code and assets for the final report.
- `output/`: Directory where processed images, metrics, and logs are stored.

## How to Run

To execute the entire Assignment 3 pipeline, run the following command from this directory:

```bash
python main.py
```

### Pipeline Steps:
1. **Assignment 2 Integration**: Runs the baseline and K-Means segmentation from the previous assignment.
2. **Dataset 1 Processing**: Extracts features from seed images using Canny and Sobel operators.
3. **Dataset 2 Processing**: Extracts features from the BSDS500 dataset for comparison.
4. **Analysis**: Performs statistical tests and generates cross-dataset comparison results.

## Requirements
- Python 3.x
- Dependencies: `numpy`, `opencv-python`, `matplotlib`, `pandas`, `scipy` (ensure they are installed in your environment).

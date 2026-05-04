# AgriTech SeedCounter – Project Guide

This project implements an automated seed counting pipeline using classical computer vision techniques.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd AgriTech_SeedCounter
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install opencv-python numpy pyyaml scipy scikit-learn
   ```

3. **Data Preparation:**
   Place your seed images in the `data/` directory. Images should be named with their ground truth count (e.g., `100.jpg` for 100 seeds).

## Running the Pipeline

### Single Image / Main Run
To run the full pipeline on all images in the `data/` folder with default settings:
```bash
python main.py
```
This will generate outputs in the `output/` directory, including:
- `output/inspection/`: Visualisations of pipeline stages for each image.
- `output/metrics/`: Performance reports and detailed CSV logs.

### Running Experiments
To run a series of comparative experiments (ablation studies) efficiently:
```bash
python run_experiments.py
```
This script runs several variants (e.g., comparing filters, thresholding methods, and segmentation sources) and outputs:
- `output/metrics/experiment_results.csv`: A summary table of all variants.
- `output/experiments/`: Sample images from interesting experiments for visual audit.

## Configuration
All pipeline parameters (kernel sizes, area thresholds, etc.) are centralised in:
`baseline_code/config.yaml`

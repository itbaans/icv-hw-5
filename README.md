# Assignment 5: CNN from Scratch and Neural Style Transfer

This repository contains the solution for Assignment 5, focusing on end-to-end deep learning methods to complement the hand-engineered methods developed in Assignments 2 and 3.

## Running on Kaggle

The code is structured to be run seamlessly on Kaggle Notebooks. 
1. Push this repository to GitHub.
2. In your Kaggle Notebook, clone the repository:
   ```bash
   !git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   pip install -r requirements.txt
   ```
3. Upload the `Assignment2` datasets directly to Kaggle, or modify the dataset paths in the config to point to Kaggle's `/kaggle/input/` directories.

## Task 1: CNN from Scratch
- `task1_cnn/train.py`: Trains Model A and Model B.
- `task1_cnn/evaluate.py`: Compares the CNN models to Clustering (A1) and Edge Detection (A2).

## Task 2: Video NST and Matting
- `task2_nst_video/matting/train.py`: Trains the human matting model on the AISegment dataset.
- `task2_nst_video/video_pipeline.py`: Runs the complete video compositing pipeline.

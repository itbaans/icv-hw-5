import os
import cv2
import yaml
import sys
from main import stage_preprocessing

with open('baseline_code/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

data_dir = 'data'
files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def dummy_save(path, img):
    # Only save the filtered image to save disk I/O and time
    if 'filtered' in path:
        # Change .png to match original extension if needed, or just save as png
        # The main.py uses f"{name}.png"
        cv2.imwrite(path, img)

print(f"Fast preprocessing {len(files)} images...")
for i, fname in enumerate(files):
    name = os.path.splitext(fname)[0]
    out_path = os.path.join('output/preprocessed_images/filtered', f"{name}.png")
    
    # Skip if already exists
    if os.path.exists(out_path):
        continue
        
    img = cv2.imread(os.path.join(data_dir, fname))
    if img is None: continue
    
    # We pass dummy_save to prevent saving edges/gray/gaussian to save time
    stage_preprocessing(img, cfg, name, save_fn=dummy_save)
    if i % 10 == 0:
        print(f"Processed {i}/{len(files)}")
print("Done!")

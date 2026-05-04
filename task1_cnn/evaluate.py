import torch
import sys
import os
import argparse
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms

# Add Assignment2 baseline code to path to import evaluate logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Assignment2', 'baseline_code')))
try:
    from evaluate import generate_performance_summary
except ImportError:
    print("Could not import generate_performance_summary from Assignment2.")
    generate_performance_summary = None

from models import ModelA, ModelB

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
    def __len__(self):
        return len(self.image_filenames)
        
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        try:
            stem = os.path.splitext(img_name)[0]
            label = int(stem)
        except ValueError:
            label = 0
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.long)
        return image, label, img_name

def evaluate_model(model, dataloader, device):
    model.eval()
    records = []
    
    with torch.no_grad():
        for inputs, labels, filenames in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(filenames)):
                records.append({
                    "filename": filenames[i],
                    "prediction": predicted[i].item(),
                    "ground_truth": labels[i].item()
                })
    return records

def check_failure_cases(records, failure_cases_path):
    if not os.path.exists(failure_cases_path):
        print(f"Warning: Failure cases file not found at {failure_cases_path}")
        return 0
        
    with open(failure_cases_path, 'r') as f:
        failure_cases = json.load(f)
        
    failure_filenames = set([case['filename'] for case in failure_cases])
    
    fixed_count = 0
    for record in records:
        if record['filename'] in failure_filenames:
            if record['prediction'] == record['ground_truth']:
                fixed_count += 1
                
    return fixed_count

def plot_confusion_matrix(records, model_name, output_dir):
    y_true = [r['ground_truth'] for r in records]
    y_pred = [r['prediction'] for r in records]
    
    # Only plot classes that actually exist in the data/predictions to keep it readable
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN Models for Seed Counting")
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle dataset paths')
    parser.add_argument('--test-run', action='store_true', help='Run quickly on a few images for testing')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    image_dir = config['data']['image_dir']
    if args.kaggle:
        image_dir = '/kaggle/input/agrivision-assignment2/output/preprocessed_images/filtered/'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = config['data']['image_size']
    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = EvalDataset(image_dir, transform=eval_transform)
    
    if args.test_run:
        print("TEST RUN: Subsetting dataset to 32 images...")
        dataset.image_filenames = dataset.image_filenames[:32]
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    
    os.makedirs('cnn_outputs', exist_ok=True)
    
    # Load Models
    model_a = ModelA(num_classes=150).to(device)
    if os.path.exists('cnn_outputs/model_a.pth'):
        model_a.load_state_dict(torch.load('cnn_outputs/model_a.pth', map_location=device, weights_only=True))
    else:
        print("Warning: model_a.pth not found. Evaluating randomly initialized model.")
        
    model_b = ModelB(num_classes=150, dropout_rate=config['model_b']['dropout_rate']).to(device)
    if os.path.exists('cnn_outputs/model_b.pth'):
        model_b.load_state_dict(torch.load('cnn_outputs/model_b.pth', map_location=device, weights_only=True))
    else:
        print("Warning: model_b.pth not found. Evaluating randomly initialized model.")

    # Evaluate Model A
    print("\n--- Evaluating Model A ---")
    records_a = evaluate_model(model_a, dataloader, device)
    
    # Evaluate Model B
    print("\n--- Evaluating Model B ---")
    records_b = evaluate_model(model_b, dataloader, device)
    
    # Calculate Metrics using Assignment 2 evaluate.py logic
    failure_cases_path = '../Assignment2/output/metrics/failure_cases.json'
    
    if generate_performance_summary:
        summary_a = generate_performance_summary(records_a, output_path=None)
        summary_b = generate_performance_summary(records_b, output_path=None)
        
        fixed_a = check_failure_cases(records_a, failure_cases_path)
        fixed_b = check_failure_cases(records_b, failure_cases_path)
        
        # Load Baseline metrics for comparison if available
        comparison_data = []
        try:
            with open('../Assignment2/output/metrics/performance_summary.json', 'r') as f:
                a2_summary = json.load(f)
            comparison_data.append({
                "Method": "Edge Detection (A2)",
                "Accuracy": a2_summary.get("accuracy_within_threshold_pct", "N/A"),
                "MAE": a2_summary.get("mae", "N/A"),
                "RMSE": a2_summary.get("rmse", "N/A"),
                "Failure Cases Fixed": "reference"
            })
        except:
            pass
            
        comparison_data.extend([
            {
                "Method": "CNN Model A",
                "Accuracy": summary_a.get("accuracy_within_threshold_pct", "N/A"),
                "MAE": summary_a.get("mae", "N/A"),
                "RMSE": summary_a.get("rmse", "N/A"),
                "Failure Cases Fixed": fixed_a
            },
            {
                "Method": "CNN Model B",
                "Accuracy": summary_b.get("accuracy_within_threshold_pct", "N/A"),
                "MAE": summary_b.get("mae", "N/A"),
                "RMSE": summary_b.get("rmse", "N/A"),
                "Failure Cases Fixed": fixed_b
            }
        ])
        
        df_comp = pd.DataFrame(comparison_data)
        df_comp.to_csv('cnn_outputs/comparison_table.csv', index=False)
        print("\nSaved comparison_table.csv")
        print(df_comp)
        
    else:
        print("Missing generate_performance_summary. Skipping metric calculations.")

    # Plotting
    plot_confusion_matrix(records_a, "Model_A", "cnn_outputs")
    plot_confusion_matrix(records_b, "Model_B", "cnn_outputs")
    print("\nEvaluation complete. Plots and CSVs saved to cnn_outputs/.")

if __name__ == "__main__":
    main()

import os
import yaml
import argparse
import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from models import AblationModel
import time

def main():
    parser = argparse.ArgumentParser(description="Ablation Study for CNN Models")
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle dataset paths')
    parser.add_argument('--test-run', action='store_true', help='Run quickly for testing')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    if args.kaggle:
        config['data']['image_dir'] = '/kaggle/input/datasets/abdullahahmedani/seeds-data/filtered'
        
    epochs = config['training']['epochs']
    if args.test_run:
        epochs = 1
        print("TEST RUN: Epochs set to 1")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(config['seed'])
    train_loader, val_loader = get_dataloaders(config)
    
    if args.test_run:
        train_loader.dataset.subset.indices = train_loader.dataset.subset.indices[:16]
        val_loader.dataset.subset.indices = val_loader.dataset.subset.indices[:16]
    
    os.makedirs('cnn_outputs', exist_ok=True)
    
    # Ablation Grid
    # We will test different layer sizes, kernel sizes, dropout, and learning rates
    grid = {
        'filters': [
            [16, 32, 64],           # 3 layers (Shallow)
            [32, 64, 128],          # 3 layers (Baseline)
            [16, 32, 64, 128],      # 4 layers (Deep)
            [32, 64, 128, 256]      # 4 layers (Very Deep)
        ],
        'kernel_size': [3, 5, 7],
        'dropout_rate': [0.2, 0.4],
        'learning_rate': [0.001, 0.0001],
        'optimizer': ['adam', 'sgd'],
        'use_residual': [True, False],
        'activation': ['relu', 'leaky_relu'],
        'weight_decay': [0.0, 1e-4]
    }
    
    if args.test_run:
        grid['filters'] = [[16, 32, 64]]
        grid['kernel_size'] = [3]
        grid['dropout_rate'] = [0.2]
        grid['learning_rate'] = [0.001]
        grid['optimizer'] = ['adam']
        grid['use_residual'] = [True, False]
        grid['activation'] = ['relu']
        grid['weight_decay'] = [0.0]

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting ablation study over {len(combinations)} configurations...")
    
    results = []
    best_overall_val_mae = float('inf')
    
    for idx, params in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] Training config: {params}")
        
        model = AblationModel(
            filters=params['filters'], 
            kernel_size=params['kernel_size'], 
            dropout_rate=params['dropout_rate'],
            use_residual=params['use_residual'],
            activation=params['activation']
        ).to(device)
        
        criterion = nn.MSELoss()
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=params['weight_decay'])
        
        best_val_mae_for_run = float('inf')
        best_epoch = 0
        final_train_mae = 0.0
        
        start_time = time.time()
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            total_train_mae = 0.0
            total_train = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                mae = torch.abs(outputs.data - labels).sum().item()
                total_train_mae += mae
                total_train += labels.size(0)
                
            train_mae = total_train_mae / total_train
            
            # Validation
            model.eval()
            total_val_mae = 0.0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    mae = torch.abs(outputs.data - labels).sum().item()
                    total_val_mae += mae
                    total_val += labels.size(0)
            
            val_mae = total_val_mae / total_val
            
            if val_mae < best_val_mae_for_run:
                best_val_mae_for_run = val_mae
                best_epoch = epoch + 1
                
                # Check if this is the absolute best model overall
                if val_mae < best_overall_val_mae:
                    best_overall_val_mae = val_mae
                    torch.save({
                        'state_dict': model.state_dict(),
                        'params': params
                    }, 'cnn_outputs/best_ablation_model.pth')
                    
            if epoch == epochs - 1:
                final_train_mae = train_mae
                
        train_time = time.time() - start_time
        print(f"Finished in {train_time:.1f}s. Best Val MAE: {best_val_mae_for_run:.4f} at epoch {best_epoch}")
        
        # Log result
        results.append({
            'Run': idx + 1,
            'Filters': str(params['filters']),
            'Kernel Size': params['kernel_size'],
            'Dropout Rate': params['dropout_rate'],
            'Learning Rate': params['learning_rate'],
            'Optimizer': params['optimizer'],
            'Residual': params['use_residual'],
            'Activation': params['activation'],
            'Weight Decay': params['weight_decay'],
            'Best Epoch': best_epoch,
            'Final Train MAE': round(final_train_mae, 4),
            'Best Val MAE': round(best_val_mae_for_run, 4),
            'Training Time (s)': round(train_time, 1)
        })
        
        # Save incrementally
        df = pd.DataFrame(results)
        df.to_csv('cnn_outputs/ablation_results.csv', index=False)
        
    print(f"\n✅ Ablation Study Complete! Results saved to cnn_outputs/ablation_results.csv")
    print(f"👑 Best overall model achieved Val MAE {best_overall_val_mae:.4f} and was saved to cnn_outputs/best_ablation_model.pth")

if __name__ == "__main__":
    main()

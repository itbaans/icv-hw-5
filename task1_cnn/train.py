import os
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from models import ModelA, ModelB
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, config, device, save_path, model_name):
    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_mae = 0.0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Regression MAE
            mae = torch.abs(outputs.data - labels).sum().item()
            total_mae += mae
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_mae = total_mae / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        total_mae_val = 0.0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                # Regression MAE
                mae = torch.abs(outputs.data - labels).sum().item()
                total_mae_val += mae
                total_val += labels.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = total_mae_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best {model_name} with val_loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {epochs_no_improve} epochs (Early stopping disabled).")
                # break

    # Save metrics to CSV
    df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_mae': train_maes,
        'val_mae': val_maes
    })
    df.to_csv(f'cnn_outputs/{model_name}_metrics.csv', index=False)
    
    # Plot loss
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_name} Loss Curve')
    plt.legend()
    plt.savefig(f'cnn_outputs/{model_name}_loss.png')
    plt.close()
    
    # Plot MAE
    plt.figure()
    plt.plot(df['epoch'], df['train_mae'], label='Train MAE')
    plt.plot(df['epoch'], df['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (Seeds)')
    plt.title(f'{model_name} MAE Curve')
    plt.legend()
    plt.savefig(f'cnn_outputs/{model_name}_mae.png')
    plt.close()

    return train_losses, val_losses, train_maes, val_maes

def main():
    parser = argparse.ArgumentParser(description="Train CNN Models for Seed Counting")
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle dataset paths')
    parser.add_argument('--test-run', action='store_true', help='Run quickly on a few images for testing')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    if args.kaggle:
        # Override paths for Kaggle
        config['data']['image_dir'] = '/kaggle/input/datasets/abdullahahmedani/seeds-data/filtered'
        
    if args.test_run:
        config['training']['epochs'] = 1
        print("TEST RUN: Epochs set to 1")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fix seed
    torch.manual_seed(config['seed'])
    
    train_loader, val_loader = get_dataloaders(config)
    
    if args.test_run:
        print("TEST RUN: Subsetting datasets to 32 images...")
        train_loader.dataset.subset.indices = train_loader.dataset.subset.indices[:32]
        val_loader.dataset.subset.indices = val_loader.dataset.subset.indices[:32]
    
    os.makedirs('cnn_outputs', exist_ok=True)
    
    # --- Train Model A ---
    print("\n--- Training Model A ---")
    model_a = ModelA().to(device)
    criterion = nn.MSELoss()
    
    if config['training']['optimizer'] == 'adam':
        optimizer_a = optim.Adam(model_a.parameters(), lr=config['training']['learning_rate'])
    else:
        optimizer_a = optim.SGD(model_a.parameters(), lr=config['training']['learning_rate'], momentum=config['training']['momentum'])
        
    train_model(model_a, train_loader, val_loader, criterion, optimizer_a, config, device, 'cnn_outputs/model_a.pth', 'Model_A')
    
    # --- Train Model B ---
    print("\n--- Training Model B ---")
    model_b = ModelB(dropout_rate=config['model_b']['dropout_rate']).to(device)
    
    # Use Adam with weight decay (L2) for Model B
    optimizer_b = optim.Adam(model_b.parameters(), lr=config['training']['learning_rate'], weight_decay=float(config['model_b']['weight_decay']))
    train_model(model_b, train_loader, val_loader, criterion, optimizer_b, config, device, 'cnn_outputs/model_b.pth', 'Model_B')
    
    print("\nTraining complete. Metrics and plots saved to cnn_outputs/.")

if __name__ == "__main__":
    main()

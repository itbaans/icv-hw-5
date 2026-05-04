import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SeedDataset(Dataset):
    def __init__(self, image_dir, labels_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image filenames from the directory
        self.image_filenames = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
    def __len__(self):
        return len(self.image_filenames)
        
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Open image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        # Calculate label from the filename (e.g. '4.jpg' -> 4)
        try:
            stem = os.path.splitext(img_name)[0]
            label = int(stem)
        except ValueError:
            label = 0
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to tensor (count of seeds)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def get_dataloaders(config):
    """
    Returns train and validation dataloaders based on config.
    """
    img_size = config['data']['image_size']
    
    # Define augmentations as per requirement
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # Zoom +- 10%
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SeedDataset(
        image_dir=config['data']['image_dir'],
        labels_path=config['data']['labels_path'],
        transform=None # We will apply manually or use a wrapper if needed
    )
    
    # Split dataset
    dataset_size = len(dataset)
    val_size = int(config['data']['validation_split'] * dataset_size)
    train_size = dataset_size - val_size
    
    # Fix seed for reproducibility
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Apply transforms (in PyTorch, applying transforms after random_split can be done via dataset wrapper)
    class TransformDatasetWrapper(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)
            
    train_dataset = TransformDatasetWrapper(train_dataset, transform=train_transform)
    val_dataset = TransformDatasetWrapper(val_dataset, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    
    return train_loader, val_loader

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class JapaneseTextDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
        self.labels = self.load_labels(label_file)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor() # Convert image to tensor and normalize
        ])

        unique_chars = set(self.labels.values())
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def load_labels(self, label_file):
        labels = {} # Labels into a dictionary
        with open(label_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split() # Split each line into parts by spaces
                if len(parts) == 3: # If line has 3 parts
                    filename, utf8_hex, character = parts # define respective parts
                    labels[os.path.splitext(filename)[0]] = character # Add filename:character as key:value pairs to dictionary
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx]) # Path to image folder and files
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        # Get label from dictionary via filename as key without file extension, returns empty string if not found    
        filename = os.path.splitext(self.image_files[idx])[0]
        label_str = self.labels.get(filename, '')
        label = self.char_to_idx.get(label_str, 0)
        return image, label

def get_dataloaders(image_dir, label_file, batch_size,
                    transform=None, test_size=0.1, val_size=0.1,
                    num_workers=os.cpu_count()//2):
    dataset = JapaneseTextDataset(image_dir, label_file, transform)

    # Split to test segments
    train_val_files, test_files = train_test_split(dataset.image_files, test_size=test_size, random_state=10) # Random state set to be able to reliably compare metrics during multiple training sessions

    # Split to train and validation segments
    train_files, val_files = train_test_split(train_val_files, test_size=val_size, random_state=10)

    # Removed pipeline function since no OCR/translation is needed anymore
    # If no OCR/translation processing is needed, simply work with the image files
    train_dataset = JapaneseTextDataset(image_dir, label_file, transform)
    train_dataset.image_files = train_files

    test_dataset = JapaneseTextDataset(image_dir, label_file, transform)
    test_dataset.image_files = test_files

    val_dataset = JapaneseTextDataset(image_dir, label_file, transform)
    val_dataset.image_files = val_files

    # Splitting data to batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True) # Shuffle set to True to remove bias from data selection
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=True, pin_memory=True) # Pin memory set to True for faster data allocation from CPU to GPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=True, pin_memory=True)

    return train_loader, test_loader, val_loader

def move_to_device(loader, device):
    """Move data to device for more efficiency."""
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
    return loader


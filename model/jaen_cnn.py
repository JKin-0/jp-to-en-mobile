import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # Progress bar for loops during training
from transformers import MarianMTModel, MarianTokenizer
import argparse
import keras_ocr
from jaen_dataloader import get_dataloaders

class CNN(nn.Module):
    def __init__(self, num_classes=700):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True) # Creates directories if they don't exist
    writer = SummaryWriter(log_dir) # Writer for TensorBoard

    for epoch in range(num_epochs):
        start_time = time.time()
        model.to(device)
        epoch_loss = 0

        # Training
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # Forward pass, compute predictions for batch
            loss = criterion(outputs, labels) # Calculate loss between predictions and true labels
            optimizer.zero_grad() # Clear old gradients from last step
            loss.backward() # Compute gradient of loss with model parameters
            optimizer.step() # Update model parameters based on computed gradients
            epoch_loss += loss.item() # Convert tensor value into numerical value

        epoch_time = time.time() - start_time

        # Evaluation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad(): # No gradient calculation
            for images, labels in tqdm(val_loader, desc=f"Validating epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) # Get predicted class labels from index 1 (maximum values)
                total += labels.size(0) # Return the size of tensor from index 0 (the total examples in batch)
                correct += (predicted == labels).sum().item() # Calculate the number of True values and convert it to single numerical value

        val_accuracy = correct / total

        with writer:  # Writer for TensorBoard
            writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Epoch Time', epoch_time, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Time: {epoch_time:.2f} seconds')

def test_model(test_loader, model, device):
    model.to(device)
    model.eval()
    print("Starting testing...")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Predicting classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test accuracy: {100 * correct / total:.2f}%')

def pipeline(image_paths, model, tokenizer, device, output_file="translated_texts.txt"):
    # Keras-OCR
    print("Starting pipeline...")
    ocr_pipeline = keras_ocr.pipeline.Pipeline()
    recognized_texts = []

    with open(output_file, "w", encoding="utf-8") as file:
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images", unit="image")):
            abs_image_path = os.path.abspath(image_path)
            image = keras_ocr.tools.read(abs_image_path)
            prediction_groups = ocr_pipeline.recognize([image]) # Identify text from image

            for predictions in prediction_groups:
                for text, box in predictions:
                    # inputs = tokenizer.encode(text, return_tensors='pt').to(device) # Tokenize text and returning result as PyTorch tensor
                    # outputs = model.generate(inputs) # Input text into MarianMT model for translation
                    # translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # Decode back to text and skip special tokens (specific symbols for language model)
                    recognized_texts.append(text)

                    file.write(f"Image {i + 1}: {text}\n")

    print("Pipeline finished.")
    return recognized_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1, help='Version number') # Add version number as command line argument for making multiple versions of model
    parser.add_argument('--data_path', type=str, required=True, help='Path to extracted image data')
    parser.add_argument('--label_file', type=str, required=True, help='Path to label file')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument ('-l', '--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = args.data_path
    label_file = args.label_file
    num_classes = 700

    # MarianMT
    # model_name = 'Helsinki-NLP/opus-mt-ja-en'
    # tokenizer = MarianTokenizer.from_pretrained(model_name) # Divide text into small parts
    # translation_model = MarianMTModel.from_pretrained(model_name).to(device)

    train_loader, test_loader, val_loader = get_dataloaders(image_dir, label_file, batch_size=args.batch_size)
    
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(train_loader, val_loader, model, criterion,
                optimizer, args.num_epochs, device=device) # Train and validate the model
    torch.save(model.state_dict(), f'jaenCNN_v{args.version}.pth') # Save trained model

    loaded_model = CNN(num_classes).to(device)
    loaded_model.load_state_dict(torch.load(f'jaenCNN_v{args.version}.pth')) # Load the saved model
    loaded_model.eval()

    test_model(test_loader, loaded_model, device=device) # Test the model

    image_paths = [os.path.join(image_dir, fname) for fname in test_loader.dataset.image_files]
    pipeline(image_paths, model=loaded_model, tokenizer=None, device=device)

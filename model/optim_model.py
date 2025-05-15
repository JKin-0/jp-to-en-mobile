import os
import time
import torch
import torch.nn as nn
import tempfile
from torch.quantization import quantize_dynamic
from jaen_cnn import CNN
from jaen_dataloader import get_dataloaders

def model_size_mb(model, filename):
    torch.save(model.state_dict(), filename)
    return os.path.getsize(filename) / 1_000_000

def eval_acc(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to('cpu'), y.to('cpu')
            outputs = model(X)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100

def measure_latency(model, num_batches=10):
    model.eval()
    total_time = 0.0
    batches_run = 0
    with torch.no_grad():
        for i, (X, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            X = X.to('cpu')
            start_time = time.time()
            _ = model(X)
            end_time = time.time()
            total_time += (end_time - start_time)
            batches_run += 1
    avg_latency_ms = (total_time / batches_run) * 1000
    return avg_latency_ms

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ["jaenCNN_v3.pth", "jaenCNN_v4.pth", "jaenCNN_v5.pth"]

    _, test_loader, _ = get_dataloaders("data/", "label.txt", batch_size=32)

    with open("ablation_table.txt", "w") as f:
        f.write(f"\nAblation table:\n{'Model':<20} | {'Size (MB)':<10} | {'Latency (ms/batch)':<20} | {'Accuracy (%)':<15}")
        f.write("-" * 70 + "\n")

        for model_path in models:
            model = CNN(700)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval().to(device)

            quantized_model = quantize_dynamic(
                model.cpu(),
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            quantized_model.eval()

            torch.save(quantized_model.state_dict(), f"{model_path.replace('.pth', '_quantized.pth')}")

            accuracy = eval_acc(quantized_model, test_loader)
            latency_ms = measure_latency(quantized_model)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                size_mb = model_size_mb(quantized_model, tmp_file.name)

            line = f"{model_path:<20} | {size_mb:.2f} MB    | {latency_ms:.2f} ms        | {accuracy:.2f}%\n"
            f.write(line)
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import psutil
import logging

# Setup logging for performance monitoring
logging.basicConfig(filename='performance.log', level=logging.INFO)

# Global variables to accumulate total time and memory
total_time = 0
peak_memory = 0

# Function to measure the execution time and memory usage of a function
def monitor_performance(func):
    def wrapper(*args, **kwargs):
        global total_time, peak_memory
        start_time = time.time()
        process = psutil.Process()
        start_mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        result = func(*args, **kwargs)
        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        exec_time = end_time - start_time
        mem_usage = end_mem - start_mem
        peak_mem = process.memory_info().vms / (1024 * 1024)  # Virtual memory size for peak usage
        logging.info(f"{func.__name__}: Time: {exec_time:.4f}s, Memory Change: {mem_usage:.2f}MB")
        total_time += exec_time
        peak_memory = max(peak_memory, peak_mem)
        return result
    return wrapper

# Class for handling the entire Alexis process
class AlexisAnomalyDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.mean = None
        self.std = None
        self.train_data = None
        self.val_data = None
        self.train_losses = []
        self.val_losses = []

    @monitor_performance
    def load_data(self):
        self.data = pd.read_csv(self.data_path).dropna()

    @monitor_performance
    def compute_checksum(self):
        last_50_values = self.data.values[-50:]
        golden_reference = np.full_like(last_50_values, 4.48)
        mse_checksum = ((last_50_values - golden_reference) ** 2).mean()
        anomaly_detected = mse_checksum > 0.01
        return mse_checksum, anomaly_detected

    @monitor_performance
    def split_data(self, test_size=0.3, random_state=42):
        self.train_data, self.val_data = train_test_split(self.data, test_size=test_size, random_state=random_state)

    @monitor_performance
    def normalize_data(self):
        tensor_train_data = torch.tensor(self.train_data.values, dtype=torch.float32)
        tensor_val_data = torch.tensor(self.val_data.values, dtype=torch.float32)
        self.mean = tensor_train_data.mean(dim=0)
        self.std = tensor_train_data.std(dim=0)
        self.tensor_train_data = (tensor_train_data - self.mean) / self.std
        self.tensor_val_data = (tensor_val_data - self.mean) / self.std

    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super(AlexisAnomalyDetector.Autoencoder, self).__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(1, 16),  # Reduced complexity
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),  # Reduced dropout
                torch.nn.Linear(16, 4),
                torch.nn.ReLU()
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(4, 16),  # Reduced complexity
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),  # Reduced dropout
                torch.nn.Linear(16, 1)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    @monitor_performance
    def train_model(self, num_epochs=50, patience=10, lr=0.005):
        self.model = self.Autoencoder()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)  # Changed optimizer to SGD
        best_val_loss = float('inf')
        count = 0

        for epoch in range(num_epochs):
            # Forward pass - Training
            self.model.train()
            train_outputs = self.model(self.tensor_train_data)
            train_loss = criterion(train_outputs, self.tensor_train_data)
            self.train_losses.append(train_loss.item())

            # Forward pass - Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.tensor_val_data)
                val_loss = criterion(val_outputs, self.tensor_val_data)
                self.val_losses.append(val_loss.item())

            # Backward and optimize - Training
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Early stopping based on validation loss
            if val_loss.item() < best_val_loss - 0.1:
                best_val_loss = val_loss.item()
                count = 0
            else:
                count += 1
                if count == patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

    @monitor_performance
    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(self.tensor_val_data)
            mse_loss = torch.nn.MSELoss()(val_outputs, self.tensor_val_data)
            mae_loss = torch.abs(val_outputs - self.tensor_val_data).mean()

        mse_per_data_point = ((val_outputs - self.tensor_val_data) ** 2).mean(dim=1)
        q1 = np.quantile(mse_per_data_point.numpy(), 0.25)
        q3 = np.quantile(mse_per_data_point.numpy(), 0.75)
        iqr = q3 - q1
        dynamic_threshold = q3 + 1.5 * iqr

        accuracy = (mse_per_data_point < dynamic_threshold).float().mean().item()
        
        return mse_loss, mae_loss, accuracy

    def plot_losses(self):
        fig, ax = plt.subplots()
        ax.plot(self.train_losses, label='Training Loss')
        ax.plot(self.val_losses, label='Validation Loss')
        ax.set_title('Training & Validation Loss', fontweight='bold')
        ax.set_xlabel('Epochs', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()

# Track memory before starting the execution
start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

# Example usage:
detector = AlexisAnomalyDetector('Filename.csv')
detector.load_data()
checksum, anomaly_detected = detector.compute_checksum()
print(f"Checksums comparison - MSE: {checksum:.4f}")
if anomaly_detected:
    print("The last 50 values deviate from the golden reference.")
else:
    print("The last 50 values are consistent with the golden reference.")

detector.split_data()
detector.normalize_data()
detector.train_model()
mse_loss, mae_loss, accuracy = detector.evaluate_model()

print("\033[1mFinal Model Performance on Validation Data\033[0m")
print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"MAE Loss: {mae_loss.item():.4f}")
print(f"Accuracy: {accuracy:.2%}")
if anomaly_detected:
    print("\033[1mAnomaly detected\033[0m")
else:
    print("\033[1mNo anomaly detected\033[0m")

detector.plot_losses()

# Track memory after execution
end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

# Print total time and memory overhead
print(f"\033[1mTotal Execution Time: {total_time:.4f} seconds\033[0m")
print(f"\033[1mTotal Memory Overhead: {end_memory - start_memory:.2f} MB\033[0m")
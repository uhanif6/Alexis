import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import seaborn as sns

# Load bitstream data from file
data = pd.read_csv('Filename.csv').dropna()


last_50_values = data.values[-50:]
golden_reference = np.full_like(last_50_values, 4.48)

# Compare the Checksums using MSE for a more accurate comparison
mse_checksum = ((last_50_values - golden_reference) ** 2).mean()
anomaly_detected = mse_checksum > 0.01  # Adjusted threshold
print(f"Checksums comparison - MSE: {mse_checksum:.4f}")
if anomaly_detected:
    print("The last 50 values deviate from the golden reference.")
else:
    print("The last 50 values are consistent with the golden reference.")

# Define losses
train_losses = []
val_losses = []
true_labels = []
predictions = []

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

# Convert bitstream data to PyTorch tensors
tensor_train_data = torch.tensor(train_data.values, dtype=torch.float32)
tensor_val_data = torch.tensor(val_data.values, dtype=torch.float32)

# Normalize input data based on training data
mean = tensor_train_data.mean(dim=0)
std = tensor_train_data.std(dim=0)
tensor_train_data = (tensor_train_data - mean) / std
tensor_val_data = (tensor_val_data - mean) / std

# Define autoencoder architecture with reduced dropout
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 64),  # Increased number of neurons
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),  # Increased number of neurons
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate autoencoder model
model = Autoencoder()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Train autoencoder on bitstream data with early stopping
num_epochs = 100
patience = 20
count = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Forward pass - Training
    model.train()
    train_outputs = model(tensor_train_data)
    train_loss = criterion(train_outputs, tensor_train_data)
    train_losses.append(train_loss.item())

    # Forward pass - Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(tensor_val_data)
        val_loss = criterion(val_outputs, tensor_val_data)
        val_losses.append(val_loss.item())

        # Collecting true labels and predictions for ROC calculation
        predictions.extend(val_outputs.numpy().flatten())
        true_labels.extend(tensor_val_data.numpy().flatten())

    # Backward and optimize - Training
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Early stopping based on validation loss
    if val_loss.item() < best_val_loss - 0.01:
        best_val_loss = val_loss.item()
        count = 0
    else:
        count += 1
        if count == patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Binarize the true labels and predictions
threshold = np.median(true_labels)  # You can also try other thresholds like mean
binary_true_labels = np.array(true_labels) > threshold
binary_predictions = np.array(predictions) > threshold

# Compute ROC Curve and AUC
fpr, tpr, _ = roc_curve(binary_true_labels, binary_predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Reconstruction Error Distribution
reconstruction_errors = np.array(predictions) - np.array(true_labels)
plt.figure()
sns.histplot(reconstruction_errors, bins=50, kde=True)
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
# plt.title('Reconstruction Error Distribution')
plt.show()

# Calculate and print performance metrics
mse_loss = criterion(torch.tensor(predictions), torch.tensor(true_labels))
mae_loss = torch.abs(torch.tensor(predictions) - torch.tensor(true_labels)).mean()

print(f"Final Model Performance")
print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"MAE Loss: {mae_loss.item():.4f}")
print(f"AUC: {roc_auc:.2f}")
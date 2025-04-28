import numpy as np
import pickle
import os
from torch.utils.tensorboard import SummaryWriter

# --- Paths ---
HISTORY_PATH = "model/history.npz"
TEST_METRICS_PATH = "model/test_metrics.npz"
CONFUSION_MATRIX_PATH = "model/confusion_matrix.pkl"
CLASS_ACCURACY_PATH = "model/class_accuracy.npz"

LOG_DIR = "logs/fit/converted"

# Create writer
writer = SummaryWriter(LOG_DIR)

# Load training history
history = np.load(HISTORY_PATH)
epochs = range(len(history['accuracy']))

print("📦 Writing training and validation metrics...")
for epoch in epochs:
    writer.add_scalar("Train/Accuracy", history['accuracy'][epoch], epoch)
    writer.add_scalar("Train/Loss", history['loss'][epoch], epoch)
    writer.add_scalar("Val/Accuracy", history['val_accuracy'][epoch], epoch)
    writer.add_scalar("Val/Loss", history['val_loss'][epoch], epoch)

# Load and write test metrics
test_metrics = np.load(TEST_METRICS_PATH)
writer.add_scalar("Test/Accuracy", test_metrics["test_acc"].item(), 0)
writer.add_scalar("Test/Loss", test_metrics["test_loss"].item(), 0)

# Optional: class-wise accuracy
class_data = np.load(CLASS_ACCURACY_PATH)
for i, acc in enumerate(class_data["class_accuracy"]):
    writer.add_scalar(f"ClassAccuracy/Class_{i}", acc, 0)

writer.flush()
writer.close()
print(f"✅ TensorBoard logs written to: {LOG_DIR}")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd
from config import *

# Load the saved model
model = load_model(MODEL_PATH)

# Load CIFAR-10 test data
(_, _), (test_images, test_labels) = cifar10.load_data()

# Normalize test images
test_images = test_images.astype('float32') / 255.0

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Save test metrics (Loss and Accuracy)
np.savez(TEST_METRICS_PATH, test_loss=test_loss, test_acc=test_acc)

# Save the confusion matrix
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, predicted_labels)

# Save confusion matrix to a pickle file
with open(CM_PATH, 'wb') as f:
    pickle.dump(cm, f)

# Calculate and save class-wise accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Save class-wise accuracy to a .npz file
np.savez(ACCURACY_CLASS_PATH, class_accuracy=class_accuracy)

data = np.load(HISTORY_PATH)
df = pd.DataFrame({key: data[key] for key in data.files})
df["epoch"] = df.index + 1
df.to_csv(METRICS_PATH, index=False)


# Print out the saved metrics
print(f"✅ Saved test metrics: Accuracy = {test_acc:.4f}, Loss = {test_loss:.4f}")
print(f"✅ Saved confusion matrix and class-wise accuracy.")

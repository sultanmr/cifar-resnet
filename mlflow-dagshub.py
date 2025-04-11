import mlflow
from tensorflow.keras.models import load_model
from dagshub import dagshub_logger
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import plot_model

from config import *

# Set up environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Load history and model
history = dict(np.load(HISTORY_PATH))
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load test data
data = np.load(TEST_DATA_PATH)
x_test = data['images'] 
y_test = data['labels']  
y_test_labels = np.argmax(y_test, axis=-1).astype(int) 
# Get predictions
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=-1).astype(int)

def plot_and_log_model(model):
    model_plot_path = 'model_architecture.png'
    mlflow.keras.log_model(model, model.__class__.__name__)
    plot_model(model, to_file=model_plot_path, 
               show_shapes=True, 
               show_layer_names=True,
               expand_nested=True)
    mlflow.log_artifact(model_plot_path)
    
# Function to plot and log training curves
def plot_and_log_training(history):
    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    # Save plot locally and log it to MLflow
    plot_path = "training_curves.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

# Function to log confusion matrix
def log_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    # Save confusion matrix plot locally and log it to MLflow
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

# Start MLflow run
with mlflow.start_run():
    # Get active run ID and print it
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id
    print(f"Run ID: {run_id}")
    
    params = model.get_config()
    for param, value in params.items():
        mlflow.log_param(param, str(value))  

    # Start logging with DagsHub logger
    with dagshub_logger() as logger:
        
        # Log hyperparameters
        logger.log_hyperparams({
            'model': model.__class__.__name__,
            'epochs': len(history['accuracy']),
            'learning_rate': model.optimizer.learning_rate.numpy(),  # Extract learning rate
            'accuracy': history['accuracy'][-1],
            'loss': history['loss'][-1],
        })
        # Log metrics after each epoch
        for i in range(len(history['accuracy'])):
            hist = {
                'train_accuracy': history['accuracy'][i],
                'val_accuracy': history['val_accuracy'][i],
                'train_loss': history['loss'][i],
                'val_loss': history['val_loss'][i],
            }
            logger.log_metrics(hist, step=i)

        # Log the model to MLflow
        plot_and_log_model(model)       

        # Call function to plot and log training curves
        plot_and_log_training(history)

        # Log confusion matrix with CIFAR-10 labels (0-9)
        log_confusion_matrix(y_test_labels, y_pred_labels)

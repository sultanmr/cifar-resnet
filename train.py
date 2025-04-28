"""
CIFAR-10 Image Classification using ResNet50 with TensorFlow/Keras

This module implements a deep learning pipeline for CIFAR-10 classification using:
- ResNet50 as base architecture
- Data augmentation techniques
- Regularization methods (L2, Dropout, BatchNorm)
- Training callbacks (EarlyStopping, ReduceLROnPlateau, TensorBoard)
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import numpy as np
import datetime
from typing import Dict, Tuple
from config import *


class CIFAR10Classifier:
    """
    A ResNet50-based classifier for CIFAR-10 dataset
    
    Attributes:
        model (Sequential): The Keras model
        history (Dict): Training history
        callbacks (list): List of training callbacks
    """
    
    def __init__(self):
        """Initialize the classifier with default parameters"""
        self.model = None
        self.history = None
        self.callbacks = []
        
    def load_data(self):
        """
        Load and preprocess CIFAR-10 data
        
        Returns:
            tuple: ((train_images, train_labels), (test_images, test_labels))
        """
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        # Normalize pixel values
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        
        return (train_images, train_labels), (test_images, test_labels)
    
    def build_model(self):
        """Build the ResNet50-based classification model"""
        # Initialize base model
        base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
        base_model.trainable = True
        
        # Create sequential model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ], name=MODEL_NAME)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Configure training callbacks"""
        # TensorBoard callback
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # Log weight histograms
            profile_batch=0,   # Disable profiling (set to batch numbers to profile)
            update_freq='epoch' # Log metrics at the end of each epoch
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Learning rate reduction
        lr_reduction = ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        )
        
        self.callbacks = [early_stopping, lr_reduction, tensorboard_callback]
        return self.callbacks
    
    def get_data_augmenter(self):
        """
        Create data augmentation generator
        
        Returns:
            ImageDataGenerator: Configured data augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def train(self, train_data, 
              test_data, 
              epochs = 50, 
              batch_size = 32):
        """
        Train the model with data augmentation
        
        Args:
            train_data: Tuple of (images, labels)
            test_data: Tuple of (images, labels)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            dict: Training history
        """
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        
        # Setup data augmentation
        datagen = self.get_data_augmenter()
        datagen.fit(train_images)
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(train_images, train_labels, batch_size=batch_size),
            validation_data=(test_images, test_labels),
            epochs=epochs,
            callbacks=self.callbacks
        )
        
        return self.history.history
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray):
        """
        Evaluate model on test data
        
        Args:
            test_images: Test images
            test_labels: Test labels
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        return self.model.evaluate(test_images, test_labels, verbose=0)
    
    def save_model(self):
        """Save the trained model to disk"""
        self.model.save(MODEL_PATH)
    
    def save_training_history(self):
        """Save training history to file"""
        if self.history is not None:
            np.savez(HISTORY_PATH, **self.history.history)
    
    def save_test_data(self, test_data):
        """Save test data for later evaluation"""
        test_images, test_labels = test_data
        np.savez(TEST_DATA_PATH, images=test_images, labels=test_labels)


def main():
    """Main execution function"""
    # Initialize classifier
    classifier = CIFAR10Classifier()
    
    # Load data
    train_data, test_data = classifier.load_data()
    
    # Build model
    model = classifier.build_model()
    model.summary()
    
    # Setup callbacks
    classifier.setup_callbacks()
    
    # Train model
    history = classifier.train(train_data, test_data, epochs=50)
    
    # Evaluate model
    test_loss, test_acc = classifier.evaluate(*test_data)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save artifacts
    classifier.save_model()
    classifier.save_training_history()
    classifier.save_test_data(test_data)


if __name__ == "__main__":
    main()
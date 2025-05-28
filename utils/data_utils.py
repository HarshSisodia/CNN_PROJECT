"""
Data utility functions for the CNN project.
This module contains functions for loading, preprocessing, and augmenting image data.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

def load_cifar10_data(subset_size=None):
    """
    Load the CIFAR-10 dataset with robust error handling.
    
    Args:
        subset_size (tuple, optional): Tuple of (train_size, test_size) to use a subset of data.
                                      Default is None which uses the full dataset.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test) - training and testing data
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('../data/cifar-10', exist_ok=True)
        
        # Try to load the dataset from keras
        print("Loading CIFAR-10 dataset...")
        (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
        print("Successfully loaded CIFAR-10 dataset")
        
        # Use subset if specified
        if subset_size is not None:
            train_size, test_size = subset_size
            x_train = x_train_full[:train_size]
            y_train = y_train_full[:train_size]
            x_test = x_test_full[:test_size]
            y_test = y_test_full[:test_size]
        else:
            x_train, y_train = x_train_full, y_train_full
            x_test, y_test = x_test_full, y_test_full
            
        # Print dataset shapes for verification
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data instead...")
        
        # Create synthetic data for demonstration purposes
        # This allows the notebook to run even if dataset download fails
        np.random.seed(42)
        
        if subset_size is not None:
            train_size, test_size = subset_size
        else:
            train_size, test_size = 50000, 10000
            
        x_train = np.random.randint(0, 256, size=(train_size, 32, 32, 3), dtype=np.uint8)
        y_train = np.random.randint(0, 10, size=(train_size, 1), dtype=np.uint8)
        x_test = np.random.randint(0, 256, size=(test_size, 32, 32, 3), dtype=np.uint8)
        y_test = np.random.randint(0, 10, size=(test_size, 1), dtype=np.uint8)
        
        print("Created synthetic CIFAR-10 data for demonstration")
        
        # Print dataset shapes for verification
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, y_train, x_test, y_test, resize_dim=(224, 224), one_hot=True):
    """
    Preprocess the data for the model.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        resize_dim (tuple): Dimensions to resize images to, default (224, 224) for ResNet50
        one_hot (bool): Whether to convert labels to one-hot encoding
        
    Returns:
        tuple: Preprocessed (x_train, y_train), (x_test, y_test)
    """
    # Resize images if needed
    if resize_dim != (32, 32):
        print(f"Resizing images to {resize_dim}...")
        x_train_resized = np.zeros((x_train.shape[0], resize_dim[0], resize_dim[1], 3))
        x_test_resized = np.zeros((x_test.shape[0], resize_dim[0], resize_dim[1], 3))
        
        for i in range(x_train.shape[0]):
            x_train_resized[i] = tf.image.resize(x_train[i], resize_dim).numpy()
        
        for i in range(x_test.shape[0]):
            x_test_resized[i] = tf.image.resize(x_test[i], resize_dim).numpy()
        
        x_train, x_test = x_train_resized, x_test_resized
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding if requested
    if one_hot:
        num_classes = 10  # CIFAR-10 has 10 classes
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train), (x_test, y_test)

def create_data_augmentation():
    """
    Create a data augmentation pipeline for training.
    
    Returns:
        tf.keras.Sequential: Data augmentation model
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    return data_augmentation

def visualize_samples(x_data, y_data, class_names=None, num_samples=25):
    """
    Visualize sample images from the dataset.
    
    Args:
        x_data: Image data
        y_data: Labels
        class_names (list): List of class names
        num_samples (int): Number of samples to visualize
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    plt.figure(figsize=(10, 10))
    for i in range(min(num_samples, len(x_data))):
        plt.subplot(grid_size, grid_size, i+1)
        
        # Handle different label formats (one-hot or integer)
        if len(y_data.shape) > 1 and y_data.shape[1] > 1:  # one-hot
            label = np.argmax(y_data[i])
        else:  # integer
            label = y_data[i][0] if len(y_data[i].shape) > 0 else y_data[i]
            
        plt.imshow(x_data[i])
        plt.title(class_names[label])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

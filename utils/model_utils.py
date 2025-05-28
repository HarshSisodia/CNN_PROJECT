"""
Model utility functions for the CNN project.
This module contains functions for building, training, and evaluating CNN models with transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=10, freeze_layers=True):
    """
    Create a transfer learning model based on ResNet50.
    
    Args:
        input_shape (tuple): Input shape for the model
        num_classes (int): Number of output classes
        freeze_layers (bool): Whether to freeze the base model layers
        
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Load the pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model layers if requested
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Create a new model with custom classification head
    model = models.Sequential([
        # Base ResNet50 model
        base_model,
        
        # Add global average pooling
        layers.GlobalAveragePooling2D(),
        
        # Add dropout for regularization
        layers.Dropout(0.3),
        
        # Add a dense layer with L2 regularization
        layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        
        # Add batch normalization
        layers.BatchNormalization(),
        
        # Add another dropout layer
        layers.Dropout(0.5),
        
        # Output layer with softmax activation
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks(checkpoint_path, patience=5):
    """
    Create callbacks for model training.
    
    Args:
        checkpoint_path (str): Path to save model checkpoints
        patience (int): Number of epochs with no improvement after which training will be stopped
        
    Returns:
        list: List of callbacks
    """
    # Create directory for checkpoints if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        
        # Model checkpoint to save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Reduce learning rate when a metric has stopped improving
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    return callbacks

def train_model(model, x_train, y_train, batch_size=32, epochs=30, validation_split=0.2, callbacks=None):
    """
    Train the model with the given data.
    
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs to train
        validation_split (float): Fraction of training data to use for validation
        callbacks (list): List of callbacks for training
        
    Returns:
        tuple: (trained_model, history)
    """
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        x_test: Test data
        y_test: Test labels
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_loss, test_accuracy

def plot_training_history(history):
    """
    Plot the training and validation metrics.
    
    Args:
        history: Training history object
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.show()

def predict_and_visualize(model, x_data, y_data, class_names=None, num_samples=10):
    """
    Make predictions and visualize results.
    
    Args:
        model: Trained model
        x_data: Input data
        y_data: True labels
        class_names (list): List of class names
        num_samples (int): Number of samples to visualize
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Make predictions
    predictions = model.predict(x_data[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:  # one-hot
        true_classes = np.argmax(y_data[:num_samples], axis=1)
    else:  # integer
        true_classes = y_data[:num_samples].flatten()
    
    # Plot the results
    plt.figure(figsize=(15, 2 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(x_data[i])
        plt.title(f"True: {class_names[true_classes[i]]}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.bar(range(len(class_names)), predictions[i])
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.title(f"Predicted: {class_names[predicted_classes[i]]}")
    
    plt.tight_layout()
    plt.show()

def unfreeze_layers(model, num_layers_to_unfreeze=10):
    """
    Unfreeze the last few layers of the base model for fine-tuning.
    
    Args:
        model: Model to modify
        num_layers_to_unfreeze (int): Number of layers to unfreeze from the end
        
    Returns:
        tf.keras.Model: Modified model
    """
    # Get the base model (first layer in the Sequential model)
    base_model = model.layers[0]
    
    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze the last few layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

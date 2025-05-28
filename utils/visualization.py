"""
Visualization utility functions for the CNN project.
This module contains functions for visualizing model architecture, feature maps, and results.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, x_test, y_test, class_names=None):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        model: Trained model
        x_test: Test data
        y_test: Test labels
        class_names (list): List of class names
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true classes
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:  # one-hot
        y_true_classes = np.argmax(y_test, axis=1)
    else:  # integer
        y_true_classes = y_test.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_feature_maps(model, image, layer_name=None):
    """
    Visualize feature maps from a specific layer for a given image.
    
    Args:
        model: Trained model
        image: Input image (should be preprocessed)
        layer_name (str): Name of the layer to visualize. If None, uses the last convolutional layer.
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # If layer_name is not provided, find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers[0].layers):  # Assuming base model is first layer in Sequential
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Create a model that outputs the feature maps
    base_model = model.layers[0]  # Assuming base model is first layer in Sequential
    layer_outputs = [layer.output for layer in base_model.layers if layer.name == layer_name]
    activation_model = Model(inputs=base_model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(image)
    
    # Plot feature maps
    if len(activations) > 0:
        activation = activations[0]
        num_features = min(16, activation.shape[-1])  # Display up to 16 features
        
        plt.figure(figsize=(12, 8))
        for i in range(num_features):
            plt.subplot(4, 4, i+1)
            plt.imshow(activation[0, :, :, i], cmap='viridis')
            plt.title(f'Feature {i+1}')
            plt.axis('off')
        
        plt.suptitle(f'Feature Maps from Layer: {layer_name}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Layer {layer_name} not found or does not produce feature maps.")

def visualize_model_architecture(model):
    """
    Visualize the model architecture.
    
    Args:
        model: Model to visualize
    """
    # Print model summary
    model.summary()
    
    # Plot model architecture if TensorFlow version supports it
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file='model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB'
        )
        
        # Display the saved image
        plt.figure(figsize=(12, 12))
        plt.imshow(plt.imread('model_architecture.png'))
        plt.axis('off')
        plt.show()
    except:
        print("Model visualization not available. TensorFlow version may not support it.")

def visualize_augmented_images(data_augmentation, image, num_augmentations=5):
    """
    Visualize augmented versions of an image.
    
    Args:
        data_augmentation: Data augmentation model
        image: Input image
        num_augmentations (int): Number of augmented versions to generate
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Create augmented versions
    augmented_images = [image]
    for _ in range(num_augmentations):
        augmented_images.append(data_augmentation(image, training=True))
    
    # Plot original and augmented images
    plt.figure(figsize=(12, 4))
    for i, aug_img in enumerate(augmented_images):
        plt.subplot(1, len(augmented_images), i+1)
        plt.imshow(aug_img[0])
        plt.title('Original' if i == 0 else f'Augmented {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_class_activation_maps(model, image, class_idx, layer_name=None):
    """
    Visualize class activation maps (CAM) to show which parts of the image are important for classification.
    
    Args:
        model: Trained model
        image: Input image (should be preprocessed)
        class_idx (int): Index of the class to visualize
        layer_name (str): Name of the layer to use for CAM. If None, uses the last convolutional layer.
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # If layer_name is not provided, find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers[0].layers):  # Assuming base model is first layer in Sequential
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Create a model that outputs both the predictions and the activations
    base_model = model.layers[0]  # Assuming base model is first layer in Sequential
    grad_model = Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradient of the predicted class with respect to the activations
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match image size
    import cv2
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[2]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    original_img = (image[0] * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Display original image and heatmap
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image[0])
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Class Activation Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Superimposed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

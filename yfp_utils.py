"""
Utility functions for YFP Facial Palsy Interview Detection
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_image_for_openface(image_path, face_detector=None, landmark_detector=None):
    """
    Preprocess image using OpenFace pipeline before interview detection
    
    Args:
        image_path: Path to input image
        face_detector: OpenFace face detector instance
        landmark_detector: OpenFace landmark detector instance
    
    Returns:
        Preprocessed image as PIL Image
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        # If face detector is provided, detect and crop face
        if face_detector is not None:
            try:
                cropped_face, dets = face_detector.get_face(image_path)
                if cropped_face is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(image)
            except Exception as e:
                print(f"Face detection failed for {image_path}: {e}")
        
        # If no face detection or it failed, use original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def create_confusion_matrix(y_true, y_pred, class_names=['No Interview', 'Interview'], save_path=None):
    """
    Create and optionally save confusion matrix visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for the classes
        save_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm


def analyze_predictions_by_confidence(model, dataloader, device):
    """
    Analyze model predictions by confidence scores
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
    
    Returns:
        Dictionary with confidence analysis results
    """
    model.eval()
    confidences = []
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            
            confidences.extend(max_probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate accuracy by confidence thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    accuracy_by_threshold = []
    
    for threshold in thresholds:
        mask = confidences >= threshold
        if mask.sum() > 0:
            acc = (predictions[mask] == true_labels[mask]).mean()
            accuracy_by_threshold.append((threshold, acc, mask.sum()))
    
    return {
        'confidences': confidences,
        'predictions': predictions,
        'true_labels': true_labels,
        'accuracy_by_threshold': accuracy_by_threshold
    }


def export_model_for_deployment(model, save_path, input_size=(1, 3, 224, 224)):
    """
    Export model for deployment (ONNX format)
    
    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        input_size: Input tensor size (batch, channels, height, width)
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX format: {save_path}")
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
        
    except Exception as e:
        print(f"Error exporting model: {e}")


def create_data_splits_stratified(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create stratified data splits ensuring balanced class distribution
    
    Args:
        dataset: Dataset instance
        train_ratio: Training data ratio
        val_ratio: Validation data ratio  
        test_ratio: Test data ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    from sklearn.model_selection import train_test_split
    
    # Get indices and labels
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]
    
    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_seed, 
        stratify=labels
    )
    
    # Second split: train vs val
    train_val_labels = [dataset.samples[i][1] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=val_ratio/(train_ratio+val_ratio), 
        random_state=random_seed, 
        stratify=train_val_labels
    )
    
    # Print class distribution
    def print_split_stats(indices, split_name):
        split_labels = [dataset.samples[i][1] for i in indices]
        class_counts = np.bincount(split_labels)
        print(f"{split_name}: {len(indices)} samples")
        print(f"  Class 0 (No Interview): {class_counts[0]} ({class_counts[0]/len(indices)*100:.1f}%)")
        print(f"  Class 1 (Interview): {class_counts[1]} ({class_counts[1]/len(indices)*100:.1f}%)")
    
    print_split_stats(train_indices, "Training")
    print_split_stats(val_indices, "Validation") 
    print_split_stats(test_indices, "Test")
    
    return train_indices, val_indices, test_indices


def benchmark_model_performance(model, dataloader, device, num_runs=10):
    """
    Benchmark model inference performance
    
    Args:
        model: Model to benchmark
        dataloader: DataLoader for benchmarking
        device: Device to run on
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with performance metrics
    """
    model.eval()
    
    # Warm up
    for i, (images, _) in enumerate(dataloader):
        if i >= 3:  # Warm up with first 3 batches
            break
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)
    
    # Benchmark
    import time
    total_time = 0
    total_samples = 0
    
    for run in range(num_runs):
        run_time = 0
        run_samples = 0
        
        for images, _ in dataloader:
            start_time = time.time()
            images = images.to(device)
            
            with torch.no_grad():
                _ = model(images)
            
            run_time += time.time() - start_time
            run_samples += images.size(0)
        
        total_time += run_time
        total_samples += run_samples
    
    avg_time_per_sample = total_time / total_samples
    samples_per_second = total_samples / total_time
    
    return {
        'avg_time_per_sample_ms': avg_time_per_sample * 1000,
        'samples_per_second': samples_per_second,
        'total_time_seconds': total_time,
        'total_samples': total_samples
    }


def visualize_predictions(model, dataloader, device, num_samples=8, save_path=None):
    """
    Visualize model predictions with images
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    
    # Get a batch of data
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.numpy()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {labels[i]}, Pred: {preds[i]}')
        axes[i].axis('off')
        
        # Color code based on correctness
        if preds[i] == labels[i]:
            axes[i].set_title(f'True: {labels[i]}, Pred: {preds[i]} ✓', color='green')
        else:
            axes[i].set_title(f'True: {labels[i]}, Pred: {preds[i]} ✗', color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
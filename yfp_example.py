"""
Example script demonstrating YFP interview detection usage
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yfp_evaluation import YFPDataset, InterviewDetector, evaluate_interview_detection, get_transforms
from yfp_train import train_interview_detection, create_data_splits_stratified
from yfp_utils import create_confusion_matrix, analyze_predictions_by_confidence
from yfp_demo import YFPInterviewDetector


def create_sample_dataset():
    """
    Create a small sample dataset for demonstration
    """
    print("Creating sample dataset for demonstration...")
    
    # This is a placeholder - in real usage, you would have actual YFP images
    # For demonstration, we'll show the structure
    
    sample_data = {
        'filename': [
            'interview/patient001_frame001.jpg',
            'interview/patient001_frame002.jpg', 
            'normal/control001_frame001.jpg',
            'normal/control001_frame002.jpg',
            'interview/patient002_frame001.jpg',
            'normal/control002_frame001.jpg'
        ],
        'label': [1, 1, 0, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_yfp_labels.csv', index=False)
    print("Sample CSV created: sample_yfp_labels.csv")
    
    return df


def example_training_workflow():
    """
    Example of complete training workflow
    """
    print("\n" + "="*60)
    print("EXAMPLE: TRAINING WORKFLOW")
    print("="*60)
    
    # Create sample data structure
    data_dir = "/path/to/yfp/data"  # Replace with actual path
    csv_file = "sample_yfp_labels.csv"
    
    # Check if we have actual data
    if not os.path.exists(data_dir):
        print(f"Note: Data directory {data_dir} does not exist.")
        print("This is a demonstration. Replace with your actual YFP dataset path.")
        return
    
    # Initialize transforms
    transform = get_transforms(224, is_training=True)
    
    # Create dataset
    dataset = YFPDataset(
        data_dir=data_dir,
        transform=transform,
        csv_file=csv_file,
        limit=100  # Use small limit for demo
    )
    
    if len(dataset) == 0:
        print("No valid samples found in dataset.")
        return
    
    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits_stratified(
        dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Training: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InterviewDetector(num_classes=2, pretrained=True)
    model = model.to(device)
    
    print(f"Model initialized on {device}")
    print("Training workflow setup complete!")
    print("To actually train, run: python yfp_train.py --data_dir /path/to/data --csv_file labels.csv")


def example_evaluation():
    """
    Example of model evaluation
    """
    print("\n" + "="*60)
    print("EXAMPLE: MODEL EVALUATION")
    print("="*60)
    
    # This would be the path to your trained model
    model_path = "./yfp_checkpoints/best_model.pth"
    data_dir = "/path/to/yfp/data"
    
    if not os.path.exists(model_path):
        print(f"Note: Model {model_path} not found.")
        print("This is a demonstration. Train a model first or provide correct path.")
        return
    
    # Initialize detector
    detector = YFPInterviewDetector(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example prediction
    image_path = "/path/to/test/image.jpg"
    if os.path.exists(image_path):
        result = detector.predict_image(image_path)
        print(f"Prediction for {image_path}:")
        print(f"  Class: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities: {result['probabilities']}")
    else:
        print(f"Test image {image_path} not found.")


def example_metrics_calculation():
    """
    Example of metrics calculation
    """
    print("\n" + "="*60)
    print("EXAMPLE: METRICS CALCULATION")
    print("="*60)
    
    # Simulated predictions and ground truth
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    
    print("Simulated Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # Create confusion matrix
    create_confusion_matrix(y_true, y_pred, save_path='confusion_matrix_demo.png')


def example_integration_with_openface():
    """
    Example of integration with OpenFace components
    """
    print("\n" + "="*60)
    print("EXAMPLE: INTEGRATION WITH OPENFACE")
    print("="*60)
    
    # This is a demonstration of how to integrate with OpenFace
    print("To integrate with OpenFace face detection:")
    print("""
    from openface.face_detection import FaceDetector
    from yfp_demo import YFPInterviewDetector
    
    # Initialize OpenFace components
    face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth')
    
    # Initialize YFP detector
    yfp_detector = YFPInterviewDetector(model_path='./yfp_checkpoints/best_model.pth')
    
    # Use with OpenFace preprocessing
    result = yfp_detector.predict_image('image.jpg', face_detector=face_detector)
    print(f"Result: {result}")
    """)


def main():
    """
    Main demonstration function
    """
    print("YFP Facial Palsy Interview Detection - Examples")
    print("=" * 60)
    
    # Create sample dataset structure
    create_sample_dataset()
    
    # Show example workflows
    example_training_workflow()
    example_evaluation()
    example_metrics_calculation()
    example_integration_with_openface()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your YFP dataset")
    print("2. Run: python yfp_pipeline.py --mode train --data_dir /path/to/data")
    print("3. Evaluate: python yfp_pipeline.py --mode eval --data_dir /path/to/data --model_path ./yfp_checkpoints/best_model.pth")
    print("4. Try demo: python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input /path/to/test/image.jpg")


if __name__ == "__main__":
    main()
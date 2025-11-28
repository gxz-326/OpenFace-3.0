import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd
import cv2


class YFPDataset(Dataset):
    """
    Dataset class for YFP Facial Palsy dataset for interview detection.
    Assumes binary classification: 0 = no interview, 1 = interview detected
    """
    def __init__(self, data_dir, transform=None, csv_file=None, limit=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            csv_file (string): Path to CSV file with image filenames and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Limit the number of samples to load for debugging.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        if csv_file and os.path.exists(csv_file):
            # Load from CSV file
            df = pd.read_csv(csv_file)
            if limit is not None:
                df = df.head(limit)
            
            for _, row in df.iterrows():
                img_path = os.path.join(data_dir, row['filename'])
                label = row['label']  # 0 or 1 for binary classification
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
        else:
            # If no CSV provided, assume folder structure: data_dir/class_name/image_files
            # or try to auto-detect pattern
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            for ext in image_extensions:
                image_files = []
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        if file.lower().endswith(ext):
                            img_path = os.path.join(root, file)
                            # Try to infer label from folder name or filename
                            if 'interview' in root.lower() or 'interview' in file.lower():
                                label = 1
                            elif 'normal' in root.lower() or 'control' in root.lower() or 'healthy' in file.lower():
                                label = 0
                            else:
                                # Default to 1 if uncertain, should be overridden by proper labeling
                                label = 1
                            image_files.append((img_path, label))
                
                if limit is not None:
                    image_files = image_files[:limit]
                self.samples.extend(image_files)
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class InterviewDetector(nn.Module):
    """
    Modified model for interview detection using facial palsy features
    """
    def __init__(self, base_model_name='tf_efficientnet_b0_ns', num_classes=2, pretrained=False):
        super(InterviewDetector, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained)
        self.base_model.classifier = nn.Identity()
        
        feature_dim = self.base_model.num_features
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        features = self.dropout(features)
        output = self.classifier(features)
        return output


def evaluate_interview_detection(model, dataloader, device, class_names=['No Interview', 'Interview']):
    """
    Evaluate the interview detection model and compute comprehensive metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    
    # Generate detailed classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Print results
    print("\n" + "="*60)
    print("INTERVIEW DETECTION EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*60)
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report
    }
    
    return metrics, all_predictions, all_labels


def get_transforms(image_size=224, is_training=False):
    """
    Get image transforms for preprocessing
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def save_results(metrics, predictions, labels, output_dir="./results"):
    """
    Save evaluation results to files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics summary
    metrics_df = pd.DataFrame([{
        'Metric': 'Accuracy',
        'Value': metrics['accuracy']
    }, {
        'Metric': 'F1 Score', 
        'Value': metrics['f1_score']
    }, {
        'Metric': 'Precision',
        'Value': metrics['precision']
    }, {
        'Metric': 'Recall',
        'Value': metrics['recall']
    }])
    
    metrics_df.to_csv(os.path.join(output_dir, 'interview_detection_metrics.csv'), index=False)
    
    # Save predictions
    results_df = pd.DataFrame({
        'True_Label': labels,
        'Predicted_Label': predictions
    })
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Save detailed classification report
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    print(f"\nResults saved to {output_dir}")


def main():
    """
    Main function for evaluating interview detection on YFP dataset
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Interview Detection on YFP Facial Palsy Dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing YFP dataset images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file with image filenames and labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for debugging')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Initialize model
    model = InterviewDetector(num_classes=2, pretrained=False)
    
    # Load model weights
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=args.device)
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using randomly initialized weights.")
    
    model = model.to(args.device)
    
    # Get transforms
    transform = get_transforms(args.image_size, is_training=False)
    
    # Create dataset and dataloader
    dataset = YFPDataset(
        data_dir=args.data_dir,
        transform=transform,
        csv_file=args.csv_file,
        limit=args.limit
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Evaluate model
    metrics, predictions, labels = evaluate_interview_detection(model, dataloader, args.device)
    
    # Save results
    save_results(metrics, predictions, labels, args.output_dir)


if __name__ == "__main__":
    main()
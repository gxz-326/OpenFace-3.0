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
import argparse
from yfp_evaluation import YFPDataset, InterviewDetector, evaluate_interview_detection, get_transforms, save_results


def train_interview_detection(model, train_loader, val_loader, device, num_epochs=50, learning_rate=1e-4, save_dir="./checkpoints"):
    """
    Train the interview detection model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_val_f1 = 0.0
    best_metrics = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({'Loss': f'{train_loss/train_total:.4f}', 'Acc': f'{train_acc:.2f}%'})
        
        # Validation phase
        val_metrics, _, _ = evaluate_interview_detection(model, val_loader, device)
        val_f1 = val_metrics['f1_score']
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"New best model saved with F1: {val_f1:.4f}")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train Acc: {100*train_correct/train_total:.2f}%")
        print(f"Val F1: {val_f1:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print("-" * 50)
    
    return best_metrics


def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create train/val/test splits from dataset
    """
    from sklearn.model_selection import train_test_split
    
    # Get indices and labels
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]
    
    # First split: train + val vs test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_seed, stratify=labels
    )
    
    # Second split: train vs val
    train_val_labels = [dataset.samples[i][1] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio/(train_ratio+val_ratio), 
        random_state=random_seed, stratify=train_val_labels
    )
    
    return train_indices, val_indices, test_indices


class SubsetDataset(Dataset):
    """
    Dataset wrapper for creating subsets
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def main():
    """
    Main function for training interview detection on YFP dataset
    """
    parser = argparse.ArgumentParser(description='Train Interview Detection on YFP Facial Palsy Dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing YFP dataset images')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file with image filenames and labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for debugging')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model weights')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test data ratio')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Create full dataset
    full_dataset = YFPDataset(
        data_dir=args.data_dir,
        transform=get_transforms(args.image_size, is_training=True),
        csv_file=args.csv_file,
        limit=args.limit
    )
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits(
        full_dataset, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create datasets with appropriate transforms
    train_dataset = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(
        YFPDataset(
            data_dir=args.data_dir,
            transform=get_transforms(args.image_size, is_training=False),
            csv_file=args.csv_file,
            limit=args.limit
        ),
        val_indices
    )
    test_dataset = SubsetDataset(
        YFPDataset(
            data_dir=args.data_dir,
            transform=get_transforms(args.image_size, is_training=False),
            csv_file=args.csv_file,
            limit=args.limit
        ),
        test_indices
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = InterviewDetector(num_classes=2, pretrained=True)
    model = model.to(args.device)
    
    # Load pretrained weights if provided
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        checkpoint = torch.load(args.pretrained_model, map_location=args.device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {args.pretrained_model}")
    
    # Train model
    print("\nStarting training...")
    best_metrics = train_interview_detection(
        model, train_loader, val_loader, args.device, 
        args.num_epochs, args.learning_rate, args.save_dir
    )
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        print(f"Loaded best model from {best_model_path}")
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics, test_predictions, test_labels = evaluate_interview_detection(model, test_loader, args.device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save test results
    save_results(test_metrics, test_predictions, test_labels, os.path.join(args.output_dir, 'test'))
    
    # Save training summary
    summary_df = pd.DataFrame([{
        'Best_Val_F1': best_metrics['f1_score'] if best_metrics else 0,
        'Test_Accuracy': test_metrics['accuracy'],
        'Test_F1': test_metrics['f1_score'],
        'Test_Precision': test_metrics['precision'],
        'Test_Recall': test_metrics['recall']
    }])
    summary_df.to_csv(os.path.join(args.output_dir, 'training_summary.csv'), index=False)
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")
    print(f"Best validation F1: {best_metrics['f1_score']:.4f}" if best_metrics else "N/A")
    print(f"Test F1: {test_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
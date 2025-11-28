#!/usr/bin/env python3
"""
YFP Facial Palsy Interview Detection Pipeline

This script provides a complete pipeline for training and evaluating
interview detection models on the YFP facial palsy dataset.

Usage Examples:
    # Train a new model
    python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data --csv_file /path/to/labels.csv

    # Evaluate an existing model
    python yfp_pipeline.py --mode eval --data_dir /path/to/yfp/data --model_path /path/to/model.pth

    # Train and evaluate in one go
    python yfp_pipeline.py --mode full --data_dir /path/to/yfp/data --csv_file /path/to/labels.csv
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def create_sample_csv(data_dir, output_csv):
    """Create a sample CSV file for labeling"""
    import pandas as pd
    from PIL import Image
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(ext):
                    rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                    image_files.append(rel_path)
    
    # Create DataFrame with default labels
    df = pd.DataFrame({
        'filename': image_files,
        'label': 1  # Default to interview detected
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Sample CSV created at: {output_csv}")
    print(f"Please edit this file to set correct labels (0=no interview, 1=interview detected)")
    

def main():
    parser = argparse.ArgumentParser(
        description='YFP Facial Palsy Interview Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'full', 'create_csv'], 
                       required=True, help='Pipeline mode to run')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, help='Directory containing YFP dataset images')
    parser.add_argument('--csv_file', type=str, help='CSV file with image filenames and labels')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, help='Path to trained model weights (for eval mode)')
    parser.add_argument('--pretrained_model', type=str, help='Path to pretrained model weights (for training)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # General arguments
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if 'cuda' in str(__import__('torch').cuda.is_available()) else 'cpu', 
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./yfp_results', help='Directory to save results')
    parser.add_argument('--checkpoints_dir', type=str, default='./yfp_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for debugging')
    
    # Data split arguments
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test data ratio')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'create_csv':
        if not args.data_dir:
            print("Error: --data_dir is required for create_csv mode")
            sys.exit(1)
        create_sample_csv(args.data_dir, 'yfp_labels.csv')
        return
    
    if args.mode in ['train', 'full']:
        if not args.data_dir:
            print("Error: --data_dir is required for training")
            sys.exit(1)
    
    if args.mode == 'eval':
        if not args.data_dir or not args.model_path:
            print("Error: --data_dir and --model_path are required for evaluation")
            sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Build common arguments
    common_args = f"--data_dir {args.data_dir} --batch_size {args.batch_size} --image_size {args.image_size} --device {args.device} --output_dir {args.output_dir}"
    
    if args.csv_file:
        common_args += f" --csv_file {args.csv_file}"
    if args.limit:
        common_args += f" --limit {args.limit}"
    
    success = True
    
    if args.mode == 'train':
        # Training mode
        train_args = f"{common_args} --num_epochs {args.num_epochs} --learning_rate {args.learning_rate} --save_dir {args.checkpoints_dir}"
        train_args += f" --train_ratio {args.train_ratio} --val_ratio {args.val_ratio} --test_ratio {args.test_ratio}"
        
        if args.pretrained_model:
            train_args += f" --pretrained_model {args.pretrained_model}"
        
        success = run_command(f"python yfp_train.py {train_args}", "Training Interview Detection Model")
    
    elif args.mode == 'eval':
        # Evaluation mode
        eval_args = f"{common_args} --model_path {args.model_path}"
        success = run_command(f"python yfp_evaluation.py {eval_args}", "Evaluating Interview Detection Model")
    
    elif args.mode == 'full':
        # Full pipeline: train then evaluate
        # Training
        train_args = f"{common_args} --num_epochs {args.num_epochs} --learning_rate {args.learning_rate} --save_dir {args.checkpoints_dir}"
        train_args += f" --train_ratio {args.train_ratio} --val_ratio {args.val_ratio} --test_ratio {args.test_ratio}"
        
        if args.pretrained_model:
            train_args += f" --pretrained_model {args.pretrained_model}"
        
        success = run_command(f"python yfp_train.py {train_args}", "Training Interview Detection Model")
        
        if success:
            # Evaluation with best model
            best_model_path = os.path.join(args.checkpoints_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                eval_args = f"{common_args} --model_path {best_model_path}"
                success = run_command(f"python yfp_evaluation.py {eval_args}", "Evaluating Best Model")
            else:
                print(f"Warning: Best model not found at {best_model_path}, skipping evaluation")
    
    if success:
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {args.output_dir}")
        print('='*60)
    else:
        print(f"\n{'='*60}")
        print("PIPELINE FAILED!")
        print("Please check the error messages above.")
        print('='*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
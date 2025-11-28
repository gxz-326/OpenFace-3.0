#!/usr/bin/env python3
"""
Demo script for YFP Facial Palsy Interview Detection

This script demonstrates how to use the interview detection model
with the existing OpenFace pipeline for real-time inference.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yfp_evaluation import InterviewDetector, get_transforms
from yfp_utils import preprocess_image_for_openface, visualize_predictions


class YFPInterviewDetector:
    """
    Wrapper class for YFP interview detection with OpenFace integration
    """
    
    def __init__(self, model_path, device='cpu', confidence_threshold=0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
            confidence_threshold: Threshold for positive prediction
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = InterviewDetector(num_classes=2, pretrained=False)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.to(device)
        self.model.eval()
        
        # Transforms for preprocessing
        self.transform = get_transforms(224, is_training=False)
        
        # Class names
        self.class_names = ['No Interview', 'Interview']
    
    def predict_image(self, image_path, face_detector=None, landmark_detector=None):
        """
        Predict interview detection for a single image
        
        Args:
            image_path: Path to input image
            face_detector: Optional OpenFace face detector
            landmark_detector: Optional OpenFace landmark detector
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image = preprocess_image_for_openface(
            image_path, face_detector, landmark_detector
        )
        
        if image is None:
            return {
                'error': 'Failed to load or preprocess image',
                'prediction': None,
                'confidence': None
            }
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)
            
            pred_class = pred_class.item()
            confidence = confidence.item()
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                pred_class = 0  # Default to no interview if confidence is low
            
            result = {
                'prediction': pred_class,
                'class_name': self.class_names[pred_class],
                'confidence': confidence,
                'probabilities': {
                    'no_interview': probs[0][0].item(),
                    'interview': probs[0][1].item()
                }
            }
            
            return result
    
    def predict_batch(self, image_paths, face_detector=None, landmark_detector=None):
        """
        Predict interview detection for multiple images
        
        Args:
            image_paths: List of image paths
            face_detector: Optional OpenFace face detector
            landmark_detector: Optional OpenFace landmark detector
        
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_image(image_path, face_detector, landmark_detector)
            result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def predict_video(self, video_path, output_path=None, frame_interval=1, face_detector=None):
        """
        Predict interview detection for video frames
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            frame_interval: Process every nth frame
            face_detector: Optional OpenFace face detector
        
        Returns:
            List of prediction results for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        results = []
        frame_count = 0
        processed_count = 0
        
        # Video writer for output
        writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % frame_interval != 0:
                continue
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, 1)
                
                pred_class = pred_class.item()
                confidence = confidence.item()
            
            # Store result
            result = {
                'frame_number': frame_count,
                'prediction': pred_class,
                'class_name': self.class_names[pred_class],
                'confidence': confidence,
                'probabilities': {
                    'no_interview': probs[0][0].item(),
                    'interview': probs[0][1].item()
                }
            }
            results.append(result)
            
            # Draw result on frame
            color = (0, 255, 0) if pred_class == 1 else (0, 0, 255)
            label = f"{self.class_names[pred_class]}: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2)
            
            # Write frame if output path specified
            if writer:
                writer.write(frame)
            
            processed_count += 1
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"Processed {processed_count} frames from {frame_count} total frames")
        return results


def main():
    parser = argparse.ArgumentParser(description='YFP Interview Detection Demo')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model weights')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (image, directory, or video)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results or video')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for positive prediction')
    parser.add_argument('--mode', type=str, choices=['image', 'directory', 'video'], 
                       default='auto', help='Input mode')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Process every nth frame for video')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YFPInterviewDetector(
        args.model_path, 
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Determine input mode
    if args.mode == 'auto':
        if os.path.isfile(args.input):
            if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                args.mode = 'video'
            else:
                args.mode = 'image'
        elif os.path.isdir(args.input):
            args.mode = 'directory'
        else:
            raise ValueError(f"Invalid input path: {args.input}")
    
    print(f"Processing {args.mode}: {args.input}")
    
    if args.mode == 'image':
        # Single image prediction
        result = detector.predict_image(args.input)
        
        print("\nPrediction Results:")
        print(f"Image: {args.input}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities:")
            print(f"  No Interview: {result['probabilities']['no_interview']:.4f}")
            print(f"  Interview: {result['probabilities']['interview']:.4f}")
    
    elif args.mode == 'directory':
        # Directory batch prediction
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.input).glob(f'*{ext}'))
            image_paths.extend(Path(args.input).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in directory: {args.input}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Batch prediction
        results = detector.predict_batch(image_paths)
        
        # Save results if output path specified
        if args.output:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
        
        # Print summary
        interview_count = sum(1 for r in results if r['prediction'] == 1)
        print(f"\nSummary:")
        print(f"Total images: {len(results)}")
        print(f"Interview detected: {interview_count}")
        print(f"No interview: {len(results) - interview_count}")
        print(f"Interview rate: {interview_count/len(results)*100:.1f}%")
    
    elif args.mode == 'video':
        # Video prediction
        results = detector.predict_video(
            args.input, 
            output_path=args.output,
            frame_interval=args.frame_interval
        )
        
        # Save results if output path specified
        if args.output and not args.output.endswith(('.mp4', '.avi', '.mov')):
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
        
        # Print summary
        interview_frames = sum(1 for r in results if r['prediction'] == 1)
        print(f"\nSummary:")
        print(f"Total frames processed: {len(results)}")
        print(f"Interview frames: {interview_frames}")
        print(f"No interview frames: {len(results) - interview_frames}")
        print(f"Interview rate: {interview_frames/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
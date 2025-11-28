# YFP Facial Palsy Interview Detection - Quick Start Guide

## 🎯 Overview

This implementation adapts the OpenFace 3.0 model for interview detection using the YFP facial palsy dataset. The system provides comprehensive evaluation metrics including Accuracy, F1 Score, Precision, and Recall.

## 📁 Files Created

1. **Core Modules**:
   - `yfp_evaluation.py` - Dataset class, model architecture, and evaluation functions
   - `yfp_train.py` - Training script with data splitting and model training
   - `yfp_pipeline.py` - Complete pipeline for training and evaluation
   - `yfp_demo.py` - Demo script for inference on images, directories, and videos
   - `yfp_utils.py` - Utility functions for visualization and analysis
   - `yfp_example.py` - Example usage and demonstrations

2. **Documentation**:
   - `YFP_README.md` - Basic usage guide
   - `YFP_DETAILED_README.md` - Comprehensive documentation

## 🚀 Quick Start

### 1. Prepare Your Data

Structure your YFP dataset:

```
yfp_data/
├── interview/          # Images with interview detected
│   ├── patient001.jpg
│   └── ...
└── normal/            # Normal/control images
    ├── control001.jpg
    └── ...
```

Or create a CSV file:

```csv
filename,label
interview/patient001.jpg,1
normal/control001.jpg,0
```

### 2. Train Model

```bash
# Basic training
python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data

# With CSV labels
python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data --csv_file labels.csv

# Custom parameters
python yfp_pipeline.py --mode train \
    --data_dir /path/to/yfp/data \
    --csv_file labels.csv \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python yfp_pipeline.py --mode eval \
    --data_dir /path/to/yfp/data \
    --model_path ./yfp_checkpoints/best_model.pth
```

### 4. Run Inference

```bash
# Single image
python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input image.jpg

# Batch processing
python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input /path/to/images/ --output results.csv

# Video processing
python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input video.mp4 --output processed_video.mp4
```

## 📊 Evaluation Metrics

The system automatically calculates:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value
- **Recall**: Sensitivity (true positive rate)

Results are saved to:
- `yfp_results/interview_detection_metrics.csv`
- `yfp_results/predictions.csv`
- `yfp_results/classification_report.csv`

## 🏗️ Model Architecture

- **Backbone**: EfficientNet-B0 (tf_efficientnet_b0_ns)
- **Input**: 224x224 RGB images
- **Output**: Binary classification (interview vs non-interview)
- **Features**: Dropout regularization, transfer learning support

## 🔧 Advanced Features

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter
- ImageNet normalization

### Confidence Thresholding
Adjust confidence threshold for predictions:

```bash
python yfp_demo.py --model_path model.pth --input image.jpg --confidence_threshold 0.7
```

### OpenFace Integration
Use with OpenFace face detection:

```python
from openface.face_detection import FaceDetector
from yfp_demo import YFPInterviewDetector

face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth')
yfp_detector = YFPInterviewDetector(model_path='./yfp_checkpoints/best_model.pth')

result = yfp_detector.predict_image('image.jpg', face_detector=face_detector)
```

## 📈 Performance Optimization

### GPU Training
```bash
python yfp_pipeline.py --mode train --data_dir /path/to/data --device cuda
```

### Batch Size Tuning
```bash
python yfp_pipeline.py --mode train --data_dir /path/to/data --batch_size 64
```

### Learning Rate Scheduling
The training includes automatic learning rate reduction on plateau.

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size
2. **Poor Performance**: Increase epochs, use pretrained weights
3. **Data Loading**: Check image formats and CSV structure

### Debug Mode
```bash
python yfp_pipeline.py --mode train --data_dir /path/to/data --limit 100
```

## 🎓 Example Usage

See `yfp_example.py` for complete examples:

```bash
python yfp_example.py
```

## 📚 Research Applications

- Medical diagnosis automation
- Rehabilitation monitoring
- Clinical decision support
- Large-scale facial palsy analysis

## 🤝 Next Steps

1. Prepare your YFP dataset
2. Run training pipeline
3. Evaluate performance
4. Deploy using demo script
5. Integrate with OpenFace if needed

For detailed documentation, see `YFP_DETAILED_README.md`.
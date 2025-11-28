# YFP Facial Palsy Interview Detection

A comprehensive toolkit for interview detection using facial palsy analysis, built on top of OpenFace 3.0.

## 🚀 Features

- **Binary Classification**: Detects interview vs non-interview scenarios from facial images
- **Comprehensive Metrics**: Evaluates using Accuracy, F1 Score, Precision, and Recall
- **Flexible Data Loading**: Supports both CSV-based labeling and automatic folder-based detection
- **Complete Pipeline**: Includes training, evaluation, and result saving
- **Transfer Learning**: Uses EfficientNet backbone with optional pretrained weights
- **Real-time Inference**: Demo script for single image, batch, and video processing
- **OpenFace Integration**: Seamlessly works with existing OpenFace face detection and landmark extraction

## 📋 Requirements

- Python 3.7+
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- pandas
- matplotlib
- seaborn
- PIL (Pillow)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
cd OpenFace-3.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The YFP interview detection scripts are included in the main repository.

## 📁 Data Preparation

### Option 1: Folder Structure
Organize your YFP facial palsy dataset as follows:

```
yfp_data/
├── interview/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── normal/
    ├── image3.jpg
    ├── image4.jpg
    └── ...
```

### Option 2: CSV File
Create a CSV file with image filenames and labels:

```csv
filename,label
interview/image1.jpg,1
normal/image3.jpg,0
```

## 🎯 Quick Start

### 1. Create a CSV template (optional)
```bash
python yfp_pipeline.py --mode create_csv --data_dir /path/to/yfp/data
```

### 2. Train a new model
```bash
python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data --csv_file labels.csv
```

### 3. Evaluate the model
```bash
python yfp_pipeline.py --mode eval --data_dir /path/to/yfp/data --model_path ./yfp_checkpoints/best_model.pth
```

### 4. Run full pipeline (train + evaluate)
```bash
python yfp_pipeline.py --mode full --data_dir /path/to/yfp/data --csv_file labels.csv
```

## 🎮 Demo Usage

### Single Image Prediction
```bash
python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input /path/to/image.jpg
```

### Batch Directory Processing
```bash
python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input /path/to/images/ --output results.csv
```

### Video Processing
```bash
python yfp_demo.py --model_path ./yfp_checkpoints/best_model.pth --input /path/to/video.mp4 --output processed_video.mp4
```

## 📊 Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value (TP / (TP + FP))
- **Recall**: Sensitivity or true positive rate (TP / (TP + FN))

## 🏗️ Model Architecture

The interview detection model uses:

- **Backbone**: EfficientNet-B0 (tf_efficientnet_b0_ns)
- **Classifier**: Linear layer with dropout regularization
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (interview vs non-interview)

## 📈 Advanced Usage

### Custom Training Parameters
```bash
python yfp_pipeline.py --mode train \
    --data_dir /path/to/yfp/data \
    --csv_file labels.csv \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 5e-5 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### Using Pretrained Weights
```bash
python yfp_pipeline.py --mode train \
    --data_dir /path/to/yfp/data \
    --pretrained_model /path/to/pretrained/model.pth \
    --csv_file labels.csv
```

### GPU Training
```bash
python yfp_pipeline.py --mode train \
    --data_dir /path/to/yfp/data \
    --device cuda \
    --csv_file labels.csv
```

## 📁 Output Files

### Training Results (`./yfp_results/`)
- `training_summary.csv`: Overall training and test metrics
- `test/interview_detection_metrics.csv`: Test set evaluation metrics
- `test/predictions.csv`: Individual predictions with ground truth
- `test/classification_report.csv`: Detailed classification report

### Model Checkpoints (`./yfp_checkpoints/`)
- `best_model.pth`: Best model based on validation F1 score

## 🔧 Integration with OpenFace

This module is designed to work alongside the existing OpenFace 3.0 toolkit:

```python
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from yfp_demo import YFPInterviewDetector

# Initialize OpenFace components
face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth')
landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl')

# Initialize YFP detector
yfp_detector = YFPInterviewDetector(model_path='./yfp_checkpoints/best_model.pth')

# Use with OpenFace preprocessing
result = yfp_detector.predict_image('image.jpg', face_detector, landmark_detector)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
   ```bash
   python yfp_pipeline.py --mode train --batch_size 16 --device cpu
   ```

2. **Poor performance**: 
   - Increase training epochs
   - Adjust learning rate
   - Use pretrained weights
   - Check data quality and labeling

3. **Data loading errors**: 
   - Verify image formats (JPG, PNG)
   - Check CSV file structure
   - Ensure image paths are correct

### Debug Mode
Use `--limit` parameter to test with a small subset:
```bash
python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data --limit 100
```

## 📚 API Reference

### YFPDataset
```python
dataset = YFPDataset(
    data_dir="/path/to/data",
    transform=transform,
    csv_file="labels.csv",
    limit=1000  # optional limit
)
```

### InterviewDetector
```python
model = InterviewDetector(num_classes=2, pretrained=True)
model = model.to(device)
```

### YFPInterviewDetector (Demo Class)
```python
detector = YFPInterviewDetector(
    model_path="model.pth",
    device="cuda",
    confidence_threshold=0.5
)

result = detector.predict_image("image.jpg")
```

## 📖 Research Applications

This toolkit can be used for:

- **Medical Diagnosis**: Automated detection of facial palsy during interviews
- **Rehabilitation Monitoring**: Track patient progress over time
- **Research Studies**: Large-scale analysis of facial palsy datasets
- **Clinical Decision Support**: Assist healthcare professionals in diagnosis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 Citation

If you use this YFP interview detection module in your research, please cite:

```bibtex
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:

1. Check the [Issues](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/issues) page
2. Create a new issue with detailed description
3. Join our community discussions

---

**Note**: This module is specifically designed for the YFP (Yale Facial Palsy) dataset but can be adapted for other facial palsy or interview detection datasets with minimal modifications.
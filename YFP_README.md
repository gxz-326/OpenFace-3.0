# YFP Facial Palsy Interview Detection

This module provides tools for training and evaluating interview detection models on the YFP facial palsy dataset.

## Features

- **Binary Classification**: Detects interview vs non-interview scenarios
- **Comprehensive Metrics**: Evaluates using Accuracy, F1 Score, Precision, and Recall
- **Flexible Data Loading**: Supports both CSV-based labeling and automatic folder-based detection
- **Complete Pipeline**: Includes training, evaluation, and result saving
- **Transfer Learning**: Uses EfficientNet backbone with optional pretrained weights

## Quick Start

### 1. Prepare Your Data

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

Or create a CSV file with image filenames and labels:

```csv
filename,label
interview/image1.jpg,1
normal/image3.jpg,0
```

### 2. Create a CSV template (optional)

```bash
python yfp_pipeline.py --mode create_csv --data_dir /path/to/yfp/data
```

### 3. Train a model

```bash
python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data --csv_file labels.csv
```

### 4. Evaluate the model

```bash
python yfp_pipeline.py --mode eval --data_dir /path/to/yfp/data --model_path ./yfp_checkpoints/best_model.pth
```

### 5. Run full pipeline (train + evaluate)

```bash
python yfp_pipeline.py --mode full --data_dir /path/to/yfp/data --csv_file labels.csv
```

## Advanced Usage

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

## Output Files

The pipeline generates the following output files:

### Training Results (`./yfp_results/`)
- `training_summary.csv`: Overall training and test metrics
- `test/interview_detection_metrics.csv`: Test set evaluation metrics
- `test/predictions.csv`: Individual predictions with ground truth
- `test/classification_report.csv`: Detailed classification report

### Model Checkpoints (`./yfp_checkpoints/`)
- `best_model.pth`: Best model based on validation F1 score

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value
- **Recall**: Sensitivity or true positive rate

## Model Architecture

The interview detection model uses:

- **Backbone**: EfficientNet-B0 (tf_efficientnet_b0_ns)
- **Classifier**: Linear layer with dropout regularization
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (interview vs non-interview)

## Data Augmentation

Training uses the following augmentations:
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet statistics

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Poor performance**: Increase epochs, adjust learning rate, or use pretrained weights
3. **Data loading errors**: Check image formats and CSV file structure

### Debug Mode

Use `--limit` parameter to test with a small subset:

```bash
python yfp_pipeline.py --mode train --data_dir /path/to/yfp/data --limit 100
```

## Integration with OpenFace

This module is designed to work alongside the existing OpenFace 3.0 toolkit. You can:

1. Use OpenFace's face detection and landmark extraction to preprocess images
2. Combine interview detection with emotion recognition and action unit detection
3. Use the multitask learning capabilities for enhanced performance

## Citation

If you use this YFP interview detection module in your research, please cite:

```
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```
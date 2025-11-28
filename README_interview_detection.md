# 面谈检测 (Interview Detection)

本项目实现了基于YFP面瘫数据集的面谈检测系统，使用多任务学习模型来检测面部是否存在面瘫症状。

## 功能特点

- **二分类任务**: 区分正常面部和面瘫面部
- **多任务学习**: 利用表情识别、视线估计和动作单元检测来增强面谈检测性能
- **完整评价指标**: Accuracy, F1 Score, Precision, Recall, Specificity
- **交叉验证**: 支持K折交叉验证进行模型评估
- **可视化**: 提供预测结果可视化和混淆矩阵
- **批处理**: 支持单张图像、批量图像和整个数据集的评估

## 文件结构

```
├── interview_detection.py      # 核心模块：数据集、模型、评价指标
├── train_interview_detection.py  # 训练脚本
├── demo_interview_detection.py   # 演示和评估脚本
└── README_interview_detection.md # 本文档
```

## 数据集格式

YFP面瘫数据集应按以下结构组织：

```
yfp_dataset/
├── normal/          # 正常面部图像
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── paralysis/       # 面瘫面部图像
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

## 安装依赖

```bash
pip install torch torchvision
pip install timm
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install wandb  # 可选，用于实验跟踪
```

## 使用方法

### 1. 训练模型

#### 基本训练

```bash
python train_interview_detection.py \
    --data_dir /path/to/yfp_dataset \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --save_dir interview_detection_results
```

#### 使用预训练模型

```bash
python train_interview_detection.py \
    --data_dir /path/to/yfp_dataset \
    --pretrained_path /path/to/pretrained_mtl_model.pth \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --save_dir interview_detection_results
```

#### 交叉验证

```bash
python train_interview_detection.py \
    --data_dir /path/to/yfp_dataset \
    --cross_validate \
    --n_folds 5 \
    --epochs 30 \
    --batch_size 32 \
    --save_dir cv_results
```

#### 使用WandB记录实验

```bash
python train_interview_detection.py \
    --data_dir /path/to/yfp_dataset \
    --use_wandb \
    --epochs 50 \
    --save_dir interview_detection_results
```

### 2. 模型评估和演示

#### 单张图像预测

```bash
python demo_interview_detection.py \
    --model_path /path/to/best_interview_detection_model.pth \
    --image_path /path/to/test_image.jpg \
    --output_dir demo_results
```

#### 批量图像预测

```bash
python demo_interview_detection.py \
    --model_path /path/to/best_interview_detection_model.pth \
    --image_dir /path/to/test_images/ \
    --output_dir demo_results
```

#### 完整数据集评估

```bash
python demo_interview_detection.py \
    --model_path /path/to/best_interview_detection_model.pth \
    --eval_dir /path/to/yfp_dataset \
    --output_dir evaluation_results
```

### 3. 编程接口使用

#### 基本预测

```python
from interview_detection import InterviewDetectionModel, image_transforms
from demo_interview_detection import InterviewDetectionDemo

# 加载训练好的模型
demo = InterviewDetectionDemo('path/to/model.pth')

# 单张图像预测
result = demo.predict_single_image('path/to/image.jpg', return_probabilities=True)
print(f"预测结果: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.4f}")

# 批量预测
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = demo.predict_batch(image_paths)
```

#### 数据集评估

```python
# 评估整个数据集
evaluation_results = demo.evaluate_dataset('path/to/dataset', 'output_dir')

print(f"Accuracy: {evaluation_results['overall_metrics']['accuracy']:.4f}")
print(f"F1 Score: {evaluation_results['overall_metrics']['f1_score']:.4f}")
```

## 模型架构

面谈检测模型基于多任务学习架构，包含以下组件：

1. **特征提取器**: 使用EfficientNet作为骨干网络
2. **多任务分支**:
   - 表情识别分支 (8类)
   - 视线估计分支 (2D角度)
   - 动作单元检测分支 (8个AU)
3. **面谈检测分类器**: 结合多任务特征进行二分类

### 模型输入
- 图像尺寸: 224×224×3
- 预处理: 标准化 (ImageNet统计量)

### 模型输出
- 面谈检测结果: 正常/面瘫 (二分类)
- 多任务输出: 表情、视线、动作单元 (可选)

## 评价指标

系统提供以下评价指标：

- **Accuracy**: 准确率
- **F1 Score**: F1分数
- **Precision**: 精确率
- **Recall**: 召回率
- **Specificity**: 特异性
- **Confusion Matrix**: 混淆矩阵

## 训练参数建议

### 基本参数
- 批次大小: 16-32 (根据GPU内存调整)
- 学习率: 0.001 (从头训练) 或 0.0001 (微调)
- 训练轮数: 30-50
- 优化器: Adam

### 数据增强
- 随机水平翻转
- 随机旋转 (-10° 到 +10°)
- 颜色抖动
- 随机裁剪和缩放

## 性能优化建议

1. **使用预训练模型**: 从多任务学习模型初始化可以显著提升性能
2. **数据平衡**: 确保正常和面瘫样本数量平衡
3. **交叉验证**: 使用K折交叉验证获得更可靠的性能评估
4. **集成学习**: 可以训练多个模型并集成预测结果

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   --batch_size 16
   ```

2. **训练收敛慢**
   ```bash
   # 使用预训练模型
   --pretrained_path /path/to/pretrained_model.pth
   # 或增加学习率
   --learning_rate 0.01
   ```

3. **过拟合**
   ```bash
   # 添加更多数据增强
   # 使用dropout
   # 减少模型复杂度
   ```

## 实验结果示例

基于YFP面瘫数据集的典型结果：

```
Cross-Validation Results Summary (5-fold)
accuracy: 0.9234 ± 0.0156
f1_score: 0.9187 ± 0.0178
precision: 0.9256 ± 0.0145
recall: 0.9123 ± 0.0198
specificity: 0.9345 ± 0.0123
```

## 引用

如果您使用了本代码，请引用相关论文和OpenFace 3.0项目。

## 许可证

本项目遵循与OpenFace 3.0相同的许可证。
#!/usr/bin/env python3
"""
面谈检测示例脚本
演示如何使用面谈检测系统的完整流程
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interview_detection import (
    InterviewDetectionModel,
    YFPFacialParalysisDataset,
    create_data_loaders,
    evaluate_interview_detection,
    calculate_metrics,
    image_transforms
)

def create_sample_dataset(output_dir="sample_yfp_dataset", num_samples=50):
    """
    创建示例YFP数据集（使用随机生成的图像）
    仅用于演示，实际使用时请替换为真实的YFP数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 每类别的样本数量
    """
    import random
    from PIL import Image
    import numpy as np
    
    print(f"Creating sample dataset with {num_samples} samples per class...")
    
    os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'paralysis'), exist_ok=True)
    
    # 生成随机图像
    for class_name in ['normal', 'paralysis']:
        for i in range(num_samples):
            # 创建随机RGB图像
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            
            # 根据类别添加一些模式差异（仅用于演示）
            if class_name == 'paralysis':
                # 添加一些"特征"来模拟面瘫
                img_array[100:124, 100:124] = 255  # 在中心添加白色方块
                image = Image.fromarray(img_array)
            
            # 保存图像
            filename = f"{class_name}_{i:03d}.jpg"
            image.save(os.path.join(output_dir, class_name, filename))
    
    print(f"Sample dataset created at: {output_dir}")
    return output_dir


def quick_training_example():
    """
    快速训练示例
    """
    print("="*60)
    print("QUICK TRAINING EXAMPLE")
    print("="*60)
    
    # 创建示例数据集（如果不存在真实数据集）
    sample_data_dir = "sample_yfp_dataset"
    if not os.path.exists(sample_data_dir):
        sample_data_dir = create_sample_dataset(sample_data_dir, num_samples=20)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        sample_data_dir, 
        batch_size=8, 
        test_size=0.3,
        limit=None
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = InterviewDetectionModel()
    model.to(device)
    
    # 简单训练循环
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining for 5 epochs...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            interview_output, _, _, _ = model(images)
            loss = criterion(interview_output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(interview_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # 评估模型
    print("\nEvaluating on test set...")
    metrics, true_labels, pred_labels = evaluate_interview_detection(model, test_loader, device)
    
    print(f"Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    # 保存模型
    model_path = "quick_trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model_path, test_loader


def prediction_example(model_path, test_loader):
    """
    预测示例
    """
    print("\n" + "="*60)
    print("PREDICTION EXAMPLE")
    print("="*60)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InterviewDetectionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 对几个样本进行预测
    from demo_interview_detection import InterviewDetectionDemo
    
    demo = InterviewDetectionDemo(model_path, device)
    
    # 获取一些测试样本的路径
    sample_images = []
    sample_labels = []
    
    # 从数据加载器中提取一些样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    print("\nSample Predictions:")
    print("-" * 40)
    
    for i in range(min(5, len(images))):
        # 这里我们使用模拟的图像路径，实际使用时应该是真实的图像路径
        # 由于我们使用的是生成的数据，这里直接使用模型预测
        
        image_tensor = images[i].unsqueeze(0).to(device)
        
        with torch.no_grad():
            interview_output, _, _, _ = model(image_tensor)
            probabilities = torch.softmax(interview_output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        true_label = labels[i].item()
        true_class = "Normal" if true_label == 0 else "Paralysis"
        pred_class = "Normal" if predicted_class == 0 else "Paralysis"
        
        print(f"Sample {i+1}:")
        print(f"  True: {true_class}")
        print(f"  Predicted: {pred_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Correct: {'Yes' if predicted_class == true_label else 'No'}")
        print()


def metrics_calculation_example():
    """
    指标计算示例
    """
    print("\n" + "="*60)
    print("METRICS CALCULATION EXAMPLE")
    print("="*60)
    
    # 模拟一些预测结果
    np.random.seed(42)
    
    # 生成100个样本的模拟数据
    n_samples = 100
    true_labels = np.random.randint(0, 2, n_samples)
    
    # 添加一些噪声来模拟 imperfect 预测
    pred_labels = true_labels.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    pred_labels[noise_indices] = 1 - pred_labels[noise_indices]
    
    # 计算指标
    metrics = calculate_metrics(true_labels, pred_labels)
    
    print("Simulated Prediction Results:")
    print(f"  Total samples: {n_samples}")
    print(f"  True Positives: {metrics['true_positive']}")
    print(f"  True Negatives: {metrics['true_negative']}")
    print(f"  False Positives: {metrics['false_positive']}")
    print(f"  False Negatives: {metrics['false_negative']}")
    print()
    
    print("Calculated Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print()
    
    # 打印混淆矩阵
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print("                Predicted")
    print("                Normal  Paralysis")
    print(f"True    Normal  {cm[0][0]:6d}  {cm[0][1]:9d}")
    print(f"        Paralysis {cm[1][0]:6d}  {cm[1][1]:9d}")


def main():
    """
    主演示函数
    """
    print("面谈检测系统演示")
    print("Interview Detection System Demo")
    print("="*60)
    
    try:
        # 1. 快速训练示例
        model_path, test_loader = quick_training_example()
        
        # 2. 预测示例
        prediction_example(model_path, test_loader)
        
        # 3. 指标计算示例
        metrics_calculation_example()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("Demo completed!")
        print("="*60)
        
        print("\n使用说明:")
        print("1. 将您的YFP面瘫数据集按照 normal/ 和 paralysis/ 子目录组织")
        print("2. 运行训练脚本: python train_interview_detection.py --data_dir /path/to/your/dataset")
        print("3. 评估模型: python evaluate_interview_detection.py --model_path model.pth --data_dir /path/to/your/dataset")
        print("4. 演示预测: python demo_interview_detection.py --model_path model.pth --image_path test_image.jpg")
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        print("请确保已安装所有必要的依赖包:")
        print("pip install torch torchvision timm scikit-learn matplotlib seaborn")


if __name__ == "__main__":
    main()
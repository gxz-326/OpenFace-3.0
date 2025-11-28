#!/usr/bin/env python3
"""
YFP面瘫数据集面谈检测完整流程演示
展示从数据准备到模型评估的完整过程
"""

import os
import sys
import argparse
import json
from datetime import datetime

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

def run_complete_pipeline(data_dir, output_dir="yfp_interview_results", epochs=20, batch_size=16):
    """
    运行完整的面谈检测流程
    
    Args:
        data_dir: YFP数据集目录
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
    """
    
    print("="*80)
    print("YFP面瘫数据集面谈检测完整流程")
    print("Complete Interview Detection Pipeline for YFP Dataset")
    print("="*80)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print(f"Output directory: {output_dir}")
    
    # 1. 数据加载和预处理
    print("\n" + "="*60)
    print("1. 数据加载和预处理")
    print("1. Data Loading and Preprocessing")
    print("="*60)
    
    try:
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            test_size=0.2,
            limit=None
        )
        
        print(f"训练样本数量: {len(train_loader.dataset)}")
        print(f"测试样本数量: {len(test_loader.dataset)}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # 统计类别分布
        train_labels = []
        test_labels = []
        
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        
        for _, labels in test_loader:
            test_labels.extend(labels.numpy())
        
        train_normal = train_labels.count(0)
        train_paralysis = train_labels.count(1)
        test_normal = test_labels.count(0)
        test_paralysis = test_labels.count(1)
        
        print(f"\n训练集分布 - Normal: {train_normal}, Paralysis: {train_paralysis}")
        print(f"测试集分布 - Normal: {test_normal}, Paralysis: {test_paralysis}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 2. 模型创建和训练
    print("\n" + "="*60)
    print("2. 模型创建和训练")
    print("2. Model Creation and Training")
    print("="*60)
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"Using device: {device}")
    
    # 创建模型
    model = InterviewDetectionModel()
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 训练历史
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    print(f"开始训练 {epochs} 个epoch...")
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            interview_output, _, _, _ = model(images)
            loss = criterion(interview_output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(interview_output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_acc:.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                interview_output, _, _, _ = model(images)
                loss = criterion(interview_output, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(interview_output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_acc = val_correct / val_total
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{val_acc:.4f}'})
        
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_path)
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # 3. 最终评估
    print("\n" + "="*60)
    print("3. 最终模型评估")
    print("3. Final Model Evaluation")
    print("="*60)
    
    # 加载最佳模型
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模型
    metrics, true_labels, pred_labels = evaluate_interview_detection(model, test_loader, device)
    
    print(f"\n最终评估结果:")
    print(f"Final evaluation results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  True Positives: {metrics['true_positive']}")
    print(f"  True Negatives: {metrics['true_negative']}")
    print(f"  False Positives: {metrics['false_positive']}")
    print(f"  False Negatives: {metrics['false_negative']}")
    
    # 4. 保存结果
    print("\n" + "="*60)
    print("4. 保存结果")
    print("4. Saving Results")
    print("="*60)
    
    # 保存详细结果
    results = {
        'dataset_info': {
            'data_dir': data_dir,
            'train_samples': len(train_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'train_distribution': {'normal': train_normal, 'paralysis': train_paralysis},
            'test_distribution': {'normal': test_normal, 'paralysis': test_paralysis}
        },
        'training_info': {
            'epochs': epochs,
            'batch_size': batch_size,
            'device': str(device),
            'best_val_accuracy': best_val_acc,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        },
        'evaluation_metrics': metrics,
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    }
    
    results_file = os.path.join(output_dir, 'complete_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: {results_file}")
    print(f"Detailed results saved to: {results_file}")
    
    # 保存训练曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 混淆矩阵
    plt.subplot(1, 3, 3)
    import seaborn as sns
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Paralysis'],
               yticklabels=['Normal', 'Paralysis'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {os.path.join(output_dir, 'training_curves.png')}")
    print(f"Training curves saved to: {os.path.join(output_dir, 'training_curves.png')}")
    
    print(f"\n" + "="*80)
    print("面谈检测流程完成！")
    print("Interview detection pipeline completed!")
    print(f"所有结果保存在: {output_dir}")
    print(f"All results saved in: {output_dir}")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YFP面瘫数据集面谈检测完整流程')
    parser.add_argument('--data_dir', type=str, required=True, help='YFP数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='yfp_interview_results', help='输出目录')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # 检查数据目录结构
    normal_dir = os.path.join(args.data_dir, 'normal')
    paralysis_dir = os.path.join(args.data_dir, 'paralysis')
    
    if not os.path.exists(normal_dir):
        print(f"警告: normal目录不存在: {normal_dir}")
        print(f"Warning: normal directory does not exist: {normal_dir}")
    
    if not os.path.exists(paralysis_dir):
        print(f"警告: paralysis目录不存在: {paralysis_dir}")
        print(f"Warning: paralysis directory does not exist: {paralysis_dir}")
    
    # 运行完整流程
    try:
        results = run_complete_pipeline(
            args.data_dir,
            args.output_dir,
            args.epochs,
            args.batch_size
        )
        
        if results:
            print("\n流程成功完成！")
            print("Pipeline completed successfully!")
        else:
            print("\n流程执行失败！")
            print("Pipeline execution failed!")
            
    except Exception as e:
        print(f"\n流程执行出错: {e}")
        print(f"Pipeline execution error: {e}")


if __name__ == "__main__":
    main()
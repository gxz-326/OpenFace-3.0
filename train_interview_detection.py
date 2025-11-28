import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
import argparse
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

from interview_detection import (
    InterviewDetectionModel, 
    YFPFacialParalysisDataset,
    evaluate_interview_detection,
    calculate_metrics,
    image_transforms,
    create_data_loaders
)


def train_interview_detection(model, train_loader, val_loader, device, epochs, learning_rate, save_dir):
    """
    训练面谈检测模型
    
    Args:
        model: 面谈检测模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
        learning_rate: 学习率
        save_dir: 模型保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
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
            
            # 前向传播
            interview_output, _, _, _ = model(images)
            loss = criterion(interview_output, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(interview_output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_acc = train_correct / train_total
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_acc:.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                interview_output, _, _, _ = model(images)
                loss = criterion(interview_output, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(interview_output.data, 1)
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算验证指标
        val_metrics = calculate_metrics(all_val_labels, all_val_preds)
        val_metrics_history.append(val_metrics)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_correct/train_total:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Metrics - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
        
        # 使用wandb记录（如果可用）
        if 'wandb' in globals():
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_f1_score': val_metrics['f1_score'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_specificity': val_metrics['specificity']
            })
        
        # 保存最佳模型
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_path = os.path.join(save_dir, 'best_interview_detection_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'val_f1': best_val_f1
            }, best_model_path)
            print(f"  New best model saved with F1: {best_val_f1:.4f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
    
    print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")
    
    return train_losses, val_losses, val_metrics_history


def cross_validate_model(data_dir, n_folds=5, batch_size=32, epochs=50, learning_rate=0.001, 
                       pretrained_path=None, limit=None):
    """
    交叉验证评估模型性能
    
    Args:
        data_dir: 数据集目录
        n_folds: 交叉验证折数
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        pretrained_path: 预训练模型路径
        limit: 样本数量限制
    
    Returns:
        所有折的评估结果
    """
    # 创建完整数据集
    full_dataset = YFPFacialParalysisDataset(data_dir, transform=image_transforms, limit=limit)
    
    # K折交叉验证
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_fold_metrics = []
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # 创建数据加载器
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = InterviewDetectionModel(pretrained_path=pretrained_path)
        model = model.to(device)
        
        # 训练模型
        save_dir = f"cv_results/fold_{fold+1}"
        train_losses, val_losses, val_metrics_history = train_interview_detection(
            model, train_loader, val_loader, device, epochs, learning_rate, save_dir
        )
        
        # 评估最终模型
        final_metrics, _, _ = evaluate_interview_detection(model, val_loader, device)
        all_fold_metrics.append(final_metrics)
        
        print(f"Fold {fold+1} Final Results:")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {final_metrics['f1_score']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall: {final_metrics['recall']:.4f}")
        print(f"  Specificity: {final_metrics['specificity']:.4f}")
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_fold_metrics[0].keys():
        if key != 'confusion_matrix':  # 不平均混淆矩阵
            values = [fold_metrics[key] for fold_metrics in all_fold_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
    
    print(f"\n{'='*50}")
    print("Cross-Validation Results Summary")
    print(f"{'='*50}")
    for key in sorted(avg_metrics.keys()):
        if key.startswith('avg_'):
            metric_name = key[4:]  # 移除'avg_'前缀
            avg_val = avg_metrics[key]
            std_val = avg_metrics[f'std_{metric_name}']
            print(f"{metric_name}: {avg_val:.4f} ± {std_val:.4f}")
    
    return all_fold_metrics, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Interview Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to YFP dataset directory')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained MTL model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='interview_detection_results', help='Directory to save results')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for debugging')
    parser.add_argument('--cross_validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # 初始化wandb（如果使用）
    if args.use_wandb:
        wandb.init(
            project="interview-detection-yfp",
            config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "pretrained": args.pretrained_path is not None
            }
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.cross_validate:
        # 交叉验证模式
        all_fold_metrics, avg_metrics = cross_validate_model(
            args.data_dir, 
            n_folds=args.n_folds,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            pretrained_path=args.pretrained_path,
            limit=args.limit
        )
        
        # 保存交叉验证结果
        import json
        with open(os.path.join(args.save_dir, 'cross_validation_results.json'), 'w') as f:
            json.dump({
                'fold_results': all_fold_metrics,
                'average_metrics': avg_metrics
            }, f, indent=2)
    
    else:
        # 标准训练模式
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            args.data_dir, 
            batch_size=args.batch_size, 
            limit=args.limit
        )
        
        # 创建模型
        model = InterviewDetectionModel(pretrained_path=args.pretrained_path)
        model = model.to(device)
        
        # 训练模型
        train_losses, val_losses, val_metrics_history = train_interview_detection(
            model, train_loader, test_loader, device, 
            args.epochs, args.learning_rate, args.save_dir
        )
        
        # 最终评估
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        
        final_metrics, true_labels, pred_labels = evaluate_interview_detection(model, test_loader, device)
        
        print(f"Final Test Results:")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {final_metrics['f1_score']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall: {final_metrics['recall']:.4f}")
        print(f"  Specificity: {final_metrics['specificity']:.4f}")
        print(f"  True Positives: {final_metrics['true_positive']}")
        print(f"  True Negatives: {final_metrics['true_negative']}")
        print(f"  False Positives: {final_metrics['false_positive']}")
        print(f"  False Negatives: {final_metrics['false_negative']}")
        
        # 保存最终结果
        import json
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = final_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Paralysis'],
                   yticklabels=['Normal', 'Paralysis'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.save_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        f1_scores = [m['f1_score'] for m in val_metrics_history]
        plt.plot(f1_scores, label='F1 Score')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
        plt.close()
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
面谈检测评估脚本
用于评估训练好的面谈检测模型在YFP面瘫数据集上的性能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from interview_detection import (
    InterviewDetectionModel,
    YFPFacialParalysisDataset,
    evaluate_interview_detection,
    calculate_metrics,
    image_transforms
)


def comprehensive_evaluation(model, test_loader, device, output_dir):
    """
    全面评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        output_dir: 输出目录
    
    Returns:
        详细的评估结果
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_features = []
    
    print("Running comprehensive evaluation...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # 获取模型输出
            interview_output, emotion_output, gaze_output, au_output = model(images)
            
            # 获取概率
            probabilities = torch.softmax(interview_output, dim=1)
            predictions = torch.argmax(interview_output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 保存特征用于后续分析
            features = {
                'emotion': emotion_output.cpu().numpy(),
                'gaze': gaze_output.cpu().numpy(),
                'au': au_output.cpu().numpy()
            }
            all_features.append(features)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # 计算基本指标
    basic_metrics = calculate_metrics(all_labels, all_predictions)
    
    # 计算详细分类报告
    class_report = classification_report(
        all_labels, all_predictions,
        target_names=['Normal', 'Paralysis'],
        output_dict=True
    )
    
    # ROC和AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall曲线
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probabilities[:, 1])
    pr_auc = auc(recall, precision)
    
    # 寻找最佳阈值
    youden_j = tpr - fpr
    best_threshold_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_idx]
    best_sensitivity = tpr[best_threshold_idx]
    best_specificity = 1 - fpr[best_threshold_idx]
    
    # 生成详细报告
    detailed_results = {
        'basic_metrics': basic_metrics,
        'classification_report': class_report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': {
            'threshold': float(best_threshold),
            'sensitivity': float(best_sensitivity),
            'specificity': float(best_specificity),
            'youden_j': float(youden_j[best_threshold_idx])
        },
        'sample_statistics': {
            'total_samples': len(all_labels),
            'normal_samples': int(np.sum(all_labels == 0)),
            'paralysis_samples': int(np.sum(all_labels == 1)),
            'predicted_normal': int(np.sum(all_predictions == 0)),
            'predicted_paralysis': int(np.sum(all_predictions == 1))
        }
    }
    
    # 保存结果
    with open(os.path.join(output_dir, 'comprehensive_evaluation.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # 绘制图表
    plot_evaluation_charts(all_labels, all_predictions, all_probabilities, 
                          fpr, tpr, roc_auc, precision, recall, pr_auc,
                          basic_metrics, output_dir)
    
    return detailed_results


def plot_evaluation_charts(labels, predictions, probabilities, fpr, tpr, roc_auc, 
                          precision, recall, pr_auc, metrics, output_dir):
    """
    绘制评估图表
    """
    plt.style.use('default')
    
    # 1. 混淆矩阵
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Paralysis'],
               yticklabels=['Normal', 'Paralysis'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. ROC曲线
    plt.subplot(2, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall曲线
    plt.subplot(2, 3, 3)
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # 4. 预测置信度分布
    plt.subplot(2, 3, 4)
    normal_probs = probabilities[labels == 0, 1]  # 正常样本被预测为面瘫的概率
    paralysis_probs = probabilities[labels == 1, 1]  # 面瘫样本被预测为面瘫的概率
    
    plt.hist(normal_probs, bins=30, alpha=0.7, label='Normal', density=True)
    plt.hist(paralysis_probs, bins=30, alpha=0.7, label='Paralysis', density=True)
    plt.xlabel('Predicted Probability of Paralysis')
    plt.ylabel('Density')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # 5. 指标雷达图
    plt.subplot(2, 3, 5, projection='polar')
    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity']
    metrics_values = [
        metrics['accuracy'],
        metrics['f1_score'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity']
    ]
    
    # 闭合雷达图
    metrics_names += [metrics_names[0]]
    metrics_values += [metrics_values[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    
    plt.fill(angles, metrics_values, 'alpha=0.25')
    plt.plot(angles, metrics_values, 'o-', linewidth=2)
    plt.xticks(angles[:-1], metrics_names[:-1])
    plt.ylim(0, 1)
    plt.title('Performance Metrics Radar Chart')
    
    # 6. 阈值分析
    plt.subplot(2, 3, 6)
    thresholds = np.arange(0, 1.01, 0.01)
    sensitivities = []
    specificities = []
    
    for threshold in thresholds:
        pred_threshold = (probabilities[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred_threshold).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    plt.plot(thresholds, sensitivities, label='Sensitivity', color='red')
    plt.plot(thresholds, specificities, label='Specificity', color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Sensitivity vs Specificity by Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_charts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存高分辨率混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Paralysis'],
               yticklabels=['Normal', 'Paralysis'],
               annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_high_res.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_errors(model, test_loader, device, output_dir, num_errors=20):
    """
    分析错误预测的样本
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        output_dir: 输出目录
        num_errors: 要分析的错误样本数量
    """
    model.eval()
    
    error_analysis = []
    
    print("Analyzing prediction errors...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Error Analysis")):
            images, labels = images.to(device), labels.to(device)
            
            interview_output, emotion_output, gaze_output, au_output = model(images)
            probabilities = torch.softmax(interview_output, dim=1)
            predictions = torch.argmax(interview_output, dim=1)
            
            for j in range(len(labels)):
                if predictions[j] != labels[j]:
                    error_info = {
                        'sample_index': i * test_loader.batch_size + j,
                        'true_label': int(labels[j].cpu()),
                        'predicted_label': int(predictions[j].cpu()),
                        'confidence': float(probabilities[j, predictions[j]].cpu()),
                        'probabilities': probabilities[j].cpu().numpy().tolist(),
                        'emotion_features': emotion_output[j].cpu().numpy().tolist(),
                        'gaze_features': gaze_output[j].cpu().numpy().tolist(),
                        'au_features': au_output[j].cpu().numpy().tolist()
                    }
                    error_analysis.append(error_info)
    
    # 按置信度排序
    error_analysis.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 保存错误分析结果
    with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
        json.dump(error_analysis[:num_errors], f, indent=2)
    
    # 统计错误类型
    false_positives = [e for e in error_analysis if e['true_label'] == 0 and e['predicted_label'] == 1]
    false_negatives = [e for e in error_analysis if e['true_label'] == 1 and e['predicted_label'] == 0]
    
    error_stats = {
        'total_errors': len(error_analysis),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'fp_avg_confidence': np.mean([e['confidence'] for e in false_positives]) if false_positives else 0,
        'fn_avg_confidence': np.mean([e['confidence'] for e in false_negatives]) if false_negatives else 0
    }
    
    with open(os.path.join(output_dir, 'error_statistics.json'), 'w') as f:
        json.dump(error_stats, f, indent=2)
    
    print(f"Error Analysis Complete:")
    print(f"  Total errors: {error_stats['total_errors']}")
    print(f"  False positives: {error_stats['false_positives']}")
    print(f"  False negatives: {error_stats['false_negatives']}")
    print(f"  FP average confidence: {error_stats['fp_avg_confidence']:.3f}")
    print(f"  FN average confidence: {error_stats['fn_avg_confidence']:.3f}")
    
    return error_analysis


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Interview Detection Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to YFP dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--analyze_errors', action='store_true', help='Perform detailed error analysis')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for debugging')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据集
    print("Loading dataset...")
    full_dataset = YFPFacialParalysisDataset(args.data_dir, transform=image_transforms, limit=args.limit)
    
    # 划分数据集
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.test_size * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    from torch.utils.data import Subset
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # 加载模型
    print("Loading model...")
    model = InterviewDetectionModel(pretrained_path=args.model_path)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # 运行全面评估
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    results = comprehensive_evaluation(model, test_loader, device, args.output_dir)
    
    # 打印主要结果
    print(f"\nMain Results:")
    print(f"  Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {results['basic_metrics']['f1_score']:.4f}")
    print(f"  Precision: {results['basic_metrics']['precision']:.4f}")
    print(f"  Recall: {results['basic_metrics']['recall']:.4f}")
    print(f"  Specificity: {results['basic_metrics']['specificity']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    print(f"  PR AUC: {results['pr_auc']:.4f}")
    
    print(f"\nBest Threshold Analysis:")
    print(f"  Threshold: {results['best_threshold']['threshold']:.3f}")
    print(f"  Sensitivity: {results['best_threshold']['sensitivity']:.3f}")
    print(f"  Specificity: {results['best_threshold']['specificity']:.3f}")
    print(f"  Youden's J: {results['best_threshold']['youden_j']:.3f}")
    
    print(f"\nSample Statistics:")
    print(f"  Total samples: {results['sample_statistics']['total_samples']}")
    print(f"  Normal samples: {results['sample_statistics']['normal_samples']}")
    print(f"  Paralysis samples: {results['sample_statistics']['paralysis_samples']}")
    print(f"  Predicted Normal: {results['sample_statistics']['predicted_normal']}")
    print(f"  Predicted Paralysis: {results['sample_statistics']['predicted_paralysis']}")
    
    # 错误分析（如果需要）
    if args.analyze_errors:
        print(f"\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        error_analysis = analyze_errors(model, test_loader, device, args.output_dir)
    
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Main report: {os.path.join(args.output_dir, 'comprehensive_evaluation.json')}")
    print(f"Visualization: {os.path.join(args.output_dir, 'evaluation_charts.png')}")


if __name__ == "__main__":
    main()
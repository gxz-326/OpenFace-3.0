import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from interview_detection import InterviewDetectionModel, image_transforms, calculate_metrics


class InterviewDetectionDemo:
    """
    面谈检测演示类
    """
    def __init__(self, model_path, device=None):
        """
        初始化演示类
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = InterviewDetectionModel()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_info = {
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'val_metrics': checkpoint.get('val_metrics', {})
            }
        else:
            self.model.load_state_dict(checkpoint)
            self.model_info = {'epoch': 'Unknown', 'val_metrics': {}}
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        if self.model_info['val_metrics']:
            print(f"Model validation metrics: {self.model_info['val_metrics']}")
        
        # 类别标签
        self.class_names = ['Normal', 'Paralysis']
        
        # 图像预处理
        self.transform = image_transforms
    
    def predict_single_image(self, image_path, return_probabilities=False):
        """
        对单张图像进行预测
        
        Args:
            image_path: 图像路径
            return_probabilities: 是否返回概率分布
        
        Returns:
            预测结果
        """
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            interview_output, emotion_output, gaze_output, au_output = self.model(image_tensor)
            
            # 获取预测类别和概率
            probabilities = F.softmax(interview_output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'predicted_label': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'normal': probabilities[0][0].item(),
                    'paralysis': probabilities[0][1].item()
                }
            }
            
            if return_probabilities:
                # 添加多任务信息
                emotion_probs = F.softmax(emotion_output, dim=1)
                result['emotion'] = {
                    'predicted': torch.argmax(emotion_probs, dim=1).item(),
                    'probabilities': emotion_probs[0].cpu().numpy().tolist()
                }
                
                result['gaze'] = {
                    'yaw': gaze_output[0][0].item(),
                    'pitch': gaze_output[0][1].item()
                }
                
                result['au_intensities'] = au_output[0].cpu().numpy().tolist()
            
            return result
    
    def predict_batch(self, image_paths, save_results=True, output_file='batch_results.json'):
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            save_results: 是否保存结果
            output_file: 输出文件名
        
        Returns:
            所有预测结果
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single_image(image_path, return_probabilities=True)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        if save_results:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def evaluate_dataset(self, data_dir, output_dir='evaluation_results'):
        """
        评估整个数据集
        
        Args:
            data_dir: 数据集目录（包含normal和paralysis子目录）
            output_dir: 输出目录
        
        Returns:
            评估指标
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集所有图像和标签
        image_paths = []
        true_labels = []
        
        # 正常样本
        normal_dir = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_dir):
            for img_file in os.listdir(normal_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(normal_dir, img_file))
                    true_labels.append(0)
        
        # 面瘫样本
        paralysis_dir = os.path.join(data_dir, 'paralysis')
        if os.path.exists(paralysis_dir):
            for img_file in os.listdir(paralysis_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(paralysis_dir, img_file))
                    true_labels.append(1)
        
        print(f"Found {len(image_paths)} images for evaluation")
        
        # 批量预测
        results = self.predict_batch(image_paths, save_results=False)
        
        # 收集预测结果
        pred_labels = []
        confidences = []
        
        for result in results:
            if 'error' not in result:
                pred_labels.append(result['predicted_class'])
                confidences.append(result['confidence'])
            else:
                # 错误情况下使用默认值
                pred_labels.append(0)
                confidences.append(0.0)
        
        # 计算指标
        metrics = calculate_metrics(true_labels, pred_labels)
        
        # 生成分类报告
        class_report = classification_report(
            true_labels, pred_labels, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # 保存结果
        evaluation_results = {
            'overall_metrics': metrics,
            'classification_report': class_report,
            'sample_count': len(image_paths),
            'normal_samples': sum(1 for label in true_labels if label == 0),
            'paralysis_samples': sum(1 for label in true_labels if label == 1),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences)
        }
        
        with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 绘制置信度分布
        plt.figure(figsize=(10, 6))
        normal_confs = [conf for conf, label in zip(confidences, true_labels) if label == 0]
        paralysis_confs = [conf for conf, label in zip(confidences, true_labels) if label == 1]
        
        plt.hist(normal_confs, bins=30, alpha=0.7, label='Normal', density=True)
        plt.hist(paralysis_confs, bins=30, alpha=0.7, label='Paralysis', density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution by Class')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()
        
        # 打印结果
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total samples: {evaluation_results['sample_count']}")
        print(f"Normal samples: {evaluation_results['normal_samples']}")
        print(f"Paralysis samples: {evaluation_results['paralysis_samples']}")
        print(f"\nMetrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"\nAverage Confidence: {evaluation_results['average_confidence']:.4f}")
        print(f"Confidence Std: {evaluation_results['confidence_std']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Normal: {metrics['true_negative']} (TN) | {metrics['false_positive']} (FP)")
        print(f"  True Paralysis: {metrics['false_negative']} (FN) | {metrics['true_positive']} (TP)")
        
        print(f"\nResults saved to: {output_dir}")
        
        return evaluation_results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        可视化单张图像的预测结果
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
        """
        result = self.predict_single_image(image_path, return_probabilities=True)
        
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title(f"Original Image\nPrediction: {result['predicted_label']}")
        axes[0, 0].axis('off')
        
        # 预测概率
        probs = [result['probabilities']['normal'], result['probabilities']['paralysis']]
        axes[0, 1].bar(self.class_names, probs, color=['green', 'red'])
        axes[0, 1].set_title(f"Prediction Probabilities\nConfidence: {result['confidence']:.3f}")
        axes[0, 1].set_ylabel("Probability")
        axes[0, 1].set_ylim(0, 1)
        
        # 情绪分布（如果有）
        if 'emotion' in result:
            emotion_probs = result['emotion']['probabilities']
            emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
            axes[1, 0].bar(range(len(emotion_probs)), emotion_probs)
            axes[1, 0].set_title(f"Emotion Distribution\nPredicted: {emotion_labels[result['emotion']['predicted']]}")
            axes[1, 0].set_xticks(range(len(emotion_labels)))
            axes[1, 0].set_xticklabels(emotion_labels, rotation=45)
            axes[1, 0].set_ylabel("Probability")
        
        # AU强度（如果有）
        if 'au_intensities' in result:
            au_intensities = result['au_intensities']
            au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15']
            axes[1, 1].bar(au_labels, au_intensities)
            axes[1, 1].set_title("Action Unit Intensities")
            axes[1, 1].set_ylabel("Intensity")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Interview Detection Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Single image path for prediction')
    parser.add_argument('--image_dir', type=str, help='Directory containing images for batch prediction')
    parser.add_argument('--eval_dir', type=str, help='Dataset directory for evaluation')
    parser.add_argument('--output_dir', type=str, default='demo_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 初始化演示类
    demo = InterviewDetectionDemo(args.model_path, device=device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image_path:
        # 单张图像预测
        print(f"\nPredicting single image: {args.image_path}")
        result = demo.predict_single_image(args.image_path, return_probabilities=True)
        
        print("\nPrediction Result:")
        print(f"  Predicted Class: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities - Normal: {result['probabilities']['normal']:.4f}, "
              f"Paralysis: {result['probabilities']['paralysis']:.4f}")
        
        # 可视化
        vis_path = os.path.join(args.output_dir, 'single_prediction_visualization.png')
        demo.visualize_prediction(args.image_path, save_path=vis_path)
    
    elif args.image_dir:
        # 批量预测
        print(f"\nBatch prediction for directory: {args.image_dir}")
        image_paths = []
        for root, dirs, files in os.walk(args.image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
        
        if image_paths:
            output_file = os.path.join(args.output_dir, 'batch_predictions.json')
            results = demo.predict_batch(image_paths, save_results=True, output_file=output_file)
            
            # 统计结果
            normal_count = sum(1 for r in results if r.get('predicted_class') == 0)
            paralysis_count = sum(1 for r in results if r.get('predicted_class') == 1)
            avg_confidence = np.mean([r.get('confidence', 0) for r in results])
            
            print(f"\nBatch Prediction Summary:")
            print(f"  Total images processed: {len(results)}")
            print(f"  Predicted Normal: {normal_count}")
            print(f"  Predicted Paralysis: {paralysis_count}")
            print(f"  Average Confidence: {avg_confidence:.4f}")
        else:
            print("No images found in the specified directory.")
    
    elif args.eval_dir:
        # 数据集评估
        print(f"\nEvaluating dataset: {args.eval_dir}")
        evaluation_results = demo.evaluate_dataset(args.eval_dir, args.output_dir)
    
    else:
        print("Please specify either --image_path, --image_dir, or --eval_dir")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()
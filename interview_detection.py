import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch.nn.functional as F
from model.MLT import MLT


class YFPFacialParalysisDataset(Dataset):
    """
    YFP面瘫数据集加载器
    数据集结构：
    - 正常面部样本：标签为0
    - 面瘫面部样本：标签为1
    """
    def __init__(self, data_dir, transform=None, limit=None):
        """
        Args:
            data_dir (string): 数据集根目录，应包含normal和paralysis两个子目录
            transform (callable, optional): 可选的图像变换
            limit (int, optional): 限制样本数量，用于调试
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 加载正常面部样本
        normal_dir = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_dir):
            normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in normal_files:
                self.samples.append((os.path.join(normal_dir, f), 0))
        
        # 加载面瘫面部样本
        paralysis_dir = os.path.join(data_dir, 'paralysis')
        if os.path.exists(paralysis_dir):
            paralysis_files = [f for f in os.listdir(paralysis_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in paralysis_files:
                self.samples.append((os.path.join(paralysis_dir, f), 1))
        
        # 应用样本数量限制
        if limit is not None:
            self.samples = self.samples[:limit]
        
        print(f"Loaded {len(self.samples)} samples from YFP dataset")
        normal_count = sum(1 for _, label in self.samples if label == 0)
        paralysis_count = sum(1 for _, label in self.samples if label == 1)
        print(f"Normal samples: {normal_count}, Paralysis samples: {paralysis_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


class InterviewDetectionModel(nn.Module):
    """
    面谈检测模型，基于MLT架构
    利用多任务学习的特征来进行面谈检测
    """
    def __init__(self, base_model_name='tf_efficientnet_b0_ns', pretrained_path=None):
        super(InterviewDetectionModel, self).__init__()
        
        # 初始化MLT模型作为特征提取器
        self.mlt_model = MLT(base_model_name=base_model_name)
        
        # 如果有预训练权重，加载它们
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            model_dict = self.mlt_model.state_dict()
            
            # 过滤掉不匹配的键
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.mlt_model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} pretrained parameters")
        
        # 获取特征维度
        feature_dim = self.mlt_model.base_model.num_features
        
        # 面谈检测分类器
        self.interview_classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),  # 结合emotion, gaze, AU特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 二分类：正常 vs 面瘫
        )
        
        # 冻结MLT的部分层（可选）
        for param in self.mlt_model.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 获取MLT的多任务特征
        emotion_output, gaze_output, au_output = self.mlt_model(x)
        
        # 获取中间特征
        features = self.mlt_model.base_model(x)
        
        # 获取各任务的专门特征
        features_emotion = self.mlt_model.relu(self.mlt_model.fc_emotion(features))
        features_gaze = self.mlt_model.relu(self.mlt_model.fc_gaze(features))
        features_au = self.mlt_model.relu(self.mlt_model.fc_au(features))
        
        # 拼接所有特征
        combined_features = torch.cat([features_emotion, features_gaze, features_au], dim=1)
        
        # 面谈检测
        interview_output = self.interview_classifier(combined_features)
        
        return interview_output, emotion_output, gaze_output, au_output


def calculate_metrics(y_true, y_pred):
    """
    计算面谈检测的评价指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        dict: 包含所有指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 计算特异性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def evaluate_interview_detection(model, dataloader, device):
    """
    评估面谈检测模型
    
    Args:
        model: 面谈检测模型
        dataloader: 测试数据加载器
        device: 设备
    
    Returns:
        dict: 评估结果
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # 获取面谈检测结果
            interview_output, _, _, _ = model(images)
            
            # 获取预测类别
            predictions = torch.argmax(interview_output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_predictions)
    
    return metrics, all_labels, all_predictions


# 图像变换
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def create_data_loaders(data_dir, batch_size=32, test_size=0.2, limit=None):
    """
    创建训练和测试数据加载器
    
    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        test_size: 测试集比例
        limit: 样本数量限制
    
    Returns:
        train_loader, test_loader
    """
    # 创建完整数据集
    full_dataset = YFPFacialParalysisDataset(data_dir, transform=image_transforms, limit=limit)
    
    # 划分训练集和测试集
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    # 创建子数据集
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
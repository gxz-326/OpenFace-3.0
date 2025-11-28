#!/usr/bin/env python3
"""
面谈检测系统测试脚本
验证所有组件是否正常工作
"""

import os
import sys
import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from interview_detection import (
        InterviewDetectionModel,
        YFPFacialParalysisDataset,
        calculate_metrics,
        image_transforms
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"导入错误: {e}")
    IMPORTS_AVAILABLE = False


class TestInterviewDetection(unittest.TestCase):
    """面谈检测系统测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        if not IMPORTS_AVAILABLE:
            cls.skipTest("无法导入必要的模块")
        
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据集
        cls.create_test_dataset()
        
        # 设置设备
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        # 删除临时目录
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def create_test_dataset(cls):
        """创建测试数据集"""
        # 创建目录结构
        normal_dir = os.path.join(cls.temp_dir, 'normal')
        paralysis_dir = os.path.join(cls.temp_dir, 'paralysis')
        os.makedirs(normal_dir)
        os.makedirs(paralysis_dir)
        
        # 创建测试图像
        for i in range(5):
            # 正常图像
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            image.save(os.path.join(normal_dir, f'normal_{i:03d}.jpg'))
            
            # 面瘫图像
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            image.save(os.path.join(paralysis_dir, f'paralysis_{i:03d}.jpg'))
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = YFPFacialParalysisDataset(self.temp_dir, transform=image_transforms)
        
        # 检查数据集大小
        self.assertEqual(len(dataset), 10)
        
        # 检查数据加载
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label.item(), [0, 1])
    
    def test_model_creation(self):
        """测试模型创建"""
        model = InterviewDetectionModel()
        
        # 检查模型结构
        self.assertIsInstance(model, torch.nn.Module)
        
        # 检查模型输出
        dummy_input = torch.randn(1, 3, 224, 224)
        interview_output, emotion_output, gaze_output, au_output = model(dummy_input)
        
        self.assertEqual(interview_output.shape, (1, 2))  # 二分类
        self.assertEqual(emotion_output.shape, (1, 8))   # 8类情感
        self.assertEqual(gaze_output.shape, (1, 2))     # 2D gaze
        self.assertEqual(au_output.shape, (1, 8))       # 8个AU
    
    def test_metrics_calculation(self):
        """测试指标计算"""
        # 创建模拟数据
        true_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        pred_labels = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        
        metrics = calculate_metrics(true_labels, pred_labels)
        
        # 检查指标是否存在且在合理范围内
        required_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'specificity']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        
        # 检查混淆矩阵
        self.assertIn('confusion_matrix', metrics)
        self.assertEqual(len(metrics['confusion_matrix']), 2)
        self.assertEqual(len(metrics['confusion_matrix'][0]), 2)
    
    def test_training_step(self):
        """测试训练步骤"""
        model = InterviewDetectionModel()
        model.to(self.device)
        
        # 创建测试数据
        dummy_input = torch.randn(4, 3, 224, 224).to(self.device)
        dummy_labels = torch.randint(0, 2, (4,)).to(self.device)
        
        # 前向传播
        interview_output, _, _, _ = model(dummy_input)
        
        # 计算损失
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(interview_output, dummy_labels)
        
        # 反向传播
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 检查损失是否为数值
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_prediction(self):
        """测试预测功能"""
        model = InterviewDetectionModel()
        model.to(self.device)
        model.eval()
        
        # 创建测试输入
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            interview_output, _, _, _ = model(dummy_input)
            
            # 获取预测结果
            probabilities = torch.softmax(interview_output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0, predicted_class]
            
            # 检查预测结果
            self.assertIn(predicted_class.item(), [0, 1])
            self.assertGreaterEqual(confidence.item(), 0.0)
            self.assertLessEqual(confidence.item(), 1.0)
    
    def test_data_loading(self):
        """测试数据加载"""
        dataset = YFPFacialParalysisDataset(self.temp_dir, transform=image_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 测试数据加载
        for images, labels in dataloader:
            self.assertEqual(images.shape[0], 2)  # batch size
            self.assertEqual(images.shape[1:], (3, 224, 224))  # image shape
            self.assertEqual(labels.shape[0], 2)  # batch size
            break  # 只测试第一个batch


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试初始化"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("无法导入必要的模块")
        
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_dataset()
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """创建测试数据集"""
        normal_dir = os.path.join(self.temp_dir, 'normal')
        paralysis_dir = os.path.join(self.temp_dir, 'paralysis')
        os.makedirs(normal_dir)
        os.makedirs(paralysis_dir)
        
        for i in range(3):
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            image.save(os.path.join(normal_dir, f'normal_{i:03d}.jpg'))
            image.save(os.path.join(paralysis_dir, f'paralysis_{i:03d}.jpg'))
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 1. 创建数据集
        dataset = YFPFacialParalysisDataset(self.temp_dir, transform=image_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 2. 创建模型
        model = InterviewDetectionModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 3. 训练几个步骤
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            interview_output, _, _, _ = model(images)
            loss = criterion(interview_output, labels)
            loss.backward()
            optimizer.step()
        
        # 4. 评估
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                interview_output, _, _, _ = model(images)
                preds = torch.argmax(interview_output, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 5. 计算指标
        metrics = calculate_metrics(all_labels, all_preds)
        
        # 检查流程是否完成
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)


def run_basic_tests():
    """运行基本测试"""
    print("运行基本功能测试...")
    print("="*50)
    
    # 检查导入
    if not IMPORTS_AVAILABLE:
        print("❌ 导入测试失败 - 无法导入必要模块")
        return False
    
    print("✅ 导入测试通过")
    
    # 检查PyTorch
    try:
        x = torch.randn(1, 3, 224, 224)
        print("✅ PyTorch基本操作测试通过")
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False
    
    # 检查其他依赖
    try:
        import sklearn
        import matplotlib
        import seaborn
        print("✅ 其他依赖包测试通过")
    except ImportError as e:
        print(f"❌ 依赖包测试失败: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("面谈检测系统测试")
    print("Interview Detection System Tests")
    print("="*60)
    
    # 运行基本测试
    if not run_basic_tests():
        print("\n基本测试失败，请检查依赖包安装")
        return
    
    print("\n运行单元测试...")
    print("="*50)
    
    # 运行单元测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestInterviewDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("🎉 所有测试通过！系统运行正常。")
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        print("❌ Some tests failed, please check the error messages.")
        print(f"失败的测试数量: {len(result.failures)}")
        print(f"错误的测试数量: {len(result.errors)}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
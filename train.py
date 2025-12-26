import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# 导入自定义模块
from config import cfg
from models import create_dual_transfer_model
from losses import EEGLoss
from metrics import EEGEvaluator
from trainer_utils import EarlyStopping, build_optimizer_and_scheduler


'''
核心功能覆盖
✅ 3D-CNN+ViT 双迁移模型
✅ EEG 精细化空域重采样增强
✅ EEG Embedding 层
✅ 域适应（DANN）+ 梯度反转层
✅ 加权焦点损失（带标签平滑）+ 组合损失
✅ 完整评估指标（准确率、F1、灵敏度、特异度、混淆矩阵）
✅ 早停机制
✅ 学习率调度器（ReduceLROnPlateau/CosineAnnealing/StepLR）
✅ 梯度监控 + 参数统计
✅ 梯度裁剪 + 数据归一化
✅ 模型保存 / 加载（兼容多 GPU）

'''

# ===================== 模拟EEG数据集（替换为你的真实数据集） =====================
class MockEEGDataset(Dataset):
    """模拟EEG数据集（用于测试代码）"""

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = np.random.randn(num_samples, cfg.eeg_time_steps, *cfg.eeg_spatial_size).astype(np.float32)
        self.labels = np.random.randint(0, cfg.num_classes, num_samples)
        self.domain_labels = np.random.randint(0, cfg.num_domains, num_samples)

        # 数据归一化
        self.data = (self.data - self.data.mean(axis=1, keepdims=True)) / (self.data.std(axis=1, keepdims=True) + 1e-8)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx]), torch.tensor(self.domain_labels[idx])


# ===================== 训练主函数 =====================
def main():
    # 1. 初始化数据集和数据加载器
    train_dataset = MockEEGDataset(num_samples=800)
    val_dataset = MockEEGDataset(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # 2. 初始化模型
    model = create_dual_transfer_model(pretrained_cnn3d_path=cfg.pretrained_cnn3d_path)
    model.count_params(save_path=cfg.params_stats_path)  # 统计参数

    # 3. 初始化损失函数、优化器、调度器、早停
    criterion = EEGLoss()
    criterion.update_class_weights(train_dataset)  # 更新类别权重
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    early_stopping = EarlyStopping()

    # 4. 训练循环
    for epoch in range(cfg.max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y, domain_label) in enumerate(train_loader):
            x, y, domain_label = x.to(cfg.device), y.to(cfg.device), domain_label.to(cfg.device)

            optimizer.zero_grad()

            # 前向传播
            cls_out, domain_out = model(x, domain_label=domain_label)

            # 计算损失
            feat_3d = model.module.cnn3d(x) if isinstance(model, nn.DataParallel) else model.cnn3d(x)
            vit_feat = model.module.vit(torch.randn(x.shape[0], 3, 224, 224).to(cfg.device), return_layer_features=True)[1][-1]
            vit_feat = vit_feat.to(cfg.device)

            loss, loss_dict = criterion(
                cls_out=cls_out,
                y_true=y,
                domain_out=domain_out,
                domain_label=domain_label,
                feat_3d=feat_3d,
                feat_vit=vit_feat
            )

            # 反向传播 + 梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item()

            # 打印训练日志
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{cfg.max_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss_dict['total_loss']:.4f}, ClsLoss: {loss_dict['cls_loss']:.4f}, "
                      f"DomainLoss: {loss_dict['domain_loss']:.4f}, AlignLoss: {loss_dict['align_loss']:.4f}")

        # 验证阶段
        model.eval()
        val_evaluator = EEGEvaluator()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, domain_label in val_loader:
                x, y, domain_label = x.to(cfg.device), y.to(cfg.device), domain_label.to(cfg.device)

                cls_out, domain_out = model(x, domain_label=domain_label)
                loss, loss_dict = criterion(
                    cls_out=cls_out,
                    y_true=y,
                    domain_out=domain_out,
                    domain_label=domain_label,
                    feat_3d=model.module.cnn3d(x) if isinstance(model, nn.DataParallel) else model.cnn3d(x),
                    feat_vit = model.module.vit(torch.randn(x.shape[0], 3, 224, 224).to(cfg.device), return_layer_features=True)[1][-1].to(cfg.device)
                )

                val_loss += loss.item()
                val_evaluator.update(cls_out, y)

        # 计算验证指标
        val_loss /= len(val_loader)
        val_metrics = val_evaluator.compute_metrics()
        val_f1 = val_metrics["macro_f1"]

        # 打印验证日志
        print(
            f"\nEpoch [{epoch + 1}/{cfg.max_epochs}], Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_metrics['accuracy']:.2%}")
        val_evaluator.print_report()

        # 更新学习率调度器
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        # 早停判断
        early_stopping(val_f1, model, epoch)
        if early_stopping.early_stop:
            break

    # 5. 训练后处理
    # 加载最佳模型
    model = early_stopping.load_best_model(model)

    # 保存梯度日志和曲线
    if hasattr(model, "module"):
        model.module.save_grad_logs()
        model.module.plot_grad_curves()
        model.module.remove_grad_hooks()
    else:
        model.save_grad_logs()
        model.plot_grad_curves()
        model.remove_grad_hooks()

    # 绘制混淆矩阵
    val_evaluator.plot_confusion_matrix()

    print("\n训练完成！所有结果已保存至指定路径。")


if __name__ == "__main__":
    main()
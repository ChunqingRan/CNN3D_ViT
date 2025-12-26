import torch

# 基础配置
cfg = {
    # 设备
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # 数据相关
    "eeg_time_steps": 588,          # EEG时间步长（匹配论文）
    "eeg_spatial_size": (7, 9),     # EEG电极空间尺寸（H×W）
    "num_classes": 2,               # 分类类别数（示例：正常/癫痫）
    "class_names": ["正常", "癫痫"], # 类别名称（用于评估）
    # 3D-CNN配置
    "cnn3d_in_channels": 1,
    "cnn3d_out_features": 196,
    # ViT配置
    "vit_model_name": "vit_base_patch16_224",
    # 训练配置
    "max_epochs": 100,
    "lr": 1e-4,
    "lr_cnn3d": 1e-4,
    "lr_vit": 1e-5,
    "weight_decay": 1e-5,
    "scheduler_type": "ReduceLROnPlateau",  # 学习率调度器类型
    "min_lr": 1e-6,
    # 数据增强配置
    "aug_prob": 0.5,
    "noise_level": 0.01,
    "time_shift_range": 5,
    # 损失函数配置
    "cls_weight": 1.0,
    "domain_weight": 0.1,
    "align_weight": 0.05,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
    # 早停配置
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "early_stop_monitor": "val_f1",
    "early_stop_mode": "max",
    "best_model_path": "best_eeg_model.pth",
    # 域适应配置
    "num_domains": 2,
    "domain_hidden_dim": 512,
    # 梯度监控
    "grad_monitor_layers": ["cnn3d.conv3d_2", "vit.vit.blocks.11.attn", "cnn3d.fc_layers.4"],
    # 路径配置
    "pretrained_cnn3d_path": None,
    "params_stats_path": "params_stats.json",
    "grad_logs_path": "grad_logs.json",
    "grad_curves_path": "grad_curves.png",
    "val_cm_path": "val_confusion_matrix.png",
}

# 转换为类（方便属性访问）
class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

# 全局配置实例
cfg = Config(cfg)
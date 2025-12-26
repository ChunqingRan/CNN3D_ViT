import torch
import os
from config import cfg


class EarlyStopping:
    """早停机制"""

    def __init__(self):
        self.patience = cfg.early_stop_patience
        self.min_delta = cfg.early_stop_min_delta
        self.monitor = cfg.early_stop_monitor
        self.mode = cfg.early_stop_mode
        self.save_path = cfg.best_model_path
        self.verbose = True

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, model, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self._save_best_model(model, epoch)
            return

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            self._save_best_model(model, epoch)
            if self.verbose:
                print(f"✅ 监控指标改进 ({self.monitor}): {current_score:.6f} → 保存最佳模型（Epoch {epoch}）")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"⚠️ 早停计数器: {self.counter}/{self.patience} (当前{self.monitor}: {current_score:.6f}, 最佳: {self.best_score:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 早停触发！最佳Epoch: {self.best_epoch}, 最佳{self.monitor}: {self.best_score:.6f}")

    def _save_best_model(self, model, epoch):
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict() if not isinstance(model,
                                                                     nn.DataParallel) else model.module.state_dict(),
            "best_score": self.best_score,
            "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("__")}
        }
        torch.save(save_dict, self.save_path)
        if self.verbose:
            print(f"📌 最佳模型已保存至: {self.save_path}")

    def load_best_model(self, model):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=cfg.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if self.verbose:
                print(f"📥 加载最佳模型（Epoch {checkpoint['epoch']}, {self.monitor}: {checkpoint['best_score']:.6f}）")
            return model
        else:
            raise FileNotFoundError(f"最佳模型文件不存在: {self.save_path}")


def build_optimizer_and_scheduler(model):
    """构建优化器和学习率调度器"""
    # 分层学习率
    params = model.get_learning_rates()

    # 优化器
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999)
    )

    # 调度器
    if cfg.scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_epochs,
            eta_min=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.7,
            verbose=True
        )
    else:
        scheduler = None

    return optimizer, scheduler
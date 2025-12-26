import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class EEGWeightedLoss(nn.Module):
    """EEG加权焦点损失（带标签平滑）"""

    def __init__(self):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.focal_gamma = cfg.focal_gamma
        self.label_smoothing = cfg.label_smoothing
        self.device = cfg.device
        self.class_weights = torch.ones(self.num_classes, device=self.device)

    def update_class_weights(self, dataset):
        class_counts = torch.bincount(torch.tensor(dataset.labels))
        self.class_weights = 1.0 / (class_counts.float() / class_counts.sum())
        self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes
        self.class_weights = self.class_weights.to(self.device)
        print(f"更新EEG类别权重：{self.class_weights.cpu().numpy()}")

    def forward(self, logits, targets):
        # 标签平滑
        one_hot = F.one_hot(targets, self.num_classes).float()
        one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        # 焦点损失
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        focal_weight = (1 - probs) ** self.focal_gamma

        # 类别加权
        class_weight = self.class_weights[targets]

        # 组合损失
        ce_loss = -one_hot * torch.log(probs)
        focal_ce_loss = ce_loss * focal_weight
        weighted_loss = focal_ce_loss.sum(dim=-1) * class_weight
        loss = weighted_loss.mean()

        return loss


class EEGLoss(nn.Module):
    """组合损失：分类损失 + 域损失 + 特征对齐损失"""

    def __init__(self):
        super().__init__()
        self.cls_criterion = EEGWeightedLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.align_criterion = nn.MSELoss()
        self.cls_weight = cfg.cls_weight
        self.domain_weight = cfg.domain_weight
        self.align_weight = cfg.align_weight

    def update_class_weights(self, dataset):
        self.cls_criterion.update_class_weights(dataset)

    def forward(self, cls_out, y_true, domain_out=None, domain_label=None, feat_3d=None, feat_vit=None):
        # 分类损失
        cls_loss = self.cls_criterion(cls_out, y_true)

        # 域损失
        domain_loss = 0.0 if domain_out is None else self.domain_criterion(domain_out, domain_label)

        # 特征对齐损失
        align_loss = 0.0
        if feat_3d is not None and feat_vit is not None:
            feat_3d_norm = F.normalize(feat_3d.reshape(feat_3d.shape[0], -1), dim=1)
            feat_vit_norm = F.normalize(feat_vit[:, 0, :], dim=1)
            align_loss = self.align_criterion(feat_3d_norm, feat_vit_norm)

        # 总损失
        total_loss = self.cls_weight * cls_loss + self.domain_weight * domain_loss + self.align_weight * align_loss

        return total_loss, {
            "cls_loss": cls_loss.item(),
            "domain_loss": domain_loss.item(),
            "align_loss": align_loss.item(),
            "total_loss": total_loss.item()
        }
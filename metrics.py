import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, recall_score, precision_score
)
from config import cfg


class EEGEvaluator:
    """EEG分类任务评估器"""

    def __init__(self):
        self.num_classes = cfg.num_classes
        self.class_names = cfg.class_names
        self.all_preds = []
        self.all_targets = []

    def update(self, preds, targets):
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        if preds.ndim == 2:
            preds = np.argmax(preds, axis=1)
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)

    def compute_metrics(self):
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # 基础指标
        accuracy = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average='macro')
        micro_f1 = f1_score(targets, preds, average='micro')
        weighted_f1 = f1_score(targets, preds, average='weighted')

        # 医疗任务指标
        if self.num_classes == 2:
            sensitivity = recall_score(targets, preds, pos_label=1)
            specificity = recall_score(targets, preds, pos_label=0)
            precision = precision_score(targets, preds, pos_label=1)
        else:
            sensitivity = recall_score(targets, preds, average='macro')
            specificity = None
            precision = precision_score(targets, preds, average='macro')

        # 混淆矩阵
        cm = confusion_matrix(targets, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "sensitivity": round(sensitivity, 4) if sensitivity else None,
            "specificity": round(specificity, 4) if specificity else None,
            "precision": round(precision, 4),
            "confusion_matrix": cm,
            "confusion_matrix_norm": cm_norm
        }

    def plot_confusion_matrix(self, save_path=cfg.val_cm_path):
        metrics = self.compute_metrics()
        cm = metrics["confusion_matrix_norm"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("EEG分类混淆矩阵")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵已保存至：{save_path}")

    def print_report(self):
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        metrics = self.compute_metrics()

        print("=" * 80)
        print("EEG分类任务评估报告")
        print("=" * 80)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.2%}")
        print(f"宏F1 (Macro F1): {metrics['macro_f1']:.4f}")
        print(f"微F1 (Micro F1): {metrics['micro_f1']:.4f}")
        print(f"加权F1 (Weighted F1): {metrics['weighted_f1']:.4f}")
        if metrics["sensitivity"]:
            print(f"灵敏度 (Sensitivity): {metrics['sensitivity']:.4f}")
        if metrics["specificity"]:
            print(f"特异度 (Specificity): {metrics['specificity']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print("-" * 80)
        print("详细分类报告：")
        print(classification_report(targets, preds, target_names=self.class_names))
        print("=" * 80)

    def reset(self):
        self.all_preds = []
        self.all_targets = []
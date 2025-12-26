import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import Attention, Mlp

warnings.filterwarnings("ignore")
from config import cfg


# ===================== 工具函数：参数统计（增强版） =====================
def count_params_module(model, module_name, prefix=""):
    module_params = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "sub_modules": {}
    }
    target_module = None
    for name, mod in model.named_modules():
        if name == module_name:
            target_module = mod
            break
    if target_module is None:
        raise ValueError(f"模块 {module_name} 不存在于模型中")

    for name, param in target_module.named_parameters(prefix=prefix):
        param_num = param.numel()
        module_params["total"] += param_num
        if param.requires_grad:
            module_params["trainable"] += param_num
        else:
            module_params["non_trainable"] += param_num

        sub_mod_name = name.split(".")[0] if "." in name else "root"
        if sub_mod_name not in module_params["sub_modules"]:
            module_params["sub_modules"][sub_mod_name] = {
                "total": 0, "trainable": 0, "non_trainable": 0
            }
        module_params["sub_modules"][sub_mod_name]["total"] += param_num
        if param.requires_grad:
            module_params["sub_modules"][sub_mod_name]["trainable"] += param_num
        else:
            module_params["sub_modules"][sub_mod_name]["non_trainable"] += param_num

    return module_params


def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"


def print_params_summary(model, save_path=None):
    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_all = total_all - trainable_all

    module_stats = {}
    if isinstance(model, EEG3DCNN):
        module_stats["3dcnn_conv"] = count_params_module(model, "conv3d_1")
        module_stats["3dcnn_fc"] = count_params_module(model, "fc_layers")
    elif isinstance(model, CustomViT):
        module_stats["vit_patch_embed"] = count_params_module(model, "vit.patch_embed")
        module_stats["vit_blocks"] = count_params_module(model, "vit.blocks")
        module_stats["vit_head"] = count_params_module(model, "vit.head")
    elif isinstance(model, DualTransferModel):
        module_stats["3dcnn"] = count_params_module(model, "3dcnn")
        module_stats["vit"] = count_params_module(model, "vit")

    print("=" * 80)
    print("模型参数统计摘要（匹配论文表1）")
    print("=" * 80)
    print(
        f"全局总参数: {format_params(total_all)} (可训练: {format_params(trainable_all)}, 冻结: {format_params(non_trainable_all)})")
    print(f"可训练参数占比: {trainable_all / total_all:.2%}")
    print("-" * 80)

    for mod_name, mod_stats in module_stats.items():
        print(f"\n【模块: {mod_name}】")
        print(f"  总参数: {format_params(mod_stats['total'])}")
        print(f"  可训练: {format_params(mod_stats['trainable'])} ({mod_stats['trainable'] / mod_stats['total']:.2%})")
        print(f"  冻结: {format_params(mod_stats['non_trainable'])}")
        print("  子模块分布:")
        for sub_mod, sub_stats in mod_stats["sub_modules"].items():
            print(
                f"    - {sub_mod}: {format_params(sub_stats['total'])} (可训练: {format_params(sub_stats['trainable'])})")

    print("=" * 80)

    if save_path:
        stats_dict = {
            "global": {
                "total": total_all,
                "trainable": trainable_all,
                "non_trainable": non_trainable_all,
                "trainable_ratio": float(trainable_all / total_all)
            },
            "modules": {k: {
                "total": v["total"],
                "trainable": v["trainable"],
                "non_trainable": v["non_trainable"],
                "sub_modules": v["sub_modules"]
            } for k, v in module_stats.items()}
        }
        with open(save_path, "w") as f:
            json.dump(stats_dict, f, indent=4)
        print(f"\n参数统计结果已保存至: {save_path}")


# ===================== 梯度监控器 =====================
class GradientMonitor:
    def __init__(self, model, target_layers=None):
        self.model = model
        self.grad_logs = {}
        self.hooks = []
        self.target_layers = target_layers or cfg.grad_monitor_layers

        for layer_name in self.target_layers:
            self.grad_logs[layer_name] = []

    def _grad_hook_fn(self, layer_name):
        def hook(grad):
            grad_norm = torch.norm(grad).item()
            grad_mean = grad.mean().item()
            grad_max = grad.max().item()
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            step = len(self.grad_logs[layer_name])
            self.grad_logs[layer_name].append({
                "step": step,
                "norm": grad_norm,
                "mean": grad_mean,
                "max": grad_max,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "grad_shape": list(grad.shape)
            })

            if has_nan or has_inf:
                warnings.warn(f"⚠️ 层 {layer_name} 梯度出现NaN/Inf！step={step}")
            return grad

        return hook

    def register_hooks(self):
        for layer_name in self.target_layers:
            layer = self.model
            try:
                for sub_name in layer_name.split("."):
                    layer = getattr(layer, sub_name)
                hook = layer.register_backward_hook(
                    lambda module, grad_in, grad_out, ln=layer_name: self._grad_hook_fn(ln)(grad_out[0])
                )
                self.hooks.append(hook)
            except AttributeError:
                warnings.warn(f"层 {layer_name} 不存在，跳过梯度监控")
        print(f"✅ 已为 {len(self.hooks)} 个层注册梯度钩子")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("✅ 已移除所有梯度钩子")

    def save_grad_logs(self, save_path=cfg.grad_logs_path):
        def serialize_log(log):
            for k, v in log.items():
                if isinstance(v, (np.integer, np.floating)):
                    log[k] = float(v)
            return log

        serialized_logs = {
            layer: [serialize_log(log) for log in logs]
            for layer, logs in self.grad_logs.items()
        }
        with open(save_path, "w") as f:
            json.dump(serialized_logs, f, indent=4)
        print(f"✅ 梯度日志已保存至: {save_path}")

    def plot_grad_curves(self, save_path=cfg.grad_curves_path):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        metrics = ["norm", "mean", "max"]
        titles = ["梯度L2范数", "梯度均值", "梯度最大值"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            for layer_name, logs in self.grad_logs.items():
                if len(logs) == 0:
                    continue
                steps = [log["step"] for log in logs]
                values = [log[metric] for log in logs]
                ax.plot(steps, values, label=layer_name, linewidth=2)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("训练步数", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 梯度曲线已保存至: {save_path}")


# ===================== EEG数据增强（含精细化空域重采样） =====================
class EEGSpatialResample(nn.Module):
    def __init__(
            self,
            spatial_size=cfg.eeg_spatial_size,
            resample_modes=["scale", "local", "perturb"],
            scale_range=(0.9, 1.1),
            local_region_size=(3, 3),
            perturb_range=(-1, 1),
            aug_prob=cfg.aug_prob
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.resample_modes = resample_modes
        self.scale_range = scale_range
        self.local_region_size = local_region_size
        self.perturb_range = perturb_range
        self.aug_prob = aug_prob

    def _scale_resample(self, x):
        B, T, H, W = x.shape
        scale_h = np.random.uniform(*self.scale_range)
        scale_w = np.random.uniform(*self.scale_range)
        new_H, new_W = int(H * scale_h), int(W * scale_w)

        x_rescaled = F.interpolate(
            x.transpose(1, 2),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).transpose(1, 2)

        if new_H > H:
            start_h = (new_H - H) // 2
            x_rescaled = x_rescaled[:, :, start_h:start_h + H, :]
        else:
            pad_h = H - new_H
            x_rescaled = F.pad(x_rescaled, (0, 0, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)

        if new_W > W:
            start_w = (new_W - W) // 2
            x_rescaled = x_rescaled[:, :, :, start_w:start_w + W]
        else:
            pad_w = W - new_W
            x_rescaled = F.pad(x_rescaled, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode='constant', value=0)

        return x_rescaled

    def _local_resample(self, x):
        B, T, H, W = x.shape
        center_h = np.random.randint(self.local_region_size[0] // 2, H - self.local_region_size[0] // 2)
        center_w = np.random.randint(self.local_region_size[1] // 2, W - self.local_region_size[1] // 2)

        h_start = center_h - self.local_region_size[0] // 2
        h_end = center_h + self.local_region_size[0] // 2 + 1
        w_start = center_w - self.local_region_size[1] // 2
        w_end = center_w + self.local_region_size[1] // 2 + 1
        local_region = x[:, :, h_start:h_end, w_start:w_end]

        local_resampled = F.interpolate(
            local_region.transpose(1, 2),
            size=(self.local_region_size[0] // 2, self.local_region_size[1] // 2),
            mode='bilinear',
            align_corners=False
        )
        local_resampled = F.interpolate(
            local_resampled,
            size=self.local_region_size,
            mode='bilinear',
            align_corners=False
        ).transpose(1, 2)

        x_local = x.clone()
        x_local[:, :, h_start:h_end, w_start:w_end] = local_resampled
        return x_local

    def _perturb_resample(self, x):
        B, T, H, W = x.shape
        perturb_h = np.random.randint(*self.perturb_range, size=(H, W))
        perturb_w = np.random.randint(*self.perturb_range, size=(H, W))

        grid_h, grid_w = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        grid_h = (grid_h + torch.tensor(perturb_h, device=x.device)) / (H - 1) * 2 - 1
        grid_w = (grid_w + torch.tensor(perturb_w, device=x.device)) / (W - 1) * 2 - 1
        grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        x_perturb = F.grid_sample(
            x.transpose(1, 2),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).transpose(1, 2)
        return x_perturb

    def forward(self, x):
        if not self.training:
            return x

        mode = np.random.choice(self.resample_modes)
        if mode == "scale" and np.random.rand() < self.aug_prob:
            x = self._scale_resample(x)
        elif mode == "local" and np.random.rand() < self.aug_prob:
            x = self._local_resample(x)
        elif mode == "perturb" and np.random.rand() < self.aug_prob:
            x = self._perturb_resample(x)
        return x


class EEGDataAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug_prob = cfg.aug_prob
        self.noise_level = cfg.noise_level
        self.time_shift_range = cfg.time_shift_range
        self.spatial_resample = EEGSpatialResample()

    def time_warp(self, x):
        if torch.rand(1) < self.aug_prob:
            T = x.shape[1]
            scale = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            new_T = int(T * scale)
            x = F.interpolate(x.transpose(1, 3), size=new_T, mode='linear', align_corners=False)
            x = x.transpose(1, 3)
            x = x[:, :T, :, :] if new_T > T else F.pad(x, (0, 0, 0, 0, 0, T - new_T), mode='constant', value=0)
        return x

    def noise_injection(self, x):
        if torch.rand(1) < self.aug_prob:
            x = x + torch.randn_like(x) * self.noise_level
        return x

    def time_shift(self, x):
        if torch.rand(1) < self.aug_prob:
            shift = torch.randint(-self.time_shift_range, self.time_shift_range + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=1)
            x[:, :shift, :, :] = 0 if shift > 0 else x[:, shift:, :, :] = 0
        return x

    def frequency_mask(self, x):
        if torch.rand(1) < self.aug_prob:
            x_fft = torch.fft.fft(x, dim=1)
            freq_start = torch.randint(0, x.shape[1] // 2, (1,)).item()
            freq_len = torch.randint(1, x.shape[1] // 10, (1,)).item()
            x_fft[:, freq_start:freq_start + freq_len, :, :] = 0
            x = torch.fft.ifft(x_fft, dim=1).real
        return x

    def forward(self, x):
        if self.training:
            x = self.time_warp(x)
            x = self.spatial_resample(x)
            x = self.noise_injection(x)
            x = self.time_shift(x)
            x = self.frequency_mask(x)
        return x


# ===================== EEG Embedding层 =====================
class EEGEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.channel_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.time_pos_embed = nn.Parameter(torch.randn(1, embed_dim, cfg.eeg_time_steps, 1, 1) * 0.02)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, *cfg.eeg_spatial_size) * 0.02)
        self.mlp = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.channel_embed.weight)
        for m in self.mlp:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.channel_embed(x.squeeze(2)).unsqueeze(2)
        x = x + self.time_pos_embed + self.spatial_pos_embed
        x = self.mlp(x)
        return x


# ===================== 3D/2D卷积块 =====================
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.9, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

        if self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.residual_bn = nn.BatchNorm3d(out_channels)
        elif self.use_residual:
            self.residual_conv = nn.Identity()
            self.residual_bn = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.act(x)
        if self.use_residual:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
            x = x + residual
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.2)

        if self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.residual_bn = nn.BatchNorm2d(out_channels)
        elif self.use_residual:
            self.residual_conv = nn.Identity()
            self.residual_bn = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.pool(x)
        if self.use_residual:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
            residual = self.pool(residual)
            x = x + residual
        return x


# ===================== EEG 3D-CNN =====================
class EEGCNN3D(nn.Module):
    def __init__(self, time_pooling_type="mean"):
        super().__init__()
        self.in_channels = cfg.cnn3d_in_channels
        self.out_features = cfg.cnn3d_out_features
        self.time_pooling_type = time_pooling_type

        # EEG Embedding层
        self.embedding = EEGEmbedding(in_channels=self.in_channels, embed_dim=64)

        # 3D卷积层
        self.conv3d_1 = Conv3DBlock(
            in_channels=64,  # 匹配Embedding输出维度
            out_channels=16,
            kernel_size=(3, 1, 3),
            padding=(1, 0, 1),
            use_residual=True
        )
        self.conv3d_2 = Conv3DBlock(
            in_channels=16,
            out_channels=25,
            kernel_size=(4, 4, 1),
            padding=0,
            use_residual=True
        )

        # 2D卷积层
        self.conv2d_block = Conv2DBlock(
            in_channels=25,
            out_channels=40,
            kernel_size=(1, 10),
            padding=0,
            use_residual=True
        )

        # 全连接层
        self.fc_input_dim = 40 * 1 * 89
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.out_features),
            nn.BatchNorm1d(self.out_features),
            nn.ReLU(inplace=True)
        )

        # 梯度钩子
        self.grad_hooks = {}
        self._register_grad_hooks()

        # 权重初始化
        self._init_weights()

    def _register_grad_hooks(self):
        def grad_hook_fn(grad):
            self.grad_norm = torch.norm(grad).item()
            return grad

        self.grad_hooks["conv3d_2"] = self.conv3d_2.register_backward_hook(
            lambda module, grad_input, grad_output: grad_hook_fn(grad_output[0])
        )
        self.grad_hooks["fc_layers"] = self.fc_layers[0].register_backward_hook(
            lambda module, grad_input, grad_output: grad_hook_fn(grad_output[0])
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Dropout3d, nn.Dropout2d, nn.Dropout)):
                m.p = 0.2 if "3d" in str(m) or "2d" in str(m) else 0.5

    def _time_dim_pooling(self, x):
        if self.time_pooling_type == "mean":
            return torch.mean(x, dim=2)
        elif self.time_pooling_type == "max":
            return torch.max(x, dim=2)[0]
        elif self.time_pooling_type == "adaptive":
            return F.adaptive_avg_pool1d(x.transpose(2, 1), 1).squeeze(1)
        else:
            raise ValueError(f"不支持的时间池化类型：{self.time_pooling_type}")

    def forward(self, x, return_intermediate=False):
        x = x.unsqueeze(1)
        intermediate_features = {"input_cnn3d": x.detach().cpu()}

        # EEG Embedding
        x = self.embedding(x)
        intermediate_features["eeg_embedding"] = x.detach().cpu()

        # 3D卷积
        x = self.conv3d_1(x)
        intermediate_features["conv3d_1"] = x.detach().cpu()
        x = self.conv3d_2(x)
        intermediate_features["conv3d_2"] = x.detach().cpu()

        # 时间池化
        x = self._time_dim_pooling(x)
        intermediate_features["time_pooled"] = x.detach().cpu()

        # 2D卷积
        x = self.conv2d_block(x)
        intermediate_features["conv2d_out"] = x.detach().cpu()

        # 全连接
        x = self.fc_layers(x)
        intermediate_features["fc_out"] = x.detach().cpu()

        # 重塑输出
        feature_map = x.view(-1, 4, 4, self.out_features)
        intermediate_features["final_feature"] = feature_map.detach().cpu()

        if return_intermediate:
            return feature_map, intermediate_features
        return feature_map

    def count_params(self, save_path=None):
        print_params_summary(self, save_path)


# ===================== 定制化ViT =====================
class CustomViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            cfg.vit_model_name,
            pretrained=True,
            num_classes=cfg.num_classes
        )

        # 定制化位置编码
        self.eeg_pos_emb = nn.Parameter(
            torch.randn(1, self.vit.pos_embed.shape[1], self.vit.pos_embed.shape[2]) * 0.02
        )
        self.vit.pos_embed.requires_grad = False
        self.eeg_pos_emb.requires_grad = True

        # 注意力掩码
        self.attn_mask = self._generate_eeg_attn_mask()
        self.extract_layer = -3

    def _generate_eeg_attn_mask(self):
        mask = torch.ones(197, 197)
        edge_patches = [i for i in range(197) if i % 14 < 2 or i % 14 > 11 or i // 14 < 2 or i // 14 > 11]
        mask[edge_patches, :] *= 0.5
        mask[:, edge_patches] *= 0.5
        return mask.to(cfg.device)

    def forward(self, x, return_layer_features=False):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 替换位置编码
        x = x + self.eeg_pos_emb

        # Transformer编码器
        x = self.vit.pos_drop(x)
        layer_features = []
        for idx, block in enumerate(self.vit.blocks):
            x = block(x, attn_mask=self.attn_mask)
            if idx >= len(self.vit.blocks) + self.extract_layer:
                layer_features.append(x.detach().cpu())

        # 分类头
        x = self.vit.norm(x)
        cls_out = x[:, 0]
        out = self.vit.head(cls_out)

        if return_layer_features:
            return out, layer_features
        return out

    def freeze_layers(self, freeze_until=-3):
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        for idx, block in enumerate(self.vit.blocks):
            if idx < len(self.vit.blocks) + freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def count_params(self, save_path=None):
        print_params_summary(self, save_path)


# ===================== 梯度反转层 + 域判别器 =====================
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_reverse = GradientReversal(alpha=1.0)
        self.fc = nn.Sequential(
            nn.Linear(cfg.cnn3d_out_features + 768, cfg.domain_hidden_dim),
        nn.BatchNorm1d(cfg.domain_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(cfg.domain_hidden_dim, cfg.domain_hidden_dim // 2),
        nn.BatchNorm1d(cfg.domain_hidden_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(cfg.domain_hidden_dim // 2, cfg.num_domains)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.grad_reverse(x)
        return self.fc(x)


# ===================== 双迁移学习联合模型 =====================
class DualTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 数据增强
        self.eeg_aug = EEGDataAugmentation()

        # 3D-CNN模块
        self.cnn3d = EEGCNN3D(time_pooling_type="mean")
        self.freeze_cnn3d(layers_to_freeze="bottom")

        # ViT模块
        self.vit = CustomViT()
        self.vit.freeze_layers(freeze_until=-3)

        # 域适应模块
        self.domain_discriminator = DomainDiscriminator()

        # 梯度监控
        self.grad_monitor = GradientMonitor(self)
        self.grad_monitor.register_hooks()

    def _load_cnn3d_pretrained(self, pretrained_path):
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location=cfg.device)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.cnn3d.load_state_dict(state_dict, strict=False)
                print(f"成功加载预训练3D-CNN权重：{pretrained_path}")

                conv3d_weight = self.cnn3d.conv3d_2.conv.weight
                if torch.all(conv3d_weight == 0):
                    warnings.warn("3D-CNN关键层权重加载失败！")
                else:
                    print("3D-CNN权重加载验证通过")
            except Exception as e:
                warnings.warn(f"加载3D-CNN预训练权重失败：{e}")
        elif pretrained_path:
            warnings.warn(f"预训练3D-CNN权重文件不存在：{pretrained_path}")

    def freeze_cnn3d(self, layers_to_freeze="bottom"):
        params = list(self.cnn3d.parameters())
        if layers_to_freeze == "bottom":
            freeze_idx = int(len(params) * 0.8)
            for idx, param in enumerate(params):
                param.requires_grad = idx >= freeze_idx
        elif layers_to_freeze == "all":
            for param in self.cnn3d.parameters():
                param.requires_grad = False
        elif layers_to_freeze == "none":
            for param in self.cnn3d.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.cnn3d.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.cnn3d.parameters())
        print(f"3D-CNN冻结完成：可训练参数 {trainable}/{total} ({trainable / total:.2%})")

    def compute_feature_correlation(self, feat_3d, feat_vit):
        feat_3d_flat = feat_3d.reshape(feat_3d.shape[0], -1).cpu().numpy()
        feat_vit_flat = feat_vit[:, 0, :].cpu().numpy()
        self.feature_corr = np.corrcoef(feat_3d_flat.T, feat_vit_flat.T)[0, 1]
        return self.feature_corr

    def forward(self, x, domain_label=None, return_all_features=False):
        # 数据增强
        x = self.eeg_aug(x)

        # 3D-CNN特征提取
        feat_3d, intermediate_3d = self.cnn3d(x, return_intermediate=True)

        # EEG→RGB映射（需根据你的数据映射逻辑实现）
        # 注：此处为占位，需替换为实际的eeg_feature_to_rgb函数
        rgb_img = torch.randn(x.shape[0], 3, 224, 224).to(cfg.device)  # 临时占位

        # ViT特征提取
        cls_out, layer_vit = self.vit(rgb_img, return_layer_features=True)

        # 特征融合
        feat_3d_flat = feat_3d.reshape(feat_3d.shape[0], -1)
        vit_feat = layer_vit[-1][:, 0, :].to(cfg.device)
        fused_feat = torch.cat([feat_3d_flat, vit_feat], dim=1)

        # 域适应分支
        domain_out = None
        if self.training and domain_label is not None:
            domain_out = self.domain_discriminator(fused_feat)

        # 特征相关性计算
        if self.training:
            self.compute_feature_correlation(feat_3d, layer_vit[-1])

        if return_all_features:
            return cls_out, domain_out, {
                "cnn3d_features": intermediate_3d,
                "vit_features": layer_vit,
                "fused_feat": fused_feat.detach().cpu(),
                "feature_corr": self.feature_corr
            }
        return (cls_out, domain_out) if self.training else (cls_out, rgb_img)

    def get_learning_rates(self):
        return [
            {"params": self.cnn3d.parameters(), "lr": cfg.lr_cnn3d},
            {"params": self.vit.eeg_pos_emb, "lr": cfg.lr_vit * 2},
            {"params": self.vit.blocks[-3:].parameters(), "lr": cfg.lr_vit},
            {"params": self.vit.head.parameters(), "lr": cfg.lr_vit * 1.5},
            {"params": self.domain_discriminator.parameters(), "lr": cfg.lr}
        ]

    def count_params(self, save_path=None):
        print_params_summary(self, save_path)

    # 梯度监控相关方法
    def save_grad_logs(self):
        self.grad_monitor.save_grad_logs()

    def plot_grad_curves(self):
        self.grad_monitor.plot_grad_curves()

    def remove_grad_hooks(self):
        self.grad_monitor.remove_hooks()


# ===================== 模型保存/加载工具 =====================
def save_model(model, path, include_optimizer=False, optimizer=None):
    save_dict = {
        "model_state_dict": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("__")}
    }
    if include_optimizer and optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(save_dict, path)
    print(f"模型已保存至：{path}")


def load_model(model, path, load_optimizer=False, optimizer=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型权重文件不存在：{path}")

    checkpoint = torch.load(path, map_location=cfg.device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)

    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"模型已从 {path} 加载完成")
    if load_optimizer:
        return model, optimizer
    return model


def create_dual_transfer_model(pretrained_cnn3d_path=None):
    model = DualTransferModel()
    if pretrained_cnn3d_path:
        model._load_cnn3d_pretrained(pretrained_cnn3d_path)
    model = model.to(cfg.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
"""
Lightweight Adaptive Multi-Scale Industrial Anomaly Detection
Dynamic Scale Router (DSR) & Lightweight Adaptive Model Architecture
Designed for AeBAD-S with View and Illumination Distribution Shifts
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DynamicScaleRouter(nn.Module):
    """
    Dynamic Scale Router (DSR)
    Formulated as a structural data-routing problem:
    Computes per-image scale gating weights pi(X) in Delta^(S-1) conditioned on
    global illumination statistics and viewpoint geometries via dual spatial pooling (GAP + GMP).
    """
    def __init__(self, in_channels_list=(64, 128, 256), proj_dim=64, bottleneck_dim=32, tau=1.0):
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.tau = tau

        # Scale-specific condition projection heads phi_s: R^(2 * C_s) -> R^proj_dim
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * c, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU()
            )
            for c in in_channels_list
        ])

        # Routing gating network: R^(S * proj_dim) -> R^bottleneck_dim -> R^S
        total_proj_dim = self.num_scales * proj_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(total_proj_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.num_scales)
        )

        # Initialize gating weights with small values for balanced initial exploration
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: list of tensors [F_1, F_2, ..., F_S] from multi-scale backbone stages
                      F_s shape: (B, C_s, H_s, W_s)
        Returns:
            routing_weights: (B, S) simplex weights summing to 1 for each image in batch
            gate_logits: (B, S) unnormalized routing logits
            routing_entropy: scalar entropy loss regularizer
        """
        descriptors = []
        for s, F_s in enumerate(features):
            # Dual spatial pooling: Global Average Pooling (ambient illumination) +
            #                       Global Max Pooling (high-contrast specular/anomaly saliency)
            gap = torch.mean(F_s, dim=(2, 3))  # (B, C_s)
            gmp = torch.amax(F_s, dim=(2, 3))  # (B, C_s)
            z_s = torch.cat([gap, gmp], dim=1)  # (B, 2 * C_s)
            v_s = self.proj_heads[s](z_s)       # (B, proj_dim)
            descriptors.append(v_s)

        # Global multi-scale condition vector u in R^(S * proj_dim)
        u = torch.cat(descriptors, dim=1)  # (B, S * proj_dim)

        # Gate logits g in R^S
        gate_logits = self.gate_mlp(u)  # (B, S)

        # Temperature-scaled simplex routing distribution: pi_s = exp(g_s / tau) / sum(exp(g_j / tau))
        routing_weights = F.softmax(gate_logits / self.tau, dim=-1)  # (B, S)

        # Routing entropy H(pi) = - sum(pi_s * log(pi_s + eps))
        eps = 1e-8
        routing_entropy = -torch.sum(routing_weights * torch.log(routing_weights + eps), dim=-1).mean()

        return routing_weights, gate_logits, routing_entropy


class LightweightDecoderBlock(nn.Module):
    """
    Compact residual convolutional block for per-scale feature reconstruction.
    Significantly more lightweight than heavy ViT-Base MAE decoders.
    """
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(channels // 2, 32)
        self.block = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class LightweightAdaptiveModel(nn.Module):
    """
    Full Lightweight Adaptive Multi-Scale Framework:
    1. Single ResNet-18 frozen feature extractor
    2. Dynamic Scale Router (DSR) for per-image condition-aware weighting
    3. Lightweight multi-scale student reconstruction heads
    """
    def __init__(self, pretrained=True, layers_to_extract=("layer1", "layer2", "layer3"), tau=1.0):
        super().__init__()
        self.layers_to_extract = layers_to_extract

        # Single lightweight backbone: ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels

        # Freeze backbone parameters
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False

        in_channels = [64, 128, 256]
        # Dynamic Scale Router
        self.router = DynamicScaleRouter(in_channels_list=in_channels, proj_dim=64, bottleneck_dim=32, tau=tau)

        # Lightweight student reconstruction heads for each scale
        self.student_heads = nn.ModuleList([
            LightweightDecoderBlock(c) for c in in_channels
        ])

    def extract_teacher_features(self, x):
        """Extract multi-scale features from frozen ResNet-18 backbone."""
        x0 = self.stem(x)
        f1 = self.layer1(x0)  # 64 x 56 x 56
        f2 = self.layer2(f1)  # 128 x 28 x 28
        f3 = self.layer3(f2)  # 256 x 14 x 14
        return [f1, f2, f3]

    def forward(self, x):
        """
        Forward pass during training:
        Computes reconstructed features, dynamic routing weights, and routing entropy.
        """
        with torch.no_grad():
            teacher_features = self.extract_teacher_features(x)

        # Predict dynamic scale routing weights
        routing_weights, gate_logits, routing_entropy = self.router(teacher_features)

        # Lightweight student reconstruction per scale
        reconstructed_features = []
        for s, f in enumerate(teacher_features):
            f_rec = self.student_heads[s](f)
            reconstructed_features.append(f_rec)

        return teacher_features, reconstructed_features, routing_weights, routing_entropy

"""
Adaptive Pipeline for Lightweight Multi-Scale Anomaly Detection
Includes condition-aware dynamic scale weighting, training, inference, and evaluation.
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from utils import compute_pixelwise_retrieval_metrics, compute_pro, save_image

LOGGER = logging.getLogger(__name__)


def cal_adaptive_anomaly_map(teacher_features, student_features, routing_weights, out_size=224):
    """
    Computes per-scale cosine anomaly map and aggregates with per-image dynamic routing weights:
        A_total(X) = sum_{s=1}^S pi_s(X) * BilinearResize(A_s)
    Args:
        teacher_features: list of tensors [F_1, ..., F_S], F_s: (B, C_s, H_s, W_s)
        student_features: list of tensors [F_hat_1, ..., F_hat_S]
        routing_weights: (B, S) simplex weights from Dynamic Scale Router
        out_size: target spatial resolution (e.g. 224)
    Returns:
        anomaly_map: np.ndarray of shape (B, out_size, out_size)
        scale_maps: list of individual per-scale maps
    """
    batch_size = teacher_features[0].shape[0]
    total_map = torch.zeros((batch_size, 1, out_size, out_size), device=teacher_features[0].device)
    scale_maps = []

    for s in range(len(teacher_features)):
        ft = teacher_features[s]
        fs = student_features[s]

        # Cosine distance: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(ft, fs, dim=1, eps=1e-6)  # (B, H_s, W_s)
        a_map_s = (1.0 - cos_sim).unsqueeze(1)                   # (B, 1, H_s, W_s)
        a_map_s_upsampled = F.interpolate(a_map_s, size=out_size, mode='bilinear', align_corners=True)  # (B, 1, H, W)

        # Dynamic scale weighting per image
        pi_s = routing_weights[:, s].view(batch_size, 1, 1, 1)  # (B, 1, 1, 1)
        total_map = total_map + pi_s * a_map_s_upsampled

        scale_maps.append(a_map_s_upsampled.squeeze(1).cpu().detach().numpy())

    total_map_np = total_map.squeeze(1).cpu().detach().numpy()
    return total_map_np, scale_maps


def adaptive_reconstruction_loss(teacher_features, student_features, routing_weights, routing_entropy,
                                lambda_entropy=0.01):
    """
    Loss function:
        L_total = sum_{s=1}^S pi_s * (1 - cos_sim(F_s, F_hat_s)) + lambda_H * H(pi)
    """
    batch_size = teacher_features[0].shape[0]
    loss_recon = 0.0

    for s in range(len(teacher_features)):
        ft = teacher_features[s]
        fs = student_features[s]
        # Mean cosine distance across spatial positions for each image
        cos_sim = F.cosine_similarity(ft, fs, dim=1, eps=1e-6)  # (B, H_s, W_s)
        scale_loss_per_img = (1.0 - cos_sim).mean(dim=(1, 2))   # (B,)

        pi_s = routing_weights[:, s]                           # (B,)
        loss_recon = loss_recon + (pi_s * scale_loss_per_img).mean()

    # Entropy regularization to avoid degenerate collapsed distributions
    loss_total = loss_recon + lambda_entropy * routing_entropy
    return loss_total, loss_recon


class AdaptivePipeline:
    """
    Pipeline managing training, inference, and multi-domain evaluation for
    Lightweight Adaptive Multi-Scale Industrial Anomaly Detection.
    """
    def __init__(self, model, optimizer, device, cfg):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.lambda_entropy = getattr(cfg.TRAIN, "lambda_entropy", 0.01)

    def fit(self, train_dataloader):
        """Train the lightweight student heads and Dynamic Scale Router."""
        epochs = self.cfg.TRAIN_SETUPS.epochs
        self.model.train()

        for epoch in range(epochs):
            total_losses = []
            recon_losses = []

            for batch in train_dataloader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                else:
                    images = batch.to(self.device)

                self.optimizer.zero_grad()
                ft, fs, routing_weights, routing_entropy = self.model(images)

                loss, l_recon = adaptive_reconstruction_loss(
                    ft, fs, routing_weights, routing_entropy, lambda_entropy=self.lambda_entropy
                )

                loss.backward()
                self.optimizer.step()

                total_losses.append(loss.item())
                recon_losses.append(l_recon.item())

            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
                LOGGER.info(
                    f"Epoch [{epoch + 1}/{epochs}] - Total Loss: {np.mean(total_losses):.4f} - "
                    f"Recon Loss: {np.mean(recon_losses):.4f}"
                )

    def evaluation(self, test_dataloader):
        """
        Evaluate on test split with adaptive scale routing.
        Returns:
            auroc_samples: Image-level AUROC
            auroc_pixel: Pixel-level AUROC
            pro_auc: Per-Region-Overlap metric
        """
        self.model.eval()

        labels_gt = []
        labels_prediction = []
        masks_gt = []
        masks_prediction = []
        aupro_list = []
        ima_path = []
        ima_name_list = []

        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, dict):
                    label_current = batch["is_anomaly"].numpy()
                    mask_current = batch["mask"].squeeze(1).numpy()
                    labels_gt.extend(label_current.tolist())
                    masks_gt.extend(mask_current.tolist())

                    ima_path.extend(batch.get("image_path", []))
                    ima_name_list.extend(batch.get("image_name", []))
                    images = batch["image"].to(self.device)
                else:
                    raise ValueError("Expected dictionary format for test batch.")

                # Forward pass
                ft, fs, routing_weights, _ = self.model(images)

                # Dynamically weighted anomaly map
                out_size = getattr(self.cfg.DATASET, "imagesize", 224)
                anomaly_maps, _ = cal_adaptive_anomaly_map(ft, fs, routing_weights, out_size=out_size)

                # Smooth each anomaly map with Gaussian filter
                for i in range(len(anomaly_maps)):
                    anomaly_maps[i] = gaussian_filter(anomaly_maps[i], sigma=4)

                # Image-level anomaly score is the max pixel value
                labels_prediction.extend(np.max(anomaly_maps.reshape(anomaly_maps.shape[0], -1), axis=1))
                masks_prediction.extend(anomaly_maps.tolist())

                # Compute PRO if pixel evaluation is enabled
                if getattr(self.cfg.TEST, "pixel_mode_verify", True):
                    if set(mask_current.astype(int).flatten()) == {0, 1}:
                        aupro_list.extend(compute_pro(anomaly_maps, mask_current.astype(int), label_current))

        auroc_samples = round(roc_auc_score(labels_gt, labels_prediction), 4)

        if getattr(self.cfg.TEST, "pixel_mode_verify", True) and len(masks_gt) > 0:
            try:
                pixel_scores = compute_pixelwise_retrieval_metrics(masks_prediction, masks_gt)
                auroc_pixel = round(pixel_scores["auroc"], 4)
            except Exception as e:
                LOGGER.warning(f"Error computing pixel AUROC: {e}")
                auroc_pixel = 0.0
            pro_auc = round(float(np.mean(aupro_list)), 4) if len(aupro_list) > 0 else 0.0
        else:
            auroc_pixel = 0.0
            pro_auc = 0.0

        masks_prediction_arr = np.stack(masks_prediction)

        if getattr(self.cfg.TEST, "save_segmentation_images", False):
            save_image(
                cfg=self.cfg,
                segmentations=masks_prediction_arr,
                masks_gt=masks_gt,
                ima_path=ima_path,
                ima_name_list=ima_name_list,
                individual_dataloader=test_dataloader
            )

        return auroc_samples, auroc_pixel, pro_auc

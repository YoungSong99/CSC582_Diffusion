import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


def to_zero_one(x, normalization_range):
    if normalization_range == "minus_one_to_one":
        return (x + 1.0) / 2.0
    if normalization_range == "zero_to_one":
        return x
    raise ValueError(f"Unknown normalization_range: {normalization_range}")


class ForegroundWeightedLoss(nn.Module):
    def __init__(self, base_loss="l1", normalization_range="minus_one_to_one", threshold=0.95, foreground_weight=4.0,):
        super().__init__()
        self.base_loss = base_loss
        self.normalization_range = normalization_range
        self.threshold = threshold
        self.foreground_weight = foreground_weight

    def forward(self, recon, target):
        target_01 = to_zero_one(target, self.normalization_range)
        brightness = target_01.mean(dim=1, keepdim=True)
        foreground_mask = (brightness < self.threshold).float()
        weights = 1.0 + self.foreground_weight * foreground_mask

        if self.base_loss == "l1":
            loss = torch.abs(recon - target)
        elif self.base_loss == "mse":
            loss = (recon - target) ** 2
        elif self.base_loss == "huber":
            loss = F.smooth_l1_loss(recon, target, reduction="none")
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        return (loss * weights).sum() / weights.sum()


class VGGFeatureLoss(nn.Module):
    def __init__(self, normalization_range="minus_one_to_one"):
        super().__init__()

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])
        self.normalization_range = normalization_range

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.feature_extractor.eval()
    
    def forward(self, y_pred, y_true):
        y_true = to_zero_one(y_true, self.normalization_range).clamp(0, 1)
        y_pred = to_zero_one(y_pred, self.normalization_range).clamp(0, 1)

        feats_true = self.feature_extractor(y_true)
        feats_pred = self.feature_extractor(y_pred)

        return F.mse_loss(feats_pred, feats_true)


class CombinedLoss(nn.Module):
    def __init__(self, pixel_loss, feature_loss, feature_weight=0.01):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.feature_loss = feature_loss
        self.feature_weight = feature_weight

    def forward(self, recon, target):
        pixel = self.pixel_loss(recon, target)
        feature = self.feature_loss(recon, target)
        return pixel + self.feature_weight * feature
    

def get_loss_fn(loss_name, params=None):
    basic_losses = {
        "l1": nn.L1Loss,
        "mse": nn.MSELoss,
        "huber": nn.SmoothL1Loss,
    }

    if loss_name in basic_losses:
        return basic_losses[loss_name]()

    if loss_name == "foreground_huber_vgg":
        pixel_loss = ForegroundWeightedLoss(
            base_loss="huber",
            normalization_range=params["data"]["normalization_range"],
            threshold=params["training"].get("foreground_threshold", 0.95),
            foreground_weight=params["training"].get("foreground_weight", 4.0),
        )

        feature_loss = VGGFeatureLoss(
            normalization_range=params["data"]["normalization_range"]
        )

        return CombinedLoss(
            pixel_loss=pixel_loss,
            feature_loss=feature_loss,
            feature_weight=params["training"].get("vgg_weight", 0.01),
        )
    
    foreground_losses = {
        "foreground_l1": "l1",
        "foreground_mse": "mse",
        "foreground_huber": "huber",
    }

    if loss_name in foreground_losses:
        return ForegroundWeightedLoss(
            base_loss=foreground_losses[loss_name],
            normalization_range=params["data"]["normalization_range"],
            threshold=params["training"].get("foreground_threshold", 0.95),
            foreground_weight=params["training"].get("foreground_weight", 4.0),
        )

    raise ValueError(f"Unknown loss: {loss_name}")


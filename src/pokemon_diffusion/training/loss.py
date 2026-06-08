import torch
import torch.nn as nn
import torch.nn.functional as F

class ForegroundWeightedLoss(nn.Module):
    def __init__(self, base_loss="l1", normalization_range="minus_one_to_one", threshold=0.95, foreground_weight=4.0,):
        super().__init__()
        self.base_loss = base_loss
        self.normalization_range = normalization_range
        self.threshold = threshold
        self.foreground_weight = foreground_weight

    def to_zero_one(self, x):
        if self.normalization_range == "minus_one_to_one":
            return (x + 1.0) / 2.0
        if self.normalization_range == "zero_to_one":
            return x
        raise ValueError(f"Unknown normalization_range: {self.normalization_range}")

    def forward(self, recon, target):
        target_01 = self.to_zero_one(target)
        brightness = target_01.mean(dim=1, keepdim=True)
        fg_mask = (brightness < self.threshold).float()
        weights = 1.0 + self.foreground_weight * fg_mask

        if self.base_loss == "l1":
            loss = torch.abs(recon - target)
        elif self.base_loss == "mse":
            loss = (recon - target) ** 2
        elif self.base_loss == "huber":
            loss = F.smooth_l1_loss(recon, target, reduction="none")
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        return (loss * weights).sum() / weights.sum()

def get_loss_fn(loss_name, params=None):
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.SmoothL1Loss()

    if loss_name.startswith("foreground_"):
        base_loss = loss_name.replace("foreground_", "")
        return ForegroundWeightedLoss(
            base_loss=base_loss,
            normalization_range=params["data"]["normalization_range"],
            threshold=params["training"].get("foreground_threshold", 0.95),
            foreground_weight=params["training"].get("foreground_weight", 4.0),
        )

    raise ValueError(f"Unknown loss: {loss_name}")
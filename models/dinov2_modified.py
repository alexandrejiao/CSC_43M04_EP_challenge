import torch
import torch.nn as nn


class DinoV2FinetuneModified(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone.norm.normalized_shape[0], 2*self.backbone.norm.normalized_shape[0]),
            nn.ReLU(),
            nn.Linear(2*self.backbone.norm.normalized_shape[0], 1)
        )

    def forward(self, x):
        x = self.backbone(x["image"])
        x = self.regression_head(x)
        return x

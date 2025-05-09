import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel  # Pour encoder le texte


class DinoV2WithText(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        # Backbone pour les images
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.image_dim = self.backbone.norm.normalized_shape[0]
        
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Encodeur textuel
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_dim = self.text_encoder.config.hidden_size

        # Fusion des features image et texte
        self.fusion_dim = self.image_dim + self.text_dim
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Encodage des images
        image_features = self.backbone(x["image"])

        # Encodage du texte
        text_inputs = self.tokenizer(
            x["text"], padding=True, truncation=True, return_tensors="pt"
        ).to(image_features.device)
        text_features = self.text_encoder(**text_inputs).pooler_output

        # Fusion des features
        combined_features = torch.cat([image_features, text_features], dim=1)

        # RÃ©gression
        output = self.regression_head(combined_features)
        return output
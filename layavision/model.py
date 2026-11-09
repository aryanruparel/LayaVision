"""
LayaVision Model Architecture
=============================
Integrates Vision Transformers (CLIP / SigLIP) with Laya's non-autoregressive
decision transformer heads for fast, calibrated object classification.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel


def resolve_model_path(model_id_or_path: str) -> str:
    """Resolve local cached HF path if available to prevent network hangs."""
    if os.path.exists(model_id_or_path):
        return model_id_or_path
    hub_name = "models--" + model_id_or_path.replace("/", "--")
    hub_dir = os.path.expanduser(os.path.join("~/.cache/huggingface/hub", hub_name, "snapshots"))
    if os.path.isdir(hub_dir):
        snapshots = [os.path.join(hub_dir, s) for s in os.listdir(hub_dir) if os.path.isdir(os.path.join(hub_dir, s))]
        for s in snapshots:
            if os.path.exists(os.path.join(s, "pytorch_model.bin")) or os.path.exists(os.path.join(s, "model.safetensors")):
                return s
        if snapshots:
            return snapshots[0]
    return model_id_or_path


class MultimodalProjector(nn.Module):
    """Projects vision features into decision embedding space with residual shortcut."""

    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )
        if vision_dim != text_dim:
            self.res_proj = nn.Linear(vision_dim, text_dim, bias=False)
        else:
            self.res_proj = nn.Identity()

        # Zero-initialize last projection layer so initial output preserves pretrained zero-shot alignment
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_proj(x) + self.net(x)


class LayaVisionModel(nn.Module):
    """
    Multimodal Vision Decision Model for Object Classification.
    Combines:
      1. Pretrained Vision Encoder (CLIP / SigLIP / ViT)
      2. Vision-to-Decision Alignment & Projector
      3. Laya Scorer & Action Heads for Calibrated Decisions
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        use_projector: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_model_name = vision_model_name
        self.embed_dim = embed_dim

        resolved_path = resolve_model_path(vision_model_name)

        # Load vision backbone
        try:
            self.vision_model = CLIPModel.from_pretrained(resolved_path)
        except Exception:
            self.vision_model = AutoModel.from_pretrained(resolved_path)

        # Determine visual and text projection dimensions
        if hasattr(self.vision_model, "visual_projection"):
            vis_dim = self.vision_model.visual_projection.out_features
            txt_dim = self.vision_model.text_projection.out_features
        elif hasattr(self.vision_model, "vision_model"):
            vis_dim = self.vision_model.vision_model.config.hidden_size
            txt_dim = getattr(self.vision_model.config, "projection_dim", vis_dim)
        else:
            vis_dim = getattr(self.vision_model.config, "hidden_size", embed_dim)
            txt_dim = vis_dim

        self.use_projector = use_projector
        if use_projector:
            self.projector = MultimodalProjector(vis_dim, txt_dim)
        else:
            self.projector = nn.Identity()

        # Laya-style typed scorer MLP
        self.scorer = nn.Sequential(
            nn.LayerNorm(txt_dim),
            nn.Linear(txt_dim, txt_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(txt_dim, 1),
        )

        # Laya-style action head (for decision routing / alerting)
        self.act_head = nn.Sequential(
            nn.Linear(txt_dim + 4, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        # Temperature calibration buffers for [choice, score, noul]
        self.register_buffer("temperature", torch.tensor([1.0, 1.0, 1.0]))

    @staticmethod
    def _extract_tensor(out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        if isinstance(out, (tuple, list)):
            return out[0]
        return getattr(out, "logits", out)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract normalized visual embeddings for input images."""
        if hasattr(self.vision_model, "get_image_features"):
            img_feats = self.vision_model.get_image_features(pixel_values=pixel_values)
        elif hasattr(self.vision_model, "vision_model"):
            outputs = self.vision_model.vision_model(pixel_values=pixel_values)
            img_feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
            if hasattr(self.vision_model, "visual_projection"):
                img_feats = self.vision_model.visual_projection(img_feats)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)
            img_feats = outputs.last_hidden_state[:, 0]

        img_feats = self._extract_tensor(img_feats)
        img_feats = self.projector(img_feats)
        return F.normalize(img_feats, p=2, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract normalized text embeddings for option prompts / criteria."""
        if hasattr(self.vision_model, "get_text_features"):
            txt_feats = self.vision_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        elif hasattr(self.vision_model, "text_model"):
            outputs = self.vision_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            txt_feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
            if hasattr(self.vision_model, "text_projection"):
                txt_feats = self.vision_model.text_projection(txt_feats)
        else:
            outputs = self.vision_model(input_ids=input_ids, attention_mask=attention_mask)
            txt_feats = outputs.last_hidden_state[:, 0]

        txt_feats = self._extract_tensor(txt_feats)
        return F.normalize(txt_feats, p=2, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
        qtype: int = 0,
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass scoring candidate options against the input image.
        """
        img_feats = self.encode_image(pixel_values)  # (B, D)
        opt_feats = self.encode_text(option_input_ids, option_attention_mask)  # (K, D)

        # Scaled cross-modal affinity logits (cosine similarity * 100 / tau)
        sim = torch.matmul(img_feats, opt_feats.T) * 100.0  # (B, K)
        logits = sim / max(tau, 1e-4)

        # Extract features for Laya act_head
        p = torch.softmax(logits.detach(), -1)
        k = max(2, opt_feats.size(0))
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1, keepdim=True) / torch.log(torch.tensor(float(k), device=p.device))

        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
            top1_top2_diff = top2[:, 0:1] - top2[:, 1:2]
            top1 = top2[:, 0:1]
        else:
            top1 = p.topk(1, -1).values
            top1_top2_diff = torch.ones_like(top1)

        k_feat = torch.full((p.size(0), 1), float(k) / 255.0, device=p.device)
        feats = torch.cat([top1, top1_top2_diff, ent, k_feat], dim=-1)

        act_logits = self.act_head(torch.cat([img_feats, feats], dim=-1))
        return logits, act_logits

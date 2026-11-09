"""
LayaVision Training & Fine-Tuning Module
=======================================
Enables fine-tuning the multimodal projector and decision heads on custom object
classification datasets using CrossEntropy and Laya's Strictly Proper Scoring Rules.
"""

import json
import os
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer

from .model import LayaVisionModel
from .dataset import generate_synthetic_image


def proper_scoring_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_sph: float = 0.5,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """
    Computes loss based on Laya's Strictly Proper Scoring Rule (Log score + Spherical score).
    Encourages well-calibrated confidence estimates.
    """
    probs = torch.softmax(logits, dim=-1)
    k = logits.size(-1)
    target_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)

    log_probs = torch.log(probs.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target_onehot * log_probs).sum(-1)
    spherical = (target_onehot * probs).sum(-1) / probs.norm(dim=-1).clamp_min(1e-9)

    reward = log_score + w_sph * spherical
    return -reward.mean()


class SyntheticObjectDataset(Dataset):
    """Synthetic dataset for quick training testing and demonstration."""

    def __init__(self, categories: List[str], samples_per_cat: int = 20):
        self.categories = categories
        self.samples = []
        for idx, cat in enumerate(categories):
            for _ in range(samples_per_cat):
                self.samples.append((cat, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat, label = self.samples[idx]
        img = generate_synthetic_image(cat)
        return img, label


def train_layavision(
    categories: List[str],
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: Optional[str] = None,
    output_dir: str = "checkpoints/layavision_custom",
    use_proper_scoring: bool = True,
):
    """
    Train or fine-tune LayaVision on a list of categories.

    Args:
        categories: List of object categories.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        device: 'cuda' or 'cpu'.
        output_dir: Directory where weights will be saved.
        use_proper_scoring: Whether to use Laya's proper scoring rule loss.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Starting LayaVision training on {device}...")

    model_name = "openai/clip-vit-base-patch32"
    model = LayaVisionModel(vision_model_name=model_name, use_projector=True).to(device)

    # Freeze base vision encoder, train projector and heads for fast convergence
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.scorer.parameters():
        param.requires_grad = True
    for param in model.act_head.parameters():
        param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare candidate prompt tokens
    candidate_texts = [f"a photo of a {c}" for c in categories]
    toks = tokenizer(candidate_texts, padding=True, truncation=True, return_tensors="pt")
    opt_ids = toks["input_ids"].to(device)
    opt_mask = toks["attention_mask"].to(device)

    dataset = SyntheticObjectDataset(categories, samples_per_cat=16)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2,
    )

    model.train()
    print(f"Training on {len(dataset)} samples across {len(categories)} classes...")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        indices = torch.randperm(len(dataset))
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = [dataset[idx][0] for idx in batch_indices]
            batch_labels = torch.tensor([dataset[idx][1] for idx in batch_indices], device=device)

            inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            optimizer.zero_grad()

            logits, act_logits = model(
                pixel_values=pixel_values,
                option_input_ids=opt_ids,
                option_attention_mask=opt_mask,
                qtype=0,
            )

            if use_proper_scoring:
                loss = proper_scoring_loss(logits, batch_labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        avg_loss = total_loss / total
        acc = (correct / total) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), weights_path)

    config = {
        "vision_model_name": model_name,
        "categories": categories,
        "classes_count": len(categories),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete! Saved to {output_dir}")

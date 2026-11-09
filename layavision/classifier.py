"""
LayaVision Classifier & High-Level Runtime
==========================================
Provides fast, calibrated object classification and structured visual System 1
reasoning using vision encoders and Laya-style decision heads.
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer

from .model import LayaVisionModel, resolve_model_path


def _confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


class LayaVision:
    """
    High-level System 1 Vision Decision Engine for Object Classification.
    Non-autoregressive, calibrated, and ultra-fast on GPU.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_projector: bool = True,
    ):
        """
        Initialize the LayaVision model and preprocessors.

        Args:
            model_name: HuggingFace model identifier (e.g. 'openai/clip-vit-base-patch32'
                        or 'google/siglip-base-patch16-224')
            device: 'cuda', 'cpu', or None (auto-detect)
            dtype: torch.float16 or torch.float32 (auto-selected if None)
            use_projector: Whether to attach the Laya projection layer
        """
        # 1. Device resolution
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU.")
                self.device = torch.device("cpu")

        # 2. Precision
        if dtype is None:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = dtype

        print(f"Loading LayaVision ({model_name}) on {str(self.device).upper()} [{self.dtype}]...")

        # 3. Model & Processor initialization
        self.model = LayaVisionModel(
            vision_model_name=model_name,
            use_projector=use_projector,
        ).to(self.device).eval()

        resolved_path = resolve_model_path(model_name)
        self.processor = AutoProcessor.from_pretrained(resolved_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)

        # Cache for precomputed class embeddings (for ultra-low latency streams)
        self._cached_embeddings: Dict[str, torch.Tensor] = {}

    def _prepare_image(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Standardize image input into a PyTorch tensor (B, C, H, W)."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device, dtype=self.dtype)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        return pixel_values

    def _format_option_prompt(self, label: str, criterion: Optional[str] = None, prompt_template: Optional[str] = None) -> str:
        """Format a class option with optional descriptive criteria for zero-shot disambiguation."""
        if criterion:
            return f"a photo of {label}, {criterion}"
        if prompt_template:
            return prompt_template.format(label)
        return f"a photo of a {label}"

    def classify(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        candidate_classes: List[str],
        instructions: Optional[str] = None,
        criteria: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Classify an image into candidate object classes with calibrated confidence.

        Args:
            image: Image file path, PIL Image, or numpy array.
            candidate_classes: List of object class names (e.g. ['cat', 'dog', 'car']).
            instructions: Optional context instruction (e.g. 'Identify the main animal').
            criteria: Optional dict mapping class -> visual description for fine-grained disambiguation.
            top_k: Number of top classes to return in results.
            temperature: Softmax temperature scaling factor.

        Returns:
            Dict containing top_class, confidence, probabilities, and latency.
        """
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None

        t0 = time.perf_counter()
        if start_time:
            start_time.record()

        # Format candidate texts
        criteria = criteria or {}
        option_texts = [
            self._format_option_prompt(c, criteria.get(c))
            for c in candidate_classes
        ]

        # Tokenize candidate texts
        toks = self.tokenizer(option_texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = toks["input_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)

        # Prepare image
        pixel_values = self._prepare_image(image)

        # Forward pass
        with torch.no_grad():
            use_amp = self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
                logits, act_logits = self.model(
                    pixel_values=pixel_values,
                    option_input_ids=input_ids,
                    option_attention_mask=attention_mask,
                    qtype=0,
                    tau=temperature,
                )

            probs = F.softmax(logits[0], dim=-1).cpu().numpy()
            act_prob = float(F.softmax(act_logits[0].float(), dim=-1)[0].cpu().numpy())

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = start_time.elapsed_time(end_time)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Calibrate confidence using normalized entropy
        num_classes = len(candidate_classes)
        conf = _confidence_from_probs(probs, num_classes)

        # Sort predictions
        ranked_indices = np.argsort(-probs)
        top_idx = int(ranked_indices[0])
        top_class = candidate_classes[top_idx]

        sorted_probs = {
            candidate_classes[i]: round(float(probs[i]), 4)
            for i in ranked_indices[:top_k]
        }

        return {
            "top_class": top_class,
            "confidence": round(conf, 4),
            "probabilities": sorted_probs,
            "action_probability": round(act_prob, 4),
            "latency_ms": round(elapsed_ms, 2),
            "num_candidates": num_classes,
        }

    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate full Laya System 1 typed decision questions against an image.

        Args:
            image: Image input.
            questions: Dictionary mapping question_id -> question definition.
                - choice: {"type": "choice", "instructions": "...", "criteria": {"optA": "...", ...}}
                - score:  {"type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...]}
                - noul:   {"type": "noul",   "instructions": "..."}

        Returns:
            Dictionary with answers, calibrated confidence, probabilities, and latency.
        """
        t0 = time.perf_counter()
        pixel_values = self._prepare_image(image)
        answers = {}

        for qid, qdef in questions.items():
            qtype = qdef.get("type", "choice")
            instructions = qdef.get("instructions", "")
            criteria = qdef.get("criteria", {})

            if qtype == "choice":
                if isinstance(criteria, list):
                    options = criteria
                    crit_dict = {}
                else:
                    options = list(criteria.keys())
                    crit_dict = criteria

                option_texts = [
                    self._format_option_prompt(opt, crit_dict.get(opt))
                    for opt in options
                ]

            elif qtype == "score":
                # Ordinal scale questions
                if isinstance(criteria, list):
                    options = [f"level_{i}" for i in range(len(criteria))]
                    option_texts = [f"a photo showing {c}" for c in criteria]
                else:
                    options = list(criteria.keys())
                    option_texts = [f"a photo showing {v}" for v in criteria.values()]

            elif qtype == "noul":
                # Binary yes/no decisions
                options = ["false", "true"]
                clean_ins = instructions.rstrip("?").strip()
                # Remove leading 'Is the ', 'Does the ', 'Are there ' for declarative text
                lower_ins = clean_ins.lower()
                for prefix in ["is the ", "are there ", "is there ", "does the "]:
                    if lower_ins.startswith(prefix):
                        clean_ins = clean_ins[len(prefix):]
                        break

                if isinstance(criteria, dict) and "false" in criteria and "true" in criteria:
                    crit_false = criteria["false"]
                    crit_true = criteria["true"]
                else:
                    crit_false = f"no {clean_ins}, not present"
                    crit_true = f"{clean_ins} present"

                option_texts = [
                    f"a photo with {crit_false}",
                    f"a photo with {crit_true}",
                ]
            else:
                raise ValueError(f"Unknown question type: {qtype}")

            toks = self.tokenizer(option_texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = toks["input_ids"].to(self.device)
            attention_mask = toks["attention_mask"].to(self.device)

            with torch.no_grad():
                logits, act_logits = self.model(
                    pixel_values=pixel_values,
                    option_input_ids=input_ids,
                    option_attention_mask=attention_mask,
                    qtype=0 if qtype == "choice" else 1 if qtype == "score" else 2,
                )

                probs = F.softmax(logits[0], dim=-1).cpu().numpy()
                act_prob = float(F.softmax(act_logits[0].float(), dim=-1)[0].cpu().numpy())

            k = len(options)
            conf = _confidence_from_probs(probs, k)

            if qtype == "choice":
                top_idx = int(probs.argmax())
                answers[qid] = {
                    "type": "choice",
                    "choice": options[top_idx],
                    "confidence": round(conf, 4),
                    "probabilities": {options[i]: round(float(probs[i]), 4) for i in range(k)},
                    "action": {"act_probability": round(act_prob, 4)},
                }
            elif qtype == "score":
                exp_score = float((np.arange(k) * probs).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "confidence": round(conf, 4),
                    "probabilities": {options[i]: round(float(probs[i]), 4) for i in range(k)},
                    "action": {"act_probability": round(act_prob, 4)},
                }
            elif qtype == "noul":
                p_true = float(probs[1])
                answers[qid] = {
                    "type": "noul",
                    "decision": p_true >= 0.5,
                    "probability": round(p_true, 4),
                    "confidence": round(conf, 4),
                    "action": {"act_probability": round(act_prob, 4)},
                }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "answers": answers,
            "total_latency_ms": round(elapsed_ms, 2),
            "num_questions": len(questions),
        }


# Alias for convenience
VisionClassifier = LayaVision

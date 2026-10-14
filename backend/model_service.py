# -*- coding: utf-8 -*-
"""
Model inference service for MedTrace AI -- Dual-ViT Branch Architecture.

Architecture
============
Branch 1 -- Original image
    The original MRI (RGB, 224x224) is fed through a ViT-Base encoder.
    Self-attention over 16x16 patches learns texture, anatomical structure
    and visual inconsistencies introduced by manipulation or AI generation.
    The CLS token after the final encoder layer is used as the feature vector:
        F_o in R^768

Branch 2 -- ELA image
    The Error-Level-Analysis representation of the same MRI is fed through
    a second, independent ViT-Base encoder.
    ELA highlights compression artefacts that differ from the surrounding
    image, which can expose spliced or AI-generated regions.
        F_e in R^768

Fusion head
    F_combined = [F_o ; F_e]  in R^1536
    -> Linear(1536 -> 512) -> ReLU -> Dropout(0.3)
    -> Linear(512 -> 2)
    -> softmax  ->  [P(fake), P(real)]

Model loading modes (in priority order)
========================================
1.  DUAL_MODEL_PATH  -- directory containing a trained DualViTClassifier
    checkpoint (torch .pt / .pth).  Full end-to-end dual architecture.

2.  MODEL_PATH -- a standard Hugging Face ViTForImageClassification checkpoint.
    Branch 1 is initialised from it; Branch 2 uses a frozen pretrained
    google/vit-base-patch16-224 encoder.  The fusion head is *untrained*
    (random weights) in this mode, so results are approximate -- useful only
    for smoke-testing until a proper DUAL_MODEL_PATH is available.

3.  No checkpoint -- heuristic fallback that operates on the forensic feature
    dict produced by forensics.py.  Clearly labelled in the summary field.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_path(env_key: str) -> str:
    """Return an absolute path from an env variable, or empty string."""
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return ""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _BACKEND_DIR / p
    return str(p.resolve())


DUAL_MODEL_PATH = _resolve_path("DUAL_MODEL_PATH")
MODEL_PATH = _resolve_path("MODEL_PATH")

# Determine which mode we are in at import time so the server can report it.
if DUAL_MODEL_PATH and os.path.isdir(DUAL_MODEL_PATH):
    _MODE = "dual_checkpoint"
elif MODEL_PATH and os.path.isdir(MODEL_PATH):
    _MODE = "single_checkpoint"
else:
    _MODE = "heuristic"

logger.info("model_service: mode=%s", _MODE)

# ---------------------------------------------------------------------------
# Dual-ViT model definition
# ---------------------------------------------------------------------------

class DualViTClassifier(nn.Module):
    """
    Two independent ViT-Base encoders (original + ELA) whose CLS-token
    representations are concatenated and passed through a small MLP.

    Parameters
    ----------
    encoder_orig : nn.Module
        Pretrained ViT encoder for the original image.  Must accept
        ``pixel_values`` and return an object with ``.last_hidden_state``.
    encoder_ela : nn.Module
        Pretrained ViT encoder for the ELA image.  Same interface.
    hidden_dim : int
        Width of the intermediate fusion layer (default 512).
    dropout : float
        Dropout probability applied before each Linear layer (default 0.3).
    """

    def __init__(
        self,
        encoder_orig: nn.Module,
        encoder_ela: nn.Module,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder_orig = encoder_orig
        self.encoder_ela = encoder_ela

        # Infer feature dimension from encoder config (768 for ViT-Base).
        feat_dim: int = getattr(
            encoder_orig.config, "hidden_size", 768
        )

        self.fusion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _cls_token(self, encoder: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract the CLS (index 0) token from the encoder's last hidden state."""
        outputs = encoder(pixel_values=pixel_values)
        # Both ViTModel and ViTForImageClassification expose last_hidden_state.
        hidden = outputs.last_hidden_state          # (B, seq_len, hidden_size)
        return hidden[:, 0, :]                      # (B, hidden_size)

    def forward(
        self,
        pixel_values_orig: torch.Tensor,
        pixel_values_ela: torch.Tensor,
    ) -> torch.Tensor:
        f_o = self._cls_token(self.encoder_orig, pixel_values_orig)   # (B, 768)
        f_e = self._cls_token(self.encoder_ela, pixel_values_ela)     # (B, 768)
        combined = torch.cat([f_o, f_e], dim=-1)                      # (B, 1536)
        logits = self.fusion_head(combined)                           # (B, 2)
        return logits


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_dual_model: DualViTClassifier | None = None
_processor_orig = None   # HF image processor for the original branch
_processor_ela = None    # HF image processor for the ELA branch


def _load_hf_vit_model(checkpoint: str):
    """
    Load a Hugging Face ViT as a feature encoder.

    Returns (model, processor) where model is a ViTModel (not
    ViTForImageClassification so we can read last_hidden_state directly).
    """
    from transformers import ViTModel, AutoImageProcessor  # type: ignore

    model = ViTModel.from_pretrained(checkpoint, local_files_only=True, ignore_mismatched_sizes=True)
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    except Exception:
        processor = None
    return model, processor


def _load_dual_checkpoint() -> DualViTClassifier:
    """Load a DualViTClassifier saved with torch.save(model.state_dict(), ...)."""
    from transformers import ViTModel, AutoImageProcessor  # type: ignore

    # Load the two base encoders.  The checkpoint directory must contain:
    #   encoder_orig/  -- a valid HF ViT checkpoint
    #   encoder_ela/   -- a valid HF ViT checkpoint  (may be identical)
    #   fusion_head.pt -- state dict for the fusion MLP
    # OR alternatively a single flat state dict file `dual_vit.pt`.

    dual_pt = os.path.join(DUAL_MODEL_PATH, "dual_vit.pt")
    encoder_orig_dir = os.path.join(DUAL_MODEL_PATH, "encoder_orig")
    encoder_ela_dir = os.path.join(DUAL_MODEL_PATH, "encoder_ela")
    fusion_pt = os.path.join(DUAL_MODEL_PATH, "fusion_head.pt")

    if os.path.isfile(dual_pt):
        # Flat checkpoint: full state dict saved directly from DualViTClassifier.
        # We must re-create the architecture with the correct backbone first.
        backbone = "google/vit-base-patch16-224"
        enc_orig = ViTModel.from_pretrained(backbone)
        enc_ela = ViTModel.from_pretrained(backbone)
        model = DualViTClassifier(enc_orig, enc_ela)
        state = torch.load(dual_pt, map_location=_device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded flat dual_vit.pt checkpoint from %s", DUAL_MODEL_PATH)

        # Processors: look for preprocessor_config.json inside DUAL_MODEL_PATH,
        # else fall back to the standard ViT-Base processor.
        try:
            proc = AutoImageProcessor.from_pretrained(DUAL_MODEL_PATH, local_files_only=True)
            proc_orig, proc_ela = proc, proc
        except Exception:
            try:
                proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                proc_orig, proc_ela = proc, proc
            except Exception:
                proc_orig, proc_ela = None, None

    elif os.path.isdir(encoder_orig_dir) and os.path.isdir(encoder_ela_dir):
        enc_orig, proc_orig = _load_hf_vit_model(encoder_orig_dir)
        enc_ela, proc_ela = _load_hf_vit_model(encoder_ela_dir)
        model = DualViTClassifier(enc_orig, enc_ela)
        if os.path.isfile(fusion_pt):
            fusion_state = torch.load(fusion_pt, map_location=_device, weights_only=True)
            model.fusion_head.load_state_dict(fusion_state)
            logger.info("Loaded fusion_head.pt from %s", DUAL_MODEL_PATH)
    else:
        raise FileNotFoundError(
            f"DUAL_MODEL_PATH={DUAL_MODEL_PATH} must contain either "
            "'dual_vit.pt' (flat state dict) or 'encoder_orig/' + 'encoder_ela/' "
            "sub-directories with HF checkpoints."
        )

    return model, proc_orig, proc_ela


def _load_single_as_dual():
    """
    Wrap a single HF ViTForImageClassification checkpoint as Branch 1 of the
    DualViTClassifier.  Branch 2 is initialised from google/vit-base-patch16-224.

    NOTE: The fusion head has random weights in this mode.  The model will
    produce valid-shaped output but the probabilities are not calibrated.
    """
    from transformers import ViTModel, AutoImageProcessor, AutoModelForImageClassification  # type: ignore

    logger.warning(
        "DUAL_MODEL_PATH not set. Using MODEL_PATH for Branch 1 and a pretrained "
        "ViT-Base for Branch 2.  Fusion head is UNTRAINED -- probabilities are "
        "approximate. Configure DUAL_MODEL_PATH for calibrated results."
    )

    # Branch 1: load the single checkpoint.  If it is a ViTForImageClassification
    # we strip the classifier head and keep only the ViT body.
    hf_model = AutoModelForImageClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    # Grab the underlying ViT body (works for google/vit-* family).
    if hasattr(hf_model, "vit"):
        enc_orig = hf_model.vit  # ViTModel-compatible sub-module
    else:
        # Fallback: treat the whole model as encoder and hope it has
        # last_hidden_state (e.g. ViTModel loaded via AutoModel).
        enc_orig = hf_model

    try:
        proc_orig = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    except Exception:
        proc_orig = None

    # Branch 2: pretrained ViT-Base (no fine-tuning for ELA branch).
    try:
        enc_ela = ViTModel.from_pretrained("google/vit-base-patch16-224")
        proc_ela = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    except Exception:
        # If HF Hub is unavailable, mirror processor from Branch 1.
        enc_ela = ViTModel.__new__(ViTModel)  # placeholder -- will fail at forward
        proc_ela = proc_orig
        logger.error("Could not load google/vit-base-patch16-224 for Branch 2.")

    model = DualViTClassifier(enc_orig, enc_ela)
    return model, proc_orig, proc_ela


def load_model() -> DualViTClassifier:
    """Initialise and cache the DualViTClassifier (called once per process)."""
    global _dual_model, _processor_orig, _processor_ela

    if _dual_model is not None:
        return _dual_model

    if _MODE == "dual_checkpoint":
        _dual_model, _processor_orig, _processor_ela = _load_dual_checkpoint()
    elif _MODE == "single_checkpoint":
        _dual_model, _processor_orig, _processor_ela = _load_single_as_dual()
    else:
        raise RuntimeError(
            "No model checkpoint configured.  Set DUAL_MODEL_PATH (preferred) "
            "or MODEL_PATH in the .env file."
        )

    _dual_model.to(_device)
    _dual_model.eval()
    logger.info("DualViTClassifier loaded on %s", _device)
    return _dual_model


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess_image(
    image: Image.Image,
    processor,
    target_size: int = 224,
) -> torch.Tensor:
    """
    Return a (3, H, W) float32 tensor ready for the ViT encoder.

    Uses the Hugging Face processor when available, otherwise applies
    standard ImageNet normalisation.
    """
    image = image.convert("RGB")

    if processor is not None:
        encoded = processor(images=image, return_tensors="pt")
        return encoded["pixel_values"].squeeze(0)

    image = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_dual_model(
    original_image: Image.Image,
    ela_image: Image.Image,
) -> Tuple[float, float]:
    """
    Forward-pass both branches and return (fake_prob, real_prob).

    label mapping: index 0 -> fake, index 1 -> real
    (matching the convention used by the single-branch checkpoint).
    """
    model = load_model()

    t_orig = _preprocess_image(original_image, _processor_orig).unsqueeze(0).to(_device)
    t_ela = _preprocess_image(ela_image, _processor_ela).unsqueeze(0).to(_device)

    with torch.inference_mode():
        logits = model(pixel_values_orig=t_orig, pixel_values_ela=t_ela)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / probs.sum()  # guard against floating-point drift

    if probs.shape[0] != 2:
        raise RuntimeError(
            f"Expected 2 output logits, got {probs.shape[0]}."
        )

    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    return fake_prob, real_prob


# ---------------------------------------------------------------------------
# Heuristic fallback (no model)
# ---------------------------------------------------------------------------

def _heuristic_fallback(features: dict) -> Tuple[float, float]:
    """
    Rule-based estimation of fake/real probabilities from forensic features.
    Used only when MODEL_PATH and DUAL_MODEL_PATH are both absent.
    """
    ela_std = features.get("ela_std", 0.0)
    hot_fraction = features.get("ela_hot_fraction", 0.0)
    noise_std = features.get("noise_std", 0.0)

    tamper_signal = float(np.clip((hot_fraction * 4.0) + (ela_std / 60.0), 0, 1))
    ai_signal = float(np.clip((1.0 - min(noise_std / 25.0, 1.0)) * 0.3, 0, 0.4))

    fake_prob = float(np.clip(0.1 + 0.6 * tamper_signal + ai_signal, 0, 1))
    real_prob = float(np.clip(1.0 - fake_prob, 0, 1))
    # Re-normalise to sum exactly to 1.
    total = fake_prob + real_prob
    return fake_prob / total, real_prob / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    original_image: Image.Image,
    ela_image: Image.Image,
    features: dict,
) -> dict:
    """
    Main entry point called by server.py.

    Returns a dict with keys:
        prediction   : "Fake" | "Real"
        confidence   : float [0, 1]
        probabilities: {"fake": float, "real": float}
        summary      : human-readable description of the analysis mode
    """
    if _MODE in ("dual_checkpoint", "single_checkpoint"):
        try:
            fake_prob, real_prob = _run_dual_model(original_image, ela_image)

            if _MODE == "dual_checkpoint":
                summary = (
                    "Dual-ViT analysis: Branch 1 processed the original MRI "
                    "(texture & anatomical patterns); Branch 2 processed the ELA "
                    "map (compression artefact patterns). Feature vectors were "
                    "concatenated (1536-d) and classified by a trained fusion head."
                )
            else:
                summary = (
                    "Dual-ViT analysis (single checkpoint mode): Branch 1 uses "
                    "the fine-tuned ViT checkpoint; Branch 2 uses a pretrained "
                    "ViT-Base on the ELA image. The fusion head is untrained -- "
                    "results are indicative only. Set DUAL_MODEL_PATH for "
                    "calibrated dual-branch predictions."
                )
        except Exception as exc:
            logger.error("DualViT inference failed (%s); falling back to heuristic.", exc)
            fake_prob, real_prob = _heuristic_fallback(features)
            summary = f"Heuristic fallback (model error: {exc})."
    else:
        fake_prob, real_prob = _heuristic_fallback(features)
        summary = (
            "Heuristic analysis -- neither DUAL_MODEL_PATH nor MODEL_PATH is "
            "configured. Set DUAL_MODEL_PATH to enable the trained Dual-ViT model."
        )

    prediction = "Fake" if fake_prob >= real_prob else "Real"
    confidence = max(fake_prob, real_prob)

    return {
        "prediction": prediction,
        "confidence": float(round(confidence, 4)),
        "probabilities": {
            "fake": float(round(fake_prob, 4)),
            "real": float(round(real_prob, 4)),
        },
        "summary": summary,
    }

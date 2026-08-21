"""
Model inference service for MedTrace AI.

This is the ONE place the rest of the backend talks to for a prediction.
Right now MODEL_READY is False, so `predict()` falls back to a heuristic
score built from real forensic features (see forensics.py) — it is grounded
in the actual image, not random, but it is NOT your trained model. This lets
you build/test the full upload -> analyze -> result pipeline before training
finishes, then swap in the real model without touching server.py.

--- HOW TO PLUG IN YOUR TRAINED DUAL-ViT MODEL ---

1. Put your weights file somewhere under backend/, e.g. backend/models/dual_vit.pt
   and add its path to backend/.env as MODEL_PATH=models/dual_vit.pt

2. Add your inference deps to requirements.txt (e.g. torch, torchvision,
   timm — whatever you trained with) and `pip install -r requirements.txt`.

3. Implement `load_model()` below to build your architecture and load the
   weights. It's called once, lazily, on the first request.

4. Implement `_run_model(original_image, ela_image)` to:
     - preprocess both images exactly the way you did during training
       (resize, normalize, to-tensor)
     - run the forward pass through your dual-input model
     - return a 3-value softmax array: [authentic, tampered, ai_generated]

5. Flip MODEL_READY to True. `predict()` will automatically use
   `_run_model()` instead of the heuristic fallback from then on.
"""

import os
from datetime import datetime, timezone
import numpy as np
from PIL import Image

MODEL_READY = False  # flip to True once _run_model() is implemented
MODEL_PATH = os.environ.get("MODEL_PATH", "")

_model = None  # cached model instance, populated by load_model()


def load_model():
    """Load the trained Dual-ViT model once and cache it. TODO: implement."""
    global _model
    if _model is not None:
        return _model

    # Example shape once you're ready:
    #
    # import torch
    # from your_arch import DualViT
    # _model = DualViT()
    # _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    # _model.eval()

    raise NotImplementedError("load_model() is not implemented yet — set MODEL_READY=True once it is.")


def _run_model(original_image: Image.Image, ela_image: Image.Image) -> np.ndarray:
    """Run the real trained model. TODO: implement preprocessing + forward pass."""
    model = load_model()

    # Example shape once you're ready:
    #
    # x_orig = preprocess(original_image).unsqueeze(0)
    # x_ela = preprocess(ela_image).unsqueeze(0)
    # with torch.no_grad():
    #     logits = model(x_orig, x_ela)
    #     probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    # return probs

    raise NotImplementedError


def _heuristic_fallback(features: dict) -> np.ndarray:
    """
    Grounded-but-not-a-model placeholder: turns real ELA/noise statistics
    into a plausible probability triple so the API contract, DB storage, and
    frontend can be exercised end-to-end before the model is ready.

    This is intentionally simple and should NOT be treated as a real
    detector — replace it by finishing _run_model() above.
    """
    ela_std = features["ela_std"]
    hot_fraction = features["ela_hot_fraction"]
    noise_std = features["noise_std"]

    # Higher localized ELA response + higher hot-pixel fraction -> lean tampered.
    tamper_signal = np.clip((hot_fraction * 4.0) + (ela_std / 60.0), 0, 1)
    # Very smooth, low-noise images lean (weakly) toward ai_generated.
    ai_signal = np.clip((1.0 - min(noise_std / 25.0, 1.0)) * 0.3, 0, 0.4)

    tampered = 0.1 + 0.6 * tamper_signal
    ai_generated = 0.05 + ai_signal
    authentic = max(0.05, 1.0 - tampered - ai_generated)

    probs = np.array([authentic, tampered, ai_generated], dtype=np.float32)
    probs = probs / probs.sum()
    return probs


def predict(original_image: Image.Image, ela_image: Image.Image, features: dict) -> dict:
    """
    Main entry point used by server.py.

    Returns a dict matching the frontend's expected result shape:
    prediction, confidence, probabilities{authentic,tampered,ai_generated}, summary
    """
    if MODEL_READY:
        probs = _run_model(original_image, ela_image)
        model_note = "Dual-ViT model prediction."
    else:
        probs = _heuristic_fallback(features)
        model_note = (
            "Placeholder heuristic based on ELA/noise statistics — the trained "
            "Dual-ViT model is not wired up yet (see model_service.py)."
        )

    labels = ["authentic", "tampered", "ai_generated"]
    prob_dict = {label: float(round(p, 4)) for label, p in zip(labels, probs)}

    top_label = max(prob_dict, key=prob_dict.get)
    prediction_map = {
        "authentic": "Authentic",
        "tampered": "Tampered",
        "ai_generated": "AI-generated",
    }

    return {
        "prediction": prediction_map[top_label],
        "confidence": prob_dict[top_label],
        "probabilities": prob_dict,
        "summary": model_note,
    }

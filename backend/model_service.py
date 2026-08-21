"""
Model inference service for MedTrace AI.

Loads a Hugging Face/Trainer checkpoint containing:
    config.json
    model.safetensors
    training_args.bin
    trainer_state.json
    ...

The checkpoint directory is supplied with MODEL_PATH.

The supplied checkpoint is a standard Hugging Face ViTForImageClassification
model with:
    - 3 input channels
    - 224x224 input size
    - 2 classes: fake (0), real (1)

This checkpoint therefore accepts ONE RGB image. It cannot consume original
+ ELA as a 6-channel input, and it cannot distinguish tampered vs
AI-generated because those are not separate training classes.
"""

import os
import numpy as np
import torch
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_READY = bool(MODEL_PATH)

_model = None
_processor = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path() -> str:
    """Return a useful checkpoint path and fail early if it is missing."""
    if not MODEL_PATH:
        raise RuntimeError(
            "MODEL_PATH is not set. Point MODEL_PATH to the directory containing "
            "config.json and model.safetensors."
        )

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"MODEL_PATH does not exist or is not a directory: {MODEL_PATH}"
        )

    config_file = os.path.join(MODEL_PATH, "config.json")
    weights_file = os.path.join(MODEL_PATH, "model.safetensors")

    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Missing config.json in {MODEL_PATH}")

    if not os.path.isfile(weights_file):
        # Transformers can also load pytorch_model.bin, so allow that form.
        pytorch_file = os.path.join(MODEL_PATH, "pytorch_model.bin")
        if not os.path.isfile(pytorch_file):
            raise FileNotFoundError(
                f"No model weights found in {MODEL_PATH}. Expected "
                "model.safetensors or pytorch_model.bin."
            )

    return MODEL_PATH


def load_model():
    """Load the trained Hugging Face image-classification model once."""
    global _model, _processor

    if _model is not None:
        return _model

    checkpoint = _checkpoint_path()

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install transformers and torch before starting "
            "the backend."
        ) from exc

    # The checkpoint shown in the project contains config.json + model.safetensors.
    # AutoModelForImageClassification reads both and reconstructs the architecture.
    _model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    # There may not be a preprocessor_config.json in the checkpoint. In that case
    # _preprocess() below uses the model config's image size and ImageNet defaults.
    try:
        _processor = AutoImageProcessor.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
    except Exception:
        _processor = None

    _model.to(_device)
    _model.eval()

    return _model


def _preprocess(image: Image.Image, model) -> torch.Tensor:
    """
    Preprocess one RGB image for the supplied ViT checkpoint.

    The checkpoint is configured for 224x224 RGB input. If a Hugging Face
    image processor exists beside the checkpoint, use it; otherwise use
    standard ImageNet normalization.
    """
    image = image.convert("RGB")

    if _processor is not None:
        encoded = _processor(images=image, return_tensors="pt")
        return encoded["pixel_values"].squeeze(0)

    size = int(getattr(model.config, "image_size", 224))
    image = image.resize((size, size), Image.Resampling.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    return (tensor - mean) / std


def _run_model(original_image: Image.Image, ela_image: Image.Image) -> np.ndarray:
    """
    Run the supplied binary ViT checkpoint.

    The config.json explicitly says:
        num_channels = 3
        image_size = 224
        fake = 0
        real = 1

    Therefore only the original RGB image is passed to the model. The ELA
    image is deliberately not concatenated because this checkpoint does not
    accept 6-channel input.

    Returns:
        [fake_probability, real_probability]
    """
    model = load_model()
    inputs = _preprocess(original_image, model).unsqueeze(0).to(_device)

    with torch.inference_mode():
        outputs = model(pixel_values=inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        probs = probs.detach().cpu().numpy()

    if probs.shape[0] != 2:
        raise RuntimeError(
            f"Expected 2 output classes from the checkpoint, got {probs.shape[0]}. "
            "Check config.json and the trained checkpoint."
        )

    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / probs.sum()
    return probs


def _binary_model_predict(original_image: Image.Image, ela_image: Image.Image):
    """Return fake/real probabilities and the top binary prediction."""
    probs = _run_model(original_image, ela_image)

    # config.json defines:
    # 0 -> fake
    # 1 -> real
    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    if fake_prob >= real_prob:
        return "Fake", fake_prob, fake_prob, real_prob

    return "Real", real_prob, fake_prob, real_prob


def _heuristic_fallback(features: dict) -> np.ndarray:
    """
    Fallback only when MODEL_PATH is not configured.
    """
    ela_std = features["ela_std"]
    hot_fraction = features["ela_hot_fraction"]
    noise_std = features["noise_std"]

    tamper_signal = np.clip((hot_fraction * 4.0) + (ela_std / 60.0), 0, 1)
    ai_signal = np.clip((1.0 - min(noise_std / 25.0, 1.0)) * 0.3, 0, 0.4)

    tampered = 0.1 + 0.6 * tamper_signal
    ai_generated = 0.05 + ai_signal
    authentic = max(0.05, 1.0 - tampered - ai_generated)

    probs = np.array([authentic, tampered, ai_generated], dtype=np.float32)
    return probs / probs.sum()


def predict(original_image: Image.Image, ela_image: Image.Image, features: dict) -> dict:
    """
    Main entry point used by server.py.

    This checkpoint is binary:
        fake = class 0
        real = class 1

    The returned probabilities are therefore binary. It would be incorrect
    to fabricate separate tampered/AI-generated probabilities from this model.
    """
    if MODEL_READY:
        prediction, confidence, fake_prob, real_prob = _binary_model_predict(
            original_image, ela_image
        )
        model_note = "Trained ViT binary classification prediction."
    else:
        # Keep the old fallback available when no checkpoint is configured.
        fallback = _heuristic_fallback(features)
        fake_prob = float(fallback[1] + fallback[2])
        real_prob = float(fallback[0])
        prediction = "Fake" if fake_prob >= real_prob else "Real"
        confidence = max(fake_prob, real_prob)
        model_note = (
            "Placeholder heuristic — MODEL_PATH is not configured."
        )

    return {
        "prediction": prediction,
        "confidence": float(round(confidence, 4)),
        "probabilities": {
            "fake": float(round(fake_prob, 4)),
            "real": float(round(real_prob, 4)),
        },
        "summary": model_note,
    }

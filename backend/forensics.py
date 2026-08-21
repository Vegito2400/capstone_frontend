"""
Real image forensics helpers for MedTrace AI.

These functions do genuine pixel-level work (no random numbers):

- generate_ela_image(): classic Error Level Analysis — resave the image at a
  known JPEG quality, diff it against the original, and amplify the result.
  Tampered/spliced regions usually show a different error signature than the
  rest of the image because they were compressed a different number of times.

- compute_forensic_features(): a handful of cheap statistical features
  (ELA intensity/variance, high-frequency noise, basic metadata flags) that
  are useful both as a heatmap-ish signal for the frontend and as input
  features for model_service.py while the trained Dual-ViT model isn't wired
  up yet.
"""

import io
from PIL import Image, ImageChops
import numpy as np


def load_image(file_bytes: bytes) -> Image.Image:
    """Open arbitrary uploaded image bytes as an RGB Pillow image."""
    image = Image.open(io.BytesIO(file_bytes))
    image = image.convert("RGB")
    return image


def generate_ela_image(image: Image.Image, quality: int = 90, scale: int = 15) -> Image.Image:
    """Return the Error Level Analysis representation of `image`."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(image, resaved)
    diff_np = np.asarray(diff).astype(np.float32)

    max_diff = diff_np.max()
    if max_diff == 0:
        max_diff = 1.0
    amplified = np.clip(diff_np * (scale * 255.0 / max_diff), 0, 255).astype(np.uint8)

    return Image.fromarray(amplified)


def _high_frequency_noise(gray: np.ndarray) -> float:
    """Rough noise estimate via a Laplacian-style high-pass filter."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    h, w = gray.shape
    for dy in range(3):
        for dx in range(3):
            k = kernel[dy, dx]
            if k == 0:
                continue
            out += k * padded[dy:dy + h, dx:dx + w]
    return float(np.std(out))


def compute_forensic_features(image: Image.Image, ela_image: Image.Image) -> dict:
    """Cheap statistical features derived from the image and its ELA map."""
    ela_np = np.asarray(ela_image).astype(np.float32)
    gray = np.asarray(image.convert("L")).astype(np.float32)

    ela_mean = float(ela_np.mean())
    ela_std = float(ela_np.std())
    ela_max = float(ela_np.max())

    # Fraction of pixels whose ELA response is a strong outlier vs the image mean —
    # a rough proxy for "localized" tampering rather than uniform recompression noise.
    threshold = ela_mean + 2 * ela_std
    hot_fraction = float((ela_np.mean(axis=-1) > threshold).mean()) if ela_np.ndim == 3 else 0.0

    noise_std = _high_frequency_noise(gray)

    return {
        "ela_mean": round(ela_mean, 4),
        "ela_std": round(ela_std, 4),
        "ela_max": round(ela_max, 4),
        "ela_hot_fraction": round(hot_fraction, 4),
        "noise_std": round(noise_std, 4),
        "width": image.width,
        "height": image.height,
    }

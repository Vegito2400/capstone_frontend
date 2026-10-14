import io
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance


def generate_ela_image(image, quality=90):
    """
    Generate ELA image from PIL Image
    """
    image = image.convert("RGB")
    # Keep the recompressed copy in memory so uploaded images are never
    # written to disk and concurrent requests cannot overwrite one another.
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    # Compute difference
    ela_image = ImageChops.difference(image, compressed)

    # Scale differences
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def save_ela_result(image_obj, output_dir="results", filename="ela_analysis.jpg"):
    # Create the directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the full file path
    final_file_path = output_path / filename
    
    # Save the image
    image_obj.save(final_file_path)
    return str(final_file_path)

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError


SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
MIN_IMAGE_DIMENSION = 96
MAX_PROCESSING_DIMENSION = 1200


def validate_soil_image(image_data: bytes | Image.Image) -> dict[str, Any]:
    if isinstance(image_data, Image.Image):
        image = image_data
    else:
        if not image_data:
            return {"valid": False, "error": "The uploaded image is empty."}
        try:
            image = Image.open(io.BytesIO(image_data))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            return {"valid": False, "error": f"The image could not be read: {error}"}
    width, height = image.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return {"valid": False, "error": "Image is too small for reliable visual assessment. Please upload a clearer soil photograph."}
    return {"valid": True, "width": width, "height": height, "format": image.format or "uploaded image"}


def preprocess_soil_image(image_data: bytes | Image.Image) -> Image.Image:
    if isinstance(image_data, Image.Image):
        image = image_data.copy()
    else:
        image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")
    image.thumbnail((MAX_PROCESSING_DIMENSION, MAX_PROCESSING_DIMENSION), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image.filter(ImageFilter.MedianFilter(size=3))


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def analyze_soil_color(image: Image.Image) -> dict[str, Any]:
    pixels = _rgb_array(image).reshape(-1, 3)
    red, green, blue = pixels.mean(axis=0)
    brightness = float(pixels.mean())
    red_ratio = red - green
    blue_ratio = blue - red
    if brightness < 0.24:
        category = "Dark brown"
    elif red_ratio > 0.08 and red > blue * 1.18:
        category = "Reddish/brown"
    elif brightness > 0.62:
        category = "Light brown"
    elif abs(red_ratio) < 0.035 and abs(blue_ratio) < 0.035:
        category = "Grayish/brown"
    else:
        category = "Brown"
    return {"category": category, "average_rgb": [round(float(value) * 255, 1) for value in (red, green, blue)], "brightness": round(brightness * 100, 1), "note": "Visual color description only; it is not a nutrient measurement."}


def estimate_visual_texture(image: Image.Image) -> dict[str, str]:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    horizontal_change = np.abs(np.diff(grayscale, axis=1)).mean()
    vertical_change = np.abs(np.diff(grayscale, axis=0)).mean()
    texture_signal = float(horizontal_change + vertical_change)
    if texture_signal < 0.09:
        category = "Fine-looking"
    elif texture_signal > 0.18:
        category = "Coarse-looking"
    else:
        category = "Medium-looking"
    return {"category": category, "note": "Approximate visual texture observation; not laboratory particle-size analysis."}


def estimate_visual_moisture(image: Image.Image) -> dict[str, str]:
    pixels = _rgb_array(image)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    if brightness < 0.25 and contrast < 0.2:
        category = "Appears wet"
    elif brightness > 0.65:
        category = "Appears dry"
    elif contrast > 0.27:
        category = "Appears moderately moist"
    else:
        category = "Uncertain"
    return {"category": category, "note": "Visible moisture indication only; no actual moisture percentage is measured."}


def assess_image_quality(image: Image.Image) -> dict[str, Any]:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(grayscale.mean())
    horizontal_edges = np.diff(grayscale, axis=1)
    vertical_edges = np.diff(grayscale, axis=0)
    sharpness_signal = float(horizontal_edges.var() + vertical_edges.var())
    width, height = image.size
    issues: list[str] = []
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        issues.append("dimensions are very small")
    if brightness < 20:
        issues.append("image is very dark")
    if brightness > 240:
        issues.append("image is overexposed")
    if sharpness_signal < 35:
        issues.append("image may be blurred or lack visible detail")
    usable = not issues
    return {"usable": usable, "status": "Usable for preliminary assessment" if usable else "Insufficient quality", "width": width, "height": height, "aspect_ratio": round(width / height, 2), "brightness": round(brightness, 1), "sharpness_signal": round(sharpness_signal, 1), "issues": issues, "message": "Image quality is sufficient for a preliminary visual assessment." if usable else "Image quality is insufficient for reliable visual assessment. Please upload a clearer soil photograph."}


def generate_visual_assessment(image_data: bytes | Image.Image) -> dict[str, Any]:
    validation = validate_soil_image(image_data)
    if not validation["valid"]:
        return {"valid": False, "error": validation["error"]}
    try:
        image = preprocess_soil_image(image_data)
        quality = assess_image_quality(image)
        return {"valid": True, "image": image, "quality": quality, "color": analyze_soil_color(image), "texture": estimate_visual_texture(image), "moisture": estimate_visual_moisture(image), "summary": "Preliminary visual observations are available. Laboratory soil values are required for fertility prediction."}
    except (OSError, ValueError, UnidentifiedImageError) as error:
        return {"valid": False, "error": f"The image could not be processed: {error}"}

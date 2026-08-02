"""
test_quality.py

Unit Tests for Contactless Fingerprint Quality Assessment
"""

import os
import cv2
import numpy as np
import pytest

from quality_assessment import (
    load_image,
    check_blur,
    check_brightness,
    check_glare,
    check_roi_completeness,
    check_ridge_clarity,
    quality_gate
)


# -------------------------------------------------------
# Helper Function
# -------------------------------------------------------

def create_dummy_image():
    """
    Creates a random image for testing.
    """
    return np.random.randint(
        0,
        256,
        (300, 300, 3),
        dtype=np.uint8
    )


# -------------------------------------------------------
# Image Loader Tests
# -------------------------------------------------------

def test_load_image_numpy():
    img = create_dummy_image()

    loaded = load_image(img)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == img.shape


def test_load_image_invalid_path():
    with pytest.raises(ValueError):
        load_image("invalid_image.jpg")


def test_load_image_invalid_input():
    with pytest.raises(TypeError):
        load_image(100)


# -------------------------------------------------------
# Blur Test
# -------------------------------------------------------

def test_check_blur():

    img = create_dummy_image()

    result = check_blur(img)

    assert "blur_score" in result
    assert "is_blurry" in result

    assert isinstance(result["blur_score"], float)
    assert isinstance(result["is_blurry"], bool)


# -------------------------------------------------------
# Brightness Test
# -------------------------------------------------------

def test_check_brightness():

    img = create_dummy_image()

    result = check_brightness(img)

    assert "brightness" in result
    assert "too_dark" in result
    assert "too_bright" in result

    assert isinstance(result["brightness"], float)


# -------------------------------------------------------
# Glare Test
# -------------------------------------------------------

def test_check_glare():

    img = create_dummy_image()

    result = check_glare(img)

    assert "has_glare" in result
    assert "glare_fraction" in result

    assert isinstance(result["glare_fraction"], float)


# -------------------------------------------------------
# ROI Test
# -------------------------------------------------------

def test_check_roi():

    img = create_dummy_image()

    result = check_roi_completeness(img)

    assert "roi_fraction" in result
    assert "roi_complete" in result


# -------------------------------------------------------
# Ridge Test
# -------------------------------------------------------

def test_check_ridge():

    img = create_dummy_image()

    result = check_ridge_clarity(img)

    assert "ridge_score" in result
    assert "ridges_clear" in result


# -------------------------------------------------------
# Quality Gate Test (NumPy Image)
# -------------------------------------------------------

def test_quality_gate_numpy():

    img = create_dummy_image()

    result = quality_gate(img)

    assert "passed" in result
    assert "composite_score" in result
    assert "guidance" in result

    assert "blur" in result
    assert "brightness" in result
    assert "glare" in result
    assert "roi" in result
    assert "ridge" in result


# -------------------------------------------------------
# Quality Gate Test (Image Path)
# -------------------------------------------------------

def test_quality_gate_image_path(tmp_path):

    img = create_dummy_image()

    image_path = tmp_path / "sample.jpg"

    cv2.imwrite(str(image_path), img)

    result = quality_gate(str(image_path))

    assert result is not None
    assert "composite_score" in result


# -------------------------------------------------------
# Composite Score Range
# -------------------------------------------------------

def test_composite_score_range():

    img = create_dummy_image()

    result = quality_gate(img)

    score = result["composite_score"]

    assert 0 <= score <= 100


# -------------------------------------------------------
# Guidance Message
# -------------------------------------------------------

def test_guidance_is_string():

    img = create_dummy_image()

    result = quality_gate(img)

    assert isinstance(result["guidance"], str)


# -------------------------------------------------------
# Passed Value
# -------------------------------------------------------

def test_passed_is_boolean():

    img = create_dummy_image()

    result = quality_gate(img)

    assert isinstance(result["passed"], bool)
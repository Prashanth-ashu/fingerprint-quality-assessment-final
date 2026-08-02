"""
quality_assessment.py

Assignment 4
Contactless Fingerprint Quality Assessment Pipeline

Author : Prashanth
Python : 3.9+

"""

import cv2
import numpy as np
from typing import Dict, Union


# ==========================================================
# DEFAULT THRESHOLDS
# ==========================================================

DEFAULT_THRESHOLDS = {
    "blur_threshold": 10.0,
    "brightness_min": 50.0,
    "brightness_max": 210.0,
    "glare_threshold": 0.05,
    "roi_threshold": 0.15,
    "ridge_threshold": 15.0
}


# ==========================================================
# IMAGE LOADER
# ==========================================================

def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Accepts either

    - image path
    - OpenCV image (numpy array)

    Returns
    -------
    BGR Image
    """

    if isinstance(image_input, str):
        image = cv2.imread(image_input)

        if image is None:
            raise ValueError(
                f"Unable to load image : {image_input}"
            )

        return image

    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    raise TypeError(
        "Input should be image path or numpy array."
    )


# ==========================================================
# COMMON HELPER
# ==========================================================

def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert image into grayscale.
    """

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )


# ==========================================================
# METRIC 1
# BLUR DETECTION
# ==========================================================

def check_blur(
    image_bgr: np.ndarray,
    threshold: float = DEFAULT_THRESHOLDS["blur_threshold"]
) -> Dict:

    """
    Blur Detection

    Uses:
    Laplacian Variance

    Higher variance
        -> sharper image

    Lower variance
        -> blurry image
    """

    gray = to_gray(image_bgr)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    blur_score = float(
        laplacian.var()
    )

    return {

        "blur_score": round(
            blur_score,
            2
        ),

        "is_blurry":
            blur_score < threshold

    }


# ==========================================================
# METRIC 2
# BRIGHTNESS
# ==========================================================

def check_brightness(
    image_bgr: np.ndarray,
    min_thresh: float = DEFAULT_THRESHOLDS["brightness_min"],
    max_thresh: float = DEFAULT_THRESHOLDS["brightness_max"]
) -> Dict:

    """
    Brightness Quality

    Uses

    Mean grayscale intensity
    """

    gray = to_gray(image_bgr)

    brightness = float(
        np.mean(gray)
    )

    return {

        "brightness":
            round(brightness, 2),

        "too_dark":
            brightness < min_thresh,

        "too_bright":
            brightness > max_thresh

    }


# ==========================================================
# METRIC 3
# GLARE
# ==========================================================

def check_glare(
    image_bgr: np.ndarray,
    max_glare_ratio: float = DEFAULT_THRESHOLDS["glare_threshold"]
) -> Dict:

    """
    Detect Specular Reflection

    Counts pixels

    intensity > 240
    """

    gray = to_gray(image_bgr)

    glare_pixels = np.sum(
        gray > 240
    )

    total_pixels = gray.size

    glare_fraction = float(
        glare_pixels / total_pixels
    )

    return {

        "has_glare":
            glare_fraction > max_glare_ratio,

        "glare_fraction":
            round(glare_fraction, 4)

    }


# ==========================================================
# NORMALIZATION HELPERS
# ==========================================================

def normalize_blur(score: float):

    return min(
        1.0,
        score / 100.0
    )


def normalize_brightness(score: float):

    value = 1.0 - abs(score - 128.0) / 128.0

    return max(
        0.0,
        min(value, 1.0)
    )


def normalize_glare(glare_ratio: float):

    return max(
        0.0,
        1.0 - glare_ratio / 0.05
    )

# ==========================================================
# METRIC 4
# ROI (REGION OF INTEREST) COMPLETENESS
# ==========================================================

def check_roi_completeness(
    image_bgr: np.ndarray,
    min_roi_ratio: float = DEFAULT_THRESHOLDS["roi_threshold"]
) -> Dict:
    """
    Estimate the finger Region of Interest (ROI)
    using Otsu thresholding.

    Returns
    -------
    roi_fraction
    roi_complete
    """

    gray = to_gray(image_bgr)

    # Reduce image noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu Threshold
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Count foreground pixels
    foreground_pixels = np.sum(thresh > 0)
    total_pixels = gray.size

    roi_fraction = float(
        foreground_pixels / total_pixels
    )

    return {

        "roi_fraction":
            round(roi_fraction, 4),

        "roi_complete":
            roi_fraction >= min_roi_ratio

    }


# ==========================================================
# METRIC 5
# RIDGE CLARITY
# ==========================================================

def check_ridge_clarity(
    image_bgr: np.ndarray,
    threshold: float = DEFAULT_THRESHOLDS["ridge_threshold"]
) -> Dict:
    """
    Estimate ridge clarity using
    Gabor filter response variance.
    """

    gray = to_gray(image_bgr)

    # Gabor Kernel
    kernel = cv2.getGaborKernel(

        ksize=(21, 21),

        sigma=5.0,

        theta=np.pi / 4,

        lambd=10.0,

        gamma=0.5,

        psi=0,

        ktype=cv2.CV_32F

    )

    filtered = cv2.filter2D(
        gray,
        cv2.CV_64F,
        kernel
    )

    ridge_score = float(
        np.var(filtered) / 100.0
    )

    return {

        "ridge_score":
            round(ridge_score, 2),

        "ridges_clear":
            ridge_score >= threshold

    }


# ==========================================================
# NORMALIZATION
# ROI
# ==========================================================

def normalize_roi(
    roi_fraction: float
):
    """
    Normalize ROI into
    0 to 1
    """

    return min(
        1.0,
        roi_fraction / 0.30
    )


# ==========================================================
# NORMALIZATION
# RIDGE
# ==========================================================

def normalize_ridge(
    ridge_score: float,
    threshold: float = DEFAULT_THRESHOLDS["ridge_threshold"]
):
    """
    Normalize ridge score.
    """

    return min(
        1.0,
        ridge_score / threshold
    )


# ==========================================================
# COMPOSITE SCORE WEIGHTS
# ==========================================================

QUALITY_WEIGHTS = {

    "blur": 0.25,

    "brightness": 0.15,

    "glare": 0.15,

    "roi": 0.20,

    "ridge": 0.25

}

# ==========================================================
# COMPOSITE QUALITY SCORE
# ==========================================================

def calculate_composite_score(
    blur_result: Dict,
    brightness_result: Dict,
    glare_result: Dict,
    roi_result: Dict,
    ridge_result: Dict
) -> float:
    """
    Calculate the final quality score (0–100)
    using normalized metrics.
    """

    n_blur = normalize_blur(
        blur_result["blur_score"]
    )

    n_brightness = normalize_brightness(
        brightness_result["brightness"]
    )

    n_glare = normalize_glare(
        glare_result["glare_fraction"]
    )

    n_roi = normalize_roi(
        roi_result["roi_fraction"]
    )

    n_ridge = normalize_ridge(
        ridge_result["ridge_score"]
    )

    composite = (

        QUALITY_WEIGHTS["blur"] * n_blur +

        QUALITY_WEIGHTS["brightness"] * n_brightness +

        QUALITY_WEIGHTS["glare"] * n_glare +

        QUALITY_WEIGHTS["roi"] * n_roi +

        QUALITY_WEIGHTS["ridge"] * n_ridge

    ) * 100.0

    return round(composite, 1)


# ==========================================================
# GUIDANCE MESSAGE
# ==========================================================

def generate_guidance(
    blur_result,
    brightness_result,
    glare_result,
    roi_result,
    ridge_result
):
    """
    Returns the highest-priority guidance message.
    """

    if blur_result["is_blurry"]:
        return (
            "Image is too blurry. "
            "Hold your phone steady and re-focus."
        )

    if brightness_result["too_dark"]:
        return (
            "Lighting is too dark. "
            "Move to a brighter place or turn on flash."
        )

    if brightness_result["too_bright"]:
        return (
            "Image is overexposed. "
            "Reduce direct light."
        )

    if glare_result["has_glare"]:
        return (
            "Glare detected. "
            "Tilt your finger slightly."
        )

    if not roi_result["roi_complete"]:
        return (
            "Finger is too far away. "
            "Move closer so it fills the frame."
        )

    if not ridge_result["ridges_clear"]:
        return (
            "Fingerprint ridges are unclear. "
            "Clean camera lens and adjust lighting."
        )

    return "Good capture — ready for processing."


# ==========================================================
# HARD FAILURE CHECK
# ==========================================================

def has_hard_failure(
    blur_result,
    brightness_result,
    glare_result,
    roi_result,
    ridge_result
):
    """
    Returns True if any mandatory metric fails.
    """

    return (

        blur_result["is_blurry"]

        or

        brightness_result["too_dark"]

        or

        brightness_result["too_bright"]

        or

        glare_result["has_glare"]

        or

        not roi_result["roi_complete"]

        or

        not ridge_result["ridges_clear"]

    )


# ==========================================================
# MASTER QUALITY GATE
# ==========================================================

def quality_gate(
    image_input: Union[str, np.ndarray]
) -> Dict:
    """
    Main Quality Assessment Pipeline.

    Parameters
    ----------
    image_input:
        image path
        OR
        numpy image

    Returns
    -------
    dictionary
    """

    image = load_image(image_input)

    blur_result = check_blur(image)

    brightness_result = check_brightness(image)

    glare_result = check_glare(image)

    roi_result = check_roi_completeness(image)

    ridge_result = check_ridge_clarity(image)

    composite_score = calculate_composite_score(

        blur_result,

        brightness_result,

        glare_result,

        roi_result,

        ridge_result

    )

    hard_failure = has_hard_failure(

        blur_result,

        brightness_result,

        glare_result,

        roi_result,

        ridge_result

    )

    passed = (

        composite_score >= 60.0

        and

        not hard_failure

    )

    guidance = generate_guidance(

        blur_result,

        brightness_result,

        glare_result,

        roi_result,

        ridge_result

    )

    return {

        "passed": passed,

        "composite_score": composite_score,

        "blur": blur_result,

        "brightness": brightness_result,

        "glare": glare_result,

        "roi": roi_result,

        "ridge": ridge_result,

        "guidance": guidance

    }


# ==========================================================
# SAMPLE EXECUTION
# ==========================================================

if __name__ == "__main__":

    IMAGE_PATH = "sample.jpg"

    try:

        result = quality_gate(IMAGE_PATH)

        print("=" * 60)
        print("CONTACTLESS FINGERPRINT QUALITY REPORT")
        print("=" * 60)

        print(f"Passed            : {result['passed']}")
        print(f"Composite Score   : {result['composite_score']}")

        print()

        print("Blur")
        print(result["blur"])

        print()

        print("Brightness")
        print(result["brightness"])

        print()

        print("Glare")
        print(result["glare"])

        print()

        print("ROI")
        print(result["roi"])

        print()

        print("Ridge")
        print(result["ridge"])

        print()

        print("Guidance")
        print(result["guidance"])

    except Exception as e:

        print(e)
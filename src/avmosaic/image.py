"""Image I/O and the single-image pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .censor import CensorOptions, apply_censor, draw_detections
from .detector import Detection, Detector, to_bgr8

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, Unicode paths included (``cv2.imread`` fails on them on Windows).

    Images with an alpha channel come back as BGRA; everything else as BGR with
    the EXIF orientation applied.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        return image
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:  # pragma: no cover - decodable once, decodable twice
        raise ValueError(f"cannot decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"cannot encode image as {ext}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def detect_image(src: str | Path, detector: Detector) -> tuple[np.ndarray, list[Detection]]:
    image = read_image(src)
    return image, detector.detect(to_bgr8(image))


def censor_image(
    src: str | Path,
    dst: str | Path,
    detector: Detector,
    options: CensorOptions,
    *,
    draw_boxes: bool = False,
    jpeg_quality: int = 95,
    manual: Sequence[tuple[float, float, float, float]] = (),
) -> list[Detection]:
    """Censor one image. ``manual`` boxes (x1, y1, x2, y2) are censored as well."""
    image, detections = detect_image(src, detector)
    regions = detections + [Detection("MANUAL", 1.0, *box) for box in manual]
    apply_censor(image, regions, options)
    if draw_boxes:
        draw_detections(image, regions)
    write_image(dst, image, jpeg_quality)
    return detections

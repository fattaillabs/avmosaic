"""Turning detections into censored pixels."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection

METHODS = ("mosaic", "blur", "fill")
SHAPES = ("rect", "ellipse")

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive), pixel coordinates


@dataclass
class CensorOptions:
    method: str = "mosaic"
    # "auto" = max(4 px, 1/100 of the long side), an int in px, or "N%" of the long side.
    block_size: int | str = "auto"
    shape: str = "rect"
    # Grow every box by this fraction of its width/height on each side.
    padding: float = 0.15
    fill_color: tuple[int, int, int] = (0, 0, 0)  # BGR

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if self.padding < 0:
            raise ValueError("padding must be >= 0")
        self.block_size = parse_block_size(self.block_size)


def parse_block_size(value: int | str) -> int | str:
    """Validate a block size spec; returns an int, ``"auto"`` or ``"N%"``."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 1:
            raise ValueError("block size must be >= 1 px")
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text.endswith("%"):
        try:
            pct = float(text[:-1])
        except ValueError:
            raise ValueError(f"invalid block size {value!r}") from None
        if not 0 < pct <= 100:
            raise ValueError("block size percentage must be in (0, 100]")
        return f"{pct:g}%"
    try:
        return parse_block_size(int(text))
    except ValueError:
        raise ValueError(f"invalid block size {value!r}; use auto, a pixel count or N%") from None


def resolve_block_size(spec: int | str, width: int, height: int) -> int:
    long_side = max(width, height)
    if spec == "auto":
        # A commonly cited mosaic guideline: tiles of at least 1/100 of the
        # image's long side and never smaller than 4 px.
        return max(4, math.ceil(long_side / 100))
    if isinstance(spec, str) and spec.endswith("%"):
        return max(2, math.ceil(long_side * float(spec[:-1]) / 100))
    return int(spec)


def expand_box(det: Detection | Sequence[float], padding: float, width: int, height: int) -> Box:
    if isinstance(det, Detection):
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    else:
        x1, y1, x2, y2 = det
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    return (
        max(0, math.floor(x1 - pw)),
        max(0, math.floor(y1 - ph)),
        min(width, math.ceil(x2 + pw)),
        min(height, math.ceil(y2 + ph)),
    )


def build_mask(height: int, width: int, boxes: Iterable[Box], shape: str = "rect") -> np.ndarray:
    """Union of all regions as a uint8 mask (255 = censor)."""
    mask = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        if shape == "rect":
            mask[y1:y2, x1:x2] = 255
        else:
            # Circumscribe the box (semi-axes scaled by sqrt 2) so the ellipse
            # still covers the whole padded rectangle, corners included.
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            axes = ((x2 - x1) / math.sqrt(2), (y2 - y1) / math.sqrt(2))
            cv2.ellipse(mask, (center, (axes[0] * 2, axes[1] * 2), 0), 255, -1)
    return mask


def _grid_bounds(mask: np.ndarray, block: int) -> tuple[int, int, int, int] | None:
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    h, w = mask.shape
    y1 = ys[0] // block * block
    x1 = xs[0] // block * block
    y2 = min(h, -(-(ys[-1] + 1) // block) * block)
    x2 = min(w, -(-(xs[-1] + 1) // block) * block)
    return x1, y1, x2, y2


def pixelate(image: np.ndarray, mask: np.ndarray, block: int) -> None:
    """Mosaic every grid block (aligned to the image origin) that the mask touches.

    Aligning to a fixed grid keeps tiles from shimmering between video frames and
    makes overlapping regions produce exactly the same tiles.
    """
    bounds = _grid_bounds(mask, block)
    if bounds is None:
        return
    x1, y1, x2, y2 = bounds
    roi = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    pad_y, pad_x = (-h) % block, (-w) % block
    padded = cv2.copyMakeBorder(roi, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
    by, bx = padded.shape[0] // block, padded.shape[1] // block
    # INTER_AREA with an integer factor is an exact per-block mean.
    small = cv2.resize(padded, (bx, by), interpolation=cv2.INTER_AREA)
    tiles = cv2.resize(small, (bx * block, by * block), interpolation=cv2.INTER_NEAREST)[:h, :w]
    if tiles.ndim < roi.ndim:
        tiles = tiles[..., np.newaxis]
    padded_mask = np.pad(roi_mask, ((0, pad_y), (0, pad_x)))
    block_hit = padded_mask.reshape(by, block, bx, block).max(axis=(1, 3)) > 0
    hit = np.repeat(np.repeat(block_hit, block, axis=0), block, axis=1)[:h, :w]
    roi[hit] = tiles[hit]


def blur(image: np.ndarray, mask: np.ndarray, boxes: Sequence[Box]) -> None:
    for x1, y1, x2, y2 in boxes:
        size = max(x2 - x1, y2 - y1)
        if size <= 0:
            continue
        sigma = max(3.0, size / 8)
        margin = int(sigma * 3)
        h, w = image.shape[:2]
        rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
        rx2, ry2 = min(w, x2 + margin), min(h, y2 + margin)
        roi = image[ry1:ry2, rx1:rx2]
        blurred = cv2.GaussianBlur(roi, (0, 0), sigma)
        if blurred.ndim < roi.ndim:
            blurred = blurred[..., np.newaxis]
        sub = mask[ry1:ry2, rx1:rx2] > 0
        roi[sub] = blurred[sub]


def apply_censor(
    image: np.ndarray,
    detections: Iterable[Detection | Sequence[float]],
    options: CensorOptions,
) -> np.ndarray:
    """Censor ``image`` in place; returns the mask that was applied."""
    h, w = image.shape[:2]
    boxes = [expand_box(d, options.padding, w, h) for d in detections]
    mask = build_mask(h, w, boxes, options.shape)
    if not boxes:
        return mask
    if options.method == "mosaic":
        pixelate(image, mask, resolve_block_size(options.block_size, w, h))
    elif options.method == "blur":
        blur(image, mask, boxes)
    else:
        color = options.fill_color
        if image.ndim == 2:
            image[mask > 0] = int(round(sum(color) / 3))
        else:
            fill = np.array(list(color) + [255] * (image.shape[2] - 3), dtype=image.dtype)
            image[mask > 0] = fill[: image.shape[2]]
    return mask


_BOX_COLORS = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (220, 80, 220), (0, 0, 255)]


def draw_detections(image: np.ndarray, detections: Iterable[Detection]) -> None:
    """Overlay boxes, classes and scores (for tuning, not for publishing)."""
    from .labels import LABELS

    thickness = max(1, round(max(image.shape[:2]) / 500))
    for det in detections:
        color = (*_BOX_COLORS[LABELS.index(det.label) % len(_BOX_COLORS)], 255)  # opaque on BGRA
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(image, p1, p2, color, thickness)
        text = f"{det.label} {det.score:.2f}"
        scale = 0.4 * thickness
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        ty = max(th + base, p1[1])
        cv2.rectangle(image, (p1[0], ty - th - base), (p1[0] + tw, ty), color, -1)
        cv2.putText(image, text, (p1[0], ty - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0, 255),
                    thickness, cv2.LINE_AA)

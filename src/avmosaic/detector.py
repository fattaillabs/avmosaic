"""ONNX detector for NudeNet v3 models (YOLOv8-style output)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .labels import LABELS


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "class": self.label,
            "score": round(self.score, 4),
            "box": [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)],
        }


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections for a BGR uint8 image of shape (H, W, 3)."""
        ...


DEVICE_PROVIDERS: dict[str, tuple[str, ...]] = {
    "cpu": ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "tensorrt": ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
    "directml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "coreml": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
    "openvino": ("OpenVINOExecutionProvider", "CPUExecutionProvider"),
}
DEVICES = ("auto", *DEVICE_PROVIDERS)
CHANNEL_ORDERS = ("rgb", "bgr")
# On an M1, CoreML ran 320n at 15.7 ms vs 33.1 ms per 1080p frame on CPU, with identical scores.
_AUTO_ORDER = ("cuda", "directml", "coreml", "cpu")


def resolve_providers(device: str, available: Sequence[str] | None = None) -> list[str]:
    if available is None:
        import onnxruntime

        available = onnxruntime.get_available_providers()
    if device == "auto":
        for name in _AUTO_ORDER:
            if DEVICE_PROVIDERS[name][0] in available:
                return [p for p in DEVICE_PROVIDERS[name] if p in available]
        return ["CPUExecutionProvider"]
    if device not in DEVICE_PROVIDERS:
        raise ValueError(f"unknown device {device!r}; choose from {', '.join(DEVICES)}")
    wanted = DEVICE_PROVIDERS[device]
    if wanted[0] not in available:
        raise ValueError(
            f"device {device!r} needs {wanted[0]}, but this onnxruntime only has "
            f"{', '.join(available)} (for CUDA install onnxruntime-gpu, for DirectML "
            "onnxruntime-directml)"
        )
    return [p for p in wanted if p in available]


def to_bgr8(image: np.ndarray) -> np.ndarray:
    """Convert gray / BGRA / 16-bit images into the 8-bit BGR the model expects."""
    if image.dtype != np.uint8:
        scale = 255.0 / np.iinfo(image.dtype).max if image.dtype.kind in "ui" else 255.0
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class NudeNetDetector:
    """Runs a NudeNet v3 ONNX model.

    Differences from ``nudenet.NudeDetector``: the score of an anchor is its best
    score among the *requested* classes (so an "exposed" score is not hidden by a
    higher "covered" score on the same anchor), thresholds are configurable, the
    execution provider can be chosen, and post-processing is vectorised. With
    ``classes=None``, ``channels="bgr"`` and the default thresholds the output
    matches NudeNet exactly.

    ``channels``: NudeNet's own reader swaps BGR->RGB and then swaps back in
    ``blobFromImage(swapRB=True)``, so the model is fed BGR. The YOLOv8 weights
    were trained on RGB, and RGB input scores higher on our face benchmark
    (320n: 44 of 55 faces), so RGB is the default here.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        classes: Sequence[str] | None = None,
        threshold: float = 0.25,
        nms_iou: float = 0.45,
        resolution: int = 320,
        providers: Sequence[str] | None = None,
        channels: str = "rgb",
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if not 0.0 < nms_iou <= 1.0:
            raise ValueError("nms_iou must be in (0, 1]")
        if resolution < 32 or resolution % 32:
            raise ValueError("resolution must be a positive multiple of 32")
        if channels not in CHANNEL_ORDERS:
            raise ValueError(f"channels must be one of {CHANNEL_ORDERS}")
        self.model_path = Path(model_path)
        self._session = self._open(providers or ["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.classes = tuple(classes) if classes else LABELS
        unknown = [c for c in self.classes if c not in LABELS]
        if unknown:
            raise ValueError(f"unknown classes: {unknown}")
        self._class_idx = None if classes is None else np.array([LABELS.index(c) for c in self.classes])
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.resolution = resolution
        self.channels = channels

    def _open(self, providers: Sequence[str]):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        return onnxruntime.InferenceSession(
            str(self.model_path), sess_options=options, providers=list(providers)
        )

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers()

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox (pad right/bottom) to a square, resize, scale to [0, 1]."""
        img = to_bgr8(image)
        h, w = img.shape[:2]
        side = max(h, w)
        padded = cv2.copyMakeBorder(img, 0, side - h, 0, side - w, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        if self.channels == "rgb":
            resized = resized[:, :, ::-1]
        blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) * (1.0 / 255.0)
        return np.ascontiguousarray(blob), side / self.resolution

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        blob, scale = self.preprocess(image)
        try:
            output = self._session.run(None, {self._input_name: blob})[0]
        except Exception as exc:
            # An accelerator can fail at run time even though the session opened, e.g.
            # CoreML cannot build a plan for 640m at --resolution 1280. Finish on CPU.
            accelerator = self.providers[0]
            if accelerator == "CPUExecutionProvider":
                raise
            reason = str(exc).strip().splitlines()[-1][:160]
            warnings.warn(f"{accelerator} failed ({reason}); continuing on CPU",
                          RuntimeWarning, stacklevel=2)
            self._session = self._open(["CPUExecutionProvider"])
            output = self._session.run(None, {self._input_name: blob})[0]
        return self.postprocess(output[0], scale, w, h)

    def postprocess(self, output: np.ndarray, scale: float, width: int, height: int) -> list[Detection]:
        preds = output.T  # (anchors, 4 + classes)
        class_scores = preds[:, 4:]
        if self._class_idx is None:
            best = class_scores.argmax(axis=1)
            labels_idx = best
        else:
            sub = class_scores[:, self._class_idx]
            best = sub.argmax(axis=1)
            labels_idx = self._class_idx[best]
            class_scores = sub
        scores = class_scores[np.arange(len(best)), best]
        keep = scores >= self.threshold
        if not keep.any():
            return []
        scores = scores[keep]
        labels_idx = labels_idx[keep]
        cx, cy, bw, bh = (preds[keep, :4] * scale).T
        x1 = np.clip(cx - bw / 2, 0, width)
        y1 = np.clip(cy - bh / 2, 0, height)
        x2 = np.clip(cx + bw / 2, 0, width)
        y2 = np.clip(cy + bh / 2, 0, height)
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        picked = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, self.nms_iou)
        detections = [
            Detection(
                LABELS[int(labels_idx[i])],
                float(scores[i]),
                float(x1[i]),
                float(y1[i]),
                float(x2[i]),
                float(y2[i]),
            )
            for i in np.asarray(picked, dtype=int).reshape(-1)
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


def create_detector(
    model: str = "320n",
    *,
    classes: Sequence[str] | None = None,
    threshold: float = 0.25,
    nms_iou: float = 0.45,
    resolution: int | None = None,
    device: str = "auto",
    channels: str = "rgb",
    allow_download: bool = True,
    download_progress=None,
) -> NudeNetDetector:
    from .models import resolve_model

    path, default_resolution = resolve_model(
        model, allow_download=allow_download, progress=download_progress
    )
    return NudeNetDetector(
        path,
        classes=classes,
        threshold=threshold,
        nms_iou=nms_iou,
        resolution=resolution or default_resolution or 320,
        providers=resolve_providers(device),
        channels=channels,
    )

"""Shared fixtures.

Pipeline tests never use real explicit material: a ``ColorDetector`` stands in for
the model and treats a red/black checkerboard "object" as the region to censor.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from avmosaic.detector import Detection
from avmosaic.video import ffmpeg_exe

DATA = Path(__file__).parent / "data"


def background(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Gray, detailed texture with no saturated red anywhere."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    checker = (((yy // 6) + (xx // 6)) % 2) * 120 + 60
    noise = rng.integers(-20, 21, size=(height, width))
    gray = np.clip(checker + noise, 0, 255).astype(np.uint8)
    return cv2.merge([gray, gray, gray])


def paint_object(frame: np.ndarray, x: int, y: int, size: int) -> None:
    """A red/black checkerboard: easy to find by colour, full of fine detail."""
    yy, xx = np.mgrid[0:size, 0:size]
    cells = ((yy // 3) + (xx // 3)) % 2 == 0
    patch = np.zeros((size, size, 3), np.uint8)
    patch[cells] = (0, 0, 255)
    frame[y:y + size, x:x + size] = patch


def object_box(index: int, size: int = 48, width: int = 320) -> tuple[int, int, int, int]:
    """Where the object is in synthetic video frame ``index`` (moves 3 px per frame)."""
    x = 20 + (3 * index) % (width - size - 40)
    y = 60
    return x, y, x + size, y + size


class ColorDetector:
    """Reports the bounding box of strongly red pixels as one detection."""

    def __init__(self, label: str = "FEMALE_GENITALIA_EXPOSED", skip: set[int] | None = None):
        self.label = label
        self.skip = skip or set()
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        call = self.calls
        self.calls += 1
        if call in self.skip:
            return []
        b, g, r = (image[..., i].astype(int) for i in range(3))
        red = (r > 150) & (g < 90) & (b < 90)
        ys, xs = np.nonzero(red)
        if ys.size < 20:
            return []
        return [Detection(self.label, 0.9, float(xs.min()), float(ys.min()),
                          float(xs.max() + 1), float(ys.max() + 1))]


def detail(image: np.ndarray, box: tuple[int, int, int, int], inset: int = 4) -> float:
    """Variance of the Laplacian inside ``box``: high for fine texture, ~0 after a mosaic."""
    x1, y1, x2, y2 = box
    roi = image[y1 + inset:y2 - inset, x1 + inset:x2 - inset]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def flat_tiles(image: np.ndarray, box: tuple[int, int, int, int], block: int,
               tol: int = 0) -> float:
    """Fraction of mosaic grid tiles lying fully inside ``box`` whose pixels are all equal."""
    x1, y1, x2, y2 = box
    xs = range(-(-x1 // block) * block, x2 - block + 1, block)
    ys = range(-(-y1 // block) * block, y2 - block + 1, block)
    tiles = [image[y:y + block, x:x + block].astype(int) for y in ys for x in xs]
    assert tiles, "box smaller than one tile"
    return sum(int(np.abs(t - t[0, 0]).max()) <= tol for t in tiles) / len(tiles)


def make_video(path: Path, frames: int = 45, width: int = 320, height: int = 240,
               fps: str = "30", audio: bool = True, audio_codec: str = "aac",
               draw: Callable[[np.ndarray, int], None] | None = None,
               video_args: tuple[str, ...] = ("-c:v", "libx264", "-crf", "8",
                                              "-preset", "veryfast", "-pix_fmt", "yuv420p"),
               ) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", fps, "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-c:a", audio_codec, "-shortest"]
    cmd += [*video_args, str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(frames):
        frame = background(height, width, seed=i)
        if draw is None:
            x1, y1, x2, _ = object_box(i, width=width)
            paint_object(frame, x1, y1, x2 - x1)
        else:
            draw(frame, i)
        proc.stdin.write(frame.tobytes())
    _, err = proc.communicate()
    assert proc.returncode == 0, err.decode()


def read_all_frames(path: Path) -> list[np.ndarray]:
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24")
    meta = next(reader)
    w, h = meta["size"]
    return [np.frombuffer(raw, np.uint8).reshape(h, w, 3) for raw in reader]


def stream_info(path: Path) -> str:
    """ffmpeg's description of the file's streams (there is no ffprobe in imageio-ffmpeg)."""
    result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                            capture_output=True)
    return result.stderr.decode(errors="replace")


@pytest.fixture
def color_detector() -> ColorDetector:
    return ColorDetector()


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("video") / "sample.mp4"
    make_video(path)
    return path

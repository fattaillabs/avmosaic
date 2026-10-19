"""Build tests/data/david_statue.mp4 and its per-frame ground truth.

Source: "Replica Michelangelo's David black background.jpg" by Jebulon, Wikimedia
Commons, CC0 1.0 (the 1280 px wide rendition):
https://commons.wikimedia.org/wiki/File:Replica_Michelangelo%27s_David_black_background.jpg

A virtual camera starts on the whole statue and moves in towards the torso. Because
the motion is generated, the region that must be censored is known exactly in every
frame. It is written to david_statue.json as [x1, y1, x2, y2] per frame.

    python tests/data/make_statue_clip.py path/to/david_black.jpg
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from avmosaic.image import read_image
from avmosaic.video import ffmpeg_exe

HERE = Path(__file__).parent
SIZE = (360, 640)  # width, height
FPS = 15
FRAMES = 45
# Marked by hand on the 1280x2122 source photo: pubic hair and genitals.
REGION = (530, 975, 630, 1115)
# Camera windows in source pixels: (centre x, centre y, window height).
START = (640, 1060, 2000)
END = (590, 950, 1000)


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def main(source: str) -> None:
    photo = read_image(source)
    if photo.shape[:2] != (2122, 1280):
        sys.exit(f"expected the 1280x2122 rendition, got {photo.shape[1]}x{photo.shape[0]}")
    width, height = SIZE
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(HERE / "david_statue.mp4")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    boxes = []
    for i in range(FRAMES):
        t = ease(i / (FRAMES - 1))
        cx, cy, win_h = (a + (b - a) * t for a, b in zip(START, END, strict=True))
        scale = height / win_h
        x0, y0 = cx - width / scale / 2, cy - win_h / 2
        affine = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
        frame = cv2.warpAffine(photo, affine, SIZE, flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(frame.tobytes())
        x1, y1, x2, y2 = REGION
        boxes.append([round((x1 - x0) * scale, 1), round((y1 - y0) * scale, 1),
                      round((x2 - x0) * scale, 1), round((y2 - y0) * scale, 1)])
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    (HERE / "david_statue.json").write_text(json.dumps(
        {"source": "Jebulon, Wikimedia Commons, CC0 1.0", "size": list(SIZE), "fps": FPS,
         "region": "pubic hair and genitals", "boxes": boxes}, indent=1) + "\n")
    print(f"wrote {FRAMES} frames")


if __name__ == "__main__":
    main(sys.argv[1])

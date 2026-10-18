"""Tracking tests on in-memory frames: exact coverage numbers, no codec noise."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from avmosaic import tracking
from avmosaic.censor import build_mask, expand_box
from avmosaic.detector import Detection
from avmosaic.tracking import (
    HoldPlanner,
    TrackingPlanner,
    TrackOptions,
    is_scene_cut,
    median_flow,
)

from .conftest import ColorDetector

H, W, SIZE = 240, 320, 40
_BG = np.random.default_rng(7).integers(70, 190, size=(H, W), dtype=np.uint8)
_TEXTURE = np.random.default_rng(11)


def static_background() -> np.ndarray:
    """A still camera: textured grey, identical in every frame, no red."""
    return cv2.merge([_BG, _BG, _BG])


_OBJECT = np.dstack([
    _TEXTURE.integers(0, 70, size=(SIZE, SIZE)),
    _TEXTURE.integers(0, 70, size=(SIZE, SIZE)),
    _TEXTURE.integers(170, 256, size=(SIZE, SIZE)),
]).astype(np.uint8)  # random (non-periodic) red texture: easy to detect, easy to follow


def frame_with_object(box: tuple[int, int, int, int] | None, scene: str = "a") -> np.ndarray:
    if scene == "a":
        frame = static_background()
    else:  # a completely different scene: bright horizontal gradient
        ramp = np.linspace(200, 255, W, dtype=np.uint8)
        frame = cv2.merge([np.tile(ramp, (H, 1))] * 3)
    if box is not None:
        x1, y1, x2, y2 = box
        frame[y1:y2, x1:x2] = _OBJECT[: y2 - y1, : x2 - x1]
    return frame


def bounce(speed: float):
    """Box moving horizontally at ``speed`` px/frame, bouncing off the edges."""
    span = W - SIZE - 20

    def path(i: int) -> tuple[int, int, int, int]:
        x = int(round(i * speed)) % (2 * span)
        x = 10 + (x if x < span else 2 * span - x)
        return x, 100, x + SIZE, 100 + SIZE

    return path


def circle(radius: float, step_px: float):
    cx, cy = W / 2, H / 2

    def path(i: int) -> tuple[int, int, int, int]:
        angle = i * step_px / radius
        x = int(round(cx + radius * math.cos(angle) - SIZE / 2))
        y = int(round(cy + radius * math.sin(angle) - SIZE / 2))
        return x, y, x + SIZE, y + SIZE

    return path


class Run:
    def __init__(self, planner, frames: int, path, skip=(), present=lambda i: True,
                 scene=lambda i: "a"):
        self.detector = ColorDetector(skip=set(skip))
        self.detected_at: list[int] = []
        self.truth: dict[int, tuple[int, int, int, int] | None] = {}
        self.regions: dict[int, list] = {}

        def detect(index, image):
            self.detected_at.append(index)
            self.detector.calls = index  # make `skip` refer to frame indices
            return self.detector.detect(image)

        for i in range(frames):
            box = path(i) if present(i) else None
            self.truth[i] = box
            for index, _, regions in planner.push(i, frame_with_object(box, scene(i)), detect):
                self.regions[index] = regions
        for index, _, regions in planner.flush():
            self.regions[index] = regions

    def mask(self, i: int) -> np.ndarray:
        boxes = [expand_box(r, 0.0, W, H) for r in self.regions[i]]
        return build_mask(H, W, boxes) > 0

    def coverage(self, i: int) -> float:
        x1, y1, x2, y2 = self.truth[i]
        return float(self.mask(i)[y1:y2, x1:x2].mean())

    def worst_coverage(self) -> float:
        return min(self.coverage(i) for i, box in self.truth.items() if box is not None)

    def mean_area(self) -> float:
        return float(np.mean([self.mask(i).sum() for i in self.truth]))


# --------------------------------------------------------------------------- building blocks


def test_median_flow_follows_a_shift():
    src = _BG.copy()
    shifted = np.roll(np.roll(src, 7, axis=1), -4, axis=0)
    moved = median_flow(src, shifted, (100.0, 80.0, 160.0, 140.0))
    assert moved == pytest.approx((107, 76, 167, 136), abs=0.5)


def test_median_flow_follows_a_zoom():
    src = _BG.copy()
    m = cv2.getRotationMatrix2D((130, 110), 0, 1.06)
    zoomed = cv2.warpAffine(src, m, (W, H))
    x1, y1, x2, y2 = median_flow(src, zoomed, (100.0, 80.0, 160.0, 140.0))
    assert (x2 - x1) / 60 == pytest.approx(1.06, abs=0.02)
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((130, 110), abs=1.0)


def test_median_flow_gives_up_on_flat_regions():
    flat = np.full((H, W), 128, np.uint8)
    assert median_flow(flat, flat, (100.0, 80.0, 160.0, 140.0)) is None


def test_scene_cut_detection():
    a = frame_with_object(None)[..., 0]
    b = frame_with_object(None, scene="b")[..., 0]
    moved = frame_with_object((50, 50, 90, 90))[..., 0]
    assert is_scene_cut(a, b)
    assert not is_scene_cut(a, moved)


def test_invalid_track_options():
    with pytest.raises(ValueError):
        TrackOptions(max_lost=-1)


# --------------------------------------------------------------------------- behaviour


def test_fast_motion_is_followed_tightly():
    path = bounce(speed=25)
    tracked = Run(TrackingPlanner(1, TrackOptions()), 40, path)
    held = Run(HoldPlanner(1, hold=5), 40, path)

    assert tracked.worst_coverage() == 1.0
    assert held.worst_coverage() == 1.0
    # The union of +-5 frames smears a 40 px object along 250 px of its path.
    assert tracked.mean_area() < 1.3 * SIZE * SIZE
    assert held.mean_area() > 2.5 * tracked.mean_area()


def test_long_detector_miss_is_followed_by_optical_flow():
    path = bounce(speed=8)
    missed = range(12, 27)  # 15 frames in a row
    tracked = Run(TrackingPlanner(1, TrackOptions(max_lost=20)), 40, path, skip=missed)
    held = Run(HoldPlanner(1, hold=5), 40, path, skip=missed)

    assert tracked.worst_coverage() >= 0.98
    assert min(held.coverage(i) for i in range(17, 22)) < 0.5  # the middle of the gap leaks


def test_gap_is_covered_even_when_optical_flow_fails(monkeypatch):
    """No flow at all (occlusion, blur, no edges): the gap still gets the straight path."""
    monkeypatch.setattr(tracking, "median_flow", lambda *args: None)
    path = bounce(speed=7)
    run = Run(TrackingPlanner(1, TrackOptions(max_lost=15, backfill=0, smooth=0)), 30, path,
              skip=range(8, 20))
    assert run.worst_coverage() >= 0.95


def test_gap_after_a_single_detection_is_interpolated(monkeypatch):
    """No flow and no velocity yet: only the look-ahead buffer can cover the gap,
    by drawing the straight path once the detector finds the region again."""
    monkeypatch.setattr(tracking, "median_flow", lambda *args: None)
    path = bounce(speed=7)
    run = Run(TrackingPlanner(1, TrackOptions(max_lost=15, backfill=0, smooth=0)), 30, path,
              skip=set(range(0, 7)) | set(range(8, 20)))
    assert min(run.coverage(i) for i in range(7, 30)) >= 0.95


def test_stopped_object_stays_covered_when_flow_fails(monkeypatch):
    monkeypatch.setattr(tracking, "median_flow", lambda *args: None)
    moving, stop = bounce(speed=6), 10
    run = Run(TrackingPlanner(1, TrackOptions(max_lost=15, smooth=0)), 25,
              lambda i: moving(min(i, stop)), skip=range(11, 25))
    assert run.worst_coverage() == 1.0


def test_sparse_detection_follows_a_curved_path():
    path = circle(radius=70, step_px=15)
    run = Run(TrackingPlanner(6, TrackOptions(backfill=0, smooth=0)), 48, path)
    assert run.detected_at == list(range(0, 48, 6))
    assert run.worst_coverage() >= 0.97


def test_frames_before_the_first_detection_are_backfilled():
    path = bounce(speed=6)
    run = Run(TrackingPlanner(1, TrackOptions(backfill=5)), 20, path, skip=range(0, 4))
    assert all(run.coverage(i) >= 0.95 for i in range(4))

    no_backfill = Run(TrackingPlanner(1, TrackOptions(backfill=0)), 20, path, skip=range(0, 4))
    assert all(no_backfill.coverage(i) == 0 for i in range(4))


def test_region_is_released_after_max_lost():
    path = bounce(speed=4)
    run = Run(TrackingPlanner(1, TrackOptions(max_lost=10)), 45, path, present=lambda i: i < 20)
    assert run.worst_coverage() == 1.0
    assert all(run.regions[i] for i in range(20, 30))  # still followed after it vanished
    assert not any(run.regions[i] for i in range(30, 45))


def test_scene_cut_ends_tracks_and_forces_a_detection():
    path = bounce(speed=4)
    run = Run(TrackingPlanner(4, TrackOptions(max_lost=30, backfill=5)), 40, path,
              present=lambda i: i < 22, scene=lambda i: "a" if i < 22 else "b")
    assert 22 in run.detected_at  # not a multiple of 4: detected because of the cut
    assert run.worst_coverage() >= 0.97
    assert not any(run.regions[i] for i in range(22, 40))


def test_two_objects_get_two_tracks():
    """Objects moving in opposite directions keep separate tracks and both stay covered."""

    def top(i):
        x = 10 + 6 * i
        return x, 10, x + SIZE, 10 + SIZE

    def bottom(i):
        x = W - SIZE - 10 - 6 * i
        return x, 180, x + SIZE, 180 + SIZE

    def detect(index, image):
        red = ((image[..., 2] > 150) & (image[..., 1] < 90) & (image[..., 0] < 90)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(red)
        return [Detection("ANUS_EXPOSED", 0.9, float(x), float(y), float(x + w), float(y + h))
                for x, y, w, h, area in stats[1:n] if area >= 50]

    planner = TrackingPlanner(2, TrackOptions())
    planned = {}
    for i in range(30):
        frame = frame_with_object(top(i))
        x1, y1, x2, y2 = bottom(i)
        frame[y1:y2, x1:x2] = _OBJECT
        for index, _, regions in planner.push(i, frame, detect):
            planned[index] = regions
        if i == 29:
            assert len([t for t in planner.tracks if t.alive]) == 2
    for index, _, regions in planner.flush():
        planned[index] = regions

    for i in range(30):
        mask = build_mask(H, W, [expand_box(r, 0.0, W, H) for r in planned[i]]) > 0
        for x1, y1, x2, y2 in (top(i), bottom(i)):
            assert mask[y1:y2, x1:x2].mean() == 1.0, i


def test_zero_lookahead_settings_work():
    run = Run(TrackingPlanner(1, TrackOptions(max_lost=0, backfill=0, smooth=0)), 20, bounce(speed=5))
    assert run.worst_coverage() == 1.0


def test_lookahead_and_order():
    planner = TrackingPlanner(3, TrackOptions(backfill=4, smooth=2, max_lost=1))
    assert planner.lookahead == 4
    assert TrackingPlanner(1, TrackOptions(max_lost=15)).lookahead == 15
    out = []
    for i in range(10):
        out += [idx for idx, _, _ in planner.push(i, frame_with_object(None), lambda *_: [])]
    out += [idx for idx, _, _ in planner.flush()]
    assert out == list(range(10))

"""Deciding which regions to censor in each video frame.

Two planners turn a stream of frames (plus detector calls) into
``(index, frame, regions)`` ready to render:

* ``HoldPlanner`` censors every box detected within ``reach`` frames of a frame.
  Simple and safe, but fast motion smears the mosaic along the path.
* ``TrackingPlanner`` links detections into tracks and follows each region with
  optical flow: between detector runs, through short detector misses, and back
  to frames shortly before a region was first detected. Tracks end at scene cuts.

Both keep a short look-ahead buffer, so a frame is rendered only once later
frames have been seen.
"""

from __future__ import annotations

import bisect
import re
import sys
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection

BoxF = tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
DetectFn = Callable[[int, np.ndarray], list[Detection]]
Planned = tuple[int, np.ndarray, list[Detection]]
ManualFrames = tuple[int, int, BoxF]  # first frame, last frame, box at the first frame
MANUAL_LABEL = "MANUAL"

FLOW_LONG_SIDE = 480  # optical flow runs on frames downscaled to this long side
FLOW_GRID = 10  # median-flow samples a 10x10 grid of points inside the box
MATCH_IOU = 0.1
MATCH_CENTER = 0.5  # or: centres closer than half the track box diagonal ...
MATCH_CENTER_GROWTH = 0.2  # ... plus this much per frame the track went undetected
CUT_CORRELATION = 0.5  # grey-histogram correlation below this = scene cut
LOST_GROWTH = 0.01  # a box with no recent detection grows 1% per frame ...
LOST_GROWTH_CAP = 0.5  # ... up to +50%, to absorb tracking drift
_LK = dict(winSize=(15, 15), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


@dataclass
class TrackOptions:
    # Keep following a region with optical flow this many frames after the
    # detector last saw it.
    max_lost: int = 15
    # Follow a new region backwards over this many frames before its first detection.
    backfill: int = 5
    # Average box positions over +-N frames (the raw box is always covered too).
    smooth: int = 2

    def __post_init__(self) -> None:
        for name in ("max_lost", "backfill", "smooth"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class ManualRegion:
    """A region a person marked because the detector misses it.

    ``box`` is where the region is at ``start`` (seconds). In a video it is followed
    with optical flow until ``end`` (None: the end of the video). In an image it is
    simply censored.
    """

    box: BoxF
    start: float = 0.0
    end: float | None = None

    @property
    def timed(self) -> bool:
        return self.start != 0.0 or self.end is not None

    def frames(self, fps: float) -> ManualFrames:
        last = sys.maxsize if self.end is None else round(self.end * fps)
        return round(self.start * fps), last, self.box

    @classmethod
    def parse(cls, text: str) -> ManualRegion:
        """``x1,y1,x2,y2`` or ``START-END:x1,y1,x2,y2`` (times as 12.5, 1:02.5 or 0:01:02)."""
        when, _, coords = text.rpartition(":") if re.search(r"-.*:", text) else ("", "", text)
        try:
            x1, y1, x2, y2 = (float(v) for v in coords.split(","))
        except ValueError:
            raise ValueError(f"invalid region {text!r}; use x1,y1,x2,y2 or START-END:x1,y1,x2,y2") \
                from None
        if x2 <= x1 or y2 <= y1 or min(x1, y1) < 0:
            raise ValueError(f"invalid region {text!r}: need 0 <= x1 < x2 and 0 <= y1 < y2")
        if not when:
            return cls((x1, y1, x2, y2))
        start_text, _, end_text = when.partition("-")
        start, end = seconds(start_text), seconds(end_text)
        if end <= start:
            raise ValueError(f"invalid region {text!r}: the end time must be after the start")
        return cls((x1, y1, x2, y2), start, end)


def seconds(text: str) -> float:
    try:
        parts = [float(p) for p in text.strip().split(":")]
    except ValueError:
        raise ValueError(f"invalid time {text!r}; use seconds (12.5) or m:ss (1:02.5)") from None
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise ValueError(f"invalid time {text!r}")
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return total


# --------------------------------------------------------------------------- geometry


def iou(a: BoxF, b: BoxF) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: BoxF, b: BoxF) -> float:
    """Distance between centres, relative to the diagonal of ``a``."""
    diag = max(1e-6, float(np.hypot(a[2] - a[0], a[3] - a[1])))
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)) / diag


def union_box(a: BoxF, b: BoxF) -> BoxF:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def grow_box(box: BoxF, fraction: float) -> BoxF:
    if fraction <= 0:
        return box
    dx = (box[2] - box[0]) * fraction / 2
    dy = (box[3] - box[1]) * fraction / 2
    return box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy


def clip_box(box: BoxF, width: int, height: int) -> BoxF:
    return (min(max(box[0], 0.0), width), min(max(box[1], 0.0), height),
            min(max(box[2], 0.0), width), min(max(box[3], 0.0), height))


# --------------------------------------------------------------------------- optical flow


def flow_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscaled grey image for optical flow, and the scale factor applied."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, FLOW_LONG_SIDE / max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def median_flow(src: np.ndarray, dst: np.ndarray, box: BoxF) -> BoxF | None:
    """Move ``box`` from grey image ``src`` to ``dst`` (both in the same pixel space).

    Tracks a grid of points with pyramidal Lucas-Kanade, keeps the half with the
    smallest forward-backward error, and applies their median shift and scale.
    Returns None when too few points can be followed (flat or vanished region).
    """
    h, w = src.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w - 1, h - 1)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    xs = np.linspace(x1 + 0.1 * bw, x2 - 0.1 * bw, FLOW_GRID)
    ys = np.linspace(y1 + 0.1 * bh, y2 - 0.1 * bh, FLOW_GRID)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2).astype(np.float32)
    nxt, st1, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, **_LK)
    if nxt is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(dst, src, nxt, None, **_LK)
    if back is None:
        return None
    ok = (st1.ravel() == 1) & (st2.ravel() == 1)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    ok &= np.isfinite(fb)
    if ok.sum() < 4:
        return None
    ok &= fb <= max(float(np.median(fb[ok])), 0.5)
    ok &= fb < 2.0
    if ok.sum() < 4:
        return None
    p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    dx, dy = np.median(p1 - p0, axis=0)
    i, j = np.triu_indices(len(p0), 1)
    d0 = np.linalg.norm(p0[i] - p0[j], axis=1)
    d1 = np.linalg.norm(p1[i] - p1[j], axis=1)
    usable = d0 > 1.0
    scale = float(np.median(d1[usable] / d0[usable])) if usable.any() else 1.0
    scale = min(max(scale, 0.9), 1.1)
    cx, cy = (box[0] + box[2]) / 2 + dx, (box[1] + box[3]) / 2 + dy
    half_w, half_h = (box[2] - box[0]) * scale / 2, (box[3] - box[1]) * scale / 2
    return float(cx - half_w), float(cy - half_h), float(cx + half_w), float(cy + half_h)


def is_scene_cut(prev: np.ndarray, cur: np.ndarray) -> bool:
    h1 = cv2.calcHist([prev], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([cur], [0], None, [32], [0, 256])
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < CUT_CORRELATION


# --------------------------------------------------------------------------- planners


class HoldPlanner:
    """Censor the union of all boxes detected within ``hold + every - 1`` frames."""

    def __init__(self, detect_every: int = 1, hold: int = 5, manual: Sequence[ManualFrames] = ()):
        self.every = detect_every
        self.reach = hold + detect_every - 1
        self._manual = list(manual)  # without tracking, marked regions stay where they were put
        self._pending: deque[tuple[int, np.ndarray]] = deque()
        self._found: dict[int, list[Detection]] = {}

    def push(self, index: int, frame: np.ndarray, detect: DetectFn) -> list[Planned]:
        if index % self.every == 0:
            self._found[index] = detect(index, frame)
        self._pending.append((index, frame))
        out = []
        while self._pending and self._pending[0][0] + self.reach <= index:
            out.append(self._emit())
        return out

    def flush(self) -> list[Planned]:
        return [self._emit() for _ in range(len(self._pending))]

    def _emit(self) -> Planned:
        index, frame = self._pending.popleft()
        regions = [d for j in range(index - self.reach, index + self.reach + 1)
                   for d in self._found.get(j, ())]
        regions += [Detection(MANUAL_LABEL, 1.0, *box) for first, last, box in self._manual
                    if first <= index <= last]
        for j in [j for j in self._found if j < index - self.reach + 1]:
            del self._found[j]
        return index, frame, regions


@dataclass
class _Frame:
    index: int
    image: np.ndarray
    gray: np.ndarray
    scale: float
    cut: bool


@dataclass
class _Track:
    label: str
    score: float
    boxes: dict[int, BoxF] = field(default_factory=dict)
    kinds: dict[int, str] = field(default_factory=dict)  # det | flow | hold | fix | back
    detections: list[int] = field(default_factory=list)  # recent frame indices, ascending
    det_boxes: dict[int, BoxF] = field(default_factory=dict)  # boxes at those frames
    alive: bool = True
    manual_until: int | None = None  # set for regions a person marked

    @property
    def last_det(self) -> int:
        return self.detections[-1]

    def add_detection(self, index: int, box: BoxF) -> None:
        self.detections.append(index)
        self.det_boxes[index] = box
        if len(self.detections) > 32:
            del self.det_boxes[self.detections.pop(0)]

    def velocity(self) -> tuple[float, float]:
        """Centre motion per frame between two recent detections (0 if unknown)."""
        if len(self.detections) < 2:
            return 0.0, 0.0
        last = self.detections[-1]
        earlier = [d for d in self.detections[:-1] if d <= last - 3]  # less jitter
        prev = earlier[-1] if earlier else self.detections[-2]
        a, b = self.det_boxes[prev], self.det_boxes[last]
        dt = last - prev
        return (b[0] + b[2] - a[0] - a[2]) / 2 / dt, (b[1] + b[3] - a[1] - a[3]) / 2 / dt


class TrackingPlanner:
    def __init__(self, detect_every: int = 1, options: TrackOptions | None = None,
                 manual: Sequence[ManualFrames] = ()):
        self.every = detect_every
        self.options = options or TrackOptions()
        self._manual = sorted(manual, key=lambda m: m[0])
        # Between two detector runs a track must survive at least every - 1 frames.
        self.max_lost = max(self.options.max_lost, detect_every - 1)
        # Buffering max_lost frames means that when the detector finds a region again
        # within max_lost frames, every frame of the gap is still unrendered and gets
        # the straight path between the two detections as well as the flow path,
        # even if flow could not follow a featureless region.
        self.lookahead = max(detect_every - 1, self.options.backfill, self.options.smooth,
                             self.max_lost)
        self._buffer: deque[_Frame] = deque()
        self._prev: _Frame | None = None
        self.tracks: list[_Track] = []

    def push(self, index: int, image: np.ndarray, detect: DetectFn) -> list[Planned]:
        gray, scale = flow_gray(image)
        cut = self._prev is not None and is_scene_cut(self._prev.gray, gray)
        cur = _Frame(index, image, gray, scale, cut)

        for track in self.tracks:
            if not track.alive:
                continue
            box = track.boxes[index - 1]
            if cut:
                if track.manual_until is not None:  # a marked region holds its place
                    track.boxes[index] = box
                    track.kinds[index] = "hold"
                else:
                    track.alive = False
                continue
            moved = self._flow(self._prev, cur, box)
            if moved is None:
                # Flow lost it (occlusion, blur, no texture). Cover both "it stopped"
                # and "it kept its recent velocity" until the detector finds it again.
                vx, vy = track.velocity()
                moved = union_box(box, (box[0] + vx, box[1] + vy, box[2] + vx, box[3] + vy))
                track.kinds[index] = "hold"
            else:
                track.kinds[index] = "flow"
            track.boxes[index] = moved

        for first, last, box in self._manual:
            if first == index:
                track = _Track(MANUAL_LABEL, 1.0, manual_until=last)
                track.boxes[index] = box
                track.kinds[index] = "manual"
                track.add_detection(index, box)
                self.tracks.append(track)

        if index % self.every == 0 or cut:
            self._associate(cur, detect(index, image))

        for track in self.tracks:
            if not track.alive:
                continue
            if track.manual_until is not None:
                ended = index > track.manual_until
            else:
                ended = index - track.last_det > self.max_lost
            if ended:
                track.alive = False
                track.boxes.pop(index, None)
                track.kinds.pop(index, None)

        self._buffer.append(cur)
        self._prev = cur
        out = []
        while self._buffer and self._buffer[0].index + self.lookahead <= index:
            out.append(self._emit())
        return out

    def flush(self) -> list[Planned]:
        return [self._emit() for _ in range(len(self._buffer))]

    # ----------------------------------------------------------------- internals

    @staticmethod
    def _flow(src: _Frame, dst: _Frame, box: BoxF) -> BoxF | None:
        s = src.scale
        moved = median_flow(src.gray, dst.gray, (box[0] * s, box[1] * s, box[2] * s, box[3] * s))
        if moved is None:
            return None
        return moved[0] / s, moved[1] / s, moved[2] / s, moved[3] / s

    def _associate(self, cur: _Frame, detections: list[Detection]) -> None:
        index = cur.index
        live = [t for t in self.tracks if t.alive and index in t.boxes and t.manual_until is None]
        candidates = []
        for ti, track in enumerate(live):
            for di, det in enumerate(detections):
                box = (det.x1, det.y1, det.x2, det.y2)
                overlap = iou(track.boxes[index], box)
                distance = center_distance(track.boxes[index], box)
                lost = max(0, index - track.last_det - (self.every - 1))
                if overlap >= MATCH_IOU or distance < MATCH_CENTER + MATCH_CENTER_GROWTH * lost:
                    candidates.append((-overlap, distance, ti, di))
        candidates.sort()
        used_tracks, used_dets = set(), set()
        for _, _, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            self._update(live[ti], index, detections[di])
        for di, det in enumerate(detections):
            if di not in used_dets:
                track = _Track(det.label, det.score)
                track.boxes[index] = (det.x1, det.y1, det.x2, det.y2)
                track.kinds[index] = "det"
                track.add_detection(index, track.boxes[index])
                self.tracks.append(track)
                self._backfill(track, cur)

    def _update(self, track: _Track, index: int, det: Detection) -> None:
        """Snap the track to a detection and spread the drift back over earlier frames."""
        box = (det.x1, det.y1, det.x2, det.y2)
        predicted = track.boxes[index]
        error = np.subtract(box, predicted)
        start = track.last_det
        span = index - start
        for frame in self._buffer:
            j = frame.index
            if start < j < index and track.kinds.get(j) in ("flow", "hold"):
                corrected = tuple(np.add(track.boxes[j], error * (j - start) / span))
                # Keep the flow estimate too: whichever was right stays covered.
                track.boxes[j] = union_box(track.boxes[j], corrected)
                track.kinds[j] = "fix"
        track.boxes[index] = box
        track.kinds[index] = "det"
        track.add_detection(index, box)
        track.label, track.score = det.label, det.score

    def _backfill(self, track: _Track, cur: _Frame) -> None:
        box = track.boxes[cur.index]
        later = cur
        for frame in list(reversed(self._buffer))[: self.options.backfill]:
            if later.cut:  # never follow a region back into the previous scene
                break
            box = self._flow(later, frame, box) or box
            track.boxes[frame.index] = box
            track.kinds[frame.index] = "back"
            later = frame

    def _emit(self) -> Planned:
        frame = self._buffer.popleft()
        index = frame.index
        height, width = frame.image.shape[:2]
        regions = []
        for track in self.tracks:
            raw = track.boxes.get(index)
            if raw is None:
                continue
            box = union_box(self._smoothed(track, index), raw)
            box = grow_box(box, self._growth(track, index))
            x1, y1, x2, y2 = clip_box(box, width, height)
            if x2 > x1 and y2 > y1:
                regions.append(Detection(track.label, track.score, x1, y1, x2, y2))
        keep_from = index + 1 - self.options.smooth
        if self._prev is not None:  # live tracks are propagated from their latest box
            keep_from = min(keep_from, self._prev.index)
        for track in self.tracks:
            for j in [j for j in track.boxes if j < keep_from]:
                del track.boxes[j]
                track.kinds.pop(j, None)
        self.tracks = [t for t in self.tracks if t.alive or t.boxes]
        return index, frame.image, regions

    def _smoothed(self, track: _Track, index: int) -> BoxF:
        radius = self.options.smooth
        boxes = [track.boxes[j] for j in range(index - radius, index + radius + 1) if j in track.boxes]
        return tuple(np.mean(boxes, axis=0))

    def _growth(self, track: _Track, index: int) -> float:
        """Extra size for boxes far from any detection of this track (drift margin)."""
        if track.manual_until is not None:
            return 0.0  # keep the size a person chose
        dets = track.detections
        pos = bisect.bisect_left(dets, index)
        gaps = [abs(index - dets[k]) for k in (pos - 1, pos) if 0 <= k < len(dets)]
        gap = min(gaps) if gaps else 0
        return min(LOST_GROWTH_CAP, LOST_GROWTH * max(0, gap - (self.every - 1)))

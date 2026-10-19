"""Benchmarking on datasets that only have per-file labels (e.g. NPDI, Pornography-2k,
folders of explicit vs. swimwear images).

``run`` samples frames from every video (an image is one sample) and stores all
detections above a low score floor in a JSON-lines file (resumable).
``summarize`` turns those files into numbers: how often explicit files get
censored at all, how much of them, how often clean files are censored by
mistake, and the file-level AUC. Nothing here shows or saves images.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from .detector import Detector, to_bgr8
from .image import IMAGE_EXTS, read_image
from .labels import LABELS

POSITIVE = r"porn"
NEGATIVE = r"no[nt][-_ ]?porn"  # nonporn, non_porn, not_porn, ...
SUBSET = ("easy", "difficult", "hard")  # folder names that mark a subset of clean files


def label_for(path: Path, positive: str = POSITIVE, negative: str = NEGATIVE) -> int | None:
    """1 for explicit, 0 for clean, None when the path says neither (negative wins)."""
    text = path.as_posix().lower()
    if re.search(negative, text):
        return 0
    if re.search(positive, text):
        return 1
    return None


def subset_for(path: Path) -> str | None:
    """A folder named exactly easy/difficult/hard (file names are not looked at)."""
    for part in reversed([p.lower() for p in path.parent.parts]):
        if part in SUBSET:
            return part
    return None


def sample_frames(path: str | Path, fps: float, start: float = 0.0,
                  end: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time in the video, frame) at ``fps`` frames per second from ``start`` to ``end``."""
    input_params = ["-ss", f"{start:.3f}"] if start > 0 else []  # seek before decoding
    output_params = ["-vf", f"fps={fps:g}"]
    if end is not None:
        output_params += ["-t", f"{end - start:.3f}"]
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24", input_params=input_params,
                                        output_params=output_params)
    try:
        meta = next(reader)
        width, height = meta["size"]
        for k, raw in enumerate(reader):
            yield start + k / fps, np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    finally:
        reader.close()


def done_paths(results: Path) -> set[str]:
    if not results.is_file():
        return set()
    done = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "error" not in record:
                done.add(record["path"])
    return done


def evaluate_file(path: Path, detector: Detector, fps: float, config: dict,
                  label_path: Path | None = None,
                  positive: str = POSITIVE, negative: str = NEGATIVE) -> dict:
    """Detect on one image (a single sample) or on frames sampled from one video.

    ``label_path`` is what the label is read from (default: ``path``); pass the
    path below the dataset folder so that, say, a repository named "...Porn..."
    does not label every file explicit.
    """
    if path.suffix.lower() in IMAGE_EXTS:
        samples = iter([(0.0, to_bgr8(read_image(path)))])
    else:
        samples = sample_frames(path, fps)
    frames, count = [], 0
    for t, frame in samples:
        count += 1
        dets = detector.detect(frame)
        if dets:
            frames.append([round(t, 2), [[LABELS.index(d.label), round(d.score, 4),
                                          *(round(v) for v in (d.x1, d.y1, d.x2, d.y2))]
                                         for d in dets]])
    label_path = label_path or path
    return {"path": str(path), "label": label_for(label_path, positive, negative),
            "subset": subset_for(label_path), "samples": count, "frames": frames, "config": config}


def run(items: Sequence[tuple[Path, Path]], results: Path, detector: Detector, *, fps: float,
        config: dict, positive: str = POSITIVE, negative: str = NEGATIVE,
        log: Callable[[str], None] = lambda msg: None) -> tuple[int, int]:
    """Evaluate every ``(path, label_path)`` not yet in ``results``. Returns (evaluated, failed)."""
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old = json.loads(line).get("config")
                if old is not None and old != config:
                    raise ValueError(f"{results} was made with different settings ({old}); "
                                     "use another output file")
                break
    done = done_paths(results)
    todo = [(p, lp) for p, lp in items if str(p) not in done]
    if len(todo) < len(items):
        log(f"resuming: {len(items) - len(todo)} of {len(items)} files already in {results}")
    evaluated = failed = 0
    results.parent.mkdir(parents=True, exist_ok=True)
    with open(results, "a", encoding="utf-8") as out:
        for n, (path, label_path) in enumerate(todo, 1):
            try:
                record = evaluate_file(path, detector, fps, config, label_path, positive, negative)
                evaluated += 1
                hits = len(record["frames"])
                log(f"[{n}/{len(todo)}] {path}: label={record['label']} "
                    f"hits in {hits}/{record['samples']} samples")
            except Exception as exc:
                record = {"path": str(path), "error": str(exc), "config": config}
                failed += 1
                log(f"[{n}/{len(todo)}] {path}: error: {exc}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return evaluated, failed


# --------------------------------------------------------------------------- summary


@dataclass
class VideoScore:
    path: str
    label: int
    subset: str | None
    samples: int
    scores: list[tuple[float, float, str]]  # (time, best score in that sample, class)

    def best(self) -> tuple[float, float, str]:
        return max(self.scores, key=lambda s: s[1], default=(0.0, 0.0, ""))

    def hit_fraction(self, threshold: float) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for _, s, _ in self.scores if s >= threshold) / self.samples


def load(results: Path, classes: Sequence[str]) -> tuple[list[VideoScore], dict, int]:
    wanted = {LABELS.index(c) for c in classes}
    videos, config, errors = [], {}, 0
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        config = record.get("config", config)
        if "error" in record:
            errors += 1
            continue
        if record["label"] is None:
            continue
        scores = []
        for t, dets in record["frames"]:
            best = max(((s, c) for c, s, *_ in dets if c in wanted), default=None)
            if best is not None:
                scores.append((t, best[0], LABELS[best[1]]))
        videos.append(VideoScore(record["path"], record["label"], record["subset"],
                                 record["samples"], scores))
    return videos, config, errors


def auc(videos: Sequence[VideoScore]) -> float | None:
    """Chance that an explicit video scores higher than a clean one (ties count half)."""
    pos = np.array([v.best()[1] for v in videos if v.label == 1])
    neg = np.array([v.best()[1] for v in videos if v.label == 0])
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def summarize(videos: Sequence[VideoScore], thresholds: Sequence[float], top: int = 10) -> str:
    pos = [v for v in videos if v.label == 1]
    neg = [v for v in videos if v.label == 0]
    groups = [("explicit", pos), ("clean", neg)]
    for name in sorted({v.subset for v in neg if v.subset}):
        groups.append((f"clean/{name}", [v for v in neg if v.subset == name]))

    lines = [f"files: {len(pos)} explicit, {len(neg)} clean"]
    score = auc(videos)
    if score is not None:
        lines.append(f"file-level AUC (best score per file): {score:.3f}")
    header = f"{'threshold':>9} " + " ".join(f"{name:>22}" for name, _ in groups)
    lines += ["", "share of files with any censored sample / mean share of censored samples",
              header]
    for t in thresholds:
        cells = []
        for _, group in groups:
            if not group:
                cells.append(f"{'-':>22}")
                continue
            any_hit = np.mean([v.best()[1] >= t for v in group])
            frames = np.mean([v.hit_fraction(t) for v in group])
            cells.append(f"{any_hit:>10.1%} / {frames:>8.1%}")
        lines.append(f"{t:>9.2f} " + " ".join(cells))

    if neg:
        lines += ["", f"clean files with the highest scores (check for false alarms, top {top}):"]
        flagged = [v for v in sorted(neg, key=lambda v: -v.best()[1]) if v.best()[1] > 0][:top]
        for v in flagged:
            t, s, c = v.best()
            lines.append(f"  {s:.2f}  {c:<26} at {t:>7.1f}s  {v.path}")
        if not flagged:
            lines.append("  (none)")
    if pos:
        lowest = min(thresholds)
        missed = [v for v in pos if v.best()[1] < lowest]
        lines += ["", f"explicit files with nothing at or above {lowest:.2f}: {len(missed)}"]
        for v in sorted(missed, key=lambda v: v.best()[1])[:top]:
            lines.append(f"  best {v.best()[1]:.2f}  {v.path}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- region checks


@dataclass
class RegionCheck:
    """How well the mosaic on one labelled image matches the regions a person marked."""

    path: Path
    regions: list[list[float]]  # marked by a person: must be censored
    coverage: list[float]  # share of each region that the mosaic covers
    spill: float  # share of the mosaic outside every marked region
    false_area: float  # share of the whole image censored outside the marked regions
    detections: list


def labelled_images(root: Path) -> list[tuple[Path, list[list[float]]]]:
    """Images under ``root`` with a ``<name>.json`` next to them."""
    found = []
    for image in sorted(root.rglob("*")):
        label = image.with_suffix(".json")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTS and label.is_file():
            found.append((image, json.loads(label.read_text(encoding="utf-8"))["regions"]))
    return found


def check_regions(image_path: Path, regions: list[list[float]], detector: Detector,
                  options) -> RegionCheck:
    from .censor import apply_censor, build_mask, expand_box

    image = read_image(image_path)
    height, width = image.shape[:2]
    detections = detector.detect(to_bgr8(image))
    mask = apply_censor(image.copy(), detections, options) > 0
    marked = build_mask(height, width, [expand_box(r, 0.0, width, height) for r in regions]) > 0
    coverage = []
    for x1, y1, x2, y2 in (expand_box(r, 0.0, width, height) for r in regions):
        area = mask[y1:y2, x1:x2]
        coverage.append(float(area.mean()) if area.size else 1.0)
    outside = int((mask & ~marked).sum())
    return RegionCheck(image_path, regions, coverage,
                       spill=outside / max(int(mask.sum()), 1),
                       false_area=outside / mask.size, detections=detections)


def summarize_regions(checks: Sequence[RegionCheck], root: Path, min_coverage: float = 0.9,
                      max_false_area: float = 0.02, top: int = 20) -> tuple[str, bool]:
    """Text report and whether every check passed."""
    with_regions = [c for c in checks if c.regions]
    clean = [c for c in checks if not c.regions]
    coverages = [v for c in with_regions for v in c.coverage]
    under = [(c, k, v) for c in with_regions for k, v in enumerate(c.coverage) if v < min_coverage]
    false = [c for c in clean if c.false_area > max_false_area]

    def name(c: RegionCheck) -> str:
        try:
            return c.path.relative_to(root).as_posix()
        except ValueError:
            return str(c.path)

    def found(c: RegionCheck) -> str:
        return ", ".join(f"{d.label} {d.score:.2f}" for d in c.detections) or "nothing detected"

    lines = [f"labelled images: {len(checks)} ({len(with_regions)} with regions, {len(clean)} clean)"]
    if coverages:
        full = sum(v >= min_coverage for v in coverages)
        lines.append(f"regions covered >= {min_coverage:.0%}: {full} / {len(coverages)} "
                     f"(mean coverage {np.mean(coverages):.1%}, worst {min(coverages):.1%})")
        spill = np.mean([c.spill for c in with_regions])
        lines.append(f"mosaic outside the marked regions: mean {spill:.1%} of the mosaic "
                     "(padding and tile edges count here)")
    if clean:
        lines.append(f"clean images censored on more than {max_false_area:.0%} of the picture: "
                     f"{len(false)} / {len(clean)}")
    if under:
        lines += ["", "under-censored regions (coverage, box, what the detector found):"]
        for c, k, v in sorted(under, key=lambda u: u[2])[:top]:
            box = [round(x) for x in c.regions[k]]
            lines.append(f"  {v:6.1%}  {box}  {name(c)}  ({found(c)})")
    if false:
        lines += ["", "clean images censored by mistake:"]
        for c in sorted(false, key=lambda c: -c.false_area)[:top]:
            lines.append(f"  {c.false_area:6.1%} of the image  {name(c)}  ({found(c)})")
    return "\n".join(lines), not under and not false

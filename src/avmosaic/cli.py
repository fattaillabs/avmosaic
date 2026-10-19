"""Command line interface."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__, evaluate
from .censor import METHODS, SHAPES, CensorOptions
from .detector import CHANNEL_ORDERS, DEVICES, Detection, create_detector, resolve_providers
from .image import IMAGE_EXTS, censor_image, detect_image, write_image
from .labels import DEFAULT_CLASSES, LABELS, PRESETS, parse_classes
from .models import DEFAULT_MODEL, MODELS, cache_dir, download_model, find_model
from .tracking import ManualRegion, seconds
from .video import VIDEO_EXTS, VideoOptions, censor_video, detect_video

COMMANDS = ("censor", "detect", "eval", "frames", "label", "classes", "models")
_KEEP_VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".mkv")


class UsageError(Exception):
    pass


# --------------------------------------------------------------------------- parser


def _detection_args(p: argparse.ArgumentParser, threshold: float = 0.25,
                    threshold_help: str = "minimum detection score, lower = censor more") -> None:
    g = p.add_argument_group("detection")
    g.add_argument("--classes", default=DEFAULT_CLASSES,
                   help="comma-separated presets and/or class names (default: %(default)s; "
                        "see `avmosaic classes`)")
    g.add_argument("--threshold", type=float, default=threshold,
                   help=f"{threshold_help} (default: %(default)s)")
    g.add_argument("--nms-iou", type=float, default=0.45,
                   help="overlap above which duplicate boxes are merged (default: %(default)s)")
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"{' | '.join(MODELS)} | path to an .onnx file (default: %(default)s)")
    g.add_argument("--resolution", type=int, default=None,
                   help="model input size in px, multiple of 32 (default: the model's own)")
    g.add_argument("--device", choices=DEVICES, default="auto",
                   help="ONNX Runtime execution provider (default: %(default)s)")
    g.add_argument("--channels", choices=CHANNEL_ORDERS, default="rgb",
                   help="channel order fed to the model; bgr reproduces nudenet exactly "
                        "(default: %(default)s)")


def _video_detect_args(g: argparse._ArgumentGroup) -> None:
    g.add_argument("--detect-every", type=int, default=1, metavar="N",
                   help="run the detector on every Nth frame only (default: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avmosaic",
        description="Detect explicit regions in images and videos and cover them with a mosaic.",
        epilog="Shorthand: `avmosaic FILE ...` is `avmosaic censor FILE ...`.",
    )
    parser.add_argument("--version", action="version", version=f"avmosaic {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("censor", help="censor images/videos (default command)")
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="image, video or directory")
    p.add_argument("-o", "--output",
                   help="output file (single input) or directory (default: next to the input)")
    p.add_argument("--suffix", default="_censored",
                   help="added to output names when -o is not given (default: %(default)s)")
    p.add_argument("-y", "--overwrite", action="store_true", help="overwrite existing outputs")
    _detection_args(p)
    g = p.add_argument_group("censoring")
    g.add_argument("--method", choices=METHODS, default="mosaic", help="(default: %(default)s)")
    g.add_argument("--block-size", default="auto",
                   help="mosaic tile: auto (max of 4 px and 1/100 of the long side), "
                        "pixels, or N%% of the long side (default: %(default)s)")
    g.add_argument("--shape", choices=SHAPES, default="rect", help="(default: %(default)s)")
    g.add_argument("--padding", type=float, default=0.15,
                   help="grow each box by this fraction of its size per side (default: %(default)s)")
    g.add_argument("--fill-color", default="black",
                   help="for --method fill: black, white, #rrggbb or r,g,b (default: %(default)s)")
    g.add_argument("--draw-boxes", action="store_true",
                   help="draw detection boxes and scores on the output (for tuning)")
    g.add_argument("--add-region", action="append", default=[], metavar="SPEC",
                   help="also censor a region the detector misses: x1,y1,x2,y2 (whole image or "
                        "video) or START-END:x1,y1,x2,y2 (video; seconds or m:ss, followed with "
                        "optical flow). Repeatable")
    g.add_argument("--jpeg-quality", type=int, default=95, help="(default: %(default)s)")
    g = p.add_argument_group("video")
    _video_detect_args(g)
    g.add_argument("--tracking", choices=("on", "off"), default="on",
                   help="follow regions with optical flow between and through detections; "
                        "off = censor the union of nearby boxes (default: %(default)s)")
    g.add_argument("--hold", type=int, default=5, metavar="N",
                   help="tracking on: follow a region back N frames before it was first detected; "
                        "off: censor boxes found up to N frames before/after (default: %(default)s)")
    g.add_argument("--max-lost", type=int, default=15, metavar="N",
                   help="tracking on: keep following a region N frames after the detector "
                        "lost it (default: %(default)s)")
    g.add_argument("--smooth", type=int, default=2, metavar="N",
                   help="tracking on: average box positions over +-N frames (default: %(default)s)")
    g.add_argument("--codec", default="libx264", help="ffmpeg video encoder (default: %(default)s)")
    g.add_argument("--crf", type=int, default=18,
                   help="quality for x264/x265/vp9/av1, lower = better (default: %(default)s)")
    g.add_argument("--preset", default="medium", help="x264/x265 preset (default: %(default)s)")
    g.add_argument("--no-audio", action="store_true", help="drop the audio track")
    g.add_argument("--ffmpeg-args", default="",
                   help='extra encoder arguments, e.g. "-tune film" (quote as one string)')
    g = p.add_argument_group("output")
    g.add_argument("--report", metavar="JSON", help="write all detections to a JSON file")
    g.add_argument("--progress", choices=("auto", "always", "never"), default="auto",
                   help="video progress: auto = a bar in a terminal; always = also a line every "
                        "10 s when output goes to a file or log (default: %(default)s)")
    g.add_argument("-q", "--quiet", action="store_true", help="only print errors")

    p = sub.add_parser("detect", help="only detect; print or save detections as JSON")
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="image, video or directory")
    p.add_argument("-o", "--output", metavar="JSON", help="write JSON here instead of stdout")
    _detection_args(p)
    _video_detect_args(p.add_argument_group("video"))
    p.add_argument("--progress", choices=("auto", "always", "never"), default="auto",
                   help="video progress, as for censor (default: %(default)s)")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    p = sub.add_parser("eval", help="benchmark on images/videos labelled explicit/clean by their path")
    esub = p.add_subparsers(dest="eval_command", metavar="ACTION", required=True)
    r = esub.add_parser("run", help="sample frames, detect, append results to a JSON-lines file")
    r.add_argument("inputs", nargs="+", metavar="INPUT",
                   help="images, videos or directories (an image counts as one sample)")
    r.add_argument("-o", "--output", required=True, metavar="JSONL",
                   help="results file; re-running skips videos already in it")
    r.add_argument("--sample-fps", type=float, default=1.0,
                   help="video frames per second to check (default: %(default)s)")
    r.add_argument("--positive", default=evaluate.POSITIVE,
                   help="regex on the path marking explicit videos (default: %(default)s)")
    r.add_argument("--negative", default=evaluate.NEGATIVE,
                   help="regex marking clean videos, checked first (default: %(default)s)")
    _detection_args(r, threshold=0.05,
                    threshold_help="lowest score recorded; `eval summary` sweeps thresholds above it")
    r.add_argument("-q", "--quiet", action="store_true", help="no per-video output")
    s = esub.add_parser("summary", help="print detection and false-alarm rates from results")
    s.add_argument("results", nargs="+", metavar="JSONL")
    s.add_argument("--classes", default=None,
                   help="classes to count (default: the classes the run used)")
    s.add_argument("--thresholds", default="0.15,0.25,0.35,0.5",
                   help="comma-separated score thresholds (default: %(default)s)")
    s.add_argument("--top", type=int, default=10, help="videos listed per section (default: %(default)s)")

    g = esub.add_parser("regions", help="check the mosaic against regions marked with `avmosaic label`")
    g.add_argument("folder", help="folder of images with <name>.json labels next to them")
    _detection_args(g)
    g.add_argument("--padding", type=float, default=0.15, help="(default: %(default)s)")
    g.add_argument("--shape", choices=SHAPES, default="rect", help="(default: %(default)s)")
    g.add_argument("--block-size", default="auto", help="(default: %(default)s)")
    g.add_argument("--min-coverage", type=float, default=0.9,
                   help="a marked region passes when this share is censored (default: %(default)s)")
    g.add_argument("--max-false-area", type=float, default=0.02,
                   help="a clean image fails when more than this share is censored "
                        "(default: %(default)s)")
    g.add_argument("--top", type=int, default=20, help="problems listed (default: %(default)s)")

    p = sub.add_parser("frames", help="save video frames as images (to label them)")
    p.add_argument("inputs", nargs="+", metavar="VIDEO")
    p.add_argument("-o", "--output", required=True, help="folder for the images")
    p.add_argument("--every", type=float, default=2.0,
                   help="seconds between frames (default: %(default)s)")
    p.add_argument("--start", default="0", help="first time to save, seconds or m:ss (default: 0)")
    p.add_argument("--end", default=None, help="stop at this time (default: the end)")

    p = sub.add_parser("label", help="mark the regions that must be censored, in your browser")
    p.add_argument("folder", help="folder of images; boxes are saved as <name>.json next to them")
    p.add_argument("--port", type=int, default=0, help="(default: any free port)")
    p.add_argument("--no-browser", action="store_true", help="only print the address")
    p.add_argument("--suggest", action="store_true",
                   help="pre-fill unlabelled images with the detector's boxes (faster, but biases "
                        "the labels towards what the detector already finds)")
    _detection_args(p)

    sub.add_parser("classes", help="list detectable classes and presets")

    p = sub.add_parser("models", help="list or download models")
    msub = p.add_subparsers(dest="models_command", metavar="ACTION")
    msub.add_parser("list", help="list models and whether they are available offline")
    d = msub.add_parser("download", help="download a model into the cache")
    d.add_argument("name", choices=list(MODELS))
    return parser


# --------------------------------------------------------------------------- helpers


def _parse_color(text: str) -> tuple[int, int, int]:
    named = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}
    t = text.strip().lower()
    if t in named:
        r, g, b = named[t]
    elif t.startswith("#") and len(t) == 7:
        r, g, b = (int(t[i:i + 2], 16) for i in (1, 3, 5))
    else:
        try:
            r, g, b = (int(v) for v in t.split(","))
        except ValueError:
            raise UsageError(f"invalid color {text!r}") from None
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise UsageError(f"invalid color {text!r}")
    return b, g, r


def media_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def collect_inputs(inputs: Sequence[str], suffix: str = "") -> list[tuple[Path, Path | None]]:
    """Expand directories into media files. Returns (file, base directory or None)."""
    found: list[tuple[Path, Path | None]] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file() or media_kind(child) is None:
                    continue
                if suffix and child.stem.endswith(suffix):
                    continue  # an output of an earlier run
                found.append((child, path))
        elif path.is_file():
            if media_kind(path) is None:
                raise UsageError(f"not a supported image or video file: {path}")
            found.append((path, None))
        else:
            raise UsageError(f"no such file or directory: {path}")
    if not found:
        raise UsageError("no images or videos found in the given inputs")
    return found


def output_path(src: Path, base: Path | None, output: str | None, suffix: str,
                single: bool) -> Path:
    ext = src.suffix
    if media_kind(src) == "video" and ext.lower() not in _KEEP_VIDEO_EXTS:
        ext = ".mp4"  # the default H.264 encoder cannot go into every container
    if output is None:
        return src.with_name(f"{src.stem}{suffix}{ext}")
    out = Path(output)
    if single and not out.is_dir() and out.suffix:
        return out
    rel = src.relative_to(base) if base is not None else Path(src.name)
    return out / rel.with_suffix(ext)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    """Frame progress on stderr.

    ``auto``: a bar redrawn in place when stderr is a terminal, nothing otherwise.
    ``always``: as auto in a terminal; otherwise a plain line every 10 seconds, so a
    background run leaves a readable log. ``never``: silent.
    """

    BAR = 24

    def __init__(self, mode: str = "auto", stream=None):
        self.stream = stream or sys.stderr
        tty = self.stream.isatty()
        self.inline = tty and mode != "never"
        self.lines = not tty and mode == "always"
        self.interval = 0.5 if self.inline else 10.0
        self.start = time.monotonic()
        self.last: float | None = None

    def __call__(self, done: int, total: int) -> None:
        if not (self.inline or self.lines):
            return
        now = time.monotonic()
        if self.last is not None and now - self.last < self.interval:
            return
        self.last = now
        msg = self.format(done, total, now - self.start)
        self.stream.write(f"\r{msg:<90}" if self.inline else msg + "\n")
        self.stream.flush()

    def format(self, done: int, total: int, elapsed: float) -> str:
        rate = done / elapsed if elapsed > 0 else 0.0
        if total <= 0:
            return f"  {done:,} frames  {rate:.1f} fps  elapsed {_clock(elapsed)}"
        frac = min(done / total, 1.0)
        filled = round(frac * self.BAR)
        bar = "#" * filled + "-" * (self.BAR - filled)
        eta = _clock(max(total - done, 0) / rate) if rate > 0 else "?"
        return (f"  [{bar}] {frac:6.1%}  {done:,}/{total:,} frames  {rate:.1f} fps  "
                f"ETA {eta}")

    def finish(self) -> None:
        if self.inline and self.last is not None:
            self.stream.write("\r" + " " * 90 + "\r")
            self.stream.flush()


def _say(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def _make_detector(args):
    try:
        classes = parse_classes(args.classes)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    quiet = getattr(args, "quiet", False)

    def on_download(done: int, total: int) -> None:
        if not quiet and sys.stderr.isatty():
            sys.stderr.write(f"\r  downloading model: {done / total:.0%}")
            sys.stderr.flush()

    try:
        detector = create_detector(
            args.model, classes=classes, threshold=args.threshold, nms_iou=args.nms_iou,
            resolution=args.resolution, device=args.device, channels=args.channels,
            download_progress=on_download,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise UsageError(str(exc)) from None
    _say(quiet, f"model {detector.model_path.name} @ {detector.resolution}px on "
                f"{detector.providers[0]}; classes: {', '.join(classes)}")
    return detector


def _settings(args, detector) -> dict:
    keys = ["classes", "threshold", "nms_iou", "model", "resolution", "device", "channels",
            "method", "block_size", "shape", "padding", "detect_every", "tracking", "hold",
            "max_lost", "smooth", "add_region"]
    data = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    data["classes"] = list(detector.classes)
    data["resolution"] = detector.resolution
    return data


def _collector(frames: list[tuple[int, float, list[Detection]]]):
    return lambda index, time_s, dets: frames.append((index, time_s, dets))


def _frames_record(frames: list[tuple[int, float, list[Detection]]]) -> list[dict]:
    return [
        {"index": i, "time": round(t, 3), "detections": [d.to_dict() for d in dets]}
        for i, t, dets in frames
        if dets
    ]


# --------------------------------------------------------------------------- commands


def cmd_censor(args) -> int:
    if args.output is None and not args.suffix:
        raise UsageError("--suffix must not be empty when -o is not given")
    try:
        manual = [ManualRegion.parse(spec) for spec in args.add_region]
        options = CensorOptions(method=args.method, block_size=args.block_size, shape=args.shape,
                                padding=args.padding, fill_color=_parse_color(args.fill_color))
        video_options = VideoOptions(detect_every=args.detect_every,
                                     tracking=args.tracking == "on", hold=args.hold,
                                     max_lost=args.max_lost, smooth=args.smooth,
                                     codec=args.codec, crf=args.crf, preset=args.preset,
                                     audio=not args.no_audio,
                                     ffmpeg_args=tuple(shlex.split(args.ffmpeg_args)),
                                     manual=tuple(manual))
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    jobs = collect_inputs(args.inputs, args.suffix if args.output is None else "")
    if any(m.timed for m in manual) and any(media_kind(src) == "image" for src, _ in jobs):
        raise UsageError("--add-region with START-END only applies to videos; "
                         "use x1,y1,x2,y2 for images")
    single = len(jobs) == 1 and jobs[0][1] is None
    plan = []
    for src, base in jobs:
        dst = output_path(src, base, args.output, args.suffix, single)
        if dst.resolve() == src.resolve():
            raise UsageError(f"output would overwrite its input: {src}")
        plan.append((src, dst))
    detector = _make_detector(args)

    records, failed = [], 0
    for n, (src, dst) in enumerate(plan, 1):
        tag = f"[{n}/{len(plan)}]" if len(plan) > 1 else ""
        if dst.exists() and not args.overwrite:
            print(f"{tag} {src}: skipped, {dst} exists (use -y to overwrite)", file=sys.stderr)
            failed += 1
            continue
        started = time.monotonic()
        record = {"input": str(src), "output": str(dst), "type": media_kind(src)}
        try:
            if record["type"] == "image":
                dets = censor_image(src, dst, detector, options, draw_boxes=args.draw_boxes,
                                    jpeg_quality=args.jpeg_quality,
                                    manual=[m.box for m in manual])
                record["detections"] = [d.to_dict() for d in dets]
                summary = f"{len(dets)} region(s)"
                if manual:
                    summary = f"{len(dets)} detected, {len(manual)} marked"
            else:
                frames: list[tuple[int, float, list[Detection]]] = []
                progress = Progress("never" if args.quiet else args.progress)
                _say(args.quiet, f"{tag} {src} → {dst}")
                try:
                    result = censor_video(
                        src, dst, detector, options, video_options, draw_boxes=args.draw_boxes,
                        progress=progress, on_detections=_collector(frames),
                    )
                finally:
                    progress.finish()
                hits = sum(1 for _, _, d in frames if d)
                record.update(frames=result.frames, detection_frames=result.detection_frames,
                              frames_with_detections=hits, audio_reencoded=result.audio_reencoded,
                              detections=_frames_record(frames))
                summary = f"{result.frames} frames, regions found in {hits} of " \
                          f"{result.detection_frames} checked"
                if manual:
                    summary += f", {len(manual)} marked region(s)"
                if result.audio_reencoded:
                    summary += ", audio re-encoded to AAC"
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep going with the other files
            print(f"{tag} {src}: error: {exc}", file=sys.stderr)
            failed += 1
            continue
        elapsed = time.monotonic() - started
        if record["type"] == "image":
            _say(args.quiet, f"{tag} {src} → {dst}: {summary} ({elapsed:.1f}s)".strip())
        else:
            _say(args.quiet, f"  done: {summary} ({elapsed:.1f}s)")
        records.append(record)

    if args.report:
        report = {"avmosaic": __version__, "settings": _settings(args, detector), "files": records}
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    if failed:
        print(f"{failed} of {len(plan)} file(s) failed or were skipped", file=sys.stderr)
    return 1 if failed else 0


def cmd_detect(args) -> int:
    if args.detect_every < 1:
        raise UsageError("--detect-every must be >= 1")
    jobs = collect_inputs(args.inputs)
    detector = _make_detector(args)
    records, failed = [], 0
    for src, _ in jobs:
        record: dict = {"input": str(src), "type": media_kind(src)}
        try:
            if record["type"] == "image":
                image, dets = detect_image(src, detector)
                record.update(width=image.shape[1], height=image.shape[0],
                              detections=[d.to_dict() for d in dets])
            else:
                frames: list[tuple[int, float, list[Detection]]] = []
                progress = Progress("never" if args.quiet else args.progress)
                try:
                    result = detect_video(src, detector, detect_every=args.detect_every,
                                          progress=progress,
                                          on_detections=_collector(frames))
                finally:
                    progress.finish()
                record.update(frames=result.frames, detection_frames=result.detection_frames,
                              detections=_frames_record(frames))
        except Exception as exc:
            print(f"{src}: error: {exc}", file=sys.stderr)
            failed += 1
            continue
        records.append(record)
    text = json.dumps({"avmosaic": __version__, "settings": _settings(args, detector),
                       "files": records}, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 1 if failed else 0


def cmd_eval(args) -> int:
    if args.eval_command == "regions":
        return _eval_regions(args)
    if args.eval_command == "summary":
        try:
            thresholds = sorted(float(t) for t in args.thresholds.split(","))
        except ValueError:
            raise UsageError(f"invalid thresholds {args.thresholds!r}") from None
        for n, results in enumerate(args.results):
            path = Path(results)
            if not path.is_file():
                raise UsageError(f"no such results file: {path}")
            videos, config, errors = evaluate.load(path, [])
            try:
                classes = parse_classes(args.classes) if args.classes else tuple(config["classes"])
            except (ValueError, KeyError) as exc:
                raise UsageError(f"cannot tell which classes to count: {exc}") from None
            videos, config, errors = evaluate.load(path, classes)
            if n:
                print()
            print(f"== {path}")
            print("settings: " + ", ".join(f"{k}={v}" for k, v in config.items() if k != "classes"))
            print(f"classes: {', '.join(classes)}")
            if errors:
                print(f"videos that failed to decode: {errors}")
            low = config.get("threshold")
            if low is not None and thresholds[0] < low:
                print(f"note: thresholds below the recorded floor {low} undercount")
            print(evaluate.summarize(videos, thresholds, args.top))
        return 0

    if args.sample_fps <= 0:
        raise UsageError("--sample-fps must be > 0")
    items = []
    for path, base in collect_inputs(args.inputs):
        # Labels come from the path below (and including) the folder given, so a
        # repository called "...Porn..." does not make every file explicit.
        items.append((path, base.name / path.relative_to(base) if base else Path(path.name)))
    unlabelled = [p for p, lp in items if evaluate.label_for(lp, args.positive, args.negative) is None]
    if unlabelled:
        print(f"note: {len(unlabelled)} file(s) match neither --positive nor --negative and "
              f"will not be counted, e.g. {unlabelled[0]}", file=sys.stderr)
    detector = _make_detector(args)
    config = {"model": args.model, "resolution": detector.resolution, "channels": args.channels,
              "classes": list(detector.classes), "threshold": args.threshold,
              "sample_fps": args.sample_fps}
    try:
        evaluated, failed = evaluate.run(
            items, Path(args.output), detector, fps=args.sample_fps, config=config,
            positive=args.positive, negative=args.negative,
            log=lambda msg: _say(args.quiet, msg))
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    _say(args.quiet, f"evaluated {evaluated} file(s), {failed} failed; results in {args.output}")
    return 1 if failed else 0


def _eval_regions(args) -> int:
    root = Path(args.folder)
    if not root.is_dir():
        raise UsageError(f"no such folder: {root}")
    labelled = evaluate.labelled_images(root)
    if not labelled:
        raise UsageError(f"no labelled images in {root}; mark some with `avmosaic label {root}`")
    try:
        options = CensorOptions(block_size=args.block_size, shape=args.shape, padding=args.padding)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    args.quiet = False
    detector = _make_detector(args)
    checks = [evaluate.check_regions(image, regions, detector, options)
              for image, regions in labelled]
    text, passed = evaluate.summarize_regions(checks, root, args.min_coverage,
                                              args.max_false_area, args.top)
    print(text)
    return 0 if passed else 1


def cmd_frames(args) -> int:
    if args.every <= 0:
        raise UsageError("--every must be > 0")
    try:
        start = seconds(args.start)
        end = None if args.end is None else seconds(args.end)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    if end is not None and end <= start:
        raise UsageError("--end must be after --start")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for video, _ in collect_inputs(args.inputs):
        if media_kind(video) != "video":
            continue
        count = 0
        for t, frame in evaluate.sample_frames(video, 1.0 / args.every, start, end):
            write_image(out / f"{video.stem}_{round(t * 1000):08d}ms.jpg", frame)
            count += 1
        print(f"{video}: {count} frame(s)", file=sys.stderr)
        total += count
    print(f"saved {total} image(s) to {out}", file=sys.stderr)
    return 0 if total else 1


def cmd_label(args) -> int:
    from . import labeler

    root = Path(args.folder)
    if not root.is_dir():
        raise UsageError(f"no such folder: {root}")
    suggest = None
    if args.suggest:
        args.quiet = False
        detector = _make_detector(args)

        def suggest(image: Path) -> list[list[float]]:
            return [[d.x1, d.y1, d.x2, d.y2] for d in detect_image(image, detector)[1]]

    store = labeler.LabelStore(root, suggest)
    server = labeler.make_server(store, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"labelling {len(store.items)} image(s) in {root} at {url}  (Ctrl+C to stop)",
          file=sys.stderr)
    if not args.no_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_classes(_args) -> int:
    print("Presets:")
    for name, labels in PRESETS.items():
        default = " (default)" if name == DEFAULT_CLASSES else ""
        shown = "every class" if labels == LABELS else ", ".join(labels)
        print(f"  {name:<10}{default:<11} {shown}")
    print("\nClasses:")
    for label in LABELS:
        print(f"  {label}")
    return 0


def cmd_models(args) -> int:
    if args.models_command == "download":
        spec = MODELS[args.name]
        local = find_model(spec)
        if local is not None:
            print(f"{spec.name} is already available: {local}")
            return 0

        def progress(done: int, total: int) -> None:
            if sys.stderr.isatty():
                sys.stderr.write(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB")
                sys.stderr.flush()

        path = download_model(spec, progress=progress)
        if sys.stderr.isatty():
            sys.stderr.write("\n")
        print(f"downloaded {spec.name} to {path}")
        return 0
    print(f"cache: {cache_dir()}")
    for spec in MODELS.values():
        local = find_model(spec)
        state = f"available ({local})" if local else "not downloaded"
        print(f"  {spec.name:<6} {spec.description}\n         {state}")
    try:
        providers = resolve_providers("auto")
        print(f"auto device uses: {providers[0]}")
    except ValueError:
        pass
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "censor")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    handler = {"censor": cmd_censor, "detect": cmd_detect, "eval": cmd_eval,
               "frames": cmd_frames, "label": cmd_label, "classes": cmd_classes,
               "models": cmd_models}[args.command]
    try:
        return handler(args)
    except UsageError as exc:
        parser.exit(2, f"avmosaic: error: {exc}\n")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

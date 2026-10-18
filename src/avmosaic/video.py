"""Video pipeline: decode with ffmpeg, censor frame by frame, re-encode, copy audio back."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from .censor import CensorOptions, apply_censor, draw_detections
from .detector import Detection, Detector
from .tracking import HoldPlanner, TrackingPlanner, TrackOptions

VIDEO_EXTS = frozenset(
    {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}
)
CRF_CODECS = frozenset({"libx264", "libx265", "libvpx-vp9", "libaom-av1", "libsvtav1"})
PRESET_CODECS = frozenset({"libx264", "libx265"})

ProgressFn = Callable[[int, int], None]
DetectionsFn = Callable[[int, float, list[Detection]], None]


@dataclass
class VideoOptions:
    # Run the detector on every Nth frame (and on scene cuts when tracking).
    detect_every: int = 1
    # Tracking: follow regions with optical flow. Off: censor the union of boxes
    # found within `hold` frames, which smears the mosaic when things move fast.
    tracking: bool = True
    # Tracking on: follow a new region back this many frames before its first
    # detection. Tracking off: also censor boxes found up to this many frames away.
    hold: int = 5
    # Tracking on: keep following a region this many frames after the detector lost it.
    max_lost: int = 15
    # Tracking on: average box positions over +-N frames.
    smooth: int = 2
    codec: str = "libx264"
    crf: int | None = 18
    preset: str | None = "medium"
    audio: bool = True
    ffmpeg_args: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.detect_every < 1:
            raise ValueError("detect_every must be >= 1")
        for name in ("hold", "max_lost", "smooth"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    def planner(self) -> HoldPlanner | TrackingPlanner:
        if self.tracking:
            return TrackingPlanner(self.detect_every, TrackOptions(
                max_lost=self.max_lost, backfill=self.hold, smooth=self.smooth))
        return HoldPlanner(self.detect_every, self.hold)

    @property
    def reach(self) -> int:
        """How many frames away a detection still counts.

        With detect_every N, a frame between two detection frames is covered by
        both of them; ``hold`` adds extra frames on top.
        """
        return self.hold + self.detect_every - 1


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    rate: str  # exact rate for ffmpeg, e.g. "30000/1001"
    duration: float

    @property
    def frames_estimate(self) -> int:
        return max(0, round(self.duration * self.fps))


@dataclass
class VideoResult:
    frames: int
    detection_frames: int
    audio_reencoded: bool = False


def ffmpeg_exe() -> str:
    """The ffmpeg binary: $IMAGEIO_FFMPEG_EXE if set, else the one bundled with imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


_NTSC_RATES = {23.98: "24000/1001", 29.97: "30000/1001", 47.95: "48000/1001",
               59.94: "60000/1001", 119.88: "120000/1001"}


def rate_string(fps: float) -> str:
    for approx, exact in _NTSC_RATES.items():
        if abs(fps - approx) < 0.006:
            return exact
    return f"{fps:.6g}"


def probe(path: str | Path) -> VideoInfo:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24")
    try:
        meta = next(reader)
    finally:
        reader.close()
    width, height = meta["size"]
    fps = float(meta.get("fps") or 0) or 30.0
    return VideoInfo(int(width), int(height), fps, rate_string(fps), float(meta.get("duration") or 0))


def iter_frames(path: str | Path, info: VideoInfo) -> Iterator[np.ndarray]:
    """Yield writable BGR frames at a constant rate (keeps the picture in sync with the audio)."""
    reader = imageio_ffmpeg.read_frames(
        str(path), pix_fmt="bgr24", output_params=["-fps_mode", "cfr", "-r", info.rate]
    )
    try:
        meta = next(reader)
        width, height = meta["size"]
        for raw in reader:
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
    finally:
        reader.close()


class _Encoder:
    def __init__(self, path: Path, info: VideoInfo, options: VideoOptions, log_path: Path):
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
            "-r", info.rate, "-i", "-", "-an", "-c:v", options.codec,
        ]
        if info.width % 2 or info.height % 2:
            # yuv420p needs even dimensions; pad one black row/column rather than scaling.
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        if options.crf is not None and options.codec in CRF_CODECS:
            cmd += ["-crf", str(options.crf)]
            if options.codec == "libvpx-vp9":
                cmd += ["-b:v", "0"]
        if options.preset and options.codec in PRESET_CODECS:
            cmd += ["-preset", options.preset]
        cmd += ["-pix_fmt", "yuv420p", *options.ffmpeg_args, str(path)]
        self._log_path = log_path
        self._log = open(log_path, "wb")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log
        )

    def _error(self, what: str) -> RuntimeError:
        self._log.flush()
        text = self._log_path.read_text(errors="replace").strip()
        return RuntimeError(f"{what}\n{text}" if text else what)

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self._proc.wait()
            raise self._error("ffmpeg encoder stopped unexpectedly") from None

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        code = self._proc.wait()
        self._log.close()
        if code != 0:
            raise self._error(f"ffmpeg encoder failed (exit code {code})")

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._log.close()


# Audio codecs that players reliably handle in each container; others become AAC.
_MP4_AUDIO = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})
_SAFE_AUDIO = {".mp4": _MP4_AUDIO, ".m4v": _MP4_AUDIO}


def audio_codecs(path: str | Path) -> list[str]:
    result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True)
    text = result.stderr.decode(errors="replace")
    return re.findall(r"Stream #\d+:\d+.*?: Audio: (\w+)", text)


def _mux(video_only: Path, src: Path, dst: Path, audio: bool) -> bool:
    """Combine the encoded picture with the source audio. Returns True if audio was re-encoded."""
    base = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_only)]
    ext = dst.suffix.lower()
    tail = ["-movflags", "+faststart"] if ext in (".mp4", ".m4v", ".mov") else []
    if not audio:
        attempts = [(False, base + ["-map", "0:v:0", "-c:v", "copy", *tail, str(dst)])]
    else:
        with_audio = base + ["-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy"]
        reencode = (True, with_audio + ["-c:a", "aac", "-b:a", "192k", *tail, str(dst)])
        safe = _SAFE_AUDIO.get(ext)
        if safe is not None and not set(audio_codecs(src)) <= safe:
            # e.g. PCM: ffmpeg would copy it into MP4, but most players stay silent.
            attempts = [reencode]
        else:
            attempts = [(False, with_audio + ["-c:a", "copy", *tail, str(dst)]), reencode]
    errors = []
    for reencoded, cmd in attempts:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            return reencoded
        errors.append(result.stderr.decode(errors="replace").strip())
    raise RuntimeError("ffmpeg could not write the output file:\n" + "\n".join(errors))


def censor_video(
    src: str | Path,
    dst: str | Path,
    detector: Detector,
    censor_options: CensorOptions,
    video_options: VideoOptions,
    *,
    draw_boxes: bool = False,
    progress: ProgressFn | None = None,
    on_detections: DetectionsFn | None = None,
) -> VideoResult:
    src, dst = Path(src), Path(dst)
    info = probe(src)
    planner = video_options.planner()
    tmpdir = Path(tempfile.mkdtemp(prefix="avmosaic-"))
    encoder = None
    detected = 0

    def detect(index: int, frame: np.ndarray) -> list[Detection]:
        nonlocal detected
        detected += 1
        dets = detector.detect(frame)
        if on_detections:
            on_detections(index, index / info.fps, dets)
        return dets

    def render(planned: list[tuple[int, np.ndarray, list[Detection]]]) -> None:
        for _, frame, regions in planned:
            apply_censor(frame, regions, censor_options)
            if draw_boxes:
                draw_detections(frame, regions)
            encoder.write(frame)

    try:
        encoder = _Encoder(tmpdir / "video.mkv", info, video_options, tmpdir / "encode.log")
        count = 0
        for index, frame in enumerate(iter_frames(src, info)):
            render(planner.push(index, frame, detect))
            count = index + 1
            if progress:
                progress(count, info.frames_estimate)
        render(planner.flush())
        encoder.close()
        if count == 0:
            raise ValueError(f"no video frames could be decoded from {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        reencoded = _mux(tmpdir / "video.mkv", src, dst, video_options.audio)
        return VideoResult(frames=count, detection_frames=detected, audio_reencoded=reencoded)
    except BaseException:
        if encoder is not None:
            encoder.abort()
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def detect_video(
    src: str | Path,
    detector: Detector,
    *,
    detect_every: int = 1,
    progress: ProgressFn | None = None,
    on_detections: DetectionsFn | None = None,
) -> VideoResult:
    info = probe(src)
    count = 0
    for index, frame in enumerate(iter_frames(src, info)):
        if index % detect_every == 0:
            dets = detector.detect(frame)
            if on_detections:
                on_detections(index, index / info.fps, dets)
        count = index + 1
        if progress:
            progress(count, info.frames_estimate)
    return VideoResult(frames=count, detection_frames=-(-count // detect_every))

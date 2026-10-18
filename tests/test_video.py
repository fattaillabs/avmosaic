import imageio_ffmpeg
import numpy as np
import pytest

from avmosaic.censor import CensorOptions
from avmosaic.video import VideoOptions, censor_video, detect_video, probe, rate_string

from .conftest import (
    ColorDetector,
    background,
    detail,
    make_video,
    object_box,
    read_all_frames,
    stream_info,
)

OPTS = CensorOptions(block_size=12, padding=0.1)
# Mean absolute difference that re-encoding the noisy synthetic texture can cause on its
# own. ffmpeg builds differ: 5.8 on macOS arm64, 6.01 on the Linux CI runners.
REENCODE_NOISE = 8


def censored_everywhere(out_frames, src_frames, slack: float = 0.15) -> list[int]:
    """Indices of frames whose object region still has most of its detail."""
    leaks = []
    for i, (out, src) in enumerate(zip(out_frames, src_frames, strict=True)):
        box = object_box(i)
        if detail(out, box) > detail(src, box) * slack:
            leaks.append(i)
    return leaks


def test_rate_string_keeps_ntsc_rates_exact():
    assert rate_string(29.97) == "30000/1001"
    assert rate_string(23.976) == "24000/1001"
    assert rate_string(25.0) == "25"
    assert rate_string(30.0) == "30"


def test_censor_video_keeps_frames_audio_and_size(tmp_path, sample_video):
    dst = tmp_path / "out.mp4"
    result = censor_video(sample_video, dst, ColorDetector(), OPTS, VideoOptions(hold=0))

    src_frames, out_frames = read_all_frames(sample_video), read_all_frames(dst)
    assert result.frames == len(src_frames) == len(out_frames) == 45
    assert result.detection_frames == 45
    assert not result.audio_reencoded
    info = stream_info(dst)
    assert "Audio: aac" in info and "320x240" in info
    in_secs = imageio_ffmpeg.count_frames_and_secs(str(sample_video))[1]
    out_secs = imageio_ffmpeg.count_frames_and_secs(str(dst))[1]
    assert abs(in_secs - out_secs) < 0.05

    assert censored_everywhere(out_frames, src_frames) == []
    # Background far from the object is preserved (only re-encoding noise).
    for out, src in zip(out_frames, src_frames, strict=True):
        diff = np.abs(out[150:].astype(int) - src[150:].astype(int)).mean()
        assert diff < REENCODE_NOISE


def test_missed_frame_is_covered_by_hold_without_tracking(tmp_path, sample_video):
    src_frames = read_all_frames(sample_video)

    leaky = tmp_path / "hold0.mp4"
    censor_video(sample_video, leaky, ColorDetector(skip={10}), OPTS,
                 VideoOptions(tracking=False, hold=0))
    assert censored_everywhere(read_all_frames(leaky), src_frames) == [10]

    safe = tmp_path / "hold2.mp4"
    censor_video(sample_video, safe, ColorDetector(skip={10}), OPTS,
                 VideoOptions(tracking=False, hold=2))
    assert censored_everywhere(read_all_frames(safe), src_frames) == []


def test_missed_frames_are_followed_by_tracking(tmp_path, sample_video):
    dst = tmp_path / "tracked.mp4"
    censor_video(sample_video, dst, ColorDetector(skip=set(range(10, 20))), OPTS,
                 VideoOptions(hold=0, max_lost=15))
    assert censored_everywhere(read_all_frames(dst), read_all_frames(sample_video)) == []


def test_detect_every_n_still_covers_frames_in_between(tmp_path, sample_video):
    detector = ColorDetector()
    dst = tmp_path / "every3.mp4"
    result = censor_video(sample_video, dst, detector, OPTS,
                          VideoOptions(detect_every=3, hold=0))
    assert detector.calls == 15 == result.detection_frames
    assert censored_everywhere(read_all_frames(dst), read_all_frames(sample_video)) == []


def test_on_detections_reports_frame_index_and_time(tmp_path, sample_video):
    seen = []
    censor_video(sample_video, tmp_path / "o.mp4", ColorDetector(), OPTS,
                 VideoOptions(detect_every=5, audio=False),
                 on_detections=lambda i, t, d: seen.append((i, round(t, 3), len(d))))
    assert seen[:3] == [(0, 0.0, 1), (5, 0.167, 1), (10, 0.333, 1)]
    assert "Audio" not in stream_info(tmp_path / "o.mp4")


def test_odd_size_ntsc_rate_and_unicode_names(tmp_path):
    src = tmp_path / "テスト 영상.mkv"  # lossless FFV1: H.264 4:2:0 cannot store odd sizes
    make_video(src, frames=12, width=321, height=241, fps="30000/1001", audio=False,
               video_args=("-c:v", "ffv1"))
    info = probe(src)
    assert info.rate == "30000/1001"
    dst = tmp_path / "출력" / "結果.mp4"
    result = censor_video(src, dst, ColorDetector(), OPTS, VideoOptions())
    assert result.frames == 12
    assert "322x242" in stream_info(dst)  # padded to even for yuv420p


def test_pcm_audio_in_avi_falls_back_to_aac_in_mp4(tmp_path):
    src = tmp_path / "pcm.avi"
    make_video(src, frames=10, audio_codec="pcm_s16le")
    dst = tmp_path / "pcm.mp4"
    result = censor_video(src, dst, ColorDetector(), OPTS, VideoOptions())
    assert result.audio_reencoded
    assert "Audio: aac" in stream_info(dst)


def test_pcm_audio_is_copied_into_mkv(tmp_path):
    src = tmp_path / "pcm.avi"
    make_video(src, frames=10, audio_codec="pcm_s16le")
    dst = tmp_path / "pcm.mkv"
    result = censor_video(src, dst, ColorDetector(), OPTS, VideoOptions())
    assert not result.audio_reencoded
    assert "Audio: pcm_s16le" in stream_info(dst)


def test_frames_without_object_are_untouched(tmp_path):
    src = tmp_path / "plain.mp4"
    make_video(src, frames=8, audio=False, draw=lambda frame, i: None)
    dst = tmp_path / "plain_out.mp4"
    result = censor_video(src, dst, ColorDetector(), OPTS, VideoOptions())
    assert result.frames == 8
    whole = (0, 0, 320, 240)
    for out, ref in zip(read_all_frames(dst), read_all_frames(src), strict=True):
        # A mosaic would flatten the texture (detail ratio near 0); re-encoding keeps ~0.99.
        assert detail(out, whole, inset=0) > 0.9 * detail(ref, whole, inset=0)
        assert np.abs(out.astype(int) - ref.astype(int)).mean() < REENCODE_NOISE


def test_detect_video_only_runs_detector(sample_video):
    detector = ColorDetector()
    hits = []
    result = detect_video(sample_video, detector, detect_every=4,
                          on_detections=lambda i, t, d: hits.append(i))
    assert result.frames == 45 and detector.calls == 12
    assert hits == list(range(0, 45, 4))


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        censor_video(bad, tmp_path / "x.mp4", ColorDetector(), OPTS, VideoOptions())


def test_invalid_video_options():
    with pytest.raises(ValueError):
        VideoOptions(detect_every=0)
    with pytest.raises(ValueError):
        VideoOptions(hold=-1)
    with pytest.raises(ValueError):
        VideoOptions(max_lost=-1)
    assert VideoOptions(detect_every=4, hold=2).reach == 5


def test_background_helper_has_no_red():
    frame = background(100, 100)
    assert ColorDetector().detect(frame) == []

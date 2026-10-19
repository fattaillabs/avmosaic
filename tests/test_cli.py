import json
import subprocess
import sys
from pathlib import Path

import pytest

from avmosaic import cli
from avmosaic.image import read_image, write_image

from .conftest import ColorDetector, background, flat_tiles, make_video, paint_object


class FakeModel(ColorDetector):
    model_path = Path("fake.onnx")
    resolution = 320
    providers = ["CPUExecutionProvider"]

    def __init__(self, classes=None, **_):
        super().__init__()
        self.classes = classes or ("FEMALE_GENITALIA_EXPOSED",)


@pytest.fixture
def fake_model(monkeypatch):
    created = []

    def factory(model, **kwargs):
        det = FakeModel(**kwargs)
        created.append((model, kwargs, det))
        return det

    monkeypatch.setattr(cli, "create_detector", factory)
    return created


def write_sample(path: Path, size: int = 50) -> None:
    img = background(160, 200)
    paint_object(img, 60, 40, size)
    write_image(path, img)


def test_shorthand_writes_next_to_input(tmp_path, fake_model):
    write_sample(tmp_path / "pic.png")
    assert cli.main([str(tmp_path / "pic.png"), "-q"]) == 0
    out = read_image(tmp_path / "pic_censored.png")
    assert flat_tiles(out, (60, 40, 110, 90), block=4) == 1.0  # auto: max(4, 200/100)


def test_options_reach_the_detector(tmp_path, fake_model):
    write_sample(tmp_path / "pic.png")
    cli.main(["censor", str(tmp_path / "pic.png"), "-q", "--classes", "genitals,breasts",
              "--threshold", "0.4", "--model", "640m", "--resolution", "512"])
    model, kwargs, _ = fake_model[0]
    assert model == "640m"
    assert kwargs["threshold"] == 0.4 and kwargs["resolution"] == 512
    assert "FEMALE_BREAST_EXPOSED" in kwargs["classes"]


def test_existing_output_needs_overwrite(tmp_path, fake_model, capsys):
    write_sample(tmp_path / "pic.png")
    (tmp_path / "pic_censored.png").write_bytes(b"keep me")
    assert cli.main([str(tmp_path / "pic.png"), "-q"]) == 1
    assert (tmp_path / "pic_censored.png").read_bytes() == b"keep me"
    assert "use -y" in capsys.readouterr().err
    assert cli.main([str(tmp_path / "pic.png"), "-q", "-y"]) == 0
    assert read_image(tmp_path / "pic_censored.png").shape == (160, 200, 3)


def test_directory_batch_mirrors_tree(tmp_path, fake_model):
    src = tmp_path / "in"
    write_sample(src / "a" / "one.png")
    write_sample(src / "b" / "two.jpg")
    make_video(src / "clip.avi", frames=6, audio=False)
    (src / "notes.txt").parent.mkdir(exist_ok=True)
    (src / "notes.txt").write_text("not media")
    out = tmp_path / "out"

    assert cli.main([str(src), "-o", str(out), "-q"]) == 0

    produced = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert produced == ["a/one.png", "b/two.jpg", "clip.mp4"]  # avi -> mp4 for H.264


def test_rerun_without_output_ignores_previous_results(tmp_path, fake_model):
    write_sample(tmp_path / "x.png")
    assert cli.main([str(tmp_path), "-q"]) == 0
    assert cli.main([str(tmp_path), "-q", "-y"]) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["x.png", "x_censored.png"]


def test_report_lists_detections(tmp_path, fake_model):
    write_sample(tmp_path / "pic.png")
    make_video(tmp_path / "clip.mp4", frames=6)
    report = tmp_path / "report.json"
    code = cli.main([str(tmp_path / "pic.png"), str(tmp_path / "clip.mp4"), "-o",
                     str(tmp_path / "out"), "--report", str(report), "-q", "--detect-every", "2"])
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    image, video = data["files"]
    assert image["type"] == "image" and len(image["detections"]) == 1
    assert image["detections"][0]["class"] == "FEMALE_GENITALIA_EXPOSED"
    assert video["frames"] == 6 and video["detection_frames"] == 3
    assert [f["index"] for f in video["detections"]] == [0, 2, 4]
    assert data["settings"]["detect_every"] == 2
    assert data["settings"]["tracking"] == "on" and data["settings"]["max_lost"] == 15


def test_detect_prints_json(tmp_path, fake_model, capsys):
    write_sample(tmp_path / "pic.png")
    assert cli.main(["detect", str(tmp_path / "pic.png"), "-q"]) == 0
    data = json.loads(capsys.readouterr().out)
    (record,) = data["files"]
    assert record["width"] == 200 and record["height"] == 160
    x1, y1, x2, y2 = record["detections"][0]["box"]
    assert (x1, y1, x2, y2) == (60, 40, 110, 90)
    assert not (tmp_path / "pic_censored.png").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--classes", "FEMALE_BREST_EXPOSED"], "Did you mean"),
        (["--block-size", "huge"], "invalid block size"),
        (["--fill-color", "300,0,0"], "invalid color"),
        (["--detect-every", "0"], "detect_every"),
        (["--max-lost", "-1"], "max_lost"),
    ],
)
def test_bad_options_exit_2(tmp_path, fake_model, capsys, args, message):
    write_sample(tmp_path / "pic.png")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path / "pic.png"), *args])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_and_unsupported_inputs(tmp_path, fake_model, capsys):
    (tmp_path / "doc.txt").write_text("x")
    for target in (tmp_path / "nope.png", tmp_path / "doc.txt"):
        with pytest.raises(SystemExit) as exc:
            cli.main([str(target)])
        assert exc.value.code == 2


def test_output_must_not_overwrite_input(tmp_path, fake_model):
    write_sample(tmp_path / "pic.png")
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "pic.png"), "-o", str(tmp_path / "pic.png")])


def test_fill_color_parsing():
    assert cli._parse_color("#ff8000") == (0, 128, 255)
    assert cli._parse_color("1,2,3") == (3, 2, 1)
    assert cli._parse_color("white") == (255, 255, 255)


def test_classes_and_models_commands(capsys):
    assert cli.main(["classes"]) == 0
    out = capsys.readouterr().out
    assert "genitals" in out and "ANUS_EXPOSED" in out
    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "320n" in out and "640m" in out


def test_python_dash_m_entry_point():
    result = subprocess.run([sys.executable, "-m", "avmosaic", "--version"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "avmosaic" in result.stdout


class _Stream:
    def __init__(self, tty: bool):
        self.tty, self.text = tty, ""

    def isatty(self):
        return self.tty

    def write(self, text):
        self.text += text

    def flush(self):
        pass


def _run_progress(mode, tty, monkeypatch, times):
    clock = iter(times)
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    stream = _Stream(tty)
    progress = cli.Progress(mode, stream)  # first time value = start
    for done in (100, 300, 500, 1000):
        progress(done, 1000)
    progress.finish()
    return stream.text


def test_progress_lines_in_logs(monkeypatch):
    text = _run_progress("always", False, monkeypatch, [0, 1, 5, 11, 22])
    lines = text.splitlines()
    assert len(lines) == 3  # at 1 s, then not before 10 s later: 11 s, 22 s
    assert lines[0] == "  [##----------------------]  10.0%  100/1,000 frames  100.0 fps  ETA 0:09"
    assert lines[-1].startswith("  [########################] 100.0%  1,000/1,000 frames")


def test_progress_bar_in_a_terminal(monkeypatch):
    text = _run_progress("auto", True, monkeypatch, [0, 1, 1.2, 2, 3])
    redraws = [chunk for chunk in text.split("\r") if chunk.strip()]
    assert len(redraws) == 3  # at 1 s, 2 s, 3 s; 1.2 s is too soon after 1 s
    assert text.endswith("\r")  # the bar is cleared at the end
    assert "[############------------]  50.0%  500/1,000 frames" in text


@pytest.mark.parametrize(("mode", "tty"), [("auto", False), ("never", True), ("never", False)])
def test_progress_silent(monkeypatch, mode, tty):
    assert _run_progress(mode, tty, monkeypatch, [0, 1, 2, 3, 4]) == ""


def test_progress_flag_reaches_video_runs(tmp_path, fake_model, capsys):
    make_video(tmp_path / "clip.mp4", frames=6, audio=False)
    assert cli.main([str(tmp_path / "clip.mp4"), "--progress", "always"]) == 0
    assert "frames" in capsys.readouterr().err


def test_add_region_on_an_image(tmp_path, fake_model):
    write_image(tmp_path / "plain.png", background(160, 200))  # nothing to detect
    assert cli.main([str(tmp_path / "plain.png"), "-q", "--block-size", "8",
                     "--padding", "0", "--add-region", "40,40,120,104"]) == 0
    out = read_image(tmp_path / "plain_censored.png")
    assert flat_tiles(out, (40, 40, 120, 104), block=8) == 1.0
    src = read_image(tmp_path / "plain.png")
    assert (out[120:] == src[120:]).all()


def test_timed_region_is_refused_for_images(tmp_path, fake_model, capsys):
    write_image(tmp_path / "plain.png", background(60, 60))
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path / "plain.png"), "--add-region", "1-2:0,0,10,10"])
    assert exc.value.code == 2 and "only applies to videos" in capsys.readouterr().err


def test_add_region_on_a_video_time_range(tmp_path, fake_model):
    from .conftest import detail, read_all_frames

    make_video(tmp_path / "plain.mp4", frames=30, audio=False, draw=lambda frame, i: None)
    assert cli.main([str(tmp_path / "plain.mp4"), "-q", "--block-size", "12", "--padding", "0",
                     "--add-region", "0.2-0.6:96,60,192,156"]) == 0
    out = read_all_frames(tmp_path / "plain_censored.mp4")
    src = read_all_frames(tmp_path / "plain.mp4")
    box = (96, 60, 192, 156)
    flattened = [detail(o, box) < 0.3 * detail(s, box) for o, s in zip(out, src, strict=True)]
    assert flattened == [6 <= i <= 18 for i in range(30)]  # 0.2 s to 0.6 s at 30 fps


def test_summary_counts_marked_regions(tmp_path, fake_model, capsys):
    write_image(tmp_path / "plain.png", background(80, 100))
    assert cli.main([str(tmp_path / "plain.png"), "--add-region", "10,10,40,40"]) == 0
    assert "0 detected, 1 marked" in capsys.readouterr().err

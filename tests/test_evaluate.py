import json
from pathlib import Path

import pytest

from avmosaic import cli, evaluate
from avmosaic.evaluate import VideoScore, auc, label_for, subset_for
from avmosaic.image import write_image

from .conftest import background, make_video, paint_object
from .test_cli import fake_model  # noqa: F401  (fixture)


@pytest.mark.parametrize(
    ("path", "label", "subset"),
    [
        ("NPDI/Porn/vid0001.mp4", 1, None),
        ("NPDI/nonPorn/difficult/vid0002.mp4", 0, "difficult"),
        ("NPDI/NonPorn/easy/vid0003.mp4", 0, "easy"),
        ("Pornography-2k/vPorn000001.mp4", 1, None),
        ("Pornography-2k/vNonPorn000001.mp4", 0, None),
        ("less_images/not_porn/bikini/x.jpg", 0, None),
        ("less_images/porn/x.jpg", 1, None),
        ("less_images/not_porn/bikini/hard-rock-hotel.jpg", 0, None),  # file names don't count
        ("data/nonporn/hardware/x.mp4", 0, None),
        ("other/clip.mp4", None, None),
    ],
)
def test_labels_come_from_the_path(path, label, subset):
    assert label_for(Path(path)) == label
    assert subset_for(Path(path)) == subset


def test_auc_counts_ties_as_half():
    def video(label, score):
        return VideoScore("x", label, None, 10, [(0.0, score, "ANUS_EXPOSED")] if score else [])

    assert auc([video(1, 0.9), video(1, 0.8), video(0, 0.3), video(0, 0.0)]) == 1.0
    assert auc([video(1, 0.5), video(0, 0.5)]) == 0.5
    assert auc([video(1, 0.2), video(0, 0.7)]) == 0.0
    assert auc([video(1, 0.9)]) is None


@pytest.fixture
def dataset(tmp_path) -> Path:
    root = tmp_path / "NPDI"
    (root / "porn").mkdir(parents=True)
    (root / "nonporn" / "easy").mkdir(parents=True)
    (root / "nonporn" / "difficult").mkdir(parents=True)
    (root / "misc").mkdir()
    make_video(root / "porn" / "a.mp4", frames=30, audio=False)
    make_video(root / "nonporn" / "easy" / "b.mp4", frames=30, audio=False,
               draw=lambda frame, i: None)
    make_video(root / "nonporn" / "difficult" / "c.mp4", frames=30, audio=False,
               draw=lambda frame, i: None)
    make_video(root / "misc" / "d.mp4", frames=30, audio=False)
    return root


def test_run_resume_and_summary(tmp_path, dataset, fake_model, capsys):  # noqa: F811
    results = tmp_path / "results.jsonl"
    argv = ["eval", "run", str(dataset), "-o", str(results), "--sample-fps", "4", "-q"]
    assert cli.main(argv) == 0
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 4
    by_name = {Path(r["path"]).name: r for r in records}
    assert by_name["a.mp4"]["label"] == 1 and by_name["a.mp4"]["samples"] == 4
    assert len(by_name["a.mp4"]["frames"]) == 4  # the object is in every sample
    assert by_name["b.mp4"]["frames"] == [] and by_name["c.mp4"]["subset"] == "difficult"
    assert by_name["d.mp4"]["label"] is None
    assert "1 file(s) match neither" in capsys.readouterr().err

    assert cli.main(argv) == 0  # resume: nothing new
    assert len(results.read_text().splitlines()) == 4

    assert cli.main(["eval", "summary", str(results), "--thresholds", "0.5,0.95"]) == 0
    out = capsys.readouterr().out
    assert "files: 1 explicit, 2 clean" in out
    assert "file-level AUC (best score per file): 1.000" in out
    row_05 = next(line for line in out.splitlines() if line.strip().startswith("0.50"))
    assert row_05.split()[1:4] == ["100.0%", "/", "100.0%"]  # explicit: every sample censored
    assert row_05.split()[4:7] == ["0.0%", "/", "0.0%"]  # clean: nothing censored
    row_095 = next(line for line in out.splitlines() if line.strip().startswith("0.95"))
    assert row_095.split()[1] == "0.0%"  # the fake detector scores 0.9
    assert "explicit files with nothing at or above 0.50: 0" in out


def test_changed_settings_need_a_new_results_file(tmp_path, dataset, fake_model, capsys):  # noqa: F811
    results = tmp_path / "results.jsonl"
    assert cli.main(["eval", "run", str(dataset / "porn"), "-o", str(results), "-q"]) == 0
    with pytest.raises(SystemExit) as exc:
        cli.main(["eval", "run", str(dataset / "porn"), "-o", str(results), "-q",
                  "--channels", "bgr"])
    assert exc.value.code == 2
    assert "different settings" in capsys.readouterr().err


def test_broken_video_is_recorded_and_skipped_in_summary(tmp_path, fake_model, capsys):  # noqa: F811
    (tmp_path / "porn").mkdir()
    (tmp_path / "porn" / "broken.mp4").write_bytes(b"not a video")
    results = tmp_path / "r.jsonl"
    assert cli.main(["eval", "run", str(tmp_path / "porn"), "-o", str(results), "-q"]) == 1
    assert "error" in json.loads(results.read_text())
    assert cli.main(["eval", "summary", str(results)]) == 0
    assert "videos that failed to decode: 1" in capsys.readouterr().out


def test_summary_text_lists_false_alarms():
    videos = [
        VideoScore("p1.mp4", 1, None, 10, [(1.0, 0.8, "ANUS_EXPOSED")]),
        VideoScore("n1.mp4", 0, "difficult", 10, [(42.0, 0.6, "FEMALE_GENITALIA_EXPOSED")]),
        VideoScore("n2.mp4", 0, "easy", 10, []),
    ]
    text = evaluate.summarize(videos, [0.25, 0.5])
    assert "clean/difficult" in text and "clean/easy" in text
    assert "0.60  FEMALE_GENITALIA_EXPOSED" in text and "42.0s  n1.mp4" in text


def test_image_folders_and_repository_name(tmp_path, fake_model, capsys):  # noqa: F811
    """Images count as one sample each; the repo name containing 'Porn' does not leak into labels."""
    root = tmp_path / "Deep-Learning-Porn-Detection" / "less_images"
    for n in range(3):
        img = background(120, 160, seed=n)
        paint_object(img, 30 + 10 * n, 20, 40)
        write_image(root / "porn" / f"p{n}.jpg", img)
    for n in range(4):
        write_image(root / "not_porn" / "bikini" / f"c{n}.png", background(120, 160, seed=10 + n))
    results = tmp_path / "images.jsonl"
    assert cli.main(["eval", "run", str(root), "-o", str(results), "-q"]) == 0
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert sorted(r["label"] for r in records) == [0, 0, 0, 0, 1, 1, 1]
    assert all(r["samples"] == 1 for r in records)
    assert cli.main(["eval", "summary", str(results)]) == 0
    out = capsys.readouterr().out
    assert "files: 3 explicit, 4 clean" in out and "(best score per file): 1.000" in out


def _labelled_folder(root: Path) -> Path:
    """Two images marked correctly, one marked where nothing is detected, one clean."""
    for n, (x, label) in enumerate([(30, [30, 20, 70, 60]), (60, [60, 20, 100, 60]),
                                    (90, [10, 70, 40, 110])]):
        img = background(120, 160, seed=n)
        paint_object(img, x, 20, 40)
        write_image(root / f"img{n}.jpg", img)
        (root / f"img{n}.json").write_text(json.dumps({"regions": [label]}))
    write_image(root / "clean.png", background(120, 160, seed=9))
    (root / "clean.json").write_text(json.dumps({"regions": []}))
    write_image(root / "unlabelled.png", background(120, 160, seed=8))
    return root


def test_eval_regions_reports_misses(tmp_path, fake_model, capsys):  # noqa: F811
    root = _labelled_folder(tmp_path / "labelled")
    assert cli.main(["eval", "regions", str(root)]) == 1
    out = capsys.readouterr().out
    assert "labelled images: 4 (3 with regions, 1 clean)" in out
    assert "regions covered >= 90%: 2 / 3" in out
    assert "  0.0%  [10, 70, 40, 110]  img2.jpg  (FEMALE_GENITALIA_EXPOSED 0.90)" in out
    assert "clean images censored on more than 2% of the picture: 0 / 1" in out


def test_eval_regions_passes_when_everything_is_covered(tmp_path, fake_model, capsys):  # noqa: F811
    root = _labelled_folder(tmp_path / "labelled")
    (root / "img2.json").write_text(json.dumps({"regions": [[90, 20, 130, 60]]}))
    assert cli.main(["eval", "regions", str(root)]) == 0
    assert "regions covered >= 90%: 3 / 3" in capsys.readouterr().out


def test_eval_regions_needs_labels(tmp_path, fake_model, capsys):  # noqa: F811
    write_image(tmp_path / "x.png", background(40, 40))
    with pytest.raises(SystemExit):
        cli.main(["eval", "regions", str(tmp_path)])
    assert "avmosaic label" in capsys.readouterr().err


def test_frames_command(tmp_path):
    make_video(tmp_path / "clip.mp4", frames=90, audio=False)  # 3 s at 30 fps
    assert cli.main(["frames", str(tmp_path / "clip.mp4"), "-o", str(tmp_path / "f"),
                     "--every", "1"]) == 0
    names = sorted(p.name for p in (tmp_path / "f").iterdir())
    assert names == ["clip_00000000ms.jpg", "clip_00001000ms.jpg", "clip_00002000ms.jpg"]

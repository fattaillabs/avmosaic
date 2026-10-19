"""Tests that run the real NudeNet model.

Faces are a class of the same detector, so a public-domain portrait (NASA's
astronaut photo, ``tests/data/astronaut.jpg``) exercises the whole real
detection path without any explicit material. A short clip made from a CC0 photo
of Michelangelo's David (``tests/data/david_statue.mp4``) checks the real model
and tracking on the region the tool is for.
"""

import json

import numpy as np
import pytest

from avmosaic import cli
from avmosaic.censor import build_mask, expand_box
from avmosaic.detector import NudeNetDetector, create_detector, resolve_providers
from avmosaic.image import read_image
from avmosaic.labels import LABELS, parse_classes
from avmosaic.models import MODELS, bundled_path, find_model, sha256_of
from avmosaic.tracking import TrackingPlanner, TrackOptions
from avmosaic.video import iter_frames, probe

from .conftest import DATA, background, flat_tiles

pytestmark = pytest.mark.model

ASTRONAUT = DATA / "astronaut.jpg"
FACE = (172, 82, 276, 181)  # where the face is in astronaut.jpg


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


@pytest.fixture(scope="module")
def model_path():
    return bundled_path(MODELS["320n"])


def test_bundled_model_is_the_registered_file(model_path):
    assert model_path is not None
    assert sha256_of(model_path) == MODELS["320n"].sha256


@pytest.mark.parametrize("seed", [None, 1, 2])
def test_matches_nudenet_reference_in_bgr_mode(model_path, seed):
    from nudenet import NudeDetector

    image = read_image(ASTRONAUT) if seed is None else background(300, 400, seed=seed)
    reference = NudeDetector().detect(image.copy())
    ours = NudeNetDetector(model_path, channels="bgr").detect(image)
    assert len(ours) == len(reference)
    for ref in sorted(reference, key=lambda d: -d["score"]):
        x, y, w, h = ref["box"]
        match = max(ours, key=lambda d: iou((d.x1, d.y1, d.x2, d.y2), (x, y, x + w, y + h)))
        assert match.label == ref["class"]
        assert match.score == pytest.approx(ref["score"], abs=1e-4)
        assert iou((match.x1, match.y1, match.x2, match.y2), (x, y, x + w, y + h)) > 0.95


@pytest.mark.parametrize("channels", ["rgb", "bgr"])
def test_finds_the_face(model_path, channels):
    det = NudeNetDetector(model_path, classes=["FACE_FEMALE", "FACE_MALE"], channels=channels)
    (face,) = det.detect(read_image(ASTRONAUT))
    assert face.label == "FACE_FEMALE" and face.score > 0.6
    assert iou((face.x1, face.y1, face.x2, face.y2), FACE) > 0.8


def test_no_genitals_on_a_clothed_portrait_or_texture(model_path):
    det = NudeNetDetector(model_path, classes=["FEMALE_GENITALIA_EXPOSED",
                                               "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"])
    assert det.detect(read_image(ASTRONAUT)) == []
    assert det.detect(background(480, 640)) == []


def test_requested_classes_are_scored_on_their_own(model_path):
    """An anchor that says 'covered 0.6, exposed 0.4' must still count as exposed."""
    covered = LABELS.index("FEMALE_GENITALIA_COVERED")
    exposed = LABELS.index("FEMALE_GENITALIA_EXPOSED")
    output = np.zeros((4 + len(LABELS), 10), np.float32)
    output[:4, 3] = (100, 100, 40, 40)  # cx, cy, w, h in model pixels
    output[4 + covered, 3] = 0.6
    output[4 + exposed, 3] = 0.4

    everything = NudeNetDetector(model_path)
    (d,) = everything.postprocess(output, 1.0, 320, 320)
    assert d.label == "FEMALE_GENITALIA_COVERED"

    genitals = NudeNetDetector(model_path, classes=["FEMALE_GENITALIA_EXPOSED"])
    (d,) = genitals.postprocess(output, 1.0, 320, 320)
    assert d.label == "FEMALE_GENITALIA_EXPOSED" and d.score == pytest.approx(0.4)
    assert (d.x1, d.y1, d.x2, d.y2) == (80, 80, 120, 120)

    strict = NudeNetDetector(model_path, classes=["FEMALE_GENITALIA_EXPOSED"], threshold=0.5)
    assert strict.postprocess(output, 1.0, 320, 320) == []


def test_boxes_scale_back_and_clip_to_the_image(model_path):
    output = np.zeros((4 + len(LABELS), 1), np.float32)
    output[:4, 0] = (310, 50, 40, 20)  # sticks out of the right edge
    output[4 + LABELS.index("ANUS_EXPOSED"), 0] = 0.9
    det = NudeNetDetector(model_path, classes=["ANUS_EXPOSED"])
    (d,) = det.postprocess(output, 2.0, 640, 400)  # scale 2: a 640x640 letterboxed frame
    assert (d.x1, d.y1, d.x2, d.y2) == (580, 80, 640, 120)


def test_cli_end_to_end_face_mosaic(tmp_path):
    out = tmp_path / "astronaut_out.png"
    assert cli.main([str(ASTRONAUT), "-o", str(out), "--classes", "faces", "-q"]) == 0
    src, dst = read_image(ASTRONAUT), read_image(out)
    assert flat_tiles(src, FACE, block=6) < 0.05
    assert flat_tiles(dst, FACE, block=6) == 1.0  # auto block: max(4, ceil(512 / 100))
    assert (dst[300:] == src[300:]).all()


def test_providers():
    assert resolve_providers("cpu") == ["CPUExecutionProvider"]
    assert resolve_providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert resolve_providers("auto", ["CUDAExecutionProvider", "CPUExecutionProvider"])[0] \
        == "CUDAExecutionProvider"
    assert resolve_providers("auto", ["CoreMLExecutionProvider", "CPUExecutionProvider"]) \
        == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(ValueError, match="onnxruntime-gpu"):
        resolve_providers("cuda", ["CPUExecutionProvider"])
    with pytest.raises(ValueError):
        resolve_providers("tpu")


def test_invalid_detector_settings(model_path):
    for kwargs in ({"threshold": 0}, {"nms_iou": 1.5}, {"resolution": 300},
                   {"channels": "bgra"}, {"classes": ["NOPE"]}):
        with pytest.raises(ValueError):
            NudeNetDetector(model_path, **kwargs)


def test_unknown_model_name():
    with pytest.raises(FileNotFoundError):
        create_detector("does-not-exist")


@pytest.mark.skipif(find_model(MODELS["640m"]) is None,
                    reason="640m not downloaded (avmosaic models download 640m)")
def test_640m_finds_the_face():
    det = create_detector("640m", classes=["FACE_FEMALE"], allow_download=False)
    assert det.resolution == 640
    (face,) = det.detect(read_image(ASTRONAUT))
    assert iou((face.x1, face.y1, face.x2, face.y2), FACE) > 0.8


def test_statue_clip_is_covered_in_every_frame(model_path):
    """A generated camera move over a CC0 photo of David: the real model plus tracking
    keeps the hand-marked region covered in every frame and censors nothing else.

    Marble is outside what NudeNet was trained on; it reports the statue's groin mostly
    as FEMALE_GENITALIA_COVERED, so that class is added to the default preset here.
    """
    clip = DATA / "david_statue.mp4"
    truth = json.loads((DATA / "david_statue.json").read_text())["boxes"]
    info = probe(clip)
    detector = NudeNetDetector(model_path, classes=parse_classes("genitals,FEMALE_GENITALIA_COVERED"))
    planner = TrackingPlanner(1, TrackOptions())
    planned = {}
    for i, frame in enumerate(iter_frames(clip, info)):
        for index, _, regions in planner.push(i, frame, lambda _i, image: detector.detect(image)):
            planned[index] = regions
    for index, _, regions in planner.flush():
        planned[index] = regions
    assert len(planned) == len(truth) == 45

    w, h = info.width, info.height
    coverage, stray = [], []
    for i, box in enumerate(truth):
        x1, y1, x2, y2 = (int(v) for v in box)
        mask = build_mask(h, w, [expand_box(r, 0.15, w, h) for r in planned[i]]) > 0
        coverage.append(float(mask[y1:y2, x1:x2].mean()))
        near = np.zeros_like(mask)
        near[max(0, y1 - 40):y2 + 40, max(0, x1 - 40):x2 + 40] = True
        stray.append(float((mask & ~near).mean()))
    assert min(coverage) >= 0.8, coverage
    assert np.mean(coverage) >= 0.9
    assert max(stray) <= 0.01, stray

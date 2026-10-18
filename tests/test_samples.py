"""Accuracy check on your own (private) images. Skipped unless AVMOSAIC_SAMPLES is set.

Mark the regions that must be censored with ``avmosaic label FOLDER`` (or write
``<name>.json`` = ``{"regions": [[x1, y1, x2, y2], ...]}`` by hand; ``[]`` = must
stay clean), then:

    AVMOSAIC_SAMPLES=~/private/samples pytest tests/test_samples.py -v
    AVMOSAIC_SAMPLES_ARGS="--model 640m --threshold 0.2" ...   # same flags as the CLI
    AVMOSAIC_MIN_COVERAGE=0.95 ...                              # default 0.9

``avmosaic eval regions FOLDER`` prints the same check as a summary.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from avmosaic import cli, evaluate
from avmosaic.censor import CensorOptions

SAMPLES = os.environ.get("AVMOSAIC_SAMPLES")
MIN_COVERAGE = float(os.environ.get("AVMOSAIC_MIN_COVERAGE", "0.9"))
MAX_FALSE_AREA = float(os.environ.get("AVMOSAIC_MAX_FALSE_AREA", "0.02"))

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not SAMPLES, reason="set AVMOSAIC_SAMPLES to a folder of labelled images"),
]


def _cases():
    return evaluate.labelled_images(Path(SAMPLES).expanduser()) if SAMPLES else []


@pytest.fixture(scope="module")
def setup():
    argv = ["censor", "unused", *shlex.split(os.environ.get("AVMOSAIC_SAMPLES_ARGS", ""))]
    args = cli.build_parser().parse_args(argv)
    detector = cli._make_detector(args)
    options = CensorOptions(block_size=args.block_size, shape=args.shape, padding=args.padding)
    return detector, options


@pytest.mark.parametrize("image_path,regions", _cases() or [pytest.param(None, None,
                         marks=pytest.mark.skip(reason="no labelled images found"))],
                         ids=lambda p: p.name if isinstance(p, Path) else "")
def test_marked_regions_are_censored(setup, image_path, regions):
    detector, options = setup
    check = evaluate.check_regions(image_path, regions, detector, options)
    found = ", ".join(f"{d.label} {d.score:.2f}" for d in check.detections) or "nothing"
    misses = [f"{[round(v) for v in box]} covered {cov:.0%}"
              for box, cov in zip(regions, check.coverage, strict=True) if cov < MIN_COVERAGE]
    assert not misses, f"under-censored: {'; '.join(misses)} (detected: {found})"
    if not regions:
        assert check.false_area <= MAX_FALSE_AREA, \
            f"censored {check.false_area:.1%} of a clean image ({found})"

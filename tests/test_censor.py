import numpy as np
import pytest

from avmosaic.censor import (
    CensorOptions,
    apply_censor,
    build_mask,
    expand_box,
    parse_block_size,
    pixelate,
    resolve_block_size,
)
from avmosaic.detector import Detection

from .conftest import background, detail


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [(320, 240, 4), (1920, 1080, 20), (3840, 2160, 39), (1080, 1920, 20), (100, 50, 4)],
)
def test_auto_block_size_is_one_percent_of_long_side_min_4(w, h, expected):
    assert resolve_block_size("auto", w, h) == expected


def test_block_size_spec_forms():
    assert parse_block_size("12") == 12
    assert parse_block_size(8) == 8
    assert parse_block_size("AUTO") == "auto"
    assert parse_block_size("2.5%") == "2.5%"
    assert resolve_block_size("2.5%", 1920, 1080) == 48
    for bad in ("0", "-3", "abc", "0%", "150%", "1.5"):
        with pytest.raises(ValueError):
            parse_block_size(bad)


def test_expand_box_pads_and_clamps():
    assert expand_box((10, 10, 30, 50), 0.1, 100, 100) == (8, 6, 32, 54)
    assert expand_box((0, 0, 100, 100), 0.5, 120, 110) == (0, 0, 120, 110)
    det = Detection("ANUS_EXPOSED", 0.5, 10.4, 10.6, 20.2, 20.9)
    assert expand_box(det, 0, 50, 50) == (10, 10, 21, 21)  # never shrinks


def test_ellipse_mask_covers_the_whole_box():
    mask = build_mask(200, 200, [(50, 60, 150, 120)], shape="ellipse")
    assert mask[60:120, 50:150].all()  # corners included
    assert not mask[0, 0] and not mask[199, 199]


def test_pixelate_makes_uniform_grid_aligned_tiles():
    img = background(97, 131)
    mask = build_mask(97, 131, [(13, 21, 77, 66)])
    out = img.copy()
    pixelate(out, mask, 8)
    # Every 8x8 tile touching the box is flat; tiles start on multiples of 8.
    for ty in range(16, 72, 8):
        for tx in range(8, 80, 8):
            tile = out[ty:ty + 8, tx:tx + 8]
            assert (tile == tile[0, 0]).all(), (tx, ty)
    # Outside the touched tiles nothing changes.
    assert (out[:16] == img[:16]).all()
    assert (out[72:] == img[72:]).all()
    assert (out[:, :8] == img[:, :8]).all()
    assert (out[:, 80:] == img[:, 80:]).all()


def test_tile_colour_is_the_block_mean():
    img = np.zeros((16, 16, 3), np.uint8)
    img[:8, :8] = (0, 0, 200)
    img[:4, :8] = (0, 0, 100)  # top half of the first tile darker
    mask = np.full((16, 16), 255, np.uint8)
    pixelate(img, mask, 8)
    assert tuple(img[0, 0]) == (0, 0, 150)


def test_overlapping_regions_give_same_result_as_their_union():
    img = background(120, 160)
    a, b = img.copy(), img.copy()
    pixelate(a, build_mask(120, 160, [(10, 10, 70, 70), (50, 40, 110, 100)]), 6)
    pixelate(b, build_mask(120, 160, [(10, 10, 70, 70)]), 6)
    pixelate(b, build_mask(120, 160, [(50, 40, 110, 100)]), 6)
    assert (a == b).all()


def test_pixelate_handles_image_edges():
    img = background(50, 50)
    out = img.copy()
    pixelate(out, build_mask(50, 50, [(40, 40, 50, 50)]), 16)
    # Tiles 32..47 and 48..63 (cut to 48..49 by the edge) are each flat.
    for ys in (slice(32, 48), slice(48, 50)):
        for xs in (slice(32, 48), slice(48, 50)):
            tile = out[ys, xs]
            assert (tile == tile[0, 0]).all()
    assert (out[:32] == img[:32]).all()
    assert (out[:, :32] == img[:, :32]).all()


@pytest.mark.parametrize("method", ["mosaic", "blur", "fill"])
def test_methods_remove_detail_only_inside(method):
    img = background(200, 300)
    det = Detection("FEMALE_GENITALIA_EXPOSED", 0.9, 100, 60, 180, 140)
    out = img.copy()
    mask = apply_censor(out, [det], CensorOptions(method=method, block_size=10, padding=0))
    assert detail(out, (100, 60, 180, 140)) < detail(img, (100, 60, 180, 140)) * 0.05
    assert (out[mask == 0] == img[mask == 0]).all()


def test_fill_colour_is_bgr():
    img = background(60, 60)
    apply_censor(img, [(10, 10, 30, 30)],
                 CensorOptions(method="fill", padding=0, fill_color=(1, 2, 3)))
    assert tuple(img[20, 20]) == (1, 2, 3)


def test_no_detections_is_a_no_op():
    img = background(40, 40)
    out = img.copy()
    mask = apply_censor(out, [], CensorOptions())
    assert not mask.any() and (out == img).all()


def test_gray_and_bgra_images():
    gray = background(64, 64)[..., 0].copy()
    apply_censor(gray, [(8, 8, 40, 40)], CensorOptions(block_size=8, padding=0))
    assert (gray[8:40, 8:40][:8, :8] == gray[8, 8]).all()

    bgra = np.dstack([background(64, 64), np.full((64, 64), 77, np.uint8)])
    apply_censor(bgra, [(8, 8, 40, 40)], CensorOptions(block_size=8, padding=0))
    assert (bgra[..., 3] == 77).all()  # alpha preserved by the mosaic
    assert (bgra[8:16, 8:16] == bgra[8, 8]).all()


def test_invalid_options():
    with pytest.raises(ValueError):
        CensorOptions(method="smudge")
    with pytest.raises(ValueError):
        CensorOptions(shape="star")
    with pytest.raises(ValueError):
        CensorOptions(padding=-0.1)

import numpy as np

from avmosaic.censor import CensorOptions
from avmosaic.image import censor_image, read_image, write_image

from .conftest import background, detail, paint_object


def test_unicode_paths_roundtrip(tmp_path):
    img = background(40, 50)
    for name in ("모자이크.png", "テスト画像.png", "space name.png"):
        path = tmp_path / "하위 폴더" / name
        write_image(path, img)
        assert (read_image(path) == img).all()


def test_alpha_is_kept_for_png_and_dropped_for_jpeg(tmp_path):
    bgra = np.dstack([background(32, 32), np.full((32, 32), 128, np.uint8)])
    write_image(tmp_path / "a.png", bgra)
    assert read_image(tmp_path / "a.png").shape == (32, 32, 4)
    write_image(tmp_path / "a.jpg", bgra)
    assert read_image(tmp_path / "a.jpg").shape == (32, 32, 3)


def test_censor_image_end_to_end(tmp_path, color_detector):
    img = background(240, 320)
    paint_object(img, 100, 80, 60)
    src, dst = tmp_path / "in.png", tmp_path / "out" / "in.png"
    write_image(src, img)

    dets = censor_image(src, dst, color_detector, CensorOptions(block_size=12))

    assert len(dets) == 1
    out = read_image(dst)
    box = (100, 80, 160, 140)
    assert detail(out, box) < detail(img, box) * 0.05
    assert (out[:60] == img[:60]).all()  # far from the object: untouched


def test_draw_boxes_marks_output(tmp_path, color_detector):
    img = background(120, 160)
    paint_object(img, 40, 30, 40)
    write_image(tmp_path / "in.png", img)
    plain, boxed = tmp_path / "p.png", tmp_path / "b.png"
    censor_image(tmp_path / "in.png", plain, color_detector, CensorOptions())
    censor_image(tmp_path / "in.png", boxed, color_detector, CensorOptions(), draw_boxes=True)
    assert (read_image(plain) != read_image(boxed)).any()

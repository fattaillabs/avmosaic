import json
import threading
import urllib.error
import urllib.request

import pytest

from avmosaic.image import read_image, write_image
from avmosaic.labeler import LabelStore, make_server

from .conftest import background


@pytest.fixture
def folder(tmp_path):
    write_image(tmp_path / "a" / "one.jpg", background(60, 80))
    write_image(tmp_path / "two.png", background(50, 70, seed=1))
    write_image(tmp_path / "three.tif", background(40, 40, seed=2))
    (tmp_path / "notes.txt").write_text("ignored")
    return tmp_path


def serve(store):
    server = make_server(store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(base + path, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            kind = resp.headers["Content-Type"]
        return json.loads(raw) if kind == "application/json" else (raw, kind)

    return server, call


def test_listing_saving_and_images(folder):
    server, call = serve(LabelStore(folder))
    try:
        assert server.server_address[0] == "127.0.0.1"
        page, kind = call("/")
        assert kind.startswith("text/html") and b"avmosaic labels" in page

        listing = call("/api/list")
        assert [x["name"] for x in listing] == ["a/one.jpg", "three.tif", "two.png"]
        assert not any(x["saved"] for x in listing)
        assert call("/api/item/0") == {"index": 0, "name": "a/one.jpg", "regions": [],
                                       "saved": False, "suggested": False}

        assert call("/api/item/0", {"regions": [[10.4, 20, 50.6, 40]]}) == {"ok": True}
        assert json.loads((folder / "a" / "one.json").read_text()) == {"regions": [[10, 20, 51, 40]]}
        item = call("/api/item/0")
        assert item["saved"] and item["regions"] == [[10, 20, 51, 40]]

        call("/api/item/2", {"regions": []})  # marked clean
        assert json.loads((folder / "two.json").read_text()) == {"regions": []}
        assert call("/api/item/2")["saved"]

        raw, kind = call("/image/0")
        assert kind == "image/jpeg" and raw == (folder / "a" / "one.jpg").read_bytes()
        raw, kind = call("/image/1")  # TIFF is converted for the browser
        assert kind == "image/png" and raw[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("path", "data", "code"),
    [
        ("/api/item/0", {"regions": [[5, 5, 5, 9]]}, 400),  # empty box
        ("/api/item/0", {"regions": [[1, 2, 3]]}, 400),
        ("/api/item/0", {"boxes": []}, 400),
        ("/api/item/99", {"regions": []}, 400),
        ("/api/item/99", None, 404),
        ("/image/../../etc/passwd", None, 404),
        ("/image/-1", None, 404),
    ],
)
def test_bad_requests_are_refused(folder, path, data, code):
    server, call = serve(LabelStore(folder))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            call(path, data)
        assert exc.value.code == code
        assert not (folder / "a" / "one.json").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_suggestions_until_saved(folder):
    store = LabelStore(folder, suggest=lambda image: [[1.2, 2.0, 30.0, 40.7]])
    item = store.item(0)
    assert item["suggested"] and not item["saved"] and item["regions"] == [[1, 2, 30, 41]]
    store.save(0, item["regions"])
    item = store.item(0)
    assert item["saved"] and not item["suggested"]


def test_label_files_sit_next_to_images(folder):
    store = LabelStore(folder)
    store.save(1, [[0, 0, 10, 10]])
    assert (folder / "three.json").is_file()
    assert read_image(folder / "three.tif").shape == (40, 40, 3)  # image itself untouched

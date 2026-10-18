"""A small local web page for marking the regions that must end up censored.

``avmosaic label DIR`` serves the images in DIR on 127.0.0.1 only. Boxes are
saved next to each image as ``<name>.json`` = ``{"regions": [[x1, y1, x2, y2], ...]}``,
the format ``avmosaic eval regions`` and ``tests/test_samples.py`` read.
``{"regions": []}`` marks an image that must stay clean.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .image import IMAGE_EXTS, read_image

BROWSER_FORMATS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                   ".webp": "image/webp", ".bmp": "image/bmp"}
SuggestFn = Callable[[Path], list[list[float]]]


def label_path(image: Path) -> Path:
    return image.with_suffix(".json")


def read_regions(image: Path) -> list[list[float]] | None:
    path = label_path(image)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["regions"]


class LabelStore:
    def __init__(self, root: Path, suggest: SuggestFn | None = None):
        self.root = root
        self.items = sorted(p for p in root.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        self.suggest = suggest

    def listing(self) -> list[dict]:
        return [{"index": i, "name": p.relative_to(self.root).as_posix(),
                 "saved": label_path(p).is_file()} for i, p in enumerate(self.items)]

    def item(self, index: int) -> dict:
        image = self.items[index]
        regions = read_regions(image)
        suggested = regions is None and self.suggest is not None
        if suggested:
            regions = [[round(v) for v in box] for box in self.suggest(image)]
        return {"index": index, "name": image.relative_to(self.root).as_posix(),
                "regions": regions or [], "saved": not suggested and regions is not None,
                "suggested": suggested}

    def save(self, index: int, regions: list) -> None:
        clean = []
        for box in regions:
            if not (isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)):
                raise ValueError(f"bad box: {box!r}")
            x1, y1, x2, y2 = (round(float(v)) for v in box)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"empty box: {box!r}")
            clean.append([x1, y1, x2, y2])
        label_path(self.items[index]).write_text(json.dumps({"regions": clean}) + "\n",
                                                 encoding="utf-8")

    def image_bytes(self, index: int) -> tuple[bytes, str]:
        image = self.items[index]
        mime = BROWSER_FORMATS.get(image.suffix.lower())
        if mime:
            return image.read_bytes(), mime
        ok, buf = cv2.imencode(".png", read_image(image))  # e.g. TIFF: browsers can't show it
        if not ok:
            raise ValueError(f"cannot convert {image}")
        return buf.tobytes(), "image/png"


def make_handler(store: LabelStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _index(self, pattern: str) -> int | None:
            match = re.fullmatch(pattern, self.path.split("?")[0])
            if not match:
                return None
            index = int(match.group(1))
            return index if 0 <= index < len(store.items) else None

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/list":
                self._json(store.listing())
            elif (i := self._index(r"/api/item/(\d+)")) is not None:
                self._json(store.item(i))
            elif (i := self._index(r"/image/(\d+)")) is not None:
                body, mime = store.image_bytes(i)
                self._send(HTTPStatus.OK, body, mime)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            i = self._index(r"/api/item/(\d+)")
            length = int(self.headers.get("Content-Length") or 0)
            if i is None or not 0 < length <= 1 << 20:
                self._json({"error": "bad request"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                store.save(i, json.loads(self.rfile.read(length))["regions"])
            except (ValueError, KeyError, TypeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True})

    return Handler


def make_server(store: LabelStore, port: int = 0) -> ThreadingHTTPServer:
    """Bound to 127.0.0.1 only: the images never leave this computer."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>avmosaic labels</title>
<style>
:root { --bg:#141416; --panel:#1d1d21; --line:#33333a; --fg:#ececf0; --muted:#8c8c96;
        --box:#ff3b6b; --sel:#ffd400; --ok:#3cc98a; --warn:#e6a700; }
* { box-sizing:border-box }
body { margin:0; height:100vh; display:flex; flex-direction:column; background:var(--bg);
       color:var(--fg); font:14px/1.4 system-ui,-apple-system,sans-serif }
header { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:8px 12px;
         background:var(--panel); border-bottom:1px solid var(--line) }
#name { flex:1; min-width:160px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap }
button { background:#2a2a30; color:var(--fg); border:1px solid #44444c; border-radius:6px;
         padding:6px 10px; font:inherit; cursor:pointer }
button:hover { border-color:#6a6a74 }
button.primary { background:var(--box); border-color:var(--box); color:#fff }
.pill { padding:2px 8px; border-radius:10px; border:1px solid var(--line); font-size:12px }
#wrap { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; padding:8px }
canvas { max-width:100%; max-height:100%; cursor:crosshair; touch-action:none }
footer { padding:6px 12px; color:var(--muted); font-size:12px; border-top:1px solid var(--line) }
</style></head>
<body>
<header>
  <button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
  <span id="pos" class="pill"></span><span id="name"></span><span id="state" class="pill"></span>
  <button id="clean" title="C">No regions (clean)</button>
  <button id="clear">Clear boxes</button>
  <button id="save" class="primary" title="S">Save</button>
</header>
<div id="wrap"><canvas id="cv" width="1" height="1"></canvas></div>
<footer>Drag a box around every region that must be censored. Click a box to select it,
Delete removes it. &larr;/&rarr; moves to another image and saves changes. S saves, C marks the
image clean.</footer>
<script>
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
let items = [], i = 0, img = new Image(), boxes = [], sel = -1, dirty = false, saved = false,
    suggested = false, drag = null;

async function api(path, opt) {
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function setText(id, text) { document.getElementById(id).textContent = text; }
function status() {
  const s = document.getElementById("state");
  s.textContent = dirty ? "unsaved" : suggested ? "detector suggestion — check it"
    : saved ? (boxes.length ? boxes.length + " region(s) saved" : "saved as clean") : "not labelled";
  s.style.borderColor = dirty || suggested ? "var(--warn)" : saved ? "var(--ok)" : "var(--line)";
}
function draw() {
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(cv.width, cv.height) / 300));
  boxes.forEach((b, k) => {
    const color = k === sel ? "255,212,0" : "255,59,107";
    ctx.lineWidth = lw;
    ctx.strokeStyle = "rgb(" + color + ")";
    ctx.fillStyle = "rgba(" + color + ",0.15)";
    ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
  });
  if (drag && drag.box) {
    const b = drag.box;
    ctx.setLineDash([6, 4]); ctx.strokeStyle = "#fff"; ctx.lineWidth = lw;
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); ctx.setLineDash([]);
  }
}
async function load(n) {
  i = (n + items.length) % items.length;
  const it = await api("/api/item/" + i);
  boxes = it.regions.map(b => b.slice()); saved = it.saved; suggested = it.suggested;
  dirty = false; sel = -1;
  setText("name", it.name); setText("pos", (i + 1) + " / " + items.length);
  await new Promise((ok, err) => { img.onload = ok; img.onerror = err; img.src = "/image/" + i; });
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  draw(); status();
}
async function save() {
  await api("/api/item/" + i, { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ regions: boxes }) });
  saved = true; dirty = false; suggested = false; status();
  items[i].saved = true;
}
async function go(n) { if (dirty) await save(); await load(n); }
function point(e) {
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * cv.width / r.width, y = (e.clientY - r.top) * cv.height / r.height;
  return [Math.min(Math.max(x, 0), cv.width), Math.min(Math.max(y, 0), cv.height)];
}
function hit(p) {
  for (let k = boxes.length - 1; k >= 0; k--) {
    const b = boxes[k];
    if (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]) return k;
  }
  return -1;
}
cv.addEventListener("pointerdown", e => {
  cv.setPointerCapture(e.pointerId);
  drag = { start: point(e), box: null };
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const p = point(e), s = drag.start;
  drag.box = [Math.min(s[0], p[0]), Math.min(s[1], p[1]),
              Math.max(s[0], p[0]), Math.max(s[1], p[1])].map(Math.round);
  draw();
});
cv.addEventListener("pointerup", () => {
  const d = drag; drag = null;
  if (!d) return;
  const b = d.box;
  if (b && b[2] - b[0] >= 4 && b[3] - b[1] >= 4) { boxes.push(b); sel = boxes.length - 1; dirty = true; }
  else sel = hit(d.start);
  draw(); status();
});
document.getElementById("prev").onclick = () => go(i - 1);
document.getElementById("next").onclick = () => go(i + 1);
document.getElementById("save").onclick = () => save();
document.getElementById("clear").onclick = () => { boxes = []; sel = -1; dirty = true; draw(); status(); };
document.getElementById("clean").onclick = async () => { boxes = []; sel = -1; await save(); draw(); };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") go(i + 1);
  else if (e.key === "ArrowLeft") go(i - 1);
  else if ((e.key === "Delete" || e.key === "Backspace") && sel >= 0) {
    boxes.splice(sel, 1); sel = -1; dirty = true; draw(); status(); e.preventDefault();
  } else if (e.key === "s") save();
  else if (e.key === "c") document.getElementById("clean").onclick();
});
(async () => {
  items = await api("/api/list");
  if (!items.length) { setText("name", "No images in this folder."); return; }
  const first = items.findIndex(x => !x.saved);
  await load(first < 0 ? 0 : first);
})();
</script>
</body></html>
"""

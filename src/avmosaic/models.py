"""Model registry, cache location and verified downloads."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    resolution: int
    sha256: str
    size: int
    urls: tuple[str, ...]
    description: str
    bundled_with_nudenet: bool = False


_RELEASE = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights"
_ASSETS = "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets"

MODELS: dict[str, ModelSpec] = {
    "320n": ModelSpec(
        name="320n",
        resolution=320,
        sha256="c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f",
        size=12_150_158,
        # The API asset URL serves anonymous downloads; the plain release URL may
        # redirect to a login page, so it is only a fallback.
        urls=(f"{_ASSETS}/176831997", f"{_RELEASE}/320n.onnx"),
        description="NudeNet v3.4 nano, 320px. Fast; ships with the nudenet package.",
        bundled_with_nudenet=True,
    ),
    "640m": ModelSpec(
        name="640m",
        resolution=640,
        sha256="04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb",
        size=103_538_690,
        urls=(f"{_ASSETS}/176832019", f"{_RELEASE}/640m.onnx"),
        description="NudeNet v3.4 medium, 640px. More accurate, much slower (~104 MB download).",
    ),
}

DEFAULT_MODEL = "320n"


def cache_dir() -> Path:
    """Where downloaded models live. Override with AVMOSAIC_HOME."""
    override = os.environ.get("AVMOSAIC_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "avmosaic"


def bundled_path(spec: ModelSpec) -> Path | None:
    if not spec.bundled_with_nudenet:
        return None
    try:
        import nudenet
    except ImportError:
        return None
    path = Path(nudenet.__file__).parent / f"{spec.name}.onnx"
    return path if path.is_file() else None


def cached_path(spec: ModelSpec) -> Path:
    return cache_dir() / f"{spec.name}.onnx"


def find_model(spec: ModelSpec) -> Path | None:
    """Return a local copy of the model if there is one, without downloading."""
    path = bundled_path(spec)
    if path is not None:
        return path
    path = cached_path(spec)
    if path.is_file() and path.stat().st_size == spec.size:
        return path
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    spec: ModelSpec,
    dest: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download ``spec`` to ``dest`` (default: the cache), verifying its SHA-256."""
    dest = dest or cached_path(spec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in spec.urls:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/octet-stream", "User-Agent": "avmosaic"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as fh:
                done = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, spec.size)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        actual = sha256_of(part)
        if actual != spec.sha256:
            part.unlink(missing_ok=True)
            errors.append(f"{url}: checksum mismatch (got {actual[:12]}…)")
            continue
        os.replace(part, dest)
        return dest
    part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download model {spec.name}:\n  " + "\n  ".join(errors))


def resolve_model(
    name_or_path: str,
    *,
    allow_download: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, int | None]:
    """Return ``(onnx_path, default_resolution)`` for a model name or file path.

    ``default_resolution`` is None for custom files; the caller picks one.
    """
    spec = MODELS.get(name_or_path)
    if spec is None:
        path = Path(name_or_path).expanduser()
        if path.is_file():
            return path, None
        known = ", ".join(MODELS)
        raise FileNotFoundError(f"model {name_or_path!r} is neither a known model ({known}) nor a file")
    path = find_model(spec)
    if path is None:
        if not allow_download:
            raise FileNotFoundError(
                f"model {spec.name} is not downloaded; run: avmosaic models download {spec.name}"
            )
        path = download_model(spec, progress=progress)
    return path, spec.resolution

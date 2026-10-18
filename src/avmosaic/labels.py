"""Detection classes and named presets.

The order of LABELS matches the class columns of the NudeNet v3 detector output,
so ``LABELS[i]`` is the class of score column ``4 + i``.
"""

from __future__ import annotations

import difflib

LABELS: tuple[str, ...] = (
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)

PRESETS: dict[str, tuple[str, ...]] = {
    # Default: what mosaic censorship (e.g. Japanese adult video) is normally about.
    "genitals": ("FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"),
    "breasts": ("FEMALE_BREAST_EXPOSED",),
    "buttocks": ("BUTTOCKS_EXPOSED",),
    "nudity": (
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
    ),
    "faces": ("FACE_FEMALE", "FACE_MALE"),
    "all": LABELS,
}

DEFAULT_CLASSES = "genitals"


def parse_classes(spec: str | None) -> tuple[str, ...]:
    """Turn ``"genitals,FEMALE_BREAST_EXPOSED"`` into a tuple of labels.

    Items may be preset names or label names, case-insensitive. The result keeps
    the canonical LABELS order and has no duplicates.
    """
    if spec is None or not spec.strip():
        spec = DEFAULT_CLASSES
    wanted: set[str] = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        key = item.lower()
        if key in PRESETS:
            wanted.update(PRESETS[key])
            continue
        label = item.upper()
        if label in LABELS:
            wanted.add(label)
            continue
        choices = list(PRESETS) + list(LABELS)
        close = difflib.get_close_matches(label, [c.upper() for c in choices], n=1)
        hint = f" Did you mean {close[0]}?" if close else ""
        raise ValueError(f"unknown class or preset: {item!r}.{hint}")
    if not wanted:
        raise ValueError("no classes selected")
    return tuple(label for label in LABELS if label in wanted)

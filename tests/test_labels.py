import pytest

from avmosaic.labels import LABELS, PRESETS, parse_classes


def test_default_is_genitals():
    assert set(parse_classes(None)) == set(PRESETS["genitals"])
    assert parse_classes("") == parse_classes(None)


def test_presets_and_labels_mix_in_canonical_order():
    got = parse_classes("female_breast_exposed, genitals")
    assert set(got) == {*PRESETS["genitals"], "FEMALE_BREAST_EXPOSED"}
    assert list(got) == sorted(got, key=LABELS.index)


def test_duplicates_collapse():
    assert parse_classes("genitals,ANUS_EXPOSED,Genitals") == parse_classes("genitals")
    assert len(parse_classes("genitals")) == 3


def test_unknown_class_suggests_close_match():
    with pytest.raises(ValueError, match="FEMALE_BREAST_EXPOSED"):
        parse_classes("FEMALE_BREST_EXPOSED")


def test_labels_match_nudenet_order():
    from nudenet import nudenet as nn

    assert LABELS == tuple(getattr(nn, "__labels"))

import pytest

from video_upscaler.models import CONTENT_TYPES, MODELS
from video_upscaler.upscale import fit_box, parse_resolution, plan_passes


@pytest.mark.parametrize(
    ("src", "box", "expected"),
    [
        ((1920, 1080), (3840, 2160), (3840, 2160)),
        ((720, 480), (3840, 2160), (3240, 2160)),  # 3:2 keeps aspect
        ((852, 480), (3840, 2160), (3834, 2160)),  # anamorphic DVD after SAR correction
        ((1920, 800), (3840, 2160), (3840, 1600)),  # scope film
        ((1280, 720), (2560, 1440), (2560, 1440)),
    ],
)
def test_fit_box(src, box, expected):
    assert fit_box(*src, box) == expected


@pytest.mark.parametrize(
    ("src", "dst", "scale", "passes"),
    [
        ((1920, 1080), (3840, 2160), 4, 1),
        ((1920, 1080), (3840, 2160), 2, 1),
        ((720, 480), (3240, 2160), 4, 1),  # 4.5x -> one 4x pass + resize
        ((720, 480), (3240, 2160), 2, 2),
        ((320, 240), (2880, 2160), 4, 2),  # 9x
        ((3840, 2160), (3840, 2160), 4, 0),
    ],
)
def test_plan_passes(src, dst, scale, passes):
    assert plan_passes(src, dst, scale) == passes


def test_parse_resolution():
    assert parse_resolution("4K") == (3840, 2160)
    assert parse_resolution("2k") == (2560, 1440)
    assert parse_resolution("1920x1080") == (1920, 1080)
    with pytest.raises(SystemExit):
        parse_resolution("huge")


def test_types_reference_known_models():
    for t in CONTENT_TYPES.values():
        assert set(t.all_models) <= MODELS.keys(), t.name

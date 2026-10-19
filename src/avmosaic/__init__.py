"""avmosaic: detect explicit regions in images and videos and cover them with a mosaic."""

__version__ = "0.1.1"

from .censor import CensorOptions, apply_censor  # noqa: E402
from .detector import Detection, NudeNetDetector, create_detector  # noqa: E402
from .image import censor_image, read_image, write_image  # noqa: E402
from .labels import LABELS, PRESETS, parse_classes  # noqa: E402
from .video import VideoOptions, censor_video  # noqa: E402

__all__ = [
    "LABELS",
    "PRESETS",
    "CensorOptions",
    "Detection",
    "NudeNetDetector",
    "VideoOptions",
    "apply_censor",
    "censor_image",
    "censor_video",
    "create_detector",
    "parse_classes",
    "read_image",
    "write_image",
]

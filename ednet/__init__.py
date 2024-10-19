__version__ = "1.0"

import os
os.environ["OMP_NUM_THREADS"] = "1"  

from ednet.models import YOLO, EDNet
from ednet.utils import SETTINGS
from ednet.utils.checks import check_yolo as checks
from ednet.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "YOLO",
    "EDNet",
    "checks",
    "download",
    "settings",
    "Explorer",
)

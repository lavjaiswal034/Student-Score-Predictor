"""
Centralized logging configuration.
All pipeline modules import `logger` from here to maintain consistent formatting.
"""

import logging
import os
from datetime import datetime

# Create a timestamped log file inside a 'logs/' directory
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(_log_dir, exist_ok=True)

_log_file = os.path.join(_log_dir, f"pipeline_{_timestamp}.log")

logging.basicConfig(
    filename=_log_file,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("score_predictor")

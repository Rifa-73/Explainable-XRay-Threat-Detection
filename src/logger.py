"""
==========================================================
Project Logger
==========================================================
"""

import logging
from pathlib import Path

from config import OUTPUT_DIR

# ---------------------------------------------------------
# Create Logs Directory
# ---------------------------------------------------------

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "project.log"

# ---------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
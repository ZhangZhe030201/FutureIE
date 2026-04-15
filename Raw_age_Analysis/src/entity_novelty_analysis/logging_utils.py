from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime


def setup_file_logger(log_dir: str | Path = "logs", log_name_prefix: str = "entity_newness_analysis") -> tuple[logging.Logger, Path]:
    """Create a file-only logger and return both logger and log path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name_prefix}_{timestamp}.log"

    logger_name = f"{log_name_prefix}_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if reused.
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path

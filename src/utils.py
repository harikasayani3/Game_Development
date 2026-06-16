from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tabulate import tabulate


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_table(data: Any, headers: Any = "keys", tablefmt: str = "github") -> str:
    """Format data into a table string using tabulate."""
    try:
        return tabulate(data, headers=headers, tablefmt=tablefmt, showindex=False)
    except Exception as error:
        logging.exception("Failed to format table: %s", error)
        return str(data)

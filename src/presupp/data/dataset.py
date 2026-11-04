"""Load the built dataset.

load_dev never returns lockbox passages, labelled or not. open_lockbox logs every opening.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from presupp import paths

LOCKBOX_LOG = paths.REPORTS_DIR / "lockbox_log.md"
_LOG_HEADER = (
    "# Lockbox log\n\n"
    "Every opening of the held-out fold. It is meant to be opened once, for the final "
    "evaluation.\n\n| when (UTC) | reason |\n|---|---|\n"
)


def _load(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    passages = pd.read_parquet(processed_dir / "passages.parquet")
    gold = pd.read_parquet(processed_dir / "gold_labels.parquet")
    return passages, gold


def _subset(passages: pd.DataFrame, gold: pd.DataFrame, split: str):
    chosen = passages[passages["split"] == split].reset_index(drop=True)
    ids = set(chosen["passage_id"])
    return chosen, gold[gold["passage_id"].isin(ids)].reset_index(drop=True)


def load_dev(processed_dir: Path = paths.PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(passages, gold labels) for the dev folds."""
    return _subset(*_load(processed_dir), "dev")


def open_lockbox(
    reason: str, processed_dir: Path = paths.PROCESSED_DIR, log_path: Path = LOCKBOX_LOG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(passages, gold labels) for the lockbox fold; logs the opening."""
    reason = " ".join(reason.split()).replace("|", "/")
    if not reason:
        raise ValueError("Give a reason for opening the lockbox")
    result = _subset(*_load(processed_dir), "lockbox")
    new_log = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="\n") as f:
        if new_log:
            f.write(_LOG_HEADER)
        f.write(f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | {reason} |\n")
    return result

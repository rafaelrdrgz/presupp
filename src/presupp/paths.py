"""Repository paths; import locations from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_V1_DIR = DATA_DIR / "raw" / "v1"
PROCESSED_DIR = DATA_DIR / "processed"  # gitignored: contains article text
REFERENCE_DIR = DATA_DIR / "reference"
ADJUDICATION_DIR = DATA_DIR / "adjudication"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = ROOT / "reports"
TAXONOMY_DIR = ROOT / "taxonomy"

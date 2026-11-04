"""Saving and loading runs.

runs/<experiment>/<run name>/ has predictions.parquet, sentences.parquet (sentence-scoring
models only) and run.json, which is written last and marks the run finished.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from presupp import paths

RUNS_DIR = paths.ROOT / "runs"
PROB_PREFIX = "p__"


def cv_run_name(round_index: int, seed: int) -> str:
    return f"round{round_index:02d}_seed{seed}"


def is_done(run_dir: Path) -> bool:
    return (run_dir / "run.json").exists()


def save_run(run_dir: Path, predictions: dict[str, tuple[list[str], np.ndarray]],
             labels: list[str], meta: dict,
             sentences: dict[str, tuple[list[str], list[np.ndarray]]] | None = None) -> None:
    """sentences: {part: (ids, one (sentences, labels[:-1]) array per passage)}."""
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for part, (ids, probs) in predictions.items():
        frame = pd.DataFrame({"passage_id": list(ids), "part": part})
        for j, label in enumerate(labels):
            frame[PROB_PREFIX + label] = np.asarray(probs)[:, j].astype(np.float32)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run_dir / "predictions.parquet", index=False)
    if sentences is not None:
        frames = []
        for part, (ids, arrays) in sentences.items():
            for pid, scores in zip(ids, arrays, strict=True):
                frame = pd.DataFrame(np.asarray(scores, dtype=np.float32),
                                     columns=[PROB_PREFIX + label for label in labels[:-1]])
                frame.insert(0, "sentence", np.arange(len(frame)))
                frame.insert(0, "part", part)
                frame.insert(0, "passage_id", pid)
                frames.append(frame)
        pd.concat(frames, ignore_index=True).to_parquet(run_dir / "sentences.parquet",
                                                        index=False)
    tmp = run_dir / "run.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    os.replace(tmp, run_dir / "run.json")


def load_meta(run_dir: Path) -> dict:
    with open(run_dir / "run.json", encoding="utf-8") as f:
        return json.load(f)


def check_same_data(run_dir: Path, fingerprint: str) -> None:
    """Refuse to reuse a run made on different data."""
    used = load_meta(run_dir).get("data_fingerprint")
    if used != fingerprint:
        raise RuntimeError(
            f"{run_dir} was made with data {used}, but the current data is {fingerprint}. "
            f"Delete {run_dir.parent} (or use another --runs-dir) and run it again."
        )


def check_same_evidence(run_dir: Path, fingerprint: str) -> None:
    """As check_same_data, for the evidence (not in data_fingerprint)."""
    used = load_meta(run_dir).get("evidence_fingerprint")
    if used != fingerprint:
        raise RuntimeError(
            f"{run_dir} was trained on evidence {used}, but the current evidence is "
            f"{fingerprint}. Delete {run_dir.parent} (or use another --runs-dir) and run it again."
        )


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    return meta, pd.read_parquet(run_dir / "predictions.parquet")


def load_sentences(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "sentences.parquet")


def finished_runs(runs_dir: Path, experiment: str, prefix: str = "round") -> list[Path]:
    root = runs_dir / experiment
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.name.startswith(prefix) and is_done(d))


def probabilities(predictions: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return predictions[[PROB_PREFIX + label for label in labels]].to_numpy(dtype=float)


def data_fingerprint(passages: pd.DataFrame, gold: pd.DataFrame) -> str:
    """Hash of ids, folds and gold labels. Fixed lineterminator, else Windows and Linux differ."""
    h = hashlib.sha1()
    h.update(passages[["passage_id", "fold", "split"]].sort_values("passage_id")
             .to_csv(index=False, lineterminator="\n").encode())
    h.update(gold[["passage_id", "category", "confidence"]].sort_values(["passage_id", "category"])
             .to_csv(index=False, lineterminator="\n").encode())
    return h.hexdigest()[:12]


def evidence_fingerprint(gold: pd.DataFrame) -> str:
    """Hash of the gold evidence, which data_fingerprint leaves out."""
    evidence = gold[["passage_id", "category"]].assign(evidence=[
        "" if ev is None else ";".join(str(int(e)) for e in ev) for ev in gold["evidence_0based"]])
    return hashlib.sha1(evidence.sort_values(["passage_id", "category"])
                        .to_csv(index=False, lineterminator="\n").encode()).hexdigest()[:12]


def git_state() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _jsonable(value):
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")

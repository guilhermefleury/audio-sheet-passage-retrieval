import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Job = Tuple[str, str]  # (piece_name, performance_name)


def _default_latest_run_manifest(processed_root: str) -> Path:
    return Path(processed_root) / "manifests" / "latest_run.json"


def discover_processed_jobs(
    processed_root: str,
    source_manifest_path: Optional[str] = None,
) -> List[Job]:
    """
    Discover available (piece, performance) jobs from processed outputs.

    Priority:
    1) source_manifest_path if provided
    2) processed_root/manifests/latest_run.json if present
    3) fallback scan of processed_root/audio/**/*.spec.npz
    """
    root = Path(processed_root)

    manifest_path: Optional[Path] = None
    if source_manifest_path is not None:
        manifest_path = Path(source_manifest_path)
    else:
        candidate = _default_latest_run_manifest(processed_root)
        if candidate.exists():
            manifest_path = candidate

    jobs: List[Job] = []

    if manifest_path is not None and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for item in payload.get("processed", []):
            piece = item.get("piece")
            performance = item.get("performance")
            if not piece or not performance:
                continue

            sheet_npz = root / "sheets" / f"{piece}.sheet.npz"
            spec_npz = root / "audio" / piece / f"{performance}.spec.npz"
            if sheet_npz.exists() and spec_npz.exists():
                jobs.append((piece, performance))
    else:
        for spec_file in sorted((root / "audio").glob("*/*.spec.npz")):
            piece = spec_file.parent.name
            performance = spec_file.stem.replace(".spec", "")
            sheet_npz = root / "sheets" / f"{piece}.sheet.npz"
            if sheet_npz.exists():
                jobs.append((piece, performance))

    # Deduplicate while preserving order.
    seen = set()
    unique_jobs: List[Job] = []
    for job in jobs:
        if job in seen:
            continue
        seen.add(job)
        unique_jobs.append(job)

    if not unique_jobs:
        raise ValueError("No processed jobs were discovered for split creation.")

    return unique_jobs


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}."
        )


def split_pieces(
    pieces: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Deterministic piece-level split.

    Every performance of a piece stays in the same split.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    unique_pieces = sorted(set(pieces))
    if not unique_pieces:
        raise ValueError("Cannot split an empty piece list.")

    rng = random.Random(seed)
    rng.shuffle(unique_pieces)

    n = len(unique_pieces)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Keep all splits non-empty when enough pieces exist.
    if n >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

    train_pieces = unique_pieces[:n_train]
    val_pieces = unique_pieces[n_train : n_train + n_val]
    test_pieces = unique_pieces[n_train + n_val :]

    return {
        "train": train_pieces,
        "val": val_pieces,
        "test": test_pieces,
    }


def split_jobs_by_piece(jobs: Sequence[Job], piece_split: Dict[str, Sequence[str]]) -> Dict[str, List[Job]]:
    piece_to_split: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for piece in piece_split.get(split_name, []):
            piece_to_split[piece] = split_name

    split_jobs: Dict[str, List[Job]] = {"train": [], "val": [], "test": []}

    for piece, performance in jobs:
        split_name = piece_to_split.get(piece)
        if split_name is None:
            continue
        split_jobs[split_name].append((piece, performance))

    return split_jobs


def build_piece_split_manifest(
    processed_root: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    source_manifest_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Build a piece-disjoint split manifest from processed jobs.
    """
    jobs = discover_processed_jobs(processed_root, source_manifest_path=source_manifest_path)
    pieces = [piece for piece, _ in jobs]
    piece_split = split_pieces(
        pieces,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_jobs = split_jobs_by_piece(jobs, piece_split)

    return {
        "processed_root": str(Path(processed_root).resolve()),
        "seed": int(seed),
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "piece_split": {
            "train": list(piece_split["train"]),
            "val": list(piece_split["val"]),
            "test": list(piece_split["test"]),
        },
        "jobs": {
            "train": [{"piece": p, "performance": perf} for p, perf in split_jobs["train"]],
            "val": [{"piece": p, "performance": perf} for p, perf in split_jobs["val"]],
            "test": [{"piece": p, "performance": perf} for p, perf in split_jobs["test"]],
        },
    }


def save_split_manifest(manifest: Dict[str, object], output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return str(out)


def load_split_manifest(split_manifest_path: str) -> Dict[str, object]:
    path = Path(split_manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

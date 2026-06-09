"""
PassageGroupDataset — one item per unique (piece, system_id) group.

Why this exists
---------------
AllPassagesDataset treats every (piece, performance, system_id) triple as a
separate item. But the sheet image for a given (piece, system_id) is identical
across all performances of that piece (different synths/tempos only change the
audio rendering). Two consequences:

  - Random batches can contain two items with identical sheet snippets but
    different audio (different synth/tempo of the same passage).
  - In-batch triplet loss treats those as negatives -> contradictory gradients
    forcing the model to discriminate between identical sheet snippets.

This dataset fixes that by making (piece, system_id) the atomic unit. The audio
variant (synth + tempo) is chosen per __getitem__ call, as augmentation:

  sample_variant="random"  -> uniformly random over available performances
                              (use for training)
  sample_variant="fixed"   -> deterministic choice (first performance after
                              alphabetical sort)
                              (use for val/test, so eval is reproducible)

This mirrors the original paper's design: SystemDataset indexes by
(piece, system) and the synth choice happens inside __getitem__.
"""

from __future__ import annotations

import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_PERF_PATTERN = re.compile(r"_tempo-(\d+)_(.+)$")


def parse_performance(performance: str) -> Optional[Dict[str, Any]]:
    """
    Extract tempo and synth from a performance name like
    'BachJS__BWV1006a__bwv-1006a_1_tempo-1000_ElectricPiano'.

    Returns dict with 'tempo' (int, 1000 means 100%) and 'synth' (str),
    or None if the format doesn't match.
    """
    m = _PERF_PATTERN.search(performance)
    if m is None:
        return None
    return {"tempo": int(m.group(1)), "synth": m.group(2)}


def filter_jobs_by_variant(
    jobs: List[Dict[str, str]],
    synths: Optional[List[str]] = None,
    tempo_range: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, str]]:
    """
    Filter a job list by allowed synths and/or tempo range (inclusive bounds
    in 1000ths; e.g. (900, 1100) keeps tempos 0.9x .. 1.1x).
    """
    if synths is None and tempo_range is None:
        return list(jobs)

    kept: List[Dict[str, str]] = []
    for job in jobs:
        perf = job.get("performance", "")
        parsed = parse_performance(perf)
        if parsed is None:
            continue
        if synths is not None and parsed["synth"] not in synths:
            continue
        if tempo_range is not None:
            lo, hi = tempo_range
            if not (lo <= parsed["tempo"] <= hi):
                continue
        kept.append(job)
    return kept


class _FileHandleCache:
    """LRU cache of NpzFile handles (same pattern as AllPassagesDataset)."""

    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = maxsize
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, path: str) -> Any:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        handle = np.load(path, mmap_mode="r")
        self._cache[path] = handle
        self._cache.move_to_end(path)
        if len(self._cache) > self._maxsize:
            _, old = self._cache.popitem(last=False)
            old.close()
        return handle


class PassageGroupDataset(Dataset):
    """
    One item per unique (piece, system_id) group; audio variant sampled per call.

    Each entry stores:
        sheet_path    : str
        sheet_indices : np.ndarray of slice rows in sheet.npz for this system
        variants      : list of (performance, spec_path, spec_indices) for every
                        performance of this piece that has this system
    """

    def __init__(
        self,
        processed_root: str,
        jobs: List[Dict[str, str]],
        sample_variant: str = "random",
        sheet_transform: Optional[Callable] = None,
        spec_transform: Optional[Callable] = None,
        return_meta: bool = True,
        cache_size: int = 64,
        skip_missing: bool = True,
    ) -> None:
        if sample_variant not in ("random", "fixed"):
            raise ValueError("sample_variant must be 'random' or 'fixed'")
        self.processed_root = Path(processed_root)
        self.sample_variant = sample_variant
        self.sheet_transform = sheet_transform
        self.spec_transform = spec_transform
        self.return_meta = return_meta
        self._cache = _FileHandleCache(maxsize=cache_size)
        self._groups: List[Dict[str, Any]] = []
        self._skipped: List[Dict] = []
        self._build_groups(jobs, skip_missing)

    @staticmethod
    def _build_index_map(sys_idx: np.ndarray) -> Dict[int, np.ndarray]:
        mapping: Dict[int, np.ndarray] = {}
        for sid in np.unique(sys_idx).tolist():
            mapping[int(sid)] = np.flatnonzero(sys_idx == sid)
        return mapping

    def _build_groups(self, jobs: List[Dict[str, str]], skip_missing: bool) -> None:
        # Group performances by piece (sheet is shared per piece).
        by_piece: Dict[str, List[str]] = {}
        for job in jobs:
            piece = job.get("piece", "")
            perf = job.get("performance", "")
            if not piece or not perf:
                continue
            by_piece.setdefault(piece, []).append(perf)

        for piece, perfs in by_piece.items():
            sheet_path = self.processed_root / "sheets" / f"{piece}.sheet.npz"
            if not sheet_path.exists():
                if skip_missing:
                    self._skipped.append({"piece": piece, "error": "sheet not found"})
                    continue
                raise FileNotFoundError(f"Missing sheet for {piece}")

            try:
                with np.load(str(sheet_path)) as f:
                    sheet_sys_idx = f["slice_system_idx"].copy()
            except Exception as exc:
                if skip_missing:
                    self._skipped.append({"piece": piece, "error": str(exc)})
                    continue
                raise

            sheet_map = self._build_index_map(sheet_sys_idx)

            perf_maps: Dict[str, Tuple[str, Dict[int, np.ndarray]]] = {}
            for perf in perfs:
                spec_path = self.processed_root / "audio" / piece / f"{perf}.spec.npz"
                if not spec_path.exists():
                    if skip_missing:
                        self._skipped.append({"piece": piece, "performance": perf,
                                              "error": "spec not found"})
                        continue
                    raise FileNotFoundError(f"Missing spec for {piece}/{perf}")
                try:
                    with np.load(str(spec_path)) as f:
                        spec_sys_idx = f["slice_system_idx"].copy()
                except Exception as exc:
                    if skip_missing:
                        self._skipped.append({"piece": piece, "performance": perf,
                                              "error": str(exc)})
                        continue
                    raise
                perf_maps[perf] = (str(spec_path), self._build_index_map(spec_sys_idx))

            if not perf_maps:
                continue

            for sid, sheet_indices in sheet_map.items():
                variants: List[Tuple[str, str, np.ndarray]] = []
                for perf, (spec_path, spec_map) in perf_maps.items():
                    if sid in spec_map:
                        variants.append((perf, spec_path, spec_map[sid]))
                if not variants:
                    continue
                # Sort variants by performance name so "fixed" mode is reproducible.
                variants.sort(key=lambda v: v[0])
                self._groups.append({
                    "piece": piece,
                    "system_id": int(sid),
                    "sheet_path": str(sheet_path),
                    "sheet_indices": sheet_indices,
                    "variants": variants,
                })

    @property
    def skipped(self) -> List[Dict]:
        return self._skipped

    @property
    def num_variants_total(self) -> int:
        return sum(len(g["variants"]) for g in self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int):
        group = self._groups[index]

        if self.sample_variant == "random":
            perf, spec_path, spec_indices = random.choice(group["variants"])
        else:
            perf, spec_path, spec_indices = group["variants"][0]

        sheet_f = self._cache.get(group["sheet_path"])
        spec_f = self._cache.get(spec_path)

        sheet_seq_np = sheet_f["sheet_slices"][group["sheet_indices"]]
        spec_seq_np = spec_f["spec_slices"][spec_indices]

        sheet_seq = torch.from_numpy(sheet_seq_np).float().unsqueeze(1) / 255.0
        spec_seq = torch.from_numpy(spec_seq_np).float().unsqueeze(1)

        if self.sheet_transform is not None:
            sheet_seq = self.sheet_transform(sheet_seq)
        if self.spec_transform is not None:
            spec_seq = self.spec_transform(spec_seq)

        if not self.return_meta:
            return sheet_seq, spec_seq

        meta = {
            "piece": group["piece"],
            "system_id": group["system_id"],
            "performance": perf,
        }
        return sheet_seq, spec_seq, meta

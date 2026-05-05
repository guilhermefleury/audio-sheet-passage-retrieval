"""
AllPassagesDataset — a single flat dataset that spans every job (piece +
performance) in a split manifest, without keeping thousands of files open.

Problem it solves
-----------------
Using ConcatDataset of N PassageSequenceDatasets opens 2×N NPZ files in
__init__ (one sheet + one spec per job).  With 5940 jobs that exceeds the
Windows OS file-descriptor limit (~512) immediately.

Solution
--------
- __init__ opens each NPZ file **briefly** only to read the tiny index arrays
  (slice_system_idx), then closes the file.  The result is a flat list of
  (sheet_path, spec_path, sheet_indices, spec_indices) entries.
- __getitem__ reads slice arrays through a small LRU file-handle cache that
  keeps at most `cache_size` files open at once (default 64, i.e. 32 pairs).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .passage_sequence_dataset import passage_sequence_collate_fn  # re-export


# ---------------------------------------------------------------------------
# LRU file-handle cache
# ---------------------------------------------------------------------------

class _FileHandleCache:
    """
    Keeps at most `maxsize` numpy NpzFile handles open simultaneously.
    Evicts the least-recently-used handle when full.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str) -> Any:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        # Open new handle
        handle = np.load(path, mmap_mode="r")
        self._cache[path] = handle
        self._cache.move_to_end(path)
        # Evict LRU if over limit
        if len(self._cache) > self._maxsize:
            _, old = self._cache.popitem(last=False)
            old.close()
        return handle


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AllPassagesDataset(Dataset):
    """
    Flat dataset over all (piece, performance, system) entries in a split.

    One item = one system passage:
        sheet_seq : [Ns, 1, 160, 180]  float32 tensor
        spec_seq  : [Na, 1,  92,  20]  float32 tensor
        meta      : dict (optional)

    Use `passage_sequence_collate_fn` as the DataLoader collate function.
    """

    def __init__(
        self,
        processed_root: str,
        jobs: List[Dict[str, str]],
        sheet_transform: Optional[Callable] = None,
        spec_transform: Optional[Callable] = None,
        return_meta: bool = True,
        cache_size: int = 64,
        skip_missing: bool = True,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.sheet_transform = sheet_transform
        self.spec_transform = spec_transform
        self.return_meta = return_meta
        self._cache = _FileHandleCache(maxsize=cache_size)

        # Each entry: (sheet_path, spec_path, sheet_indices, spec_indices, meta)
        self._entries: List[Tuple] = []
        self._skipped: List[Dict] = []

        self._build_entries(jobs, skip_missing)

    def _build_entries(self, jobs: List[Dict[str, str]], skip_missing: bool) -> None:
        for job in jobs:
            piece = job.get("piece", "")
            perf  = job.get("performance", "")
            if not piece or not perf:
                continue

            sheet_path = self.processed_root / "sheets" / f"{piece}.sheet.npz"
            spec_path  = self.processed_root / "audio" / piece / f"{perf}.spec.npz"

            if not sheet_path.exists() or not spec_path.exists():
                if skip_missing:
                    self._skipped.append({"piece": piece, "performance": perf,
                                          "error": "file not found"})
                    continue
                raise FileNotFoundError(
                    f"Missing files for {piece}/{perf}"
                )

            try:
                # Open briefly to read tiny index arrays, then close
                with np.load(str(sheet_path)) as f:
                    sheet_sys_idx = f["slice_system_idx"].copy()
                with np.load(str(spec_path)) as f:
                    spec_sys_idx  = f["slice_system_idx"].copy()
            except Exception as exc:
                if skip_missing:
                    self._skipped.append({"piece": piece, "performance": perf,
                                          "error": str(exc)})
                    continue
                raise

            sheet_map = self._build_index_map(sheet_sys_idx)
            spec_map  = self._build_index_map(spec_sys_idx)
            system_ids = sorted(set(sheet_map.keys()) & set(spec_map.keys()))

            if not system_ids:
                self._skipped.append({"piece": piece, "performance": perf,
                                      "error": "no overlapping systems"})
                continue

            for sid in system_ids:
                self._entries.append((
                    str(sheet_path),
                    str(spec_path),
                    sheet_map[sid],   # np.ndarray of row indices
                    spec_map[sid],
                    {"piece": piece, "performance": perf, "system_id": sid},
                ))

    @staticmethod
    def _build_index_map(sys_idx: np.ndarray) -> Dict[int, np.ndarray]:
        mapping: Dict[int, np.ndarray] = {}
        for sid in np.unique(sys_idx).tolist():
            mapping[int(sid)] = np.flatnonzero(sys_idx == sid)
        return mapping

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int):
        sheet_path, spec_path, sheet_idx, spec_idx, meta = self._entries[index]

        # Read slices through the LRU file-handle cache
        sheet_f = self._cache.get(sheet_path)
        spec_f  = self._cache.get(spec_path)

        sheet_seq_np = sheet_f["sheet_slices"][sheet_idx]   # [Ns, 160, 180]
        spec_seq_np  = spec_f["spec_slices"][spec_idx]      # [Na,  92,  20]

        # Convert to channel-first float tensors
        sheet_seq = torch.from_numpy(sheet_seq_np).float().unsqueeze(1) / 255.0
        spec_seq  = torch.from_numpy(spec_seq_np).float().unsqueeze(1)

        if self.sheet_transform is not None:
            sheet_seq = self.sheet_transform(sheet_seq)
        if self.spec_transform is not None:
            spec_seq  = self.spec_transform(spec_seq)

        if not self.return_meta:
            return sheet_seq, spec_seq
        return sheet_seq, spec_seq, meta

    @property
    def skipped(self) -> List[Dict]:
        return self._skipped

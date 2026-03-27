from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for PerformancePairDataset. Install torch to use this module."
    ) from exc


class PerformancePairDataset(Dataset):
    """
    One PyTorch dataset for one (piece, performance) pointer triplet.

    It resolves samples using pointer arrays stored in pairs files:
        sheet = sheet_slices[pair_sheet_idx[i]]
        spec  = spec_slices[pair_spec_idx[i]]

    Expected files:
    - sheet store: <piece>.sheet.npz
    - spec store: <performance>.spec.npz
    - pair store: <performance>.pairs.npz
    """

    def __init__(
        self,
        sheet_npz_path: str,
        spec_npz_path: str,
        pairs_npz_path: str,
        sheet_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        spec_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_meta: bool = True,
        mmap_mode: str = "r",
    ) -> None:
        self.sheet_npz_path = str(sheet_npz_path)
        self.spec_npz_path = str(spec_npz_path)
        self.pairs_npz_path = str(pairs_npz_path)

        self.sheet_transform = sheet_transform
        self.spec_transform = spec_transform
        self.return_meta = return_meta

        # Memory-mapped loads keep RAM usage low for large corpora.
        self.sheet_npz = np.load(self.sheet_npz_path, mmap_mode=mmap_mode)
        self.spec_npz = np.load(self.spec_npz_path, mmap_mode=mmap_mode)
        self.pairs_npz = np.load(self.pairs_npz_path, mmap_mode=mmap_mode)

        self.sheet_slices = self.sheet_npz["sheet_slices"]
        self.spec_slices = self.spec_npz["spec_slices"]
        self.pair_sheet_idx = self.pairs_npz["pair_sheet_idx"]
        self.pair_spec_idx = self.pairs_npz["pair_spec_idx"]
        self.pair_system_idx = self.pairs_npz["pair_system_idx"]

        if self.pair_sheet_idx.shape[0] != self.pair_spec_idx.shape[0]:
            raise ValueError("pair_sheet_idx and pair_spec_idx must have the same length.")

    def __len__(self) -> int:
        return int(self.pair_sheet_idx.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        si = int(self.pair_sheet_idx[index])
        ai = int(self.pair_spec_idx[index])
        sysi = int(self.pair_system_idx[index])

        sheet = self.sheet_slices[si]  # (160, 180), uint8
        spec = self.spec_slices[ai]    # (92, 20), float32

        # Channel-first tensors expected by CNN models.
        sheet_t = torch.from_numpy(sheet).float().unsqueeze(0) / 255.0
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)

        if self.sheet_transform is not None:
            sheet_t = self.sheet_transform(sheet_t)
        if self.spec_transform is not None:
            spec_t = self.spec_transform(spec_t)

        if not self.return_meta:
            return sheet_t, spec_t  # type: ignore[return-value]

        meta = {
            "pair_index": int(index),
            "sheet_index": si,
            "spec_index": ai,
            "system_index": sysi,
        }
        return sheet_t, spec_t, meta

    @classmethod
    def from_processed_root(
        cls,
        processed_root: str,
        piece_name: str,
        performance_name: str,
        **kwargs: Any,
    ) -> "PerformancePairDataset":
        """
        Build dataset using the normalized processed_pairs folder structure.
        """
        root = Path(processed_root)
        sheet_npz = root / "sheets" / f"{piece_name}.sheet.npz"
        spec_npz = root / "audio" / piece_name / f"{performance_name}.spec.npz"
        pairs_npz = root / "pairs" / piece_name / f"{performance_name}.pairs.npz"

        missing = [p for p in (sheet_npz, spec_npz, pairs_npz) if not p.exists()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Required dataset files not found: {missing_str}")

        return cls(
            sheet_npz_path=str(sheet_npz),
            spec_npz_path=str(spec_npz),
            pairs_npz_path=str(pairs_npz),
            **kwargs,
        )

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for PassageSequenceDataset. Install torch to use this module."
    ) from exc


class PassageSequenceDataset(Dataset):
    """
    Paper-style dataset: one sample = one full system passage sequence.

    For each system, returns two ordered sequences:
    - sheet sequence: [Ns, 1, 160, 180]
    - spec sequence:  [Na, 1,  92,  20]

    Ns and Na may differ, which matches the recurrent setup in the paper.
    """

    def __init__(
        self,
        sheet_npz_path: str,
        spec_npz_path: str,
        sheet_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        spec_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_meta: bool = True,
        mmap_mode: str = "r",
    ) -> None:
        self.sheet_npz_path = str(sheet_npz_path)
        self.spec_npz_path = str(spec_npz_path)

        self.sheet_transform = sheet_transform
        self.spec_transform = spec_transform
        self.return_meta = return_meta

        self.sheet_npz = np.load(self.sheet_npz_path, mmap_mode=mmap_mode)
        self.spec_npz = np.load(self.spec_npz_path, mmap_mode=mmap_mode)

        self.sheet_slices = self.sheet_npz["sheet_slices"]
        self.sheet_slice_system_idx = self.sheet_npz["slice_system_idx"]

        self.spec_slices = self.spec_npz["spec_slices"]
        self.spec_slice_system_idx = self.spec_npz["slice_system_idx"]

        self.sheet_system_to_indices = self._build_system_index_map(self.sheet_slice_system_idx)
        self.spec_system_to_indices = self._build_system_index_map(self.spec_slice_system_idx)

        # Keep only systems that exist in both modalities.
        self.system_ids = sorted(
            set(self.sheet_system_to_indices.keys()).intersection(self.spec_system_to_indices.keys())
        )

        if not self.system_ids:
            raise ValueError("No overlapping systems found between sheet and spec stores.")

    @staticmethod
    def _build_system_index_map(system_idx_array: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Build ordered mapping from system id to slice indices.
        """
        mapping: Dict[int, np.ndarray] = {}
        unique_ids = np.unique(system_idx_array)
        for sid in unique_ids.tolist():
            mapping[int(sid)] = np.flatnonzero(system_idx_array == sid)
        return mapping

    def __len__(self) -> int:
        return len(self.system_ids)

    def __getitem__(self, index: int):
        system_id = int(self.system_ids[index])

        sheet_indices = self.sheet_system_to_indices[system_id]
        spec_indices = self.spec_system_to_indices[system_id]

        # Ordered full sequences for this system.
        sheet_seq_np = self.sheet_slices[sheet_indices]  # [Ns, 160, 180]
        spec_seq_np = self.spec_slices[spec_indices]     # [Na, 92, 20]

        # Convert to channel-first tensors.
        sheet_seq = torch.from_numpy(sheet_seq_np).float().unsqueeze(1) / 255.0  # [Ns, 1, 160, 180]
        spec_seq = torch.from_numpy(spec_seq_np).float().unsqueeze(1)             # [Na, 1, 92, 20]

        if self.sheet_transform is not None:
            sheet_seq = self.sheet_transform(sheet_seq)
        if self.spec_transform is not None:
            spec_seq = self.spec_transform(spec_seq)

        if not self.return_meta:
            return sheet_seq, spec_seq

        meta = {
            "dataset_index": int(index),
            "system_index": system_id,
            "sheet_seq_len": int(sheet_seq.shape[0]),
            "spec_seq_len": int(spec_seq.shape[0]),
        }
        return sheet_seq, spec_seq, meta

    @classmethod
    def from_processed_root(
        cls,
        processed_root: str,
        piece_name: str,
        performance_name: str,
        **kwargs: Any,
    ) -> "PassageSequenceDataset":
        root = Path(processed_root)
        sheet_npz = root / "sheets" / f"{piece_name}.sheet.npz"
        spec_npz = root / "audio" / piece_name / f"{performance_name}.spec.npz"

        missing = [p for p in (sheet_npz, spec_npz) if not p.exists()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Required dataset files not found: {missing_str}")

        return cls(
            sheet_npz_path=str(sheet_npz),
            spec_npz_path=str(spec_npz),
            **kwargs,
        )


def passage_sequence_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    """
    Collate variable-length system sequences for recurrent models.

    Returns a dict with padded tensors and original lengths:
    - sheet_seq: [B, max_Ns, 1, 160, 180]
    - spec_seq:  [B, max_Na, 1,  92,  20]
    - sheet_len: [B]
    - spec_len:  [B]
    - meta: list[dict]
    """
    if len(batch) == 0:
        raise ValueError("Empty batch in passage_sequence_collate_fn.")

    if len(batch[0]) == 2:
        # return_meta=False path
        batch = [(b[0], b[1], {}) for b in batch]  # type: ignore[assignment]

    sheet_list = [b[0] for b in batch]
    spec_list = [b[1] for b in batch]
    meta_list = [b[2] for b in batch]

    sheet_lens = torch.tensor([x.shape[0] for x in sheet_list], dtype=torch.long)
    spec_lens = torch.tensor([x.shape[0] for x in spec_list], dtype=torch.long)

    bsz = len(batch)
    max_sheet_len = int(sheet_lens.max().item())
    max_spec_len = int(spec_lens.max().item())

    sheet_shape_tail = sheet_list[0].shape[1:]  # [1, 160, 180]
    spec_shape_tail = spec_list[0].shape[1:]    # [1, 92, 20]

    padded_sheet = torch.zeros((bsz, max_sheet_len, *sheet_shape_tail), dtype=sheet_list[0].dtype)
    padded_spec = torch.zeros((bsz, max_spec_len, *spec_shape_tail), dtype=spec_list[0].dtype)

    for i, (sheet_seq, spec_seq) in enumerate(zip(sheet_list, spec_list)):
        padded_sheet[i, : sheet_seq.shape[0]] = sheet_seq
        padded_spec[i, : spec_seq.shape[0]] = spec_seq

    return {
        "sheet_seq": padded_sheet,
        "spec_seq": padded_spec,
        "sheet_len": sheet_lens,
        "spec_len": spec_lens,
        "meta": meta_list,
    }

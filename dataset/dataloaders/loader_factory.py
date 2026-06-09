from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from .all_passages_dataset import AllPassagesDataset, passage_sequence_collate_fn
from .passage_group_dataset import PassageGroupDataset, filter_jobs_by_variant
from .passage_sequence_dataset import PassageSequenceDataset
from .performance_pair_dataset import PerformancePairDataset
from .splits import build_piece_split_manifest, load_split_manifest, save_split_manifest


def create_passage_sequence_dataloader(
    processed_root: str,
    piece_name: str,
    performance_name: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    sheet_transform: Optional[Any] = None,
    spec_transform: Optional[Any] = None,
    return_meta: bool = True,
    mmap_mode: str = "r",
) -> Tuple[PassageSequenceDataset, DataLoader]:
    """
    Build paper-style passage-sequence Dataset + DataLoader.

    One sample is one full system sequence pair:
      sheet_seq: [Ns, 1, 160, 180]
      spec_seq:  [Na, 1,  92,  20]

    Loader uses passage_sequence_collate_fn to pad variable lengths.
    """
    dataset = PassageSequenceDataset.from_processed_root(
        processed_root=processed_root,
        piece_name=piece_name,
        performance_name=performance_name,
        sheet_transform=sheet_transform,
        spec_transform=spec_transform,
        return_meta=return_meta,
        mmap_mode=mmap_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=passage_sequence_collate_fn,
    )
    return dataset, loader


def create_slice_pair_dataloader(
    processed_root: str,
    piece_name: str,
    performance_name: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    sheet_transform: Optional[Any] = None,
    spec_transform: Optional[Any] = None,
    return_meta: bool = True,
    mmap_mode: str = "r",
) -> Tuple[PerformancePairDataset, DataLoader]:
    """
    Build slice-level Dataset + DataLoader.

    One sample is one paired slice:
      sheet: [1, 160, 180]
      spec:  [1,  92,  20]
    """
    dataset = PerformancePairDataset.from_processed_root(
        processed_root=processed_root,
        piece_name=piece_name,
        performance_name=performance_name,
        sheet_transform=sheet_transform,
        spec_transform=spec_transform,
        return_meta=return_meta,
        mmap_mode=mmap_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, loader


def create_and_save_piece_split_manifest(
    processed_root: str,
    split_manifest_path: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    source_manifest_path: Optional[str] = None,
) -> Tuple[Dict[str, object], str]:
    """
    Build and save a piece-disjoint train/val/test split manifest.

    The split unit is piece name, so all performances of the same piece
    are guaranteed to remain in one split.
    """
    if split_manifest_path is None:
        split_manifest_path = str(Path(processed_root) / "manifests" / "piece_split_latest.json")

    manifest = build_piece_split_manifest(
        processed_root=processed_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        source_manifest_path=source_manifest_path,
    )
    out_path = save_split_manifest(manifest, split_manifest_path)
    return manifest, out_path


def _build_sequence_concat_dataset(
    processed_root: str,
    jobs: List[Dict[str, str]],
    sheet_transform: Optional[Any],
    spec_transform: Optional[Any],
    return_meta: bool,
    mmap_mode: str,
    skip_missing: bool,
):
    # AllPassagesDataset replaces ConcatDataset of N PassageSequenceDatasets.
    # Opening 2×N files simultaneously in __init__ hit the OS file-descriptor
    # limit (~512 on Windows) with large splits.  AllPassagesDataset only opens
    # files briefly to read index arrays, then manages a small LRU handle cache
    # during __getitem__.
    ds = AllPassagesDataset(
        processed_root=processed_root,
        jobs=jobs,
        sheet_transform=sheet_transform,
        spec_transform=spec_transform,
        return_meta=return_meta,
        skip_missing=skip_missing,
    )
    if len(ds) == 0:
        raise ValueError("No valid passage entries were found for this split.")
    return ds, ds.skipped


def create_passage_sequence_split_dataloaders(
    processed_root: str,
    split_manifest_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    sheet_transform: Optional[Any] = None,
    spec_transform: Optional[Any] = None,
    return_meta: bool = True,
    mmap_mode: str = "r",
    skip_missing: bool = True,
) -> Tuple[Dict[str, AllPassagesDataset], Dict[str, DataLoader], Dict[str, List[Dict[str, str]]]]:
    """
    Build train/val/test DataLoaders for paper-style passage sequences.

    split_manifest_path must point to a piece-level split manifest generated by
    create_and_save_piece_split_manifest.
    """
    manifest = load_split_manifest(split_manifest_path)
    jobs_by_split = manifest.get("jobs")
    if not isinstance(jobs_by_split, dict):
        raise ValueError("Invalid split manifest: missing jobs map.")

    split_names = ("train", "val", "test")
    datasets: Dict[str, AllPassagesDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    skipped_summary: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

    for split in split_names:
        split_jobs = jobs_by_split.get(split, [])
        if not isinstance(split_jobs, list):
            raise ValueError(f"Invalid split manifest: jobs[{split}] must be a list.")

        ds, skipped = _build_sequence_concat_dataset(
            processed_root=processed_root,
            jobs=split_jobs,
            sheet_transform=sheet_transform,
            spec_transform=spec_transform,
            return_meta=return_meta,
            mmap_mode=mmap_mode,
            skip_missing=skip_missing,
        )

        shuffle = split == "train"
        drop_last = drop_last_train if split == "train" else False

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=passage_sequence_collate_fn,
        )

        datasets[split] = ds
        loaders[split] = dl
        skipped_summary[split] = skipped

    return datasets, loaders, skipped_summary


def create_passage_group_split_dataloaders(
    processed_root: str,
    split_manifest_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = True,
    train_synths: Optional[List[str]] = None,
    train_tempo_range: Optional[Tuple[int, int]] = None,
    eval_synths: Optional[List[str]] = None,
    eval_tempo_range: Optional[Tuple[int, int]] = None,
    train_sheet_transform: Optional[Any] = None,
    train_spec_transform: Optional[Any] = None,
    eval_sheet_transform: Optional[Any] = None,
    eval_spec_transform: Optional[Any] = None,
    return_meta: bool = True,
    skip_missing: bool = True,
) -> Tuple[Dict[str, PassageGroupDataset], Dict[str, DataLoader], Dict[str, List[Dict[str, str]]]]:
    """
    Build train/val/test DataLoaders using PassageGroupDataset.

    Each item is one unique (piece, system_id); the audio variant is sampled
    per __getitem__ call ("random" for train, "fixed" for val/test). The
    train split is also filtered to the paper's training synths/tempo range;
    val and test are filtered to the held-out evaluation synth/tempo.

    Argument conventions:
      *_synths        : list of allowed synth names; None means no filtering
      *_tempo_range   : (lo, hi) inclusive bounds in 1000ths (1000 = 100%);
                        None means no filtering
    """
    manifest = load_split_manifest(split_manifest_path)
    jobs_by_split = manifest.get("jobs")
    if not isinstance(jobs_by_split, dict):
        raise ValueError("Invalid split manifest: missing jobs map.")

    split_configs = {
        "train": {
            "synths": train_synths,
            "tempo_range": train_tempo_range,
            "mode": "random",
            "sheet_transform": train_sheet_transform,
            "spec_transform": train_spec_transform,
        },
        "val": {
            "synths": eval_synths,
            "tempo_range": eval_tempo_range,
            "mode": "fixed",
            "sheet_transform": eval_sheet_transform,
            "spec_transform": eval_spec_transform,
        },
        "test": {
            "synths": eval_synths,
            "tempo_range": eval_tempo_range,
            "mode": "fixed",
            "sheet_transform": eval_sheet_transform,
            "spec_transform": eval_spec_transform,
        },
    }

    datasets: Dict[str, PassageGroupDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    skipped_summary: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

    for split, cfg in split_configs.items():
        split_jobs = jobs_by_split.get(split, [])
        if not isinstance(split_jobs, list):
            raise ValueError(f"Invalid split manifest: jobs[{split}] must be a list.")

        filtered = filter_jobs_by_variant(
            split_jobs,
            synths=cfg["synths"],
            tempo_range=cfg["tempo_range"],
        )
        if not filtered:
            raise ValueError(
                f"No jobs left in '{split}' after variant filtering. "
                f"Check synth allowlist and tempo range."
            )

        ds = PassageGroupDataset(
            processed_root=processed_root,
            jobs=filtered,
            sample_variant=cfg["mode"],
            sheet_transform=cfg["sheet_transform"],
            spec_transform=cfg["spec_transform"],
            return_meta=return_meta,
            skip_missing=skip_missing,
        )
        if len(ds) == 0:
            raise ValueError(
                f"No valid (piece, system) groups found for '{split}' split."
            )

        shuffle = split == "train"
        drop_last = drop_last_train if split == "train" else False

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=passage_sequence_collate_fn,
        )
        datasets[split] = ds
        skipped_summary[split] = ds.skipped

    return datasets, loaders, skipped_summary

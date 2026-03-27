import json
from pathlib import Path

import numpy as np

from audio import get_spec_systems, load_performance_spec, system_spec_slicer
from sheet import get_sheet_systems


def _collect_sheet_system_slices(file_path, sheet_stride=90):
    """
    Collect sheet slices grouped by global system index and keep page mapping.
    """
    piece_name = Path(file_path).name
    coords_dir = Path(file_path) / "scores" / f"{piece_name}_ly" / "coords"
    systems_files = sorted(coords_dir.glob("systems_*.npy"))
    if not systems_files:
        raise FileNotFoundError(f"No systems files found in {coords_dir}")

    systems_slices = []
    system_page_map = []
    global_idx = 0

    for page_num in range(1, len(systems_files) + 1):
        _, page_systems_sliced = get_sheet_systems(file_path, page=page_num, stride=sheet_stride)
        for local_system_idx, slices in enumerate(page_systems_sliced):
            systems_slices.append(slices)
            system_page_map.append(
                {
                    "system_index": global_idx,
                    "page": page_num,
                    "system_in_page": local_system_idx,
                }
            )
            global_idx += 1

    return systems_slices, system_page_map


def build_sheet_store(file_path, sheet_stride=90):
    """
    Build sheet store arrays once per piece (shared across all performances).
    """
    piece_name = Path(file_path).name
    systems_slices, system_page_map = _collect_sheet_system_slices(file_path, sheet_stride=sheet_stride)

    sheet_slices = []
    slice_system_idx = []
    slice_page = []
    slice_system_in_page = []

    for sys_idx, slices in enumerate(systems_slices):
        for slc in slices:
            if slc.shape != (160, 180):
                continue
            sheet_slices.append(slc.astype(np.uint8))
            slice_system_idx.append(sys_idx)
            slice_page.append(system_page_map[sys_idx]["page"])
            slice_system_in_page.append(system_page_map[sys_idx]["system_in_page"])

    if not sheet_slices:
        raise ValueError("No valid sheet slices produced.")

    return {
        "piece": piece_name,
        "sheet_stride": sheet_stride,
        "sheet_slices": np.stack(sheet_slices, axis=0),
        "slice_system_idx": np.asarray(slice_system_idx, dtype=np.int32),
        "slice_page": np.asarray(slice_page, dtype=np.int16),
        "slice_system_in_page": np.asarray(slice_system_in_page, dtype=np.int16),
    }


def build_spec_store(file_path, performance_name, spec_window_frames=20, spec_stride_frames=10):
    """
    Build audio spectrogram slice store once per performance.
    """
    piece_name = Path(file_path).name

    _, spec_systems_sliced = get_spec_systems(
        file_path,
        performance_name,
        window_frames=spec_window_frames,
        stride_frames=spec_stride_frames,
        fps=20,
    )

    spec_slices = []
    slice_system_idx = []

    for sys_idx, slices in enumerate(spec_systems_sliced):
        for slc in slices:
            if slc.shape != (92, spec_window_frames):
                continue
            spec_slices.append(slc.astype(np.float32))
            slice_system_idx.append(sys_idx)

    if not spec_slices:
        raise ValueError("No valid spectrogram slices produced.")

    return {
        "piece": piece_name,
        "performance": performance_name,
        "spec_mode": "timing",
        "spec_window_frames": spec_window_frames,
        "spec_stride_frames": spec_stride_frames,
        "spec_slices": np.stack(spec_slices, axis=0),
        "slice_system_idx": np.asarray(slice_system_idx, dtype=np.int32),
    }


def build_spec_store_equal_split(
    file_path,
    performance_name,
    n_systems,
    spec_window_frames=20,
    spec_stride_frames=10,
):
    """
    Fallback: split full spectrogram equally across systems, then slice each segment.
    """
    piece_name = Path(file_path).name
    spec = load_performance_spec(file_path, performance_name)

    total_frames = spec.shape[1]
    if n_systems <= 0:
        raise ValueError("n_systems must be > 0 for equal-split fallback.")

    boundaries = np.linspace(0, total_frames, n_systems + 1).astype(np.int32)

    spec_slices = []
    slice_system_idx = []

    for sys_idx in range(n_systems):
        start = int(boundaries[sys_idx])
        end = int(boundaries[sys_idx + 1])
        if end <= start:
            continue

        system_spec = spec[:, start:end]
        slices = system_spec_slicer(
            system_spec,
            window_frames=spec_window_frames,
            stride_frames=spec_stride_frames,
        )

        for slc in slices:
            if slc.shape != (spec.shape[0], spec_window_frames):
                continue
            spec_slices.append(slc.astype(np.float32))
            slice_system_idx.append(sys_idx)

    if not spec_slices:
        raise ValueError("No valid spectrogram slices produced in equal-split fallback.")

    return {
        "piece": piece_name,
        "performance": performance_name,
        "spec_mode": "equal_split_fallback",
        "spec_window_frames": spec_window_frames,
        "spec_stride_frames": spec_stride_frames,
        "spec_slices": np.stack(spec_slices, axis=0),
        "slice_system_idx": np.asarray(slice_system_idx, dtype=np.int32),
    }


def _system_offsets(system_idx_array, n_systems):
    """
    Compute per-system [start, count] on flattened slices.
    """
    starts = np.full(n_systems, -1, dtype=np.int32)
    counts = np.zeros(n_systems, dtype=np.int32)

    for i, sys_idx in enumerate(system_idx_array):
        if starts[sys_idx] == -1:
            starts[sys_idx] = i
        counts[sys_idx] += 1

    return starts, counts


def build_pair_index(sheet_store, spec_store):
    """
    Build pointer-based pairs as integer indices into sheet/spec stores.
    """
    if sheet_store["piece"] != spec_store["piece"]:
        raise ValueError("Sheet and spec stores belong to different pieces.")

    sheet_sys = sheet_store["slice_system_idx"]
    spec_sys = spec_store["slice_system_idx"]
    n_systems = int(max(sheet_sys.max(initial=0), spec_sys.max(initial=0)) + 1)

    sheet_start, sheet_count = _system_offsets(sheet_sys, n_systems)
    spec_start, spec_count = _system_offsets(spec_sys, n_systems)

    pair_sheet_idx = []
    pair_spec_idx = []
    pair_system_idx = []

    for sys_idx in range(n_systems):
        ns = int(sheet_count[sys_idx])
        na = int(spec_count[sys_idx])
        if ns == 0 or na == 0:
            continue

        pair_count = max(ns, na)
        sheet_last = ns - 1
        spec_last = na - 1

        for i in range(pair_count):
            sl = int(round(i * sheet_last / max(pair_count - 1, 1)))
            al = int(round(i * spec_last / max(pair_count - 1, 1)))

            pair_sheet_idx.append(int(sheet_start[sys_idx] + sl))
            pair_spec_idx.append(int(spec_start[sys_idx] + al))
            pair_system_idx.append(sys_idx)

    if not pair_sheet_idx:
        raise ValueError("No valid pair indices produced.")

    return {
        "pair_sheet_idx": np.asarray(pair_sheet_idx, dtype=np.int32),
        "pair_spec_idx": np.asarray(pair_spec_idx, dtype=np.int32),
        "pair_system_idx": np.asarray(pair_system_idx, dtype=np.int16),
    }


def save_normalized_pointer_dataset(
    file_path,
    performance_name,
    output_root,
    sheet_stride=90,
    spec_window_frames=20,
    spec_stride_frames=10,
    overwrite=False,
    allow_equal_split_fallback=False,
):
    """
    Save normalized stores:
    - sheets/<piece>.sheet.npz
    - audio/<piece>/<performance>.spec.npz
    - pairs/<piece>/<performance>.pairs.npz
    """
    piece_name = Path(file_path).name
    root = Path(output_root)

    sheet_dir = root / "sheets"
    audio_dir = root / "audio" / piece_name
    pairs_dir = root / "pairs" / piece_name
    for d in (sheet_dir, audio_dir, pairs_dir):
        d.mkdir(parents=True, exist_ok=True)

    sheet_npz = sheet_dir / f"{piece_name}.sheet.npz"
    sheet_meta = sheet_dir / f"{piece_name}.sheet.meta.json"

    spec_npz = audio_dir / f"{performance_name}.spec.npz"
    spec_meta = audio_dir / f"{performance_name}.spec.meta.json"

    pairs_npz = pairs_dir / f"{performance_name}.pairs.npz"
    pairs_meta = pairs_dir / f"{performance_name}.pairs.meta.json"

    if overwrite or not sheet_npz.exists():
        sheet_store = build_sheet_store(file_path=file_path, sheet_stride=sheet_stride)
        np.savez_compressed(
            sheet_npz,
            sheet_slices=sheet_store["sheet_slices"],
            slice_system_idx=sheet_store["slice_system_idx"],
            slice_page=sheet_store["slice_page"],
            slice_system_in_page=sheet_store["slice_system_in_page"],
        )
        with open(sheet_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "piece": piece_name,
                    "sheet_stride": sheet_stride,
                    "sheet_shape": list(sheet_store["sheet_slices"].shape),
                    "num_sheet_slices": int(sheet_store["sheet_slices"].shape[0]),
                },
                f,
                indent=2,
            )

    try:
        spec_store = build_spec_store(
            file_path=file_path,
            performance_name=performance_name,
            spec_window_frames=spec_window_frames,
            spec_stride_frames=spec_stride_frames,
        )
    except Exception:
        if not allow_equal_split_fallback:
            raise

        sheet_data_for_nsystems = np.load(sheet_npz)
        n_systems = int(sheet_data_for_nsystems["slice_system_idx"].max()) + 1
        spec_store = build_spec_store_equal_split(
            file_path=file_path,
            performance_name=performance_name,
            n_systems=n_systems,
            spec_window_frames=spec_window_frames,
            spec_stride_frames=spec_stride_frames,
        )
    np.savez_compressed(
        spec_npz,
        spec_slices=spec_store["spec_slices"],
        slice_system_idx=spec_store["slice_system_idx"],
    )
    with open(spec_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "piece": piece_name,
                "performance": performance_name,
                "spec_mode": spec_store["spec_mode"],
                "spec_window_frames": spec_window_frames,
                "spec_stride_frames": spec_stride_frames,
                "spec_shape": list(spec_store["spec_slices"].shape),
                "num_spec_slices": int(spec_store["spec_slices"].shape[0]),
            },
            f,
            indent=2,
        )

    sheet_data = np.load(sheet_npz)
    sheet_store_loaded = {
        "piece": piece_name,
        "sheet_slices": sheet_data["sheet_slices"],
        "slice_system_idx": sheet_data["slice_system_idx"],
    }
    pair_idx = build_pair_index(sheet_store_loaded, spec_store)

    np.savez_compressed(
        pairs_npz,
        pair_sheet_idx=pair_idx["pair_sheet_idx"],
        pair_spec_idx=pair_idx["pair_spec_idx"],
        pair_system_idx=pair_idx["pair_system_idx"],
    )
    with open(pairs_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "piece": piece_name,
                "performance": performance_name,
                "num_pairs": int(pair_idx["pair_sheet_idx"].shape[0]),
            },
            f,
            indent=2,
        )

    return {
        "sheet_npz": str(sheet_npz),
        "spec_npz": str(spec_npz),
        "pairs_npz": str(pairs_npz),
        "num_pairs": int(pair_idx["pair_sheet_idx"].shape[0]),
    }


class PointerPairDataset:
    """
    Lightweight loader that resolves pairs through index pointers.
    """

    def __init__(self, sheet_npz_path, spec_npz_path, pairs_npz_path):
        self.sheet_data = np.load(sheet_npz_path)
        self.spec_data = np.load(spec_npz_path)
        self.pair_data = np.load(pairs_npz_path)

        self.sheet_slices = self.sheet_data["sheet_slices"]
        self.spec_slices = self.spec_data["spec_slices"]
        self.pair_sheet_idx = self.pair_data["pair_sheet_idx"]
        self.pair_spec_idx = self.pair_data["pair_spec_idx"]
        self.pair_system_idx = self.pair_data["pair_system_idx"]

    def __len__(self):
        return int(self.pair_sheet_idx.shape[0])

    def __getitem__(self, index):
        si = int(self.pair_sheet_idx[index])
        ai = int(self.pair_spec_idx[index])
        sysi = int(self.pair_system_idx[index])
        return self.sheet_slices[si], self.spec_slices[ai], {
            "pair_index": int(index),
            "sheet_index": si,
            "spec_index": ai,
            "system_index": sysi,
        }

    @classmethod
    def from_root(cls, output_root, piece_name, performance_name):
        root = Path(output_root)
        sheet_npz = root / "sheets" / f"{piece_name}.sheet.npz"
        spec_npz = root / "audio" / piece_name / f"{performance_name}.spec.npz"
        pairs_npz = root / "pairs" / piece_name / f"{performance_name}.pairs.npz"
        return cls(sheet_npz, spec_npz, pairs_npz)


def build_aligned_pairs(file_path, performance_name, sheet_stride=90, spec_window_frames=20, spec_stride_frames=10):
    """
    Build aligned sheet/audio pairs at slice level.

    Each pair contains:
    - sheet_slice: image snippet of shape (160, 180)
    - spec_slice: spectrogram snippet of shape (92, 20)

    Pairing rule per system:
    - If one modality has more slices, indices are linearly mapped so both sides are fully used.

    Returns:
        dict with keys:
            - sheet_slices: np.ndarray [N, 160, 180]
            - spec_slices: np.ndarray [N, 92, 20]
            - pair_meta: list[dict]
    """
    piece_name = Path(file_path).name

    # Gather sheet systems and sheet slices in global system order (across pages)
    score_name = piece_name
    coords_dir = Path(file_path) / "scores" / f"{score_name}_ly" / "coords"
    systems_files = sorted(coords_dir.glob("systems_*.npy"))
    if not systems_files:
        raise FileNotFoundError(f"No systems files found in {coords_dir}")

    all_sheet_systems_sliced = []
    system_page_map = []
    global_idx = 0

    for page_num in range(1, len(systems_files) + 1):
        _, systems_sliced = get_sheet_systems(file_path, page=page_num, stride=sheet_stride)
        for local_system_idx, slices in enumerate(systems_sliced):
            all_sheet_systems_sliced.append(slices)
            system_page_map.append(
                {
                    "system_index": global_idx,
                    "page": page_num,
                    "system_in_page": local_system_idx,
                }
            )
            global_idx += 1

    # Gather spec systems and spec slices in the same global system order
    _, all_spec_systems_sliced = get_spec_systems(
        file_path,
        performance_name,
        window_frames=spec_window_frames,
        stride_frames=spec_stride_frames,
        fps=20,
    )

    if len(all_sheet_systems_sliced) != len(all_spec_systems_sliced):
        raise ValueError(
            "Mismatch between number of sheet systems and spectrogram systems: "
            f"{len(all_sheet_systems_sliced)} vs {len(all_spec_systems_sliced)}"
        )

    sheet_pairs = []
    spec_pairs = []
    pair_meta = []

    # Build aligned slice pairs system by system
    for sys_idx, (sheet_slices, spec_slices) in enumerate(
        zip(all_sheet_systems_sliced, all_spec_systems_sliced)
    ):
        if len(sheet_slices) == 0 or len(spec_slices) == 0:
            continue

        pair_count = max(len(sheet_slices), len(spec_slices))
        sheet_last = len(sheet_slices) - 1
        spec_last = len(spec_slices) - 1

        for i in range(pair_count):
            sheet_idx = int(round(i * sheet_last / max(pair_count - 1, 1)))
            spec_idx = int(round(i * spec_last / max(pair_count - 1, 1)))

            sheet_slice = sheet_slices[sheet_idx]
            spec_slice = spec_slices[spec_idx]

            # Keep only fixed-size pairs expected by training
            if sheet_slice.shape != (160, 180):
                continue
            if spec_slice.shape != (92, spec_window_frames):
                continue

            sheet_pairs.append(sheet_slice.astype(np.uint8))
            spec_pairs.append(spec_slice.astype(np.float32))

            pair_meta.append(
                {
                    "piece": piece_name,
                    "performance": performance_name,
                    "system_index": sys_idx,
                    "page": system_page_map[sys_idx]["page"],
                    "system_in_page": system_page_map[sys_idx]["system_in_page"],
                    "sheet_slice_index": sheet_idx,
                    "spec_slice_index": spec_idx,
                }
            )

    if not sheet_pairs:
        raise ValueError("No valid pairs were produced.")

    return {
        "sheet_slices": np.stack(sheet_pairs, axis=0),
        "spec_slices": np.stack(spec_pairs, axis=0),
        "pair_meta": pair_meta,
    }


def save_pairs_dataset(
    file_path,
    performance_name,
    output_dir,
    dataset_name=None,
    sheet_stride=90,
    spec_window_frames=20,
    spec_stride_frames=10,
):
    """
    Export preprocessed aligned pairs to disk for fast training reload.

    Output files:
    - <dataset_name>.npz: arrays (sheet_slices, spec_slices)
    - <dataset_name>.meta.json: metadata and per-pair index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    piece_name = Path(file_path).name
    if dataset_name is None:
        dataset_name = f"{piece_name}__{performance_name}"

    dataset = build_aligned_pairs(
        file_path=file_path,
        performance_name=performance_name,
        sheet_stride=sheet_stride,
        spec_window_frames=spec_window_frames,
        spec_stride_frames=spec_stride_frames,
    )

    npz_path = output_path / f"{dataset_name}.npz"
    meta_path = output_path / f"{dataset_name}.meta.json"

    np.savez_compressed(
        npz_path,
        sheet_slices=dataset["sheet_slices"],
        spec_slices=dataset["spec_slices"],
    )

    metadata = {
        "piece": piece_name,
        "performance": performance_name,
        "sheet_stride": sheet_stride,
        "spec_window_frames": spec_window_frames,
        "spec_stride_frames": spec_stride_frames,
        "num_pairs": int(dataset["sheet_slices"].shape[0]),
        "sheet_shape": list(dataset["sheet_slices"].shape),
        "spec_shape": list(dataset["spec_slices"].shape),
        "pairs": dataset["pair_meta"],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "num_pairs": metadata["num_pairs"],
        "sheet_shape": metadata["sheet_shape"],
        "spec_shape": metadata["spec_shape"],
    }


def load_pairs_dataset(npz_path):
    """
    Fast loader for exported pair datasets.
    """
    data = np.load(npz_path)
    return data["sheet_slices"], data["spec_slices"]


if __name__ == "__main__":
    info = save_pairs_dataset(
        file_path="../data/msmd/BachJS__BWV779__bach-invention-08",
        performance_name="BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804",
        output_dir="../data/processed_pairs",
        dataset_name="bwv779_tempo1000_overlap50",
        sheet_stride=90,
        spec_window_frames=20,
        spec_stride_frames=10,
    )
    print("Saved dataset:")
    print(info)

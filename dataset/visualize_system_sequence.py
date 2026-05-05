from pathlib import Path

import cv2
import numpy as np

from pair_export import PointerPairDataset


def normalize_spec(spec_slice):
    s = spec_slice.astype(np.float32)
    s_min = float(s.min())
    s_max = float(s.max())
    if s_max - s_min < 1e-8:
        return np.zeros_like(s, dtype=np.uint8)
    s = (s - s_min) / (s_max - s_min)
    return (s * 255.0).astype(np.uint8)


def draw_labeled_cell(img, label):
    out = img.copy()
    cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def build_sequence_image(dataset, system_index):
    pair_sys = dataset.pair_system_idx
    seq_indices = np.where(pair_sys == system_index)[0].tolist()

    if not seq_indices:
        raise ValueError(f"No pairs found for system {system_index}")

    cell_w = 220
    cell_h = 180
    spacer_w = 10
    spacer_h = 8

    rows = []

    for k, pair_idx in enumerate(seq_indices):
        sheet, spec, meta = dataset[pair_idx]

        sheet_vis = cv2.cvtColor(sheet.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        sheet_vis = cv2.resize(sheet_vis, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        sheet_vis = draw_labeled_cell(sheet_vis, f"sheet seq={k} pair={pair_idx}")

        spec_u8 = normalize_spec(spec)
        spec_vis = cv2.applyColorMap(spec_u8, cv2.COLORMAP_VIRIDIS)
        spec_vis = cv2.resize(spec_vis, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        spec_vis = draw_labeled_cell(spec_vis, f"spec seq={k} pair={pair_idx}")

        spacer = np.full((cell_h, spacer_w, 3), 30, dtype=np.uint8)
        row = np.hstack([sheet_vis, spacer, spec_vis])
        rows.append(row)

    canvas = rows[0]
    pad = np.full((spacer_h, canvas.shape[1], 3), 30, dtype=np.uint8)
    for r in rows[1:]:
        canvas = np.vstack([canvas, pad, r])

    return canvas, len(seq_indices)


def main():
    output_root = "../data/processed_pairs"
    piece = "BachJS__BWV779__bach-invention-08"
    performance = "BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804"

    dataset = PointerPairDataset.from_root(
        output_root=output_root,
        piece_name=piece,
        performance_name=performance,
    )

    # Full sequential view of system 0
    canvas0, n0 = build_sequence_image(dataset, system_index=0)

    # Full sequential view of final system (10)
    canvas10, n10 = build_sequence_image(dataset, system_index=10)

    # Replace placeholder titles after counts are known
    title0 = np.full((36, canvas0.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(title0, f"System 0 Full Sequence | pairs={n0}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas0[:36, :, :] = title0

    title10 = np.full((36, canvas10.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(title10, f"System 10 Full Sequence | pairs={n10}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas10[:36, :, :] = title10

    out_dir = Path("../data/processed_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out0 = out_dir / "pair_sequence_system0.png"
    out10 = out_dir / "pair_sequence_system10.png"

    cv2.imwrite(str(out0), canvas0)
    cv2.imwrite(str(out10), canvas10)

    print(f"Saved: {out0}")
    print(f"Saved: {out10}")


if __name__ == "__main__":
    main()

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


def to_bgr(img_gray):
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


def put_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    output_root = "../data/processed_pairs"
    piece = "BachJS__BWV779__bach-invention-08"
    performance = "BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804"

    ds = PointerPairDataset.from_root(
        output_root=output_root,
        piece_name=piece,
        performance_name=performance,
    )

    pair_count = len(ds)
    print(f"Total pairs: {pair_count}")

    # Basic consistency checks against previous testing assumptions
    sheet_shape = ds.sheet_slices.shape
    spec_shape = ds.spec_slices.shape
    print(f"Sheet store shape: {sheet_shape}")
    print(f"Spec store shape: {spec_shape}")

    system_ids, system_counts = np.unique(ds.spec_data["slice_system_idx"], return_counts=True)
    print("Spec slices per system:")
    for sid, cnt in zip(system_ids.tolist(), system_counts.tolist()):
        print(f"  system {sid}: {cnt}")

    # Pick representative pairs: first, quarter, middle, 3/4, last
    indices = sorted(set([0, pair_count // 4, pair_count // 2, (3 * pair_count) // 4, pair_count - 1]))

    rows = []
    cell_w = 220
    cell_h = 180

    for idx in indices:
        sheet, spec, meta = ds[idx]

        # Sheet is already grayscale 160x180
        sheet_u8 = sheet.astype(np.uint8)
        sheet_vis = to_bgr(sheet_u8)
        sheet_vis = cv2.resize(sheet_vis, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        sheet_vis = put_label(sheet_vis, f"sheet pair={idx} sys={meta['system_index']}")

        # Spec is 92x20, normalize then upscale for viewing
        spec_u8 = normalize_spec(spec)
        spec_u8 = cv2.applyColorMap(spec_u8, cv2.COLORMAP_VIRIDIS)
        spec_vis = cv2.resize(spec_u8, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        spec_vis = put_label(spec_vis, f"spec pair={idx} sys={meta['system_index']}")

        spacer = np.full((cell_h, 12, 3), 25, dtype=np.uint8)
        row = np.hstack([sheet_vis, spacer, spec_vis])
        rows.append(row)

    pad = np.full((12, rows[0].shape[1], 3), 25, dtype=np.uint8)
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, pad, r])

    out_path = Path("../data/processed_pairs/pair_preview_bwv779_tempo1000.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)

    print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()

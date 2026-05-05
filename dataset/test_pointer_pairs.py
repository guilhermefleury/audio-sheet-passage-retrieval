from pair_export import save_normalized_pointer_dataset, PointerPairDataset

file_path = "../data/msmd/BachJS__BWV779__bach-invention-08"
performance = "BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804"
output_root = "../data/processed_pairs"

info = save_normalized_pointer_dataset(
    file_path=file_path,
    performance_name=performance,
    output_root=output_root,
    sheet_stride=90,
    spec_window_frames=20,
    spec_stride_frames=10,
)

print("Saved normalized pointer dataset:")
print(info)

dataset = PointerPairDataset.from_root(
    output_root=output_root,
    piece_name="BachJS__BWV779__bach-invention-08",
    performance_name=performance,
)

print(f"Total pairs: {len(dataset)}")

sheet, spec, meta = dataset[0]
print(f"Sample 0 sheet shape: {sheet.shape}, dtype={sheet.dtype}")
print(f"Sample 0 spec shape: {spec.shape}, dtype={spec.dtype}")
print(f"Sample 0 meta: {meta}")

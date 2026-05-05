from torch.utils.data import DataLoader

from dataloaders import PassageSequenceDataset, passage_sequence_collate_fn


def main():
    ds = PassageSequenceDataset.from_processed_root(
        processed_root="../data/processed_pairs",
        piece_name="BachJS__BWV779__bach-invention-08",
        performance_name="BachJS__BWV779__bach-invention-08_tempo-1000_grand-piano-YDP-20160804",
    )

    print(f"Number of system samples: {len(ds)}")

    sheet_seq, spec_seq, meta = ds[0]
    print("Single sample:")
    print("  system:", meta["system_index"])
    print("  sheet_seq:", sheet_seq.shape, sheet_seq.dtype)
    print("  spec_seq:", spec_seq.shape, spec_seq.dtype)

    loader = DataLoader(
        ds,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=passage_sequence_collate_fn,
    )

    batch = next(iter(loader))
    print("\nBatch sample:")
    print("  sheet_seq:", batch["sheet_seq"].shape)
    print("  spec_seq:", batch["spec_seq"].shape)
    print("  sheet_len:", batch["sheet_len"])
    print("  spec_len:", batch["spec_len"])
    print("  systems:", [m.get("system_index") for m in batch["meta"]])


if __name__ == "__main__":
    main()

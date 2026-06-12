"""
Evaluation script for the cross-modal audio-sheet music passage retrieval model.

Reports R@1, R@10, R@25, MRR and Median Rank for both retrieval directions
(Audio-to-Sheet and Sheet-to-Audio), matching Table 2 of the paper:

  Carvalho & Widmer, "Passage Summarization with Recurrent Models for
  Audio-Sheet Music Retrieval", ISMIR 2023.

Usage:
    python eval.py --checkpoint experiments/model_emb64.pt \
                   --split_manifest data/processed_pairs/manifests/piece_split_full.json

    # Evaluate on val split instead of test:
    python eval.py --checkpoint experiments/model_emb64.pt --split val

    # Save results to JSON:
    python eval.py --checkpoint experiments/model_emb64.pt --save_results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model import CrossModalEncoder
from dataset.dataloaders import (
    create_passage_group_split_dataloaders,
    create_passage_sequence_split_dataloaders,
)


_PAPER_EVAL_SYNTHS = ("grand-piano-YDP-20160804",)


def _parse_csv_str(s):
    if s is None or s == "":
        return None
    return [item.strip() for item in s.split(",") if item.strip()]


def _parse_csv_int_pair(s):
    if s is None or s == "":
        return None
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'lo,hi' for tempo range, got {s!r}")
    return (parts[0], parts[1])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_passages(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over every passage in `loader` and collect embeddings.

    Returns:
        sheet_embs : [N, emb_dim]  numpy float32
        audio_embs : [N, emb_dim]  numpy float32
    """
    model.eval()
    sheet_list, audio_list = [], []

    for batch in tqdm(loader, desc="  embedding", ncols=80, leave=False):
        se, ae = model(
            batch["sheet_seq"].to(device),
            batch["sheet_len"],
            batch["spec_seq"].to(device),
            batch["spec_len"],
        )
        sheet_list.append(se.cpu())
        audio_list.append(ae.cpu())

    return (
        torch.cat(sheet_list).numpy(),
        torch.cat(audio_list).numpy(),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """
    For each query x[i], rank all candidates y[j] by cosine distance.
    The correct match is always y[i].

    Returns R@1, R@5, R@10, R@25 (%), MRR, and Median Rank.
    """
    n = x.shape[0]
    dists = cdist(x, y, metric="cosine")                       # [n, n]
    sorted_idx = np.argsort(dists, axis=1)
    ranks = (np.arange(n)[:, None] == sorted_idx).nonzero()[1] + 1  # [n]

    hits = {k: int((ranks <= k).sum()) for k in (1, 5, 10, 25)}
    return {
        "r@1":      100.0 * hits[1]  / n,
        "r@5":      100.0 * hits[5]  / n,
        "r@10":     100.0 * hits[10] / n,
        "r@25":     100.0 * hits[25] / n,
        "mrr":      float(np.mean(1.0 / ranks)),
        "med_rank": float(np.median(ranks)),
        "n":        n,
    }


def print_results_table(s2a: dict, a2s: dict) -> None:
    """Print a results table matching Table 2 style from the paper."""
    header = f"{'Direction':<12} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@25':>7} {'MRR':>7} {'Med.Rk':>8}  {'N':>5}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for label, m in [("S2A", s2a), ("A2S", a2s)]:
        print(
            f"{label:<12} "
            f"{m['r@1']:>7.2f} "
            f"{m['r@5']:>7.2f} "
            f"{m['r@10']:>7.2f} "
            f"{m['r@25']:>7.2f} "
            f"{m['mrr']:>7.4f} "
            f"{m['med_rank']:>8.1f}  "
            f"{m['n']:>5}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---- Load checkpoint ---------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Prefer hyperparameters stored in the checkpoint; fall back to CLI args
    saved_args = ckpt.get("args", {})
    snippet_emb_dim = saved_args.get("snippet_emb_dim", args.snippet_emb_dim)
    rnn_hidden      = saved_args.get("rnn_hidden",      args.rnn_hidden)
    emb_dim         = saved_args.get("emb_dim",         args.emb_dim)
    trained_epoch   = ckpt.get("epoch", "?")
    val_mrr         = ckpt.get("val_s2a", {}).get("mrr", None)

    print(f"  Trained epoch : {trained_epoch}")
    print(f"  Val S2A MRR  : {val_mrr:.4f}" if val_mrr is not None else "  Val MRR: (not stored)")
    print(f"  emb_dim={emb_dim}  rnn_hidden={rnn_hidden}  snippet_emb_dim={snippet_emb_dim}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model -------------------------------------------------------------
    # Detect whether the checkpoint was saved by a model that had
    # TemporalBatchNorm on the audio path. v1 / v2 predate that change; v3+
    # have it. Without this detection, every Sequential index in the audio
    # CNN is shifted by 1 and the load fails on size-mismatch.
    state_dict = ckpt["model_state"]
    has_tbn = "audio_enc.cnn.cnn.0.bn.running_mean" in state_dict
    if not has_tbn:
        print("  [info] checkpoint predates TemporalBatchNorm on audio path; "
              "building model without it for compatibility.")

    model = CrossModalEncoder(
        snippet_emb_dim=snippet_emb_dim,
        rnn_hidden=rnn_hidden,
        emb_dim=emb_dim,
        audio_normalize_input=has_tbn,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing key(s): {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected key(s): {unexpected[:3]}")

    # ---- Dataloader --------------------------------------------------------
    manifest = Path(args.split_manifest)
    if not manifest.exists():
        print(f"[ERROR] Split manifest not found: {manifest}")
        sys.exit(1)

    print(f"\nBuilding '{args.split}' loader from: {manifest.name}")
    if args.use_group_dataset:
        # eval.py reads a single split (--split). The factory still builds all
        # three to honour piece-disjoint partitioning, so apply the user's
        # synth/tempo filter to every split — the unused splits being smaller
        # is harmless, and ensures the requested split is filtered correctly
        # whichever one it is.
        filt_synths = _parse_csv_str(args.eval_synths)
        filt_tempo  = _parse_csv_int_pair(args.eval_tempo_range)
        _, loaders, skipped = create_passage_group_split_dataloaders(
            processed_root=args.processed_root,
            split_manifest_path=str(manifest),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last_train=False,
            train_synths=filt_synths, train_tempo_range=filt_tempo,
            val_synths=filt_synths,   val_tempo_range=filt_tempo,
            test_synths=filt_synths,  test_tempo_range=filt_tempo,
        )
    else:
        _, loaders, skipped = create_passage_sequence_split_dataloaders(
            processed_root=args.processed_root,
            split_manifest_path=str(manifest),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    if skipped.get(args.split):
        print(f"  [warn] {len(skipped[args.split])} jobs skipped in '{args.split}'")

    loader = loaders[args.split]
    n_passages = len(loader.dataset)
    print(f"  {n_passages} passages to evaluate\n")

    # ---- Embed -------------------------------------------------------------
    print("Embedding passages...")
    sheet_embs, audio_embs = embed_passages(model, loader, device)
    print(f"  sheet_embs: {sheet_embs.shape}  audio_embs: {audio_embs.shape}\n")

    # ---- Metrics -----------------------------------------------------------
    print("Computing retrieval metrics...")
    # S2A: sheet query → audio database
    s2a = compute_retrieval_metrics(sheet_embs, audio_embs)
    # A2S: audio query → sheet database
    a2s = compute_retrieval_metrics(audio_embs, sheet_embs)

    print(f"\nResults on '{args.split}' split  |  {n_passages} passage pairs\n")
    print_results_table(s2a, a2s)

    # ---- Save --------------------------------------------------------------
    if args.save_results:
        results = {
            "checkpoint":   str(ckpt_path),
            "split":        args.split,
            "n_passages":   n_passages,
            "trained_epoch": trained_epoch,
            "emb_dim":      emb_dim,
            "s2a":          s2a,
            "a2s":          a2s,
        }
        out_path = ckpt_path.with_name(
            ckpt_path.stem + f"_eval_{args.split}.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained cross-modal passage retrieval model"
    )

    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint saved by train.py")

    # Paths
    parser.add_argument("--split_manifest",
                        default="data/processed_pairs/manifests/piece_split_full.json",
                        help="Piece-disjoint split manifest used during training")
    parser.add_argument("--processed_root", default="data/processed_pairs",
                        help="Root directory of preprocessed data")

    # Eval settings
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding (no memory constraint from gradients)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers. Use 0 on Windows; 4-8 on Linux.")

    # Dataset selection (must match the train-time choice to interpret metrics correctly)
    parser.add_argument("--use_group_dataset", action="store_true", default=True,
                        help="Use PassageGroupDataset (atomic (piece, system)). "
                             "Match the choice used at training time.")
    parser.add_argument("--no_group_dataset", dest="use_group_dataset", action="store_false")
    parser.add_argument("--eval_synths", type=str,
                        default=",".join(_PAPER_EVAL_SYNTHS),
                        help="Comma-separated synth allowlist for eval "
                             "(paper test_aug: grand-piano-YDP-20160804). Empty disables.")
    parser.add_argument("--eval_tempo_range", type=str, default="1000,1000",
                        help="Comma-separated 'lo,hi' tempo bounds for eval "
                             "(paper test_aug: 1000,1000).")

    # Model overrides (only needed if checkpoint has no stored args)
    parser.add_argument("--snippet_emb_dim", type=int, default=32)
    parser.add_argument("--rnn_hidden",      type=int, default=128)
    parser.add_argument("--emb_dim",         type=int, default=64)

    # Output
    parser.add_argument("--save_results", action="store_true",
                        help="Save results as JSON next to the checkpoint")

    main(parser.parse_args())

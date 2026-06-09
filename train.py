"""
Training script for cross-modal audio-sheet music passage retrieval.

Replicates the RNN model from:
  Carvalho & Widmer, "Passage Summarization with Recurrent Models for
  Audio-Sheet Music Retrieval", ISMIR 2023.

Usage:
    python train.py --split_manifest data/processed_pairs/manifests/piece_split_latest.json

Tip: create the split manifest first if it doesn't exist:
    python -c "
    import sys; sys.path.insert(0, '.')
    from dataset.dataloaders import create_and_save_piece_split_manifest
    create_and_save_piece_split_manifest('data/processed_pairs')
    "
"""

import argparse
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import RandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model import CrossModalEncoder, triplet_loss
from dataset.dataloaders import (
    build_sheet_train_transform,
    create_passage_group_split_dataloaders,
    create_passage_sequence_split_dataloaders,
)


# Paper full_aug training synths (msmd_config.yaml `full_aug.synths`).
_PAPER_TRAIN_SYNTHS = (
    "acoustic_piano_imis_1",
    "ElectricPiano",
    "YamahaGrandPiano",
)
# Paper held-out test synth (msmd_config.yaml `test_aug.synths`).
_PAPER_EVAL_SYNTHS = (
    "grand-piano-YDP-20160804",
)


def _parse_csv_str(s: Optional[str]) -> Optional[list]:
    if s is None or s == "":
        return None
    return [item.strip() for item in s.split(",") if item.strip()]


def _parse_csv_int_pair(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None or s == "":
        return None
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'lo,hi' for tempo range, got {s!r}")
    return (parts[0], parts[1])


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """
    For each query x[i], rank all y[j] by cosine distance and report the rank
    of the matching y[i].  x and y must have the same length; pair (i, i) is
    the ground-truth match.

    Returns R@1, R@10, R@25 (percentage), MRR, and Median Rank.
    """
    n = x.shape[0]
    dists = cdist(x, y, metric="cosine")                   # [n, n]
    sorted_idx = np.argsort(dists, axis=1)                 # [n, n]
    # rank of the correct item for each query
    ranks = (np.arange(n)[:, None] == sorted_idx).nonzero()[1] + 1  # [n]

    hit_rates = {k: int((ranks <= k).sum()) for k in (1, 10, 25)}
    return {
        "mrr":      float(np.mean(1.0 / ranks)),
        "med_rank": float(np.median(ranks)),
        "r@1":      100.0 * hit_rates[1]  / n,
        "r@10":     100.0 * hit_rates[10] / n,
        "r@25":     100.0 * hit_rates[25] / n,
    }


# ---------------------------------------------------------------------------
# Optimizer with decoupled weight-decay (from original repo)
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """
    Three parameter groups so batch-norm weights and biases are never decayed.
    Matches the setup used in the original lcasr training script.
    """
    pg_nodecay, pg_decay, pg_bias = [], [], []
    for m in model.modules():
        if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
            pg_bias.append(m.bias)
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            pg_nodecay.append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            pg_decay.append(m.weight)

    optimizer = torch.optim.AdamW(pg_nodecay, lr=lr)
    optimizer.add_param_group({"params": pg_decay,  "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg_bias})
    return optimizer


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, loss_fn, optimizer, device, scaler=None, accum_steps=1) -> float:
    model.train()
    losses = []
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="  train", ncols=80, leave=False)):
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                sheet_emb, audio_emb = model(
                    batch["sheet_seq"].to(device),
                    batch["sheet_len"],
                    batch["spec_seq"].to(device),
                    batch["spec_len"],
                )
                loss = loss_fn(sheet_emb, audio_emb) / accum_steps
            scaler.scale(loss).backward()
        else:
            sheet_emb, audio_emb = model(
                batch["sheet_seq"].to(device),
                batch["sheet_len"],            # must stay on CPU for pack_sequence
                batch["spec_seq"].to(device),
                batch["spec_len"],
            )
            loss = loss_fn(sheet_emb, audio_emb) / accum_steps
            loss.backward()

        losses.append(loss.item() * accum_steps)  # restore scale for logging

        is_last_batch = (i + 1 == len(loader))
        if (i + 1) % accum_steps == 0 or is_last_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.sheet_enc.gru.parameters(), max_norm=0.5)
                clip_grad_norm_(model.audio_enc.gru.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip GRU gradients to prevent exploding gradients
                clip_grad_norm_(model.sheet_enc.gru.parameters(), max_norm=0.5)
                clip_grad_norm_(model.audio_enc.gru.parameters(), max_norm=0.5)
                optimizer.step()
            optimizer.zero_grad()

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# One evaluation epoch  (loss + retrieval metrics in both directions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> tuple:
    """
    Returns (mean_loss, s2a_metrics, a2s_metrics).
    s2a: sheet-to-audio  (query=sheet, database=audio)
    a2s: audio-to-sheet  (query=audio, database=sheet)
    """
    model.eval()
    sheet_embs, audio_embs, losses = [], [], []

    for batch in tqdm(loader, desc="  eval ", ncols=80, leave=False):
        se, ae = model(
            batch["sheet_seq"].to(device),
            batch["sheet_len"],
            batch["spec_seq"].to(device),
            batch["spec_len"],
        )
        losses.append(loss_fn(se, ae).item())
        sheet_embs.append(se.cpu())
        audio_embs.append(ae.cpu())

    x = torch.cat(sheet_embs).numpy()   # [N, emb_dim]
    y = torch.cat(audio_embs).numpy()   # [N, emb_dim]

    s2a = compute_retrieval_metrics(x, y)   # sheet queries → audio database
    a2s = compute_retrieval_metrics(y, x)   # audio queries → sheet database
    return float(np.mean(losses)), s2a, a2s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- Dataloaders -------------------------------------------------------
    manifest = Path(args.split_manifest)
    if not manifest.exists():
        print(f"[ERROR] Split manifest not found: {manifest}")
        print("Create it first with:")
        print("  from dataset.dataloaders import create_and_save_piece_split_manifest")
        print(f"  create_and_save_piece_split_manifest('{args.processed_root}')")
        sys.exit(1)

    print("Building dataloaders...")
    if args.use_group_dataset:
        train_sheet_tf = build_sheet_train_transform(
            translation=args.sheet_translation,
            scale_range=(args.sheet_scale_low, args.sheet_scale_high),
        )
        datasets, loaders, skipped = create_passage_group_split_dataloaders(
            processed_root=args.processed_root,
            split_manifest_path=str(manifest),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last_train=True,
            train_synths=_parse_csv_str(args.train_synths),
            train_tempo_range=_parse_csv_int_pair(args.train_tempo_range),
            eval_synths=_parse_csv_str(args.eval_synths),
            eval_tempo_range=_parse_csv_int_pair(args.eval_tempo_range),
            train_sheet_transform=train_sheet_tf,
        )
        print(f"  using PassageGroupDataset (atomic (piece, system); variant sampled per epoch)")
        for split, ds in datasets.items():
            print(f"    {split:<5}: {len(ds):>6} groups, {ds.num_variants_total:>7} total variants")
    else:
        datasets, loaders, skipped = create_passage_sequence_split_dataloaders(
            processed_root=args.processed_root,
            split_manifest_path=str(manifest),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last_train=True,
        )
    unit = "groups" if args.use_group_dataset else "passages"
    print(f"  train : {len(datasets['train']):>6} {unit}")
    print(f"  val   : {len(datasets['val']):>6} {unit}")
    print(f"  test  : {len(datasets['test']):>6} {unit}")
    for split, skip_list in skipped.items():
        if skip_list:
            print(f"  [warn] skipped {len(skip_list)} jobs in {split}")

    # Replace default train/val samplers with fixed-size random samplers (mirrors
    # the original repo's subsampling, which prevents multi-hour epochs).
    # When n_train > dataset size, sample with replacement so the per-epoch
    # iteration count stays constant; each group is then seen multiple times
    # per epoch with different audio variants — the desired augmentation effect.
    from torch.utils.data import DataLoader as _DL
    from dataset.dataloaders import passage_sequence_collate_fn

    n_train_total = len(datasets["train"])
    if args.n_train is not None:
        use_replacement = args.n_train > n_train_total
        sampler = RandomSampler(datasets["train"], replacement=use_replacement,
                                num_samples=args.n_train)
        loaders["train"] = _DL(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            collate_fn=passage_sequence_collate_fn,
        )
        ratio_str = (
            f"{args.n_train / n_train_total * 100:.1f}% of total"
            if not use_replacement
            else f"~{args.n_train / n_train_total:.1f}x oversample"
        )
        print(f"  Sampling train: {args.n_train} {unit}/epoch ({ratio_str})")

    n_val_total = len(datasets["val"])
    if args.n_val is not None and args.n_val < n_val_total:
        sampler = RandomSampler(datasets["val"], replacement=False,
                                num_samples=args.n_val)
        loaders["val"] = _DL(
            datasets["val"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=passage_sequence_collate_fn,
        )
        print(f"  Subsampling val   to {args.n_val} passages/epoch "
              f"({args.n_val / n_val_total * 100:.1f}% of total)")

    # ---- Model -------------------------------------------------------------
    model = CrossModalEncoder(
        snippet_emb_dim=args.snippet_emb_dim,
        rnn_hidden=args.rnn_hidden,
        emb_dim=args.emb_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Embedding dim:    {args.emb_dim}")
    print(f"Batch size:       {args.batch_size}  (effective: {args.batch_size * args.grad_accum})\n")

    # ---- Loss, optimiser, scheduler ----------------------------------------
    loss_fn = partial(triplet_loss, margin=args.loss_margin)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.patience, min_lr=2e-6
    )

    scaler = torch.amp.GradScaler("cuda") if (args.mixed_precision and device.type == "cuda") else None
    if scaler is not None:
        print("Mixed precision (AMP) enabled.")

    # ---- Checkpoint path ---------------------------------------------------
    exp_dir = Path(args.exp_root)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / f"model_emb{args.emb_dim}.pt"

    best_val_mrr = -1.0
    no_improve_count = 0
    start_epoch = 1
    early_stop_patience = args.patience * 2   # stop after 2x LR scheduler patience

    if args.resume:
        if not ckpt_path.exists():
            print(f"[ERROR] --resume specified but no checkpoint found at {ckpt_path}")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_mrr = ckpt["val_s2a"]["mrr"]
        no_improve_count = ckpt.get("no_improve_count", 0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  (best S2A MRR={best_val_mrr:.4f})\n")

    print(f"{'Epoch':>5}  {'tr_loss':>8}  {'va_loss':>8}  "
          f"{'S2A MRR':>8}  {'R@1':>6}  {'R@10':>6}  "
          f"{'A2S MRR':>8}  {'med_rk':>7}  {'LR':>8}  {'time':>5}")
    print("-" * 95)

    for epoch in range(start_epoch, args.n_epochs + 1):
        t0 = time.monotonic()

        tr_loss = train_epoch(model, loaders["train"], loss_fn, optimizer, device, scaler=scaler, accum_steps=args.grad_accum)
        va_loss, va_s2a, va_a2s = eval_epoch(model, loaders["val"], loss_fn, device)

        # Primary metric: S2A MRR (paper: S2A consistently outperforms A2S)
        val_mrr = va_s2a["mrr"]
        scheduler.step(val_mrr)

        elapsed = time.monotonic() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>5}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
            f"{va_s2a['mrr']:>8.4f}  {va_s2a['r@1']:>6.1f}  {va_s2a['r@10']:>6.1f}  "
            f"{va_a2s['mrr']:>8.4f}  {va_s2a['med_rank']:>7.1f}  "
            f"{lr_now:>8.2e}  {elapsed:>4.0f}s"
        )

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "no_improve_count": 0,
                "val_s2a": va_s2a,
                "val_a2s": va_a2s,
                "args": vars(args),
            }, ckpt_path)
            print(f"       -> checkpoint saved  (S2A MRR={val_mrr:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"({no_improve_count} epochs without improvement).")
                break

    print(f"\nBest val S2A MRR : {best_val_mrr:.4f}")
    print(f"Checkpoint saved : {ckpt_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the cross-modal passage retrieval model"
    )

    # Paths
    parser.add_argument("--processed_root", default="data/processed_pairs",
                        help="Root directory of preprocessed data")
    parser.add_argument("--split_manifest",
                        default="data/processed_pairs/manifests/piece_split_latest.json",
                        help="Piece-disjoint split manifest (JSON)")
    parser.add_argument("--exp_root", default="experiments",
                        help="Directory for saving checkpoints")

    # Model
    parser.add_argument("--snippet_emb_dim", type=int, default=32,
                        help="CNN output dimension per snippet")
    parser.add_argument("--rnn_hidden",      type=int, default=128,
                        help="GRU hidden size (128 per paper)")
    parser.add_argument("--emb_dim",         type=int, default=64,
                        help="Final passage embedding dimension (64 = paper best)")

    # Training
    parser.add_argument("--n_epochs",     type=int,   default=180)
    parser.add_argument("--batch_size",   type=int,   default=64,
                        help="Passages per step. Paper uses 64. Lower if you hit OOM on small GPUs.")
    parser.add_argument("--num_workers",  type=int,   default=4,
                        help="DataLoader workers. Use 0 on Windows; 4-8 on Linux.")
    parser.add_argument("--lr",           type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss_margin",  type=float, default=0.3,
                        help="Triplet loss margin (alpha in equation 1)")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="LR scheduler patience; early stop at 2x this value")
    parser.add_argument("--seed",         type=int,   default=123)
    parser.add_argument("--n_train",      type=int,   default=16384,
                        help="Passages sampled per training epoch (None = use all). "
                             "Original repo uses 16384.")
    parser.add_argument("--n_val",        type=int,   default=4096,
                        help="Passages sampled for val eval per epoch (None = use all). "
                             "Keeps epoch time manageable without losing signal.")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable AMP (torch.cuda.amp) for faster training on CUDA.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the existing checkpoint in --exp_root.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum). "
                             "Use >1 only when the GPU forces you below batch_size=64.")

    # Dataset / augmentation
    parser.add_argument("--use_group_dataset", action="store_true", default=True,
                        help="Use PassageGroupDataset (atomic (piece, system); audio variant "
                             "sampled per epoch). Recommended; matches paper semantics.")
    parser.add_argument("--no_group_dataset", dest="use_group_dataset", action="store_false",
                        help="Fall back to AllPassagesDataset (old behaviour; one item "
                             "per (piece, performance, system) triple).")
    parser.add_argument("--train_synths", type=str,
                        default=",".join(_PAPER_TRAIN_SYNTHS),
                        help="Comma-separated synth allowlist for the train split "
                             "(paper full_aug: acoustic_piano_imis_1,ElectricPiano,YamahaGrandPiano). "
                             "Empty string disables filtering.")
    parser.add_argument("--train_tempo_range", type=str, default="900,1100",
                        help="Comma-separated 'lo,hi' tempo bounds in 1000ths for train "
                             "(paper full_aug: 900,1100 = 0.9x..1.1x). Empty disables.")
    parser.add_argument("--eval_synths", type=str,
                        default=",".join(_PAPER_EVAL_SYNTHS),
                        help="Comma-separated synth allowlist for val/test "
                             "(paper test_aug: grand-piano-YDP-20160804). Empty disables.")
    parser.add_argument("--eval_tempo_range", type=str, default="1000,1000",
                        help="Comma-separated 'lo,hi' tempo bounds for val/test "
                             "(paper test_aug: 1000,1000 = original tempo only).")
    parser.add_argument("--sheet_translation", type=int, default=5,
                        help="Random vertical shift of sheet snippets in pixels (paper: 5).")
    parser.add_argument("--sheet_scale_low", type=float, default=0.95,
                        help="Random sheet scaling lower bound (paper: 0.95).")
    parser.add_argument("--sheet_scale_high", type=float, default=1.05,
                        help="Random sheet scaling upper bound (paper: 1.05).")

    main(parser.parse_args())

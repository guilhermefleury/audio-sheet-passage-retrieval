import argparse
import json
from datetime import datetime
from pathlib import Path

from audio import get_performance_list
from pair_export import save_normalized_pointer_dataset


def discover_piece_dirs(msmd_root):
    piece_dirs = []
    for entry in sorted(msmd_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "performances").exists() and (entry / "scores").exists():
            piece_dirs.append(entry)
    return piece_dirs


def run_batch(
    msmd_root,
    output_root,
    overwrite=False,
    max_pieces=None,
    max_performances_per_piece=None,
    allow_equal_split_fallback=False,
    sheet_stride=90,
    spec_window_frames=20,
    spec_stride_frames=10,
):
    piece_dirs = discover_piece_dirs(msmd_root)
    if max_pieces is not None:
        piece_dirs = piece_dirs[:max_pieces]

    started_at = datetime.now().isoformat(timespec="seconds")

    summary = {
        "started_at": started_at,
        "msmd_root": str(msmd_root),
        "output_root": str(output_root),
        "overwrite": overwrite,
        "piece_count": len(piece_dirs),
        "sheet_stride": sheet_stride,
        "spec_window_frames": spec_window_frames,
        "spec_stride_frames": spec_stride_frames,
        "processed": [],
        "failures": [],
    }

    total_jobs = 0
    for piece_dir in piece_dirs:
        perfs = get_performance_list(str(piece_dir))
        if max_performances_per_piece is not None:
            perfs = perfs[:max_performances_per_piece]
        total_jobs += len(perfs)

    print(f"Found {len(piece_dirs)} pieces and {total_jobs} performance jobs.")

    job_idx = 0
    for piece_dir in piece_dirs:
        piece_name = piece_dir.name
        performances = get_performance_list(str(piece_dir))
        if max_performances_per_piece is not None:
            performances = performances[:max_performances_per_piece]

        if not performances:
            summary["failures"].append(
                {
                    "piece": piece_name,
                    "performance": None,
                    "error": "No performances found",
                }
            )
            continue

        for perf in performances:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Processing {piece_name} | {perf}")
            try:
                info = save_normalized_pointer_dataset(
                    file_path=str(piece_dir),
                    performance_name=perf,
                    output_root=str(output_root),
                    sheet_stride=sheet_stride,
                    spec_window_frames=spec_window_frames,
                    spec_stride_frames=spec_stride_frames,
                    overwrite=overwrite,
                    allow_equal_split_fallback=allow_equal_split_fallback,
                )
                summary["processed"].append(
                    {
                        "piece": piece_name,
                        "performance": perf,
                        "num_pairs": info["num_pairs"],
                        "sheet_npz": info["sheet_npz"],
                        "spec_npz": info["spec_npz"],
                        "pairs_npz": info["pairs_npz"],
                    }
                )
            except Exception as exc:
                summary["failures"].append(
                    {
                        "piece": piece_name,
                        "performance": perf,
                        "error": str(exc),
                    }
                )

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["ok_jobs"] = len(summary["processed"])
    summary["failed_jobs"] = len(summary["failures"])

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = manifests_dir / f"run_{stamp}.json"
    latest_file = manifests_dir / "latest_run.json"

    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBatch finished")
    print(f"Successful jobs: {summary['ok_jobs']}")
    print(f"Failed jobs: {summary['failed_jobs']}")
    print(f"Run manifest: {run_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Process all MSMD pieces into pointer-based pair datasets.")
    parser.add_argument("--msmd-root", default="../data/msmd", help="Path to MSMD root folder")
    parser.add_argument("--output-root", default="../data/processed_pairs", help="Output root folder")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing files")
    parser.add_argument("--max-pieces", type=int, default=None, help="Limit number of pieces (for testing)")
    parser.add_argument(
        "--max-performances-per-piece",
        type=int,
        default=None,
        help="Limit performances per piece (for testing)",
    )
    parser.add_argument(
        "--allow-equal-split-fallback",
        action="store_true",
        help="Fallback to equal system-time split if timing-based slicing fails",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help=(
            "Fração de sobreposição entre snippets consecutivos (padrão 0.5 = 50%%). "
            "Use 0.1 para reproduzir a configuração de stride do trabalho de referência."
        ),
    )

    args = parser.parse_args()

    msmd_root = Path(args.msmd_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not msmd_root.exists():
        raise FileNotFoundError(f"MSMD root does not exist: {msmd_root}")

    if not (0.0 <= args.overlap < 1.0):
        raise ValueError(f"--overlap deve estar em [0, 1). Recebido: {args.overlap}")

    sheet_window = 180
    spec_window = 20
    sheet_stride = int(round(sheet_window * (1.0 - args.overlap)))
    spec_stride_frames = int(round(spec_window * (1.0 - args.overlap)))

    print(
        f"Sobreposição: {args.overlap:.0%} | "
        f"sheet_stride={sheet_stride} (janela {sheet_window}) | "
        f"spec_stride_frames={spec_stride_frames} (janela {spec_window})"
    )

    run_batch(
        msmd_root=msmd_root,
        output_root=output_root,
        overwrite=args.overwrite,
        max_pieces=args.max_pieces,
        max_performances_per_piece=args.max_performances_per_piece,
        allow_equal_split_fallback=args.allow_equal_split_fallback,
        sheet_stride=sheet_stride,
        spec_window_frames=spec_window,
        spec_stride_frames=spec_stride_frames,
    )


if __name__ == "__main__":
    main()

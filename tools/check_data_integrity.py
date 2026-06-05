"""
Recursively check files in a directory tree for corruption.

For each file the script applies the strictest parser available for its
extension. A file is reported BAD if any parse/read/decode step raises.

  .npz, .npy   numpy load + access every array
  .png .jpg .jpeg .bmp .tif .tiff   PIL verify() + load() (full decode)
  .json        json.load
  .wav .flac .ogg .aiff   soundfile read (if soundfile installed)
  anything else   full stream-read (catches disk / truncation errors)

Usage
-----
    python check_data_integrity.py D:/path/to/processed_pairs
    python check_data_integrity.py D:/path/to/processed_pairs --workers 4
    python check_data_integrity.py D:/path/to/processed_pairs --log bad.txt

Exit code is 0 if every file passes, 1 otherwise.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

CHUNK = 1 << 20  # 1 MB stream-read buffer
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def check_file(path):
    try:
        size = path.stat().st_size
    except OSError as e:
        return path, f"stat: {e}", 0

    ext = path.suffix.lower()
    try:
        if ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                for name in z.files:
                    _ = z[name].shape
        elif ext == ".npy":
            _ = np.load(path, allow_pickle=False).shape
        elif ext in IMG_EXT:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
        elif ext == ".json":
            with open(path, "rb") as f:
                json.load(f)
        elif ext in AUDIO_EXT and HAS_SF:
            with sf.SoundFile(str(path)) as f:
                f.read(dtype="float32")
        else:
            h = hashlib.md5()
            with open(path, "rb") as f:
                while True:
                    buf = f.read(CHUNK)
                    if not buf:
                        break
                    h.update(buf)
    except Exception as e:
        return path, f"{type(e).__name__}: {e}", size
    return path, None, size


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root", type=Path, help="Directory to scan recursively")
    ap.add_argument("--workers", "-j", type=int, default=8,
                    help="Parallel workers (default: 8). Lower to 2-4 on HDDs.")
    ap.add_argument("--log", type=Path,
                    help="Write tab-separated list of bad files to this path")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress progress bar; only summary + errors")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"not a directory: {args.root}")

    print(f"Scanning {args.root} ...", flush=True)
    files = [p for p in args.root.rglob("*") if p.is_file()]
    n_total = len(files)
    print(f"Found {n_total} files. Checking with {args.workers} workers ...",
          flush=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    bad = []
    n_ok = 0
    bytes_seen = 0
    t0 = time.time()
    pbar = tqdm(total=n_total, unit="file") if tqdm and not args.quiet else None

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed({ex.submit(check_file, p): p for p in files}):
            path, err, size = fut.result()
            bytes_seen += size
            if err:
                bad.append((path, err))
                msg = f"BAD  {path}  --  {err}"
                if pbar is not None:
                    tqdm.write(msg)
                else:
                    print(msg, flush=True)
            else:
                n_ok += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"{n_ok} ok / {len(bad)} bad / {bytes_seen/1e9:.2f} GB"
                )

    if pbar is not None:
        pbar.close()

    dt = time.time() - t0
    print()
    print(f"Checked {n_total} files in {dt:.1f}s "
          f"({bytes_seen / 1e9:.2f} GB, {bytes_seen / max(dt, 1e-9) / 1e6:.1f} MB/s).")
    print(f"  OK:  {n_ok}")
    print(f"  BAD: {len(bad)}")

    if bad and args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            for path, err in bad:
                f.write(f"{path}\t{err}\n")
        print(f"Wrote {len(bad)} bad-file entries to {args.log}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

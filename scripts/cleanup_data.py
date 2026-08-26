"""Disk GC for collected dagger datasets.

Groups data/dagger_*.npz by board config and deletes round files beyond
a retention window. Safe by design:
  - never touches base datasets (bc_*.npz) - they anchor every fit
  - never touches the newest KEEP_ROUNDS rounds per config
  - never touches files modified in the last MIN_AGE_HOURS hours
  - dry-run by default; pass --apply to actually delete
Manifest sidecars (.jsonl) are deleted alongside their npz.

Usage:
    python scripts/cleanup_data.py                 # dry-run report
    python scripts/cleanup_data.py --apply --keep-rounds 2
"""
import argparse
import glob
import os
import re
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")

ROUND_RE = re.compile(r"^(dagger_.+?)_r(\d+[ab]?)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run)")
    ap.add_argument("--keep-rounds", type=int, default=3,
                    help="newest N rounds per config to always keep")
    ap.add_argument("--min-age-hours", type=float, default=24,
                    help="never delete files newer than this")
    args = ap.parse_args()

    now = time.time()
    groups = {}
    for path in glob.glob(os.path.join(DATA_DIR, "dagger_*.npz")):
        stem = os.path.basename(path)[:-4]
        m = ROUND_RE.match(stem)
        if not m:
            continue
        cfg = m.group(1)
        age_h = (now - os.path.getmtime(path)) / 3600
        groups.setdefault(cfg, []).append((path, age_h))

    total_free = 0
    for cfg, files in sorted(groups.items()):
        files.sort(key=lambda p: os.path.getmtime(p[0]), reverse=True)
        for rank, (path, age_h) in enumerate(files):
            protected = (rank < args.keep_rounds
                         or age_h < args.min_age_hours)
            if not protected:
                size = os.path.getsize(path)
                total_free += size
                manifest = path[:-4] + ".npz.manifest.jsonl"
                extra = f" + {os.path.basename(manifest)}" \
                    if os.path.isfile(manifest) else ""
                print(f"{'DELETE' if args.apply else 'WOULD DELETE'} "
                      f"{path} ({size/1e6:.1f} MB){extra}")
                if args.apply:
                    os.remove(path)
                    if os.path.isfile(manifest):
                        os.remove(manifest)
            else:
                reason = "recent" if age_h < args.min_age_hours \
                    else "kept-round"
                print(f"keep ({reason}) {os.path.basename(path)}")
        if not files:
            continue

    print(f"\n{'deleted' if args.apply else 'would free'}: "
          f"{total_free/1e6:.1f} MB total")


if __name__ == "__main__":
    main()

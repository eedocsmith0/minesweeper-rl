"""One-off converter: rewrite legacy float16 observation npz files as
lossless uint8 (~50% smaller). Originals are preserved as *.f16bak.

Usage:
    python scripts/convert_data_u8.py [--data-dir data]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from agents.bc_pretrain import encode_obs_u8  # noqa: E402


def payload_len_hint(chk):
    return len(chk["acts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    total_before = total_after = 0
    converted = skipped = failed = 0
    import gc
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.npz")),
                       key=os.path.getsize):
        try:
            d = np.load(path)
            if "obs" not in d.files or d["obs"].dtype == np.uint8:
                skipped += 1
                continue
            enc = encode_obs_u8(d["obs"])          # f16 in, u8 out
            payload = {
                "obs": enc,
                "masks": np.asarray(d["masks"], np.int8),
                "acts": np.asarray(d["acts"], np.int64),
                "rets": (np.asarray(d["rets"], np.float32)
                         if "rets" in d.files else None),
                "certain": (np.asarray(d["certain"], np.int8)
                            if "certain" in d.files else None),
            }
            del d                                   # free decompressed src
            gc.collect()

            payload = {k: v for k, v in payload.items() if v is not None}
            tmp = path + ".u8tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **payload)
            del payload, enc
            gc.collect()

            # verify before touching the original
            chk = np.load(tmp)
            assert chk["obs"].dtype == np.uint8
            assert len(chk["acts"]) == payload_len_hint(chk)

            os.replace(path, path + ".f16bak")
            os.replace(tmp, path)
            before = os.path.getsize(path + ".f16bak")
            after = os.path.getsize(path)
            total_before += before
            total_after += after
            converted += 1
            print(f"{os.path.basename(path)}: "
                  f"{before/1e6:.1f} -> {after/1e6:.1f} MB", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAILED {path}: {e}")
        gc.collect()

    print(f"\nconverted={converted} skipped(already-u8/no-obs)={skipped} "
          f"failed={failed}")
    print(f"disk: {total_before/1e6:.0f} MB -> {total_after/1e6:.0f} MB")
    print("originals kept as *.f16bak - delete them once satisfied")


if __name__ == "__main__":
    main()

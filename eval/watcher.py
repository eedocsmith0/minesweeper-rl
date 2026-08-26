"""Auto-eval watcher: benchmark every new checkpoint that appears in models/.

Designed to run alongside a separate training session. Uses only filesystem
observation - no coordination with the trainer required.

Checkpoint config resolution (first match wins):
1. `_<W>x<H>_<M>m` pattern anywhere in the filename
2. sidecar JSON: <checkpoint>.json with {"width":..,"height":..,"mines":..}

Results are logged to runs/auto-eval/metrics.jsonl which the dashboard
renders like any other run.

Usage:
    python eval/watcher.py [--models-dir models] [--games 500]
                           [--interval 30] [--seed0 900000]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.metrics import MetricLogger
from env.minesweeper_env import MinesweeperEnv

PATTERN = re.compile(r"(\d+)x(\d+)_(\d+)m")


def resolve_config(path):
    stem = os.path.basename(path)
    m = PATTERN.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    sidecar = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(sidecar):
        try:
            cfg = json.load(open(sidecar))
            return cfg["width"], cfg["height"], cfg["mines"]
        except Exception:
            pass
    return None


def benchmark(model_path, width, height, num_mines, n_games, seed0,
              device="cpu", board_logger=None, tta=False):
    from sb3_contrib import MaskablePPO
    if tta:
        from agents.bc_pretrain import tta_pick

    model = MaskablePPO.load(model_path, device=device)
    wins = 0
    moves_total = 0
    opened_total = 0
    safe_total = 0
    max_moves = 3 * width * height
    board_every = max(1, n_games // 10)
    for seed in range(seed0, seed0 + n_games):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed)
        obs, _ = env.reset(seed=seed)
        done = False
        moves = 0
        info = {"remaining_safe": width * height - num_mines}
        while not done and moves < max_moves:
            mask = env.action_mask()
            if not mask.any():
                break
            if tta and width == height:
                action = tta_pick(model, obs.astype(np.float32), mask,
                                  width, height, device)
                action = int(action)
            else:
                action, _ = model.predict(obs, action_masks=mask,
                                          deterministic=True)
                action = int(action)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            moves += 1
        won = info["remaining_safe"] == 0
        wins += won
        moves_total += moves
        opened_total += int(env.revealed.sum())
        safe_total += width * height - num_mines
        if board_logger and (won or seed % board_every == 0):
            board_logger.log_board(env, "win" if won else "loss", moves,
                                   extra={"mines": num_mines,
                                          "model": os.path.basename(model_path)})
    return {
        "win_rate": wins / n_games,
        "avg_moves": moves_total / n_games,
        "avg_opened": opened_total / max(1, safe_total),
    }


def scan_once(models_dir, seen, logger, n_games, seed0, tta=False,
              device="cpu", max_failures=3):
    fails = scan_once._fails
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".zip"):
            continue
        path = os.path.join(models_dir, fname)
        key = (fname, os.path.getmtime(path))
        if key in seen or fname.endswith("replay_buffer.zip"):
            continue
        # wait until file stops growing to avoid reading partial saves
        try:
            size1 = os.path.getsize(path)
            time.sleep(2)
            if os.path.getsize(path) != size1:
                continue
        except OSError:
            continue

        cfg = resolve_config(path)
        if cfg is None:
            print(f"[watcher] skipping {fname}: no WxH_Mm pattern "
                  f"or sidecar json", flush=True)
            seen.add(key)
            continue
        width, height, num_mines = cfg
        print(f"[watcher] evaluating {fname} on "
              f"{width}x{height}/{num_mines}m ({n_games} games)...",
              flush=True)
        try:
            res = benchmark(path, width, height, num_mines, n_games, seed0,
                            device=device, board_logger=logger, tta=tta)
            logger.log(phase="eval", win_rate=res["win_rate"],
                       avg_moves=res["avg_moves"],
                       avg_opened=res["avg_opened"], games=n_games,
                       model=fname, width=width, height=height,
                       mines=num_mines)
            print(f"[watcher] {fname}: win_rate={res['win_rate']:.3f} "
                  f"avg_moves={res['avg_moves']:.1f}", flush=True)
            fails.pop(fname, None)
        except Exception as e:
            # transient failures (partial zip, device hiccup) are retried
            # next cycle; give up only after repeated failures
            fails[fname] = fails.get(fname, 0) + 1
            print(f"[watcher] FAILED {fname} ({e}); attempt "
                  f"{fails[fname]}/{max_failures}", flush=True)
            if fails[fname] >= max_failures:
                print(f"[watcher] giving up on {fname}", flush=True)
                seen.add(key)
                fails.pop(fname, None)
        else:
            seen.add(key)


scan_once._fails = {}  # fname -> consecutive failure count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--run", default="auto-eval")
    ap.add_argument("--nice", type=int, default=10,
                    help="yield CPU to desktop apps when they need it")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="symmetry-averaged inference (matches the "
                         "training session's published numbers)")
    ap.add_argument("--device", default="cpu",
                    help="device for model inference during benchmarks")
    args = ap.parse_args()
    try:
        os.nice(args.nice)
    except PermissionError:
        print(f"[watcher] could not apply nice={args.nice}")
    print(f"[watcher] tta={'on' if args.tta else 'off'} "
          f"device={args.device}", flush=True)

    inbox_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coordination", "to-training.jsonl")
    started = time.time()
    last_seen_ts = started
    last_inbox_check = 0.0

    def poll_inbox():
        """Print new messages once each; a {"action":"stop"} exits."""
        nonlocal last_inbox_check, last_seen_ts
        if time.time() - last_inbox_check < 60:
            return
        last_inbox_check = time.time()
        try:
            with open(inbox_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("ts", 0) > last_seen_ts:
                        last_seen_ts = r["ts"]
                        print(f"[inbox {time.strftime('%H:%M:%S')}] "
                              f"{r.get('from')}/{r.get('type')}: "
                              f"{str(r.get('body',''))[:200]}", flush=True)
                        if r.get("action") == "stop":
                            print("[watcher] stop requested - exiting",
                                  flush=True)
                            sys.exit(0)
        except OSError:
            pass

    logger = MetricLogger(args.run)
    seen = set()
    # treat everything present at startup as already evaluated
    for fname in os.listdir(args.models_dir):
        if fname.endswith(".zip"):
            p = os.path.join(args.models_dir, fname)
            seen.add((fname, os.path.getmtime(p)))
    print(f"[watcher] watching {args.models_dir} every "
          f"{args.interval}s; {len(seen)} existing checkpoints ignored",
          flush=True)
    while True:
        try:
            poll_inbox()
            scan_once(args.models_dir, seen, logger, args.games, args.seed0,
                      tta=args.tta, device=args.device)
        except Exception as e:
            print(f"[watcher] scan error: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

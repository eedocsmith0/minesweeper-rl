"""Quick TTA win-rate probe for a checkpoint (smaller N than full eval)."""
import argparse
import os
import sys

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.bc_pretrain import tta_pick
from env.minesweeper_env import MinesweeperEnv


@th.no_grad()
def tta_probs(model, obs_np, mask_np, width, height, device):
    """Symmetry-averaged action probability vector for one state."""
    from agents.bc_pretrain import dihedral_transform, _cell_perm, \
        torch_where

    n = width * height
    obs = th.as_tensor(obs_np, dtype=th.float32, device=device).unsqueeze(0)
    mask = th.as_tensor(mask_np.astype(np.bool_), device=device).unsqueeze(0)
    src = th.arange(2 * n, device=device)
    acc = th.zeros(2 * n, device=device)
    views = range(8) if width == height else [0]
    for k in views:
        o, m, _, _ = dihedral_transform(
            obs, mask, th.zeros(1, dtype=th.long, device=device), k,
            height, width)
        dist = model.policy.get_distribution(o, action_masks=m)
        p_new = dist.distribution.probs.squeeze(0)
        if width != height:
            return p_new
        perm = _cell_perm(k, height, width, device)
        flat = torch_where(src >= n, src - n, src)
        new_flat = perm[flat]
        new_idx = torch_where(src >= n, new_flat + n, new_flat)
        acc = acc + p_new[new_idx]
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, nargs="+",
                   help="one or more checkpoints; multiple are ensembled")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--mines", type=int, default=40)
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=998000)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    models = []
    for path in args.model:
        m = MaskablePPO.load(path, device=args.device)
        m.policy.eval()
        models.append(m)
    print(f"ensembling {len(models)} model(s) x 8 views")
    wins = 0
    for g in range(args.games):
        env = MinesweeperEnv(width=args.width, height=args.height,
                             num_mines=args.mines,
                             seed=args.seed0 + g)
        obs, _ = env.reset(seed=args.seed0 + g)
        done = False
        moves = 0
        while not done and moves < 3 * args.width * args.height:
            mask = env.action_mask()
            if not mask.any():
                break
            acc = None
            for m in models:
                pr = tta_probs(m, obs, mask, args.width, args.height,
                              args.device)
                acc = pr if acc is None else acc + pr
            a = int(acc.argmax())
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            moves += 1
        wins += info["remaining_safe"] == 0
    print(f"TTA win rate {wins / args.games:.3f} over {args.games} games "
          f"({args.width}x{args.height}/{args.mines}m, "
          f"seeds {args.seed0}+)")


if __name__ == "__main__":
    main()

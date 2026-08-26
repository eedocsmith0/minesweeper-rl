"""Final evaluation: RL agent vs rule-based baseline on identical boards.

Examples:
  PYTHONPATH=. python eval/evaluate.py --model models/bc_9x9_sym_r2.zip \
      --width 9 --height 9 --mines 10 --games 2000
  PYTHONPATH=. python eval/evaluate.py --baseline --width 9 --height 9 \
      --mines 10 --games 2000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.minesweeper_env import MinesweeperEnv
from agents.rule_solver import RuleSolver


def play_with_model(model, env, obs, max_moves, tta=False):
    if tta:
        from agents.bc_pretrain import tta_pick

    done = False
    moves = 0
    info = {"remaining_safe": int((~env.mines & ~env.revealed).sum())}
    while not done and moves < max_moves:
        mask = env.action_mask()
        if not mask.any():
            break
        if tta and env.width == env.height:
            action = tta_pick(model, obs.astype(np.float32), mask,
                              env.width, env.height, "cuda")
            action = int(action)
        else:
            action, _ = model.predict(obs, action_masks=mask,
                                      deterministic=True)
            action = int(action)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        moves += 1
    won = info["remaining_safe"] == 0
    return won, moves


def play_with_solver(env, solver):
    done = False
    moves = 0
    max_moves = 3 * env.height * env.width
    info = {"remaining_safe": int((~env.mines & ~env.revealed).sum())}
    while not done and moves < max_moves:
        mask = env.action_mask()
        if not mask.any():
            break
        action = solver.next_action()
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        moves += 1
    won = info["remaining_safe"] == 0
    return won, moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=800_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tta", action="store_true",
                   help="symmetry-averaged inference (square boards)")
    args = p.parse_args()

    assert args.model or args.baseline, "need --model or --baseline"
    if args.model:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.model, device=args.device)

    wins, moves_total, opened_total, safe_total = 0, 0, 0, 0
    for seed in range(args.seed0, args.seed0 + args.games):
        env = MinesweeperEnv(width=args.width, height=args.height,
                             num_mines=args.mines, seed=seed)
        obs, _ = env.reset(seed=seed)
        if args.model:
            won, moves = play_with_model(model, env, obs,
                                         3 * args.width * args.height,
                                         tta=args.tta)
        else:
            won, moves = play_with_solver(env, RuleSolver(env))
        wins += won
        moves_total += moves
        opened_total += int(env.revealed.sum())
        safe_total += args.width * args.height - args.mines

    print(f"agent={'rule-solver' if args.baseline else args.model}")
    print(f"board={args.width}x{args.height} mines={args.mines} "
          f"games={args.games} seeds=[{args.seed0},"
          f"{args.seed0 + args.games})")
    print(f"win rate:        {wins / args.games:.3f}")
    print(f"avg moves/game:  {moves_total / args.games:.1f}")
    print(f"avg safe cells opened: {opened_total / safe_total:.3f}")


if __name__ == "__main__":
    main()

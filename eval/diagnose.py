"""Diagnose solver behavior and agent failures on a given board size."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.minesweeper_env import MinesweeperEnv
from agents.rule_solver import RuleSolver


def diag_solver(width, height, num_mines, n_games, seed0):
    stats = dict(games=0, wins=0, moves=0, certain_reveals=0,
                 flag_moves=0, guesses=0, guess_deaths=0,
                 fallback_moves=0, max_component=0)
    for g in range(n_games):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed0 + g)
        env.reset(seed=seed0 + g)
        solver = RuleSolver(env)
        died_on_guess = False
        moves_this = 0
        while True:
            mask = env.action_mask()
            if not mask.any():
                break
            # instrument: what does the solver do and why?
            cons = solver._build_constraints()
            safes, mines = solver._reduce(cons)
            is_certain = bool(safes) or any(
                not env.flagged[r, c] for r, c in mines)
            if is_certain:
                if safes:
                    stats["certain_reveals"] += 1
                else:
                    stats["flag_moves"] += 1
            else:
                comps = []
                fresh = solver._build_constraints()
                adj = {}
                for unk, _ in fresh:
                    for cell in unk:
                        adj[cell] = None
                seen = set()
                biggest = 0
                for start in adj:
                    if start in seen:
                        continue
                    stack, comp = [start], set()
                    while stack:
                        cell = stack.pop()
                        if cell in comp:
                            continue
                        comp.add(cell)
                        seen.add(cell)
                        for unk, _cnt in fresh:
                            if cell in unk:
                                stack.extend(unk - comp)
                    biggest = max(biggest, len(comp))
                stats["max_component"] = max(stats["max_component"],
                                             biggest)
                if biggest > RuleSolver.MAX_ENUM_CELLS:
                    stats["fallback_moves"] += 1
                stats["guesses"] += 1

            action = solver.next_action()
            obs_prev_mine = env.mines[action % (height * width) // width,
                                      action % (height * width) % width]
            _, _, term, _, info = env.step(action)
            moves_this += 1
            if term and info["remaining_safe"] > 0:
                flat = action % (height * width)
                if not (action >= height * width):
                    r, c = divmod(flat, width)
                    if env.mines[r, c]:
                        died_on_guess = True
                break
        stats["games"] += 1
        stats["moves"] += moves_this
        stats["wins"] += info["remaining_safe"] == 0
        stats["guess_deaths"] += died_on_guess
    s = stats
    print(f"board {width}x{height}/{num_mines}: "
          f"win={s['wins']/s['games']:.3f} moves/game={s['moves']/s['games']:.0f}")
    print(f"  certain_reveals={s['certain_reveals']} flags={s['flag_moves']} "
          f"guesses={s['guesses']} ({100*s['guesses']/max(1,s['moves']):.1f}% of moves)")
    print(f"  guess_deaths={s['guess_deaths']} "
          f"({100*s['guess_deaths']/s['games']:.1f}% of games)")
    print(f"  fallback_moves={s['fallback_moves']} "
          f"max_component={s['max_component']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--mines", type=int, default=40)
    p.add_argument("--games", type=int, default=300)
    args = p.parse_args()
    diag_solver(args.width, args.height, args.mines, args.games, 700000)

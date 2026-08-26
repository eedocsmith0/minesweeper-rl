import math

import numpy as np

from env.minesweeper_env import MinesweeperEnv


class RuleSolver:
    """Constraint-based Minesweeper solver.

    Uses single-constraint and subset deductions to find guaranteed safe
    reveals and guaranteed mines. When stuck, computes per-cell mine
    probabilities via exact enumeration of constraint components. Large
    components are handled by a fast incremental-counter DFS with a node
    budget fallback, and exact components are conditioned on the global
    remaining mine count via binomial completion through interior cells.
    Ties are broken randomly (true 50-50 situations resolve randomly).
    """

    MAX_ENUM_CELLS = 64
    MAX_ENUM_NODES = 250_000

    def __init__(self, env):
        self.env = env
        self.h = env.height
        self.w = env.width
        self.rng = np.random.default_rng()

    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w:
                    yield rr, cc

    def _build_constraints(self):
        """Return list of (unknown_set, mine_count)."""
        env = self.env
        constraints = []
        for r in range(self.h):
            for c in range(self.w):
                if not env.revealed[r, c]:
                    continue
                unknown = set()
                flagged = 0
                for rr, cc in self._neighbors(r, c):
                    if env.flagged[rr, cc]:
                        flagged += 1
                    elif not env.revealed[rr, cc]:
                        unknown.add((rr, cc))
                count = int(env.counts[r, c]) - flagged
                if unknown:
                    constraints.append([unknown, count])
        return constraints

    MAX_CONSTRAINTS = 500
    MAX_PASSES = 20

    @classmethod
    def _reduce(cls, constraints):
        """Apply trivial + subset rules. Returns (safes, mines)."""
        safes, mines = set(), set()
        seen = set()  # signatures of constraints already processed
        changed = True
        passes = 0
        while changed and passes < cls.MAX_PASSES:
            changed = False
            passes += 1
            new_constraints = []
            for unk, cnt in constraints:
                removed_mines = len(unk & mines)
                unk = unk - mines
                cnt -= removed_mines
                if not unk:
                    continue
                if cnt <= 0:
                    safes |= unk
                    changed = True
                elif cnt >= len(unk):
                    mines |= unk
                    changed = True
                else:
                    sig = frozenset(unk)
                    if sig not in seen:
                        seen.add(sig)
                        new_constraints.append([unk, cnt])
            constraints = new_constraints
            # subset rule: derive residual B - A only when it shrinks the
            # constraint (or is trivial), to keep the fixpoint bounded
            out = []
            n = len(constraints)
            for i in range(n):
                a_unk, a_cnt = constraints[i]
                for j in range(i + 1, n):
                    b_unk, b_cnt = constraints[j]
                    if b_unk < a_unk:
                        diff, d_cnt = a_unk - b_unk, a_cnt - b_cnt
                    elif a_unk < b_unk:
                        diff, d_cnt = b_unk - a_unk, b_cnt - a_cnt
                    else:
                        continue
                    if not diff or d_cnt < 0:
                        continue
                    trivial = d_cnt == 0 or d_cnt == len(diff)
                    shrinking = len(diff) < min(len(a_unk), len(b_unk))
                    if not (trivial or shrinking):
                        continue
                    sig = frozenset(diff)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    out.append([diff, d_cnt])
                    changed = True
            constraints += out
            if len(constraints) > cls.MAX_CONSTRAINTS:
                break
        return safes, mines

    class _BudgetExceeded(Exception):
        pass

    def _split_components(self, constraints):
        """Partition constraints into connected components.

        Returns list of (cells_most_constrained_first, cons) where cons is
        a list of (frozenset_of_cell_indices, count). Preserves the
        most-constrained-first ordering the enumerator relies on.
        """
        cell_cons = {}
        for unk, cnt in constraints:
            for cell in unk:
                cell_cons.setdefault(cell, []).append((unk, cnt))
        constraint_count = {}
        for unk, _cnt in constraints:
            for cell in unk:
                constraint_count[cell] = constraint_count.get(cell, 0) + 1
        seen = set()
        comps = []
        for start in cell_cons:
            if start in seen:
                continue
            comp_cells, comp_cons, comp_sig = set(), [], set()
            stack = [start]
            while stack:
                cell = stack.pop()
                if cell in comp_cells:
                    continue
                comp_cells.add(cell)
                seen.add(cell)
                for unk, cnt in cell_cons[cell]:
                    sig = id(unk)
                    if sig not in comp_sig:
                        comp_sig.add(sig)
                        comp_cons.append((unk, cnt))
                        stack.extend(unk - comp_cells)
            cells = sorted(comp_cells,
                           key=lambda c: (-constraint_count.get(c, 0), c))
            index = {cell: i for i, cell in enumerate(cells)}
            cons = [(frozenset(index[c] for c in unk), cnt)
                    for unk, cnt in comp_cons]
            comps.append((cells, cons))
        return comps

    def _enum_component(self, n, cons):
        """Enumerate one component exactly.

        Returns (dist, tally): dist[m] counts solutions using m mines,
        tally[i][m] counts those where cell i is a mine. Incremental
        per-constraint counters avoid re-summing at every DFS node.
        Raises self._BudgetExceeded when the node budget is exhausted.
        """
        cnts = [cnt for _, cnt in cons]
        cell_cons = [[] for _ in range(n)]
        for ci, (idxset, _cnt) in enumerate(cons):
            for i in idxset:
                cell_cons[i].append(ci)
        rem = [len(idxset) for idxset, _cnt in cons]
        mk = [0] * len(cons)
        dist = [0] * (n + 1)
        tally = [[0] * (n + 1) for _ in range(n)]
        mine_stack = []
        nodes = 0

        def dfs(i, used):
            nonlocal nodes
            nodes += 1
            if nodes > self.MAX_ENUM_NODES:
                raise self._BudgetExceeded()
            if i == n:
                dist[used] += 1
                for i2 in mine_stack:
                    tally[i2][used] += 1
                return
            for val in (False, True):
                ok = True
                if val:
                    mine_stack.append(i)
                for ci in cell_cons[i]:
                    rem[ci] -= 1
                    if val:
                        mk[ci] += 1
                        if mk[ci] > cnts[ci]:
                            ok = False
                    if ok and cnts[ci] - mk[ci] > rem[ci]:
                        ok = False
                if ok:
                    dfs(i + 1, used + (1 if val else 0))
                for ci in cell_cons[i]:
                    rem[ci] += 1
                    if val:
                        mk[ci] -= 1
                if val:
                    mine_stack.pop()

        dfs(0, 0)
        return dist, tally

    def _component_probs(self, constraints, mines_left):
        """Per-cell mine probabilities conditioned on mines_left.

        Returns (probs dict cell->p, interior_estimate or None). When all
        components enumerate exactly, probabilities are conditioned on the
        global remaining mine count; otherwise per-component unconditioned
        estimates plus the classic interior density heuristic are used.
        """
        env = self.env
        comps = self._split_components(constraints)

        exact = []      # (cells, dist, tally)
        heuristic = {}  # cell -> p (fallback estimates)
        for cells, cons in comps:
            n = len(cells)
            if n <= self.MAX_ENUM_CELLS:
                try:
                    dist, tally = self._enum_component(n, cons)
                    if sum(dist) == 0:
                        for cell in cells:
                            heuristic[cell] = 0.5
                        continue
                    exact.append((cells, dist, tally))
                    continue
                except self._BudgetExceeded:
                    pass
            # fallback: local density heuristic (cons use cell indices)
            best = [None] * n
            for unk, cnt in cons:
                p = cnt / len(unk)
                for i in unk:
                    if best[i] is None or p > best[i]:
                        best[i] = p
            for i, cell in enumerate(cells):
                heuristic[cell] = best[i] if best[i] is not None else 0.5

        boundary_cells = set(heuristic)
        for cells, _dist, _tally in exact:
            boundary_cells.update(cells)
        interior = [
            (r, c)
            for r in range(self.h) for c in range(self.w)
            if not env.revealed[r, c] and not env.flagged[r, c]
            and (r, c) not in boundary_cells
        ]
        n_int = len(interior)

        probs = dict(heuristic)
        interior_p = None

        def comb(k):
            return math.comb(n_int, k) if 0 <= k <= n_int else 0

        if exact and not heuristic:

            def conv(a, b):
                out = [0] * (len(a) + len(b) - 1)
                for i, x in enumerate(a):
                    if x:
                        for j, y in enumerate(b):
                            if y:
                                out[i + j] += x * y
                return out

            dists = [dist for _c, dist, _t in exact]
            full = [1]
            for d in dists:
                full = conv(full, d)
            max_b = len(full) - 1
            total = sum(full)

            if total > 0:
                # comb(n_int, k) is 1 iff k==0 when n_int==0, so an empty
                # interior simply selects solutions using exactly
                # mines_left boundary mines
                weights = [full[b] * comb(mines_left - b)
                           for b in range(max_b + 1)]
                z = sum(weights)
                if z > 0:
                    # exact conditional probabilities
                    for j, (cells, dist, tally) in enumerate(exact):
                        g = [1]
                        for k, d in enumerate(dists):
                            if k != j:
                                g = conv(g, d)
                        for x, cell in enumerate(cells):
                            num = 0
                            trow = tally[x]
                            for mj, t in enumerate(trow):
                                if t:
                                    for m2, gv in enumerate(g):
                                        if gv:
                                            w = comb(mines_left - mj - m2)
                                            if w:
                                                num += t * gv * w
                            probs[cell] = num / z
                    if n_int > 0:
                        int_w = [weights[b] * (mines_left - b) / n_int
                                 for b in range(max_b + 1)]
                        interior_p = max(0.0, min(1.0, sum(int_w) / z))
                if z == 0:
                    # unconditioned normalization (mine-count inconsistent)
                    exp_b = (sum(b * full[b] for b in range(max_b + 1))
                             / total)
                    for cells, dist, tally in exact:
                        tot = sum(dist)
                        if tot == 0:
                            continue
                        for x, cell in enumerate(cells):
                            probs[cell] = sum(tally[x]) / tot
                    if n_int > 0:
                        interior_p = float(np.clip(
                            (mines_left - exp_b) / n_int, 0.0, 1.0))
        elif n_int > 0:
            # heuristic component(s) present: classic estimate, accounting
            # for expected mines consumed by every boundary cell
            # (exact comps contribute their enumerated expectation,
            # heuristic comps their density-based probabilities)
            exp_b = sum(heuristic.values())
            for cells, dist, tally in exact:
                tot = sum(dist)
                if tot:
                    exp_b += sum(m * dist[m]
                                 for m in range(len(dist))) / tot
            interior_p = float(np.clip(
                (mines_left - exp_b) / n_int, 0.0, 1.0))

        return probs, interior_p

    def next_action(self, return_info=False):
        env = self.env
        constraints = self._build_constraints()
        safes, mines = self._reduce(constraints)

        # guaranteed moves first
        if safes:
            r, c = sorted(safes)[0]
            act = r * self.w + c
            if return_info:
                info = {"guessed": False,
                        "certain": {a for r, c in safes
                                    for a in (r * self.w + c,)}}
                info["certain"] |= {
                    self.h * self.w + r * self.w + c
                    for r, c in mines if not env.flagged[r, c]}
                return act, info
            return act
        unflagged_mines = [(r, c) for r, c in sorted(mines)
                           if not env.flagged[r, c]]
        if unflagged_mines:
            r, c = unflagged_mines[0]
            act = self.h * self.w + r * self.w + c
            if return_info:
                info = {"guessed": False,
                        "certain": {self.h * self.w + r2 * self.w + c2
                                    for r2, c2 in unflagged_mines}}
                return act, info
            return act

        # guessing: compute probabilities
        fresh_constraints = self._build_constraints()
        candidates_any = [
            (r * self.w + c)
            for r in range(self.h) for c in range(self.w)
            if not env.revealed[r, c] and not env.flagged[r, c]
        ]
        if not candidates_any:
            # no legal moves left (game effectively over); reveal anything
            act = 0
            if return_info:
                return act, {"guessed": True, "certain": set()}
            return act
        if not fresh_constraints:
            # no information: pick uniformly among unrevealed unflagged
            act = int(self.rng.choice(candidates_any))
            if return_info:
                return act, {"guessed": True, "certain": set()}
            return act

        mines_left = env.num_mines - int(env.flagged.sum())
        probs, interior_p = self._component_probs(fresh_constraints, mines_left)
        boundary_cells = set(probs)
        interior = [
            (r, c)
            for r in range(self.h) for c in range(self.w)
            if not env.revealed[r, c] and not env.flagged[r, c]
            and (r, c) not in boundary_cells
        ]
        candidates = {
            cell: p for cell, p in probs.items()
            if not env.revealed[cell[0], cell[1]] and not env.flagged[cell[0], cell[1]]
        }
        if interior and interior_p is not None:
            for cell in interior:
                candidates[cell] = interior_p
        elif interior:
            # no usable estimate (e.g. inconsistent state): uniform prior
            fallback_p = mines_left / max(
                1, sum(1 for r in range(self.h) for c in range(self.w)
                       if not env.revealed[r, c] and not env.flagged[r, c]))
            for cell in interior:
                candidates[cell] = fallback_p

        best = min(candidates.values())
        tied = [cell for cell, p in candidates.items() if p <= best + 1e-9]
        r, c = tied[self.rng.integers(len(tied))]
        act = r * self.w + c
        if return_info:
            return act, {"guessed": True, "certain": set()}
        return act


def play_game(width=9, height=9, num_mines=10, seed=None, verbose=False):
    env = MinesweeperEnv(width=width, height=height,
                         num_mines=num_mines, seed=seed)
    obs, info = env.reset(seed=seed)
    solver = RuleSolver(env)
    steps = 0
    while True:
        action = solver.next_action()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if verbose:
            print(f"action={action} reward={reward:.2f}")
            print(env.render())
            print()
        if terminated:
            won = info["remaining_safe"] == 0
            return won, steps


if __name__ == "__main__":
    import sys
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wins = 0
    total_steps = 0
    for seed in range(num_games):
        won, steps = play_game(seed=seed)
        wins += won
        total_steps += steps
    print(f"games={num_games} win_rate={wins/num_games:.3f} "
          f"avg_steps={total_steps/num_games:.1f}")

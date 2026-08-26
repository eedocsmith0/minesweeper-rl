import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinesweeperEnv(gym.Env):
    """Minesweeper environment.

    Observation: float32 tensor of shape (13, H, W)
        channel 0: unrevealed mask (unrevealed and unflagged)
        channels 1-9: one-hot revealed numbers 0-8
        channel 10: flags
        channel 11: count of adjacent unrevealed cells / 8
        channel 12: count of adjacent flagged cells / 8

    Action space: Discrete(2*H*W)
        action < H*W: reveal cell (row = action // W, col = action % W)
        action >= H*W: flag cell
    Illegal actions are masked externally via action_mask().
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, width=9, height=9, num_mines=10, seed=None):
        super().__init__()
        assert 1 <= num_mines < width * height
        self.width = width
        self.height = height
        self.num_mines = num_mines

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13, height, width), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2 * height * width)

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.mines = None          # (H, W) bool
        self.revealed = None       # (H, W) bool
        self.flagged = None        # (H, W) bool
        self.counts = None         # (H, W) int8 adjacent mine counts
        self.first_move = True

    # ------------------------------------------------------------------ #
    def _neighbor_counts(self):
        counts = np.zeros((self.height, self.width), dtype=np.int8)
        for r in range(self.height):
            for c in range(self.width):
                if self.mines[r, c]:
                    continue
                n = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.height and 0 <= cc < self.width:
                            if self.mines[rr, cc]:
                                n += 1
                counts[r, c] = n
        return counts

    def _place_mines(self, safe_r, safe_c):
        """Place mines avoiding the first-clicked cell (and neighbors)."""
        safe = {(safe_r + dr, safe_c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        safe = {(r, c) for r, c in safe
                if 0 <= r < self.height and 0 <= c < self.width}
        candidates = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if (r, c) not in safe
        ]
        mine_cells = self.np_random.choice(
            len(candidates), size=self.num_mines, replace=False
        )
        mines = np.zeros((self.height, self.width), dtype=bool)
        for idx in mine_cells:
            r, c = candidates[idx]
            mines[r, c] = True
        return mines

    def _neighbor_sum(self, mask):
        """Sum a boolean mask over the 8-neighborhood of each cell."""
        padded = np.zeros((self.height + 2, self.width + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = mask
        total = np.zeros((self.height, self.width), dtype=np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r0, r1 = 1 + dr, 1 + dr + self.height
                c0, c1 = 1 + dc, 1 + dc + self.width
                total += padded[r0:r1, c0:c1]
        return total

    def _get_obs(self):
        obs = np.zeros((13, self.height, self.width), dtype=np.float32)
        unrevealed = ~self.revealed & ~self.flagged
        obs[0] = unrevealed
        for v in range(9):
            obs[1 + v][self.revealed & (self.counts == v)] = 1.0
        obs[10] = self.flagged
        obs[11] = self._neighbor_sum(unrevealed) / 8.0
        obs[12] = self._neighbor_sum(self.flagged) / 8.0
        return obs

    def _get_info(self):
        remaining_safe = int((~self.mines & ~self.revealed).sum())
        return {
            "remaining_safe": remaining_safe,
            "mines_total": self.num_mines,
            "flags_used": int(self.flagged.sum()),
        }

    def action_mask(self):
        """Boolean mask over 2*H*W actions."""
        mask = np.zeros(2 * self.height * self.width, dtype=np.int8)
        open_cells = ~self.revealed & ~self.flagged
        mask[: self.height * self.width] = open_cells.flatten()
        mask[self.height * self.width:] = open_cells.flatten()
        return mask.astype(bool)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.mines = np.zeros((self.height, self.width), dtype=bool)
        self.revealed = np.zeros((self.height, self.width), dtype=bool)
        self.flagged = np.zeros((self.height, self.width), dtype=bool)
        self.counts = np.zeros((self.height, self.width), dtype=np.int8)
        self.first_move = True
        return self._get_obs(), self._get_info()

    def step(self, action):
        h, w = self.height, self.width
        is_flag = action >= h * w
        flat = action % (h * w)
        r, c = divmod(flat, w)

        if self.first_move and not is_flag:
            self.mines = self._place_mines(r, c)
            self.counts = self._neighbor_counts()
            self.first_move = False

        reward = -0.01
        terminated = False

        if is_flag:
            if not self.flagged[r, c]:
                self.flagged[r, c] = True
                reward += 0.5 if self.mines[r, c] else -0.5
        else:
            if self.mines[r, c]:
                reward -= 10.0
                self.revealed[r, c] = True
                terminated = True
            else:
                opened = self._flood_reveal(r, c)
                reward += 1.0 + 0.05 * (opened - 1)

        info = self._get_info()
        if not terminated and info["remaining_safe"] == 0:
            terminated = True
            reward += 50.0
        return self._get_obs(), reward, terminated, False, info

    def _flood_reveal(self, start_r, start_c):
        """Reveal a cell; flood-fill zeros. Returns number of cells opened."""
        opened = 0
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < self.height and 0 <= c < self.width):
                continue
            if self.revealed[r, c] or self.flagged[r, c] or self.mines[r, c]:
                continue
            self.revealed[r, c] = True
            opened += 1
            if self.counts[r, c] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr != 0 or dc != 0:
                            stack.append((r + dr, c + dc))
        return opened

    # ------------------------------------------------------------------ #
    def render(self):
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if self.revealed[r, c]:
                    row.append(str(self.counts[r, c]) if self.counts[r, c]
                               else ".")
                elif self.flagged[r, c]:
                    row.append("F")
                else:
                    row.append("#")
            lines.append(" ".join(row))
        return "\n".join(lines)


if __name__ == "__main__":
    env = MinesweeperEnv(width=9, height=9, num_mines=10, seed=0)
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    rng = np.random.default_rng(0)
    while True:
        legal = np.flatnonzero(env.action_mask())
        action = rng.choice(legal)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            break
    print(env.render())
    print(f"steps={steps} total_reward={total_reward:.2f} info={info}")

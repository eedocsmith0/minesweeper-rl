import argparse
import os

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.minesweeper_env import MinesweeperEnv


def make_env(width, height, num_mines, rank, seed_base):
    def _init():
        env = MinesweeperEnv(
            width=width, height=height, num_mines=num_mines,
            seed=seed_base + rank,
        )
        env = ActionMasker(env, action_mask_fn)
        env = TimeLimit(env, max_episode_steps=3 * width * height)
        env = Monitor(env)
        return env
    return _init


class BoardCNN(BaseFeaturesExtractor):
    """Configurable 3x3-kernel CNN for boards of any size >= 2x2."""

    def __init__(self, observation_space, features_dim=256,
                 channels=64, layers=3):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        convs = []
        ch = n_in
        for _ in range(layers):
            convs += [nn.Conv2d(ch, channels, 3, padding=1), nn.ReLU()]
            ch = channels
        self.cnn = nn.Sequential(*convs, nn.Flatten())
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


def action_mask_fn(env):
    return env.action_mask()


class WinRateCallback(BaseCallback):
    """Evaluates win rate on a fixed seed set every `eval_freq` steps."""

    def __init__(self, width, height, num_mines, eval_freq=10_000,
                 n_eval_games=200, verbose=1):
        super().__init__(verbose)
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.best_win_rate = -1.0
        self.last_eval_step = 0
        self.history = []

    def _win_rate(self):
        wins = 0
        for seed in range(self.n_eval_games):
            if self._play_one(seed):
                wins += 1
        return wins / self.n_eval_games

    def _play_one(self, seed):
        env = MinesweeperEnv(width=self.width, height=self.height,
                             num_mines=self.num_mines, seed=seed)
        obs, info = env.reset(seed=seed)
        max_moves = 3 * self.width * self.height
        for _ in range(max_moves):
            mask = env.action_mask()
            if not mask.any():
                return False  # everything flagged/revealed without winning
            action, _ = self.model.predict(obs, action_masks=mask,
                                           deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            if term or trunc:
                return info["remaining_safe"] == 0
        return False

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        wr = self._win_rate()
        self.history.append((self.num_timesteps, wr))
        if self.verbose:
            print(f"steps={self.num_timesteps} win_rate={wr:.3f}", flush=True)
        if wr >= self.best_win_rate:
            self.best_win_rate = wr
            path = os.path.join("models", "best.zip")
            self.model.save(path)
            if self.verbose:
                print(f"saved new best ({wr:.3f}) to {path}", flush=True)
        return True


def train(width, height, num_mines, total_steps, n_envs, seed_base,
          model_path=None, eval_freq=10_000, n_eval_games=100,
          device="cpu", learning_rate=3e-4, ent_coef=0.01):
    os.makedirs("models", exist_ok=True)
    vec = DummyVecEnv([
        make_env(width, height, num_mines, rank, seed_base)
        for rank in range(n_envs)
    ])
    ppo_kwargs = dict(
        n_steps=512,
        batch_size=512,
        learning_rate=learning_rate,
        gamma=0.99,
        ent_coef=ent_coef,
        policy_kwargs={
            "normalize_images": False,
            "features_extractor_class": BoardCNN,
            "features_extractor_kwargs": {"features_dim": 256},
        },
        verbose=1,
        seed=seed_base,
        device=device,
    )
    if model_path and os.path.exists(model_path):
        model = MaskablePPO.load(model_path, env=vec, device=device,
                                 custom_objects={"learning_rate": learning_rate,
                                                 "ent_coef": ent_coef})
        print(f"resumed from {model_path}")
    else:
        model = MaskablePPO("CnnPolicy", vec, **ppo_kwargs)
    cb = WinRateCallback(width, height, num_mines,
                         eval_freq=eval_freq, n_eval_games=n_eval_games)
    model.learn(total_timesteps=total_steps, callback=cb,
                reset_num_timesteps=model_path is None)
    final_path = f"models/ppo_{width}x{height}_{num_mines}m"
    model.save(final_path)
    print(f"training complete. best win rate: {cb.best_win_rate:.3f}")
    print(f"final model: {final_path}.zip")
    for step, wr in cb.history:
        print(f"  {step:>8} steps  win_rate={wr:.3f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--mines", type=int, default=4)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--model", type=str, default=None,
                   help="resume from this model path")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    args = p.parse_args()
    train(args.width, args.height, args.mines, args.steps,
          args.n_envs, args.seed_base, model_path=args.model,
          eval_freq=args.eval_freq, n_eval_games=args.eval_games,
          device=args.device, learning_rate=args.lr,
          ent_coef=args.ent_coef)

"""Curriculum training: same board size, escalating mine counts."""
import argparse

from agents.train import train

STAGES = [
    (2, 200_000),
    (3, 300_000),
    (4, 500_000),
]


def main(width, height, steps_per_stage=None, n_envs=8, device="cpu",
         start_stage=0, resume=None):
    model_path = None
    for i, (mines, default_steps) in enumerate(STAGES):
        if i < start_stage:
            continue
        steps = steps_per_stage[i] if steps_per_stage else default_steps
        print(f"=== stage {i}: {width}x{height}, {mines} mines, "
              f"{steps} steps ===", flush=True)
        model = train(width, height, mines, steps, n_envs,
                      seed_base=1000 * (i + 1), model_path=model_path,
                      eval_freq=25_000, n_eval_games=100, device=device)
        model_path = f"models/ppo_{width}x{height}_{mines}m.zip"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--stage-steps", type=int, nargs="*", default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--start-stage", type=int, default=0,
                   help="skip earlier stages; resumes from the previous "
                        "stage's final model")
    args = p.parse_args()
    main(args.width, args.height, args.stage_steps, args.n_envs,
         args.device, args.start_stage)

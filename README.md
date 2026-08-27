# Most of the significant agent iterations are located in release V1.0

# Overview

This is a personal project for me to learn more about the reinforcement learning aspect of ML, especially working with DAgger. 

# Minesweeper RL

Train an AI to play Minesweeper by imitating a perfect-ish rule-based
teacher, then let it practice on its own (a technique called
DAgger). Works on three board sizes, up to expert 16x30 with 99 mines.

Official results (1000-game benchmarks, held-out seeds):

| Board | Rule-solver teacher | Trained agent |
|---|---|---|
| Beginner 9x9 / 10 mines | 97.4% | 95%+ |
| Intermediate 16x16 / 40 | 85.7% | 76.2% (ensemble of 4) |
| Expert 16x30 / 99 | 50.0% | 18.4% (ensemble of 4) |

You get: the game environment, the solver (exact probability
enumeration), the full training pipeline, evaluation tools, a live
web dashboard to watch games, and the campaign notes in PLAN.md.

Everything runs on an ordinary laptop CPU. A GPU only makes training
faster; it is not required.

Total time for training on my personal system (7800x3d, RX7800xt, 32GB DDR5) was ~16hrs.
OS used was fedora43.

---

## 0. Requirements

- Linux or macOS (Windows should work via WSL but im not sure)
- Python 3.10 or newer


## 1. Setup

Open a terminal **inside this project folder** and run these four
commands, one at a time:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

What just happened: command 1 created a private Python sandbox inside
the folder, command 2 switched your terminal into it (you will see
`(.venv)` at the start of each line), command 4 installed the four
libraries the project needs (PyTorch, NumPy, Gymnasium,
Stable-Baselines3).

If you close the terminal later, run `source .venv/bin/activate`
again before doing anything else. Every command below assumes the
sandbox is active.

Optional (NVIDIA GPU): install the CUDA build of PyTorch following
pytorch.org, and add `--device cuda` to training commands. AMD GPUs
work too via ROCm. Everything else is identical.

## 2. Train your first agent

Copy-paste these three commands. They create practice data, train the
network on it, and then grade the result on 500 fresh boards it has
never seen.

```bash
# Step A: collect ~30 minutes of expert play as training material
PYTHONPATH=. python agents/bc_pretrain.py --phase gen \
    --width 9 --height 9 --mines 10 --games 2000 \
    --data data/my_first_agent.npz

# Step B: train the neural network to imitate that play
PYTHONPATH=. python agents/bc_pretrain.py --phase train \
    --width 9 --height 9 --mines 10 \
    --data data/my_first_agent.npz --epochs 8 \
    --model-out models/my_first_agent --run demo

# Step C: test it against the rule-solver on unseen boards
PYTHONPATH=. python eval/evaluate.py \
    --model models/my_first_agent.zip \
    --width 9 --height 9 --mines 10 --games 500
```

The `PYTHONPATH=.` prefix tells Python where the project code lives.
Keep it exactly as written.

**Reading step C's output:** `win rate` is the fraction of boards your
agent cleared without clicking a mine. The first line printed is your
agent; run the same command again with `--baseline` instead of
`--model` to see the teacher's score on the same boards. Expect
roughly 60-80% after one pass; more data and rounds push it toward the
teacher's 97%.

Note: `models/my_first_agent.zip` is your trained brain file. Keep it;
everything else can be regenerated.

## 3. Watch games live in your browser

Open a second terminal, activate the sandbox, and start the dashboard:

```bash
viz/dashboard.py --port 8787
```

Then open http://localhost:8787 in a browser. You will see every
evaluation game rendered move by move, plus training curves from any
runs named with `--run`.

To make an agent play manually in the viewer, use the Solver page at
http://localhost:8787/solver .

Press Ctrl+C in that terminal when you are done.

### Optional: automatic benchmarking

The watcher re-tests every new checkpoint the moment it appears, so
curves fill in without manual work:

```bash
eval/watcher.py --models-dir models --games 30 --interval 2 \
    --run auto-eval
```

## 4. Make it better: practice rounds (DAgger)

One imitation pass plateaus below the teacher. The fix: let the agent
play its own games while the teacher labels what it *should* have
done, mix those corrections into the data, and retrain. Each cycle is
one "dagger round":

```bash
# Round 1: agent plays 2000 games, teacher corrects every move
PYTHONPATH=. python agents/bc_pretrain.py --phase dagger \
    --model models/my_first_agent.zip \
    --width 9 --height 9 --mines 10 --games 2000 \
    --out data/dagger_r1.npz --workers 4

# Mix original + correction data and retrain
PYTHONPATH=. python agents/bc_pretrain.py --phase train \
    --width 9 --height 9 --mines 10 \
    --data data/my_first_agent.npz,data/dagger_r1.npz --epochs 8 \
    --model-out models/my_first_agent_r1 --run demo
```

Repeat 2-3 times. On beginner boards this reaches 90%+ within two or
three rounds.

Two useful flags:
- `--workers 4` spreads collection across CPU cores (any number works)
- `--priority-weights` (add to BOTH dagger and train) tells training
  to double-count positions where the student disagreed with the
  teacher - faster learning once the basics are in place

## 5. Bigger boards

Same three commands, different numbers. The standard ladder:

| Board | Flags | Notes |
|---|---|---|
| Beginner | `--width 9 --height 9 --mines 10` | easy, start here |
| Intermediate | `--width 16 --height 16 --mines 40` | needs several rounds |
| Expert | `--width 16 --height 30 --mines 99` | research-grade hard |

On square boards the evaluator automatically uses symmetry averaging
(TTA), which is worth many points; nothing for you to configure.

For intermediate/expert, single models plateau well below the teacher.
The remedy that worked here: train several *different* members (vary
`--seed`, network size `--channels/--layers`, or start from an earlier
model with `--init-from`) and average their votes:

```bash
PYTHONPATH=. python eval/tta_probe.py \
    --model models/memberA.zip models/memberB.zip models/memberC.zip \
    --width 16 --height 16 --mines 40 --games 300
```

Details, dead ends, and the full campaign history live in PLAN.md.

## 6. Housekeeping tools

```bash
scripts/cleanup_data.py                 # preview old training files that can be deleted
scripts/cleanup_data.py --apply         # actually delete them
keepalive.sh &                          # auto-restart watcher+dashboard if they crash
```

## 7. Command cheat sheet

| I want to... | Command |
|---|---|
| Create practice data | `agents/bc_pretrain.py --phase gen ...` |
| Train / retrain | `agents/bc_pretrain.py --phase train ...` |
| Correction round | `agents/bc_pretrain.py --phase dagger ...` |
| Grade an agent | `eval/evaluate.py --model X.zip ...` |
| Grade the teacher | `eval/evaluate.py --baseline ...` |
| Test a group vote | `eval/tta_probe.py --model A.zip B.zip ...` |
| Live dashboard | `viz/dashboard.py --port 8787` |
| Auto-benchmark new models | `eval/watcher.py ...` |

All board sizes take `--width W --height H --mines M`. Training also
takes `--run NAME` to label its curves on the dashboard.

Ox Alpha (GLM 5.3 Flash) was used to assist in this project as an agent

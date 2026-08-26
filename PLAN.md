# Minesweeper RL Solver - Plan

Goal: train a reinforcement learning agent that can solve games of Minesweeper start to finish.

Decisions:
- Language: Python
- Hardware: AMD RX 7800 XT (ROCm preferred, CPU fallback is viable since the model is small)
- Flagging: included; win condition remains "reveal all non-mine cells"
- Forced guesses: use best-probability cell; random choice only on true 50-50 ties
- Observation: raw board state first (no derived features); add features later only if learning stalls

## Stack
- Python 3.11+, PyTorch, Gymnasium, Stable-Baselines3 + sb3-contrib (MaskablePPO), TensorBoard, NumPy

## Project Structure
```
minesweeper-rl/
├── env/
│   └── minesweeper_env.py    # Gymnasium env with action masking
├── agents/
│   ├── rule_solver.py        # constraint/probability baseline
│   └── train.py              # MaskablePPO training script
├── eval/
│   └── evaluate.py           # win-rate benchmarking vs baseline
└── models/                   # saved checkpoints
```

## Milestone 1: Simulator (env/minesweeper_env.py)
- Gymnasium env
- Observation: raw tensor of shape (12, H, W):
  - channel 0: unrevealed mask
  - channels 1-9: one-hot revealed numbers 0-8
  - channel 10: flagged cells
  - channel 11: spare (reserved)
- Action space: Discrete(2*H*W); index < H*W reveals cell, >= H*W flags cell
  - Illegal actions masked (revealing an already-revealed cell, flagging a revealed cell)
- Rewards:
  - Reveal safe cell: +1 (+ bonus proportional to number of cells opened by flood fill)
  - Correct flag: +0.5; wrong flag: -0.5 (flagging never ends the episode)
  - Hit mine: -10, episode ends
  - Win (all non-mine cells revealed): +50
  - Step penalty: -0.01
- Test script that plays random games and prints boards

## Milestone 2: Rule-Based Baseline (agents/rule_solver.py)
- Single-point + subset constraint solving on mine-count equations
- When no certain move exists: compute per-cell mine probabilities from constraints, reveal lowest-probability cell
- If all candidate cells tied at 50-50: choose randomly among them
- Benchmark: win rate over 1000 seeded beginner games (9x9, 10 mines)
- Purposes: benchmark, sanity reference for the RL agent

## Milestone 3: First RL Training (agents/train.py)
- MaskablePPO + CNN policy on 6x6 / 4 mines
- Vec env with 8-16 parallel environments
- Track: win rate every 10k steps via TensorBoard
- Success check: agent beats random play

## Milestone 4: Curriculum Scaling
- Promote board size when eval win rate > 90% on a fixed seed set
- Ladder: 6x6/4 -> 9x9/10 -> 16x16/40 (expert 16x30/99 optional stretch)

## Milestone 5: Final Evaluation (eval/evaluate.py)
- Fixed seeded test set of 1000 boards per difficulty
- Report: win %, avg moves per game
- Compare RL agent vs rule solver
- Note: some boards require guessing (~unavoidable on expert), so target is matching/beating the probabilistic baseline

## Risks / Escalation
- PPO may plateau on large boards; escalation path is MCTS-guided self-play (AlphaZero-lite) if needed
- ROCm setup issues: fall back to CPU training (model is tiny)

## Status Log
- [x] Step 0: environment setup (CPU venv `.venv`, GPU venv `.venv-rocm` w/ ROCm torch, verified ~2200 fps)
- [x] Milestone 1: simulator (13-channel obs; first click safe; episode length caps)
- [x] Milestone 2: rule-based baseline - 94.3% on 9x9/10m, 95.6% on 6x6/4m
- [x] Milestone 3: first RL training - see "Method evolution" below
- [x] Milestone 4: curriculum scaling
- [x] Milestone 5: final evaluation (see Results)

## Method evolution (what actually worked)
1. Pure PPO from scratch: plateaued at 12% (6x6/4m). Fixed-board test=100%,
   so wiring was fine; model-free RL cannot discover constraint logic.
2. + derived obs channels: no improvement.
3. + mine-count curriculum: no plateau break.
4. Behavioral cloning (BC) from rule-solver moves: 81% on 6x6/4m.
   Key detail: pretrain value head separately on Monte-Carlo returns with
   trunk frozen; joint training destabilizes the policy.
5. DAgger rounds (roll with learned policy, label states w/ solver):
   79% -> 81% -> 92% -> 95.5% on 6x6/4m (= solver level).
6. PPO fine-tuning after BC repeatedly DESTROYED the policy (81% -> 14%),
   even with fitted vf / tiny lr / zero entropy. Dropped; BC+DAgger is the
   winning recipe for this game.
7. Dihedral symmetry augmentation (8 transforms, per-batch random):
   large gains once the old->new cell permutation bug was fixed.

## Results (fixed seed sets)
| board    | agent            | win rate | avg moves | safe cells opened |
|----------|------------------|----------|-----------|-------------------|
| 9x9/10m  | rule solver      | 93.5%    | 19.8      | 98.7%             |
| 9x9/10m  | RL agent (r2)    | 87.6%    | 18.2      | 97.7%             |
| 6x6/4m   | RL agent (r3)    | 95.5%    | -         | -                 |
| 6x6/4m   | rule solver      | 95.6%    | -         | -                 |

DAgger progression on 9x9/10m: 14.9 -> 42.9 -> 59.2 -> 71.3 -> 74.8 ->
(+symmetry) 80.8 -> 82.4 -> 86.9 -> 87.6%

## Intermediate campaign (16x16) - in progress
Key discoveries:
- TTA (symmetry-averaged inference over 8 dihedral views) is worth +40
  points at this scale (r8: 25% plain -> 79.6% TTA). Positional bias in
  the CNN was masking real pattern knowledge.
- Same-size weight transfer across mine counts works via set_parameters:
  20m-trained weights bootstrap 40m directly.
- Soft-target experiment (uniform over certain actions) REGRESSED play and
  was reverted; hard NLL on solver actions remains the recipe.

Progress line at 16x16 (all with TTA unless noted):
- 20 mines: BC 1.2% -> DAgger rounds -> r10 = 87.7% / 1000 games
- 40 mines: transfer from r10 + BC (r1) = 47% / 300 games
  -> r2 53% -> r3 51% -> r4 59.7% -> r5 55% -> r6 64.7% -> r7 60.7%
  -> r8 60.3% (plateau at 3-layer net)
  -> BREAKTHROUGH: 6-layer net (13x13 receptive field) + bigger DAgger
     batches: r9 = 70.3% -> r10 = 74.3% / 300 games
- FINAL (1000 games, seeds 850000+): agent16_16x16_40m_r11 = **72.7%**
  vs rule solver = 78.8% on identical boards
- Winning recipe changes for 40m endgame: layers 3->6, DAgger batches
  ~2x larger, dataset pruning to recent rounds (RAM), init-from previous
  round each time

## Expert campaign (16x30 / 99 mines) - in progress
Teacher reality: our rule solver wins only ~18.4-21.6% at expert (published
strong solvers ~35-40%) - teacher quality is the bottleneck. S3 hybrid
agreed with ops: collect with current teacher while ops upgrades solver
enumeration (cap 28->64, node-budget fallback, exact large-component probs);
switch labels to improved teacher when it deploys.
Progression (TTA / 300g unless noted, seeds 970000+):
- r0 BC (trunk transfer from intermediate r11): 8.3%
- r1 DAgger round: 9.3%
- r2 DAgger round: 10.0%
- r3 DAgger round (upgraded teacher labels live): 9.7%
- r4 DEPTH BUMP 6->8 layers + 320dim: 15.3%
- r5 DAgger round: 13.0% | r6 warm-restart dud: 9.0%
- ENSEMBLES (the real payoff): [r4+r5] 15.5%, [r3+r4+r5] 16.5%/200g,
  [r4+r5+r6] 21.0%/200g - best expert result
- r7 wide-net-from-scratch: collapsed solo (0%), neutral in ensemble
- r8 10-LAYER trunk-transfer: weak solo (8.5%) but diverse - 4-model
  [r4,r5,r6,r8] jumped to 29.0%/200g
- r9 warm-start from r8: 12.5% solo; adding it DILUTED ensemble (22%)
- RIGOROUS 1000g (seeds 850000+): best ensemble [r4,r5,r6,r8] = 18.4%
  vs deployed solver = 50.0% same seeds
Seed-set variance at expert is large (29%/200g probe -> 18.4%/1000g);
teacher measures 44.8-50% depending on set. Honest agent estimate:
~18-25%. Ensemble composition matters: competent-but-diverse members help,
fully collapsed or too-similar members dilute.
Teacher reference on same seeds: deployed upgraded solver wins 45%/200g
(18.4-21.6% was the pre-upgrade teacher). Intermediate-era ceiling notes:
40m teacher now measures 85.7% (was long believed 78.8%).
Round recipe: 8k plain parallel rolls (--workers 6) + 1k TTA serial rolls,
fresh retrain init-from previous round. ~2h per round.
Note: top1 agreement reads ~99% because expert transitions are dominated by
long certain-safe stretches; TTA win rate is the real metric.

## 8-Hour Expert Block - COMPLETE (results)
Weighted sampling live from E10 onward (w=2 rate ~0.5% of transitions -
student/teacher certainty disagreement is rare; r4 well-calibrated).
- E10 depth-12 member: solo 12.0%, DILUTED pool (29->20 probe) - correlated
  with r8's errors; deep members cluster
- E11 diverse-seed twin (8L/256): solo 14.0%; 5-model still diluted (19.5%)
- E12 consolidation (8L/320, --max-transitions planner, weighted): solo
  **16.5% = NEW BEST SOLO** (beats r4's 15.3%)
- Composition search found new best trio: [r4, e11, e12]
- RIGOROUS 1000g (seeds 850000+): [r4,e11,e12] = **20.9%** vs old pool's
  18.4%; teacher same seeds ~50%
Lessons: [r4,r5,r6,r8] was a sharp local optimum (every addition diluted);
escaping required a consolidation retrain + fresh twins, not more members.
Solo quality gains DID transfer to a new composition.

## Key files
- env/minesweeper_env.py        simulator (13-ch obs, action masking, caps)
- agents/rule_solver.py         constraint solver = baseline + teacher
- agents/train.py               MaskablePPO training (superseded by BC path)
- agents/bc_pretrain.py         dataset gen / BC+DAgger training / eval
- agents/metrics.py             JSONL metric + board logging
- eval/evaluate.py              head-to-head benchmark
- viz/dashboard.py              live training dashboard (stdlib web server)
- models/bc_9x9_sym_r2.zip      best beginner agent (87.6%)
- models/bc_dagger_r3.zip       best 6x6 agent (95.5%)

## Dashboard
Start: `python viz/dashboard.py --port 8787` then open http://localhost:8787

Training scripts accept `--run NAME`; all metrics go to `runs/NAME/*.jsonl`.
Panels:
- learning curve: per-epoch top1 agreement (blue) and NLL loss (red)
- eval win-rate timeline across checkpoints / DAgger rounds
- transitions per dataset (shows DAgger growth)
- gallery of recent real boards the agent played (WIN/LOSS tagged)
Auto-refreshes every 3s. Runs "history" (9x9/6x6 documented progression)
and "demo" are seeded examples.

Follow-training mode (default ON): every refresh scans /proc for running
project training scripts and auto-switches the selected run to the
trainer's --run name. Selecting a run manually turns following off;
re-check the box to re-enable. If the trainer omits --run, the status bar
warns (its epoch curves cannot be captured retroactively).

Live solver viewer: http://localhost:8787/solver
- watch the rule solver OR any trained model play move-by-move in the browser
- presets (beginner/intermediate/custom), speed slider, pause/resume
- last move highlighted; blue = move, red outline = mine hit
- agent dropdown lists everything in models/
- "Mirror live training": reads the running trainer's PID/config from /proc,
  plays games with the same checkpoint (or newest matching board config),
  follows it as checkpoints advance; auto-restarts games and self-heals
  after network/server hiccups

## Auto-eval watcher
Start: `python eval/watcher.py --models-dir models --games 500 --interval 30`

Watches models/ for new checkpoints (filesystem only - zero coordination
with the training session). Every new zip is benchmarked on a fixed seed set
and logged to runs/auto-eval/ -> appears on the dashboard automatically.

Checkpoint config resolution:
1. filename contains `_WxH_Mm` (e.g. `agent_16x16_40m.zip`)  <- preferred
2. sidecar `<name>.json` with {"width","height","mines"}
3. otherwise skipped with a log message

Existing checkpoints at startup are ignored (no re-eval spam); anything new
or modified afterwards gets evaluated.

## Checkpoint naming convention (for dashboard sync)
- All future agent saves: `models/agent16_{W}x{H}_{M}_r{R}.zip`
  e.g. `models/agent16_16x16_20_r4.zip`
- Helper: `agents/naming.py` (`model_path`, `model_glob`)
- Legacy names (bc_*, ppo_*) predate the convention.

## Status Log (training session, mirrored)
- [x] 16x16/20m stage COMPLETE: final agent16_16x16_20m_r10 = 87.7% (1000g)
- [~] 16x16/40m stage STARTED (Aug 23 ~20:00): r1 = 47.9% TTA (1000g,
      ops-anchored; plain mode = 6.7% - TTA gap ~40 pts confirmed).
      Dataset bc_16x16_40m_v2.npz generated
- [x] INTERMEDIATE TARGET HIT (Aug 24): agent16_16x16_40m_r11 = 73.0%
      TTA independently confirmed on disjoint seeds (training: 72.7% on
      850000+; ops: 73.0% on 900000+), vs solver ceiling 78.8%.
      Training session paused; r11 = release candidate.

## Inter-session communication
`coordination/PROTOCOL.md` + `coordination/send.py`: file-based mailboxes
(`to-training.jsonl`, `to-ops.jsonl`). Project `AGENTS.md` instructs both
sessions to check inboxes before new work and post after stages.
Ops never kills training processes; training owns models/ and data/.

Persistent services (ops): watcher + dashboard run via `keepalive.sh`
(auto-restarts either within 60s if they die) and a cron @reboot entry
relaunches keepalive after reboots. Service logs live in project
`logs/` (not /tmp - volatile). Note: checkpoints that appear while the
watcher is down are intentionally not retro-evaled (startup snapshot);
touch their zip to force evaluation.

Parallel rollouts landed (ops): `--workers N` on gen/dagger phases spawns
N processes (spawn context, CUDA-safe), each playing a contiguous seed
range into a shard npz that is merged with full schema. Serial path
unchanged at --workers 1.

Desktop-friendly priority: all training/watcher processes run at nice 10
by default (`--nice` to change) - full speed on an idle machine,
proportional CPU yield when apps/games need it. GPU scheduling unaffected.

Crash-safe serial dagger (ops): `--flush-every N` writes atomic checkpoint
shards (temp+rename, grange metadata inside) + `.manifest.jsonl`; killed
runs resume with the same command plus `--resume` (adopts shards,
continues at recorded index, warns on checkpoint mismatch). Full
kill->resume cycle verified on 6x6.

Bounded-RAM training (ops): train phase caps combined datasets via
`--max-transitions` (default 900k): newest files kept whole, older
proportionally downsampled. Verified: 3.31M-transition dataset -> 300k
used / 2.0GB RAM instead of ~21GB.
Known tradeoffs: recency bias can self-reinforce policy blind spots
(25% old-reserve is an untuned mitigation); cross-round metric deltas
carry subset noise; disk usage still grows unbounded; full-history fits
need explicit --max-transitions override.

Disk halving (ops): observations now stored lossless uint8 (ch11/12 as
k*32) in buffers/shards/merged npz - ~50% smaller files and train RAM;
legacy float16 npz auto-converted on load. GC tool
scripts/cleanup_data.py (dry-run default) deletes dagger rounds beyond
--keep-rounds 3 / --min-age-hours 24; never touches base datasets.

## Expert campaign (16x30/99m) - in progress
Strategy: S3 hybrid (locked Aug 25). Teacher upgraded first (ops,
deployed 02:33): MAX_ENUM_CELLS 28->64, 250k-node budget DFS with
incremental counters, exact enumeration extended to large frontier
components, probabilities conditioned on remaining mine count via
interior binomial completion. NO pattern rules - redundant under exact
enumeration (any pattern deduction is some cell at p=0).

Teacher v2 verified numbers (ops, Aug 25):
- expert 16x30/99m: 44.8% +/-2.2 over 500g seeds 800000-800499
  (pre-upgrade baseline on same seeds: 21.6%)
- 9x9/10m: 97.4%/1000g; 16x16/40m: 85.7%/300g
- NOTE: deploy-time announcement of 26.2%/95.0% was stale intermediate
  measurements; the numbers above are ground truth for current code.
- CEILING CORRECTION: 16x16/40m solver ceiling is ~85.7%, not 78.8%
  as previously documented - ensemble RC (76.2%) has ~9pt DAgger
  headroom if that stage resumes.

Campaign log (trainer): r0 BC = 8.3% -> r1 = 9.3% -> r2 training in
flight (init-from r1). Bulk collection plain --workers 6 (no TTA) +
serial TTA rolls per round for deep-state coverage.

Milestone (trainer, Aug 25 16:22; rigorous 1000g seeds 850000+): best
expert ensemble [r4,r5,r6,r8] = 18.4% official (same ensemble reads 29%
on probe seeds - large seed variance, use 1000g for claims). Deployed
solver same seeds: 50.0%. Composition findings: competent-diverse
members help, fully-collapsed members neutral-to-negative, too-similar
dilute; r8 = 10-layer trunk-transfer member was the diversity win.
Bottleneck: teacher gap (agent ~18-25% vs teacher ~45-50%).
Membership so far: r4 solo 15.3%, r5/r6 fresh-seed rounds, r7 wide-net
(collapsed solo), r8 trunk-transfer, r9 watcher 16.7%.

## Remaining ideas (not done)
- More DAgger rounds on 9x9 (~+1-2%/round, diminishing)
- Bigger CNN / features_dim for late-game reasoning
- Intermediate 16x16/40: ATTEMPTED - BC reaches 66% per-move agreement, but
  with ~100 moves/game that compounds to ~0% wins (need >95% per-move
  accuracy). Solver ceiling re-measured Aug 25 at ~85.7% (was believed
  79%). Next steps if resumed: several more DAgger rounds with larger net
  (features_dim 512+), or curriculum through mine counts at this size first.

## How to use
Train (per size): 
```
PYTHONPATH=. .venv-rocm/bin/python agents/bc_pretrain.py --phase gen \
  --width W --height H --mines M --games G --data data/X.npz
PYTHONPATH=. .venv-rocm/bin/python agents/bc_pretrain.py --phase train ...
# repeat dagger+train until win rate plateaus
PYTHONPATH=. .venv-rocm/bin/python agents/bc_pretrain.py --phase dagger ...
PYTHONPATH=. .venv-rocm/bin/python eval/evaluate.py --model models/M.zip \
  --width W --height H --mines M --games N
```

## Ops status: solver enumeration upgrade (2026-08-25 ~02:40)

Teacher upgrade shipped under locked S3 plan (trainer-scoped: cap/ordering
raise + exact enumeration on large frontiers, NO pattern rules). Changes to
agents/rule_solver.py, built on top of the cap-28 state as agreed:
- MAX_ENUM_CELLS 28 -> 64 plus MAX_ENUM_NODES=250k budget per component;
  oversized/over-budget components fall back to local density heuristic.
- Incremental-counter DFS (no per-node constraint re-summing) with
  most-constrained-first cell ordering preserved.
- Solutions bucketed by mines-used per component; components combined and
  conditioned on global remaining-mine count via binomial completion through
  interior cells (exact when all components enumerate). Unconditioned
  fallback kept for mine-count-inconsistent states.
Verified: 163/163 randomized states match brute-force enumeration exactly
(two real bugs found+fixed pre-deploy: n_int>0 gate skipping conditioning on
closed frontiers; interior_p-None fallback clobbering conditioned probs).
Results: expert 16x30/99 same-seed A/B 21.6% -> 26.2% (500g, seeds
800000+); 9x9/10 regression 87.7% -> 95.0% (1000g); ~0.7 ms/move expert,
zero budget aborts. Pre-upgrade solver backed up at
/tmp/opencode/rule_solver_pre_upgrade_backup.py (volatile).

## Ops infra: memguard RAM sentinel (2026-08-25 09:45; total-usage thresholds + keepalive-supervised since 10:13)

scripts/memguard.sh (supervised by keepalive.sh every 60s, nice 19):
samples to logs/memguard.log. Thresholds on TOTAL USAGE
(MemTotal - MemAvailable): WARN >=24576MB (24GB) -> trainer mailbox issue
+ out-of-band human ping; CRIT >=28672MB -> escalated warning only (WARN-ONLY mode
per human directive: no auto-stop ever); state machine re-arms when
usage <23076MB. Installed per human instruction during expert campaign.

## Ops: priority-weighted sampling (2026-08-25 ~17:10, deployed)

Built per trainer spec (mailbox id 8) during human-pause window. Dagger
collections record a per-transition weight column `prio`: w=2 when the
CURRENT policy's greedy pick fell outside the teacher's certain/safe set
(student-error states), else w=1; train phase consumes it as weighted-
mean NLL in Phase A (--priority-weights). Default OFF = byte-identical
legacy flows; legacy npz without prio load as all-ones. Works in serial
and --workers paths; merge/load/downsampling keep alignment. Verified:
buffer roundtrip, legacy-shard merge, downsample alignment, weighted
hard+soft loss vs manual numpy, full micro e2e (gen->train->dagger->
train) on 6x6 with flag on/off. Pre-change backup:
/tmp/opencode/bc_pretrain_pre_pw_backup.py (volatile). Usage for next
rounds: add --priority-weights to dagger AND train commands.

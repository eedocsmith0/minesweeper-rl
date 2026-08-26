# Full Project Audit - 2024-08-24 (from ops session)

Scope: every Python file (~3,300 lines), env semantics, eval tooling,
dashboard, coordination infra. Items marked [verified] were reproduced
empirically. Please review and reply per-item: agree / disagree /
fix-owner (ops vs training). Ops will not change anything until we
converge.

## A. Bugs (correctness)

A1 [HIGH][verified] env/minesweeper_env.py step():
   Revealing an already-revealed SAFE cell awards +0.94 reward for a
   no-op (flood_reveal returns 0 opened; reward = 1.0 + 0.05*(0-1)).
   Action masks hide this from SB3 paths, but any raw caller can farm
   infinite reward - poisons RL fine-tuning on the raw env.

A2 [MED][verified] env/minesweeper_env.py step():
   Flags placed BEFORE the first reveal are scored against an empty
   mine grid (mines not yet placed) -> always -0.51 even if the cell
   ends up being a real mine. Corrupts exploration data subtly.

A3 [HIGH] eval/watcher.py poll_inbox():
   Watcher reads coordination/to-training.jsonl. Consequence: an
   --action stop meant for TRAINING SCRIPTS also kills the watcher,
   while messages addressed --to ops never reach it. Should read
   to-ops.jsonl.

A4 [HIGH] eval/evaluate.py play_with_model():
   --tta hardcodes device="cuda" for tta_pick regardless of --device.
   Running "--tta --device cpu" crashes on device mismatch.

A5 [MED] agents/bc_pretrain.py pretrain() soft-target branch:
   Soft targets spread label mass over ALL certain actions including
   certain-MINE FLAGS equally with safe reveals. Teaches flag-spam;
   plausible contributor to the r7 regression. Suggest restricting
   soft-target certain-set to reveal actions (< H*W) or down-weighting
   flags.

A6 [LOW] viz/dashboard.py start_session() called inside global LOCK:
   torch model load (seconds) blocks ALL endpoints incl. metrics
   polling during mirror-game starts.

A7 [LOW] env/minesweeper_env.py __init__:
   Config validation allows num_mines > W*H-9 which crashes later at
   first move (_place_mines cannot fit mines around guaranteed-safe
   3x3). Validate at construction.

## B. Performance

B1 Serial TTA-dagger ~2-3s/game; --workers fix landed but unused so
   far. Biggest single lever available.
B2 pretrain(): dataset tensors on CPU, sliced per batch -> unpinned
   H2D transfers each step. Pinning or GPU-resident data would cut
   epoch time when VRAM allows.
B3 rule_solver._component_probs: id() dedupe set rebuilt per visited
   cell (O(cons^2)/component). Harmless today.

## C. Design / consistency

C1 keepalive.sh restarts watcher with hardcoded --games 30 (docs say
   default 500).
C2 agents/naming.py model_glob() is dead code.
C3 viz/dashboard.py line_chart() is dead Python code.
C4 MetricLogger appends from mp workers are unlocked (practically OK
   on Linux for small lines).
C5 send.py --read does not display the "action" field.

## D. Verified good
Env obs channels match docstring; _reduce seen-set logic sound;
resume/recovery handles variable transitions-per-game correctly;
shard merge preserves all arrays; keepalive loop works.

## Proposed fix order (pending agreement)
1. A3 + A4 (one-liners, bite current workflows)
2. A1 + A2 together (env guards + consistent first-move flag scoring)
3. A5 (soft targets reveal-only)
4. B2 pinning before expert-stage marathon

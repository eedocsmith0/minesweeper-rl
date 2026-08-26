# OPS SESSION HANDOFF — Minesweeper RL

Import this file into any new session taking over the ops role.

## 1. PRIMARY ROLE

This is the **ops session** for the two-session setup on this project:

- **training session** (separate agent fork): owns model/dataset production.
  Runs `agents/bc_pretrain.py` phases (gen/dagger/train/eval). Owns
  `models/`, `data/`.
- **ops session (you)**: periodic "sync" reports on demand, evaluation,
  dashboards/tooling, solver improvements, docs, inter-agent comms.
  NEVER kill training processes. NEVER modify models/ or data/ except via
  agreed GC/converter tools.

### Sync routine (on user request "sync")
1. `date "+%H:%M:%S"`
2. Check running training processes:
   `ps aux | grep -E "[b]c_pretrain|[a]gents/train.py"`
3. Newest checkpoints: `ls -lt models/*.zip | head -4`
4. Newest data: `ls -t data/ | head -4`
5. Inbox: last message in `coordination/to-ops.jsonl`
6. Latest eval numbers from `runs/intermediate/metrics.jsonl` and
   `runs/default/metrics.jsonl` (trainer sometimes omits --run; records
   fall back to runs/default)
7. Watcher log tail: `logs/watcher.log`
8. If a run is ACTIVE: report what/progress/ETA. Compute pace from the
   partial epoch records in metrics.jsonl (~15s cadence) or epoch counts.
9. Always include system resources when asked, and flag RAM pressure
   (dataset growth) if free memory < 8 GB.

## 2. CURRENT TASK (as of export)

**Expert 16x30/99m campaign strategy discussion** — awaiting the training
session's reply in its inbox.

- Measured baseline: rule solver wins **18.4% ± 3.4%** on expert
  (500 games). Published strong solvers: ~35-40% → teacher headroom.
- Three strategies sent to trainer (mailbox, ~00:56):
  - S1: improve teacher first (raise MAX_ENUM_CELLS, better enumeration;
    ops work, no GPU idle)
  - S2: run campaign immediately with current teacher
  - S3 (ops lean): HYBRID — start round-1 collection now with current
    teacher while ops upgrades solver in parallel; improved labels from
    round 2 onward
- Three questions posed: (a) agree with S3? (b) games-per-round +
  worker count for expert (serial TTA ≈ 5-10 s/game; --workers 6 → 10k
  games in 2.5-5 h)? (c) keep TTA during expert collection?
- On their reply: reconcile, lock plan, execute. Ops-owned piece under
  every strategy = solver enumeration upgrade (raise cap, exact
  component probabilities on large frontiers).

## 3. STATE SNAPSHOT (at export time)

- Board-config ladder completed: 6x6/4m ✓, 9x9/10m ✓ (87.7%),
  intermediate 16x16/40m release candidate r11 = 73.0% TTA
  independently confirmed vs solver ceiling 78.8%
- Fork currently finishing round 14 (wide-net ensemble member,
  96ch/6L/320dim) + plans 4-model ensemble [r11,r12,r13,r14]
- Expert stage NOT started; strategy discussion above decides launch
- Services: watcher (TTA on, cuda) + dashboard :8787 running under
  keepalive.sh; cron @reboot entry installed
- Audit report (7 bugs, 5 perf items) sent for review — response still
  pending; details in coordination/audit-report.md

## 4. KEY FILES

| path | purpose |
|------|---------|
| PLAN.md | master plan + status log + tradeoffs |
| AGENTS.md | both-session conventions |
| coordination/PROTOCOL.md | mailbox rules |
| coordination/send.py | send/read mail (--action stop supported) |
| agents/bc_pretrain.py | gen/dagger/train/eval pipeline (shared w/ trainer) |
| agents/rule_solver.py | constraint solver = baseline + DAgger teacher |
| eval/watcher.py | auto-benchmarks new checkpoints (TTA on by default) |
| eval/evaluate.py | head-to-head benchmark (--tta supported) |
| scripts/cleanup_data.py | disk GC (dry-run default) |
| scripts/convert_data_u8.py | one-off f16->u8 converter (already run) |
| viz/dashboard.py | web dashboard + live solver viewer (/solver) |
| keepalive.sh | restarts watcher+dashboard within 60s; cron @reboot |

## 5. INFRASTRUCTURE

- Services run niced (nice 10), logs in project `logs/` (NOT /tmp).
- Keepalive checks every 60s; cron @reboot line installed.
- Dashboard: http://localhost:8787 (follow-training mode ON default).
  Solver viewer: http://localhost:8787/solver (mirror mode available).
- Watcher config in production: --games 30 --interval 2 --tta
  --device cuda. Startup ignores pre-existing zips; touch a zip to force
  re-eval. Checkpoint naming must match `_WxH_Mm` or have sidecar json.

## 6. GOTCHAS LEARNED (do not relearn these)

- `pkill/pgrep -f X` matches YOUR OWN command string containing X — use
  `[b]racket` trick or filter by comm.
- Each Bash tool call may start fresh at $HOME — always cd/workdir per
  call; never rely on previous cwd.
- Long-lived services must log inside the project (logs/), never /tmp
  (tmpfs wiped, killed services once).
- Trainer's stdout logs land in /tmp/opencode/*.log (volatile but useful
  mid-run); their commands often omit --run so records land in
  runs/default.
- Watcher ignores checkpoints present at startup; new ones during outages
  need a manual `touch` after adding sidecar/config.
- Serial dagger writes npz only after ALL games — timeout kills lose
  everything unless --flush-every checkpointing was used.
- Train phase has NO checkpoint/resume — a timeout there loses the fit
  but collected dagger data survives on disk.
- TTA (symmetry-averaged inference) adds ~40 points on this project's
  boards; ALWAYS benchmark with TTA for comparable numbers.
- uint8 obs encoding is lossless (ch11/12 stored k*32); legacy float16
  npz auto-convert on load; *.f16bak backups exist in data/.

## 7. COMMUNICATION WITH TRAINING SESSION

- Mailboxes: coordination/to-training.jsonl and to-ops.jsonl via
  `coordination/send.py`. Both sessions' AGENTS.md instruct checking
  inbox before new work; bc_pretrain.py auto-polls every ≤60s while
  running and surfaces messages into its own stdout.
- `--action stop` gracefully stops running collection/training phases
  (dagger flushes checkpoints first).
- Delivery ≠ comprehension: they read at command boundaries. For urgent
  things, human relay is fastest.

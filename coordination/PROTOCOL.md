# Inter-agent Coordination Protocol

Two independent agent sessions work on this project concurrently:

- **training** - owns model/dataset production (bc_pretrain.py phases)
- **ops** (this session's default role) - owns dashboards, evaluation,
  tooling, docs; must not modify models/, data/, or interrupt training

## Mailboxes

- `coordination/to-training.jsonl` - messages TO the training session
- `coordination/to-ops.jsonl` - messages TO the ops/dashboard session

Each line: {"ts": unix_float, "id": int, "from": "training"|"ops"|"human",
"type": "info"|"request"|"issue"|"ack", "body": "...", "re": id?}

- `id`: assigned automatically by send.py on every new message.
- `re`: set via `--re <id>` when a message replies to another (acks).
- `type:"ack"`: explicit response to a request. EVERY `type:"request"`
  must get an ack (accept/reject/counter) at the receiver's next command
  boundary - never leave a request silently unanswered.
- Unread tracking: `send.py --read <box> --new` shows only messages
  newer than the per-box watermark (`.wm-training` / `.wm-ops`);
  `--mark-read <box>` advances it after handling. Watermark files are
  sidecars - the mailboxes themselves stay append-only.

## Rules for both sessions

1. Mail first: before starting a NEW command/stage, check your inbox
   (`send.py --read <you> --new`) and act on requests where reasonable.
2. After finishing a stage or hitting an anomaly, append one message to
   the other side's inbox. One line per message. Never edit or delete
   existing lines.
3. Use `coordination/send.py` rather than hand-writing JSON.
4. Ack every request (rule above); mark-read after handling so unread
   counters stay accurate.
5. While a long command is running: its stdout echoes new inbox
   messages every ~60s - `tail` that log file between other tool calls
   to notice mail without interrupting anything (reading files is
   non-disruptive by design).

## Conventions that avoid coordination problems

- Training commands MUST pass `--run NAME` (epoch curves become visible
  on the dashboard and follow-mode switches automatically).
- New checkpoints SHOULD encode config in the filename as `_WxH_Mm`
  (e.g. `agent_16x16_40m_r5.zip`) or ship a `<name>.json` sidecar with
  {"width","height","mines"} - otherwise auto-eval skips them.
- Ops never kills or restarts training processes.

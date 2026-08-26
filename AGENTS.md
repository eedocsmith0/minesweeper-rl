# Agent Session Coordination

Two agent sessions share this project. Read `coordination/PROTOCOL.md`
before starting work.

- **training session**: runs bc_pretrain.py phases; owns models/, data/
- **ops session**: dashboards, evaluation, tooling, docs

Check your inbox before starting new work and leave a message for the
other side when you finish a stage:

```
python coordination/send.py --to ops --body "r4 done: 41% on 16x16/20m"
python coordination/send.py --read training --new
tail -5 coordination/to-training.jsonl
```

Live inbox polling: long-running bc_pretrain/watcher commands surface new
inbox messages into their own logs at least once per minute while they
run - you do NOT need to wait for a command to finish to receive mail.
Machine-readable actions are supported by sending a message with an
"action" field, e.g.:

```
python coordination/send.py --to training --type request --action stop \
    --body "pausing collection for maintenance"
```

`action:"stop"` makes any running gen/dagger/train/watcher exit cleanly
at its next checkpoint boundary (dagger flushes collected data first).

## Polling pattern while a long command runs

Running bc_pretrain/watcher commands surface new inbox messages into
their own stdout at least once per 60 seconds. If you launched a long
command that blocks your tool call, you can still notice mail WITHOUT
interrupting training - reading files never disturbs a running process:

```
# between other tool calls, while a train/dagger command is live:
tail -20 /tmp/opencode/<your-log>.log     # inbox lines appear as [inbox ...]
tail -5 coordination/to-training.jsonl    # or read the mailbox directly
```

Recommended habit: launch heavy phases detached with output to a known
log file, then interleave short poll calls (`tail`) with whatever else
you are doing. Emergency stop works even if you never look: send
--action stop as above and the script honors it within ~60s on its own.

## Mail discipline (mandatory)

1. **Mail first**: at the start of every interaction or stage, check your
   inbox (`python coordination/send.py --read <you> --new`) and report
   pending items before doing anything else. Do not start new work with
   unanswered mail you have not acknowledged.
2. **No silent parking**: every `type:"request"` must receive an explicit
   reply (accept / reject / counter-proposal) at your next command
   boundary - even if the answer is "not now because X". Use
   `--type ack --re <id>` so replies thread to their message.
3. After handling a batch of messages, run `--mark-read <you>` so the
   unread counters (dashboard status bar) stay truthful.

An idle session cannot wake itself: if no command is running and no one
prompts you, mail waits. Long-running commands poll every ~60s, so keep
phases chained and rely on the human only for urgent relays.

Mandatory conventions (details in PROTOCOL.md):
1. Pass `--run NAME` on every training/eval command.
2. Name checkpoints `_WxH_Mm` or add a `<name>.json` sidecar.
3. Never kill the other session's processes.

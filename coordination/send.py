"""Send/read inter-agent coordination messages.

Usage:
    python coordination/send.py --to training --type request \
        --body "use --run NAME on training commands"
    python coordination/send.py --read training   # read inbox for training
    python coordination/send.py --read ops
    python coordination/send.py --read ops --new  # only unread messages
    python coordination/send.py --mark-read ops   # advance unread watermark
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

BOXES = ("training", "ops")


def _box_path(to):
    return os.path.join(DIR, f"to-{to}.jsonl")


def _wm_path(to):
    return os.path.join(DIR, f".wm-{to}")


def _next_id(path):
    """One past the largest existing numeric id (missing ids count as 0)."""
    last = 0
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    v = int(json.loads(line).get("id", 0))
                    last = max(last, v)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
    return last + 1


def send(to, sender, mtype, body, action=None, re=None):
    assert to in BOXES
    path = _box_path(to)
    rec = {"ts": time.time(), "id": _next_id(path), "from": sender,
           "type": mtype, "body": body}
    if action:
        rec["action"] = action
    if re is not None:
        rec["re"] = re
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"sent to {to} as id {rec['id']}: {body}")


def _iter_msgs(path):
    """Yield parsed messages, skipping torn/partial trailing lines
    (writers append concurrently and may be mid-write while we read)."""
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read(to, n=10):
    path = _box_path(to)
    if not os.path.isfile(path):
        print("(no messages)")
        return
    lines = list(_iter_msgs(path))[-n:]
    for r in lines:
        t = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
        mid = f" #{r['id']}" if "id" in r else ""
        re = f" re:#{r['re']}" if "re" in r else ""
        print(f"[{t}] {r['from']} ({r['type']}){mid}{re}: {r['body']}")


def read_new(to):
    """Messages newer than the stored watermark; prints count when none."""
    path = _box_path(to)
    wm = 0.0
    if os.path.isfile(_wm_path(to)):
        with open(_wm_path(to)) as f:
            wm = float(f.read().strip() or 0)
    msgs = []
    if os.path.isfile(path):
        for r in _iter_msgs(path):
            if r.get("ts", 0) > wm:
                msgs.append(r)
    if not msgs:
        print("(no unread messages)")
        return
    for r in msgs:
        t = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
        mid = f" #{r['id']}" if "id" in r else ""
        re = f" re:#{r['re']}" if "re" in r else ""
        print(f"[{t}] {r['from']} ({r['type']}){mid}{re}: {r['body']}")
    print(f"({len(msgs)} unread; run --mark-read {to} after handling)")


def mark_read(to):
    path = _box_path(to)
    newest = 0.0
    if os.path.isfile(path):
        for r in _iter_msgs(path):
            newest = max(newest, float(r.get("ts", 0)))
    with open(_wm_path(to), "w") as f:
        f.write(str(newest))
    print(f"watermark for {to} advanced to {newest:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--to", choices=BOXES)
    p.add_argument("--from", dest="sender", default="human",
                   choices=["training", "ops", "human"])
    p.add_argument("--type", default="info",
                   choices=["info", "request", "issue", "ack"])
    p.add_argument("--body", type=str)
    p.add_argument("--action", type=str, default=None,
                   help="machine-readable action for the receiver "
                        "(e.g. 'stop' = graceful shutdown)")
    p.add_argument("--re", type=int, default=None,
                   help="message id this one replies to")
    p.add_argument("--read", choices=BOXES)
    p.add_argument("--new", action="store_true",
                   help="with --read: show only messages newer than the "
                        "unread watermark")
    p.add_argument("--mark-read", choices=BOXES, dest="mark_read")
    args = p.parse_args()
    if args.read and args.new:
        read_new(args.read)
    elif args.read:
        read(args.read)
    elif args.mark_read:
        mark_read(args.mark_read)
    elif args.body:
        send(args.to, args.sender, args.type, args.body, action=args.action,
             re=args.re)
    else:
        p.print_help()

"""Zero-dependency training dashboard + live solver viewer.

Serves a live web UI from JSONL logs written by agents/metrics.py, plus a
real-time view of the rule solver (or a trained model) playing games.

Usage:
    python viz/dashboard.py [--port 8787] [--root runs]

Then open http://localhost:8787          (training metrics)
     and http://localhost:8787/solver    (watch the solver play)
"""
import argparse
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ------------------------------------------------------------------ #
# coordination mailbox status (read-only; watermarks are sidecar files)

def mail_status():
    out = {}
    coord = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coordination")
    for box in ("training", "ops"):
        path = os.path.join(coord, f"to-{box}.jsonl")
        wmpath = os.path.join(coord, f".wm-{box}")
        wm = 0.0
        if os.path.isfile(wmpath):
            try:
                with open(wmpath) as f:
                    wm = float(f.read().strip() or 0)
            except ValueError:
                wm = 0.0
        unread = total = 0
        reqs, acked = {}, set()
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if r.get("ts", 0) > wm:
                        unread += 1
                    rid = r.get("id")
                    if r.get("type") == "request" and isinstance(rid, int):
                        reqs[rid] = r
                    if r.get("type") == "ack" and isinstance(r.get("re"), int):
                        acked.add(r["re"])
        out[box] = {
            "unread": unread,
            "total": total,
            "open_requests": [
                {"id": i, "from": reqs[i].get("from"),
                 "body": reqs[i].get("body", "")[:80]}
                for i in sorted(reqs) if i not in acked],
        }
    return out

# ------------------------------------------------------------------ #
# live solver session state

SESSION = {
    "env": None,
    "solver": None,
    "model": None,
    "model_path": None,
    "cfg": None,
    "last_action": None,
    "status": "idle",       # idle | playing | win | loss
    "moves": 0,
}
LOCK = threading.Lock()
MIRROR_LOCK = threading.Lock()
_MIRROR_LOGGER = None


def mirror_logger():
    global _MIRROR_LOGGER
    with MIRROR_LOCK:
        if _MIRROR_LOGGER is None:
            from agents.metrics import MetricLogger
            _MIRROR_LOGGER = MetricLogger("mirror")
        return _MIRROR_LOGGER


def session_state_payload():
    env = SESSION["env"]
    if env is None:
        return {"status": SESSION["status"]}
    h, w = env.height, env.width
    cells = []
    for r in range(h):
        row = []
        for c in range(w):
            if env.revealed[r, c]:
                row.append(11 if env.mines[r, c]
                           else int(env.counts[r, c]) + 1)
            elif env.flagged[r, c]:
                row.append(10)
            else:
                row.append(0)
        cells.append(row)
    last = None
    if SESSION["last_action"] is not None:
        flat = SESSION["last_action"] % (h * w)
        r, c = divmod(flat, w)
        last = {"r": r, "c": c,
                "flag": SESSION["last_action"] >= h * w,
                "mine": bool(env.mines[r, c]) if env.revealed[r, c] else None}
    return {
        "cells": cells,
        "width": w,
        "height": h,
        "moves": SESSION["moves"],
        "mines": env.num_mines,
        "flags": int(env.flagged.sum()),
        "safe_left": int((~env.mines & ~env.revealed).sum()),
        "status": SESSION["status"],
        "last": last,
    }


def find_matching_checkpoint(width, height, num_mines):
    """Newest zip in models/ whose sidecar/filename matches this board."""
    import glob
    cands = []
    for path in glob.glob(os.path.join(MODELS_DIR, "*.zip")):
        stem = os.path.basename(path)
        m = re.search(r"(\d+)x(\d+)_(\d+)m", stem)
        if m:
            ok = (int(m.group(1)), int(m.group(2)), int(m.group(3))) == \
                 (width, height, num_mines)
        else:
            sidecar = os.path.splitext(path)[0] + ".json"
            try:
                cfg = json.load(open(sidecar))
                ok = (cfg.get("width"), cfg.get("height"),
                      cfg.get("mines")) == (width, height, num_mines)
            except Exception:
                continue
        if ok:
            cands.append((os.path.getmtime(path), path))
    return max(cands)[1] if cands else None


def start_session(width, height, num_mines, model_path=None):
    from env.minesweeper_env import MinesweeperEnv
    from agents.rule_solver import RuleSolver
    env = MinesweeperEnv(width=width, height=height,
                         num_mines=num_mines, seed=None)
    env.reset()
    SESSION["env"] = env
    SESSION["solver"] = RuleSolver(env) if model_path is None else None
    if model_path != SESSION["model_path"]:
        SESSION["model"] = None
    if model_path:
        if SESSION["model"] is None:
            from sb3_contrib import MaskablePPO
            SESSION["model"] = MaskablePPO.load(model_path, device="cpu")
            SESSION["model_path"] = model_path
        SESSION["status"] = "playing"
    else:
        SESSION["model_path"] = None
        SESSION["status"] = "playing"
    SESSION["last_action"] = None
    SESSION["moves"] = 0
    SESSION["end_logged"] = False


def step_session():
    """Advance one move. Returns True if game is over."""
    env, solver, model = SESSION["env"], SESSION["solver"], SESSION["model"]
    mask = env.action_mask()
    if not mask.any():
        SESSION["status"] = "loss" if \
            (~env.mines & ~env.revealed).any() else "win"
        return True
    if model is not None:
        obs = env._get_obs()
        action, _ = model.predict(obs, action_masks=mask,
                                  deterministic=True)
        action = int(action)
    else:
        action = solver.next_action()
    _, _, term, _, info = env.step(action)
    SESSION["last_action"] = action
    SESSION["moves"] += 1
    if term:
        won = info["remaining_safe"] == 0
        SESSION["status"] = "win" if won else "loss"
        return True
    return False


# ------------------------------------------------------------------ #


def list_runs():
    """Run dirs sorted by most recent activity (metrics mtime), newest first."""
    if not os.path.isdir(ROOT):
        return []
    entries = []
    for d in os.listdir(ROOT):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        mpath = os.path.join(p, "metrics.jsonl")
        bpath = os.path.join(p, "boards.jsonl")
        mt = max(
            os.path.getmtime(x) for x in [mpath, bpath] if os.path.isfile(x)
        ) if any(os.path.isfile(x) for x in [mpath, bpath]) else \
            os.path.getmtime(p)
        entries.append((mt, d))
    entries.sort(reverse=True)
    return [d for _, d in entries]


def read_jsonl(path, tail=5000):
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as f:
        lines = f.readlines()[-tail:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run_data(run):
    d = os.path.join(ROOT, run)
    return {
        "run": run,
        "metrics": read_jsonl(os.path.join(d, "metrics.jsonl")),
        "boards": read_jsonl(os.path.join(d, "boards.jsonl"), 60),
    }


def find_training_procs():
    """Scan /proc for running project training scripts (no ps dependency)."""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                parts = f.read().decode("utf-8", "ignore").split("\0")
        except OSError:
            continue
        parts = [p for p in parts if p]
        if len(parts) < 2:
            continue
        script = next((p for p in parts
                       if p.endswith("bc_pretrain.py")
                       or p.endswith("agents/train.py")), None)
        if not script:
            continue

        def arg(flag):
            if flag in parts:
                i = parts.index(flag)
                return parts[i + 1] if i + 1 < len(parts) else None
            return None

        try:
            start_ticks = ""
            with open(f"/proc/{pid}/stat") as f:
                start_ticks = f.read().split(") ")[13].split()[19]
        except Exception:
            pass
        out.append({
            "pid": int(pid),
            "script": os.path.basename(script),
            "phase": arg("--phase") or ("ppo-train"
                                        if script.endswith("train.py")
                                        else "?"),
            "run": arg("--run"),
            # trainers without --run still log into the default bucket
            "effective_run": arg("--run") or (
                "default" if script.endswith("bc_pretrain.py") else None),
            "width": arg("--width"),
            "height": arg("--height"),
            "mines": arg("--mines"),
            "model": os.path.basename(arg("--model") or "")
            if arg("--model") else None,
            "start_ticks": start_ticks,
        })
    out.sort(key=lambda d: (d["start_ticks"], d["pid"]), reverse=True)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif url.path == "/solver":
            self._send(200, SOLVER_HTML.encode(), "text/html; charset=utf-8")
        elif url.path == "/api/runs":
            body = json.dumps(list_runs()).encode()
            self._send(200, body, "application/json")
        elif url.path == "/api/data":
            qs = parse_qs(url.query)
            run = (qs.get("run") or ["default"])[0]
            if run not in list_runs():
                self._send(404, b"{}", "application/json")
                return
            body = json.dumps(run_data(run)).encode()
            self._send(200, body, "application/json")
        elif url.path == "/api/training":
            body = json.dumps(find_training_procs()).encode()
            self._send(200, body, "application/json")
        elif url.path == "/api/mail":
            body = json.dumps(mail_status()).encode()
            self._send(200, body, "application/json")
        elif url.path == "/api/models":
            models = []
            if os.path.isdir(MODELS_DIR):
                for f in sorted(os.listdir(MODELS_DIR)):
                    if f.endswith(".zip"):
                        models.append(f)
            body = json.dumps(models).encode()
            self._send(200, body, "application/json")
        elif url.path == "/api/solver/state":
            with LOCK:
                body = json.dumps(session_state_payload()).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/solver/new":
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                model_path = None
                if req.get("model"):
                    cand = os.path.join(MODELS_DIR, str(req["model"]))
                    if os.path.isfile(cand):
                        model_path = cand
                with LOCK:
                    start_session(int(req.get("width", 9)),
                                  int(req.get("height", 9)),
                                  int(req.get("mines", 10)),
                                  model_path)
                    body = json.dumps(session_state_payload()).encode()
                self._send(200, body, "application/json")
            elif url.path == "/api/solver/step":
                req = {}
                steps = 1
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(length) or b"{}")
                    steps = max(1, min(50, int(req.get("steps", 1))))
                except Exception:
                    pass
                with LOCK:
                    if SESSION["env"] is None or \
                            SESSION["status"] not in ("idle", "playing"):
                        body = json.dumps(session_state_payload()).encode()
                    else:
                        for _ in range(steps):
                            try:
                                if step_session():
                                    break
                            except Exception as e:
                                SESSION["status"] = "error"
                                body = json.dumps(
                                    {"error": str(e),
                                     **session_state_payload()}).encode()
                                self._send(200, body,
                                           "application/json")
                                return
                        body = json.dumps(session_state_payload()).encode()
                    if req.get("log") and \
                            SESSION["status"] in ("win", "loss") and \
                            not SESSION.get("end_logged"):
                        try:
                            mirror_logger().log_board(
                                SESSION["env"], SESSION["status"],
                                SESSION["moves"],
                                extra={"mines": SESSION["env"].num_mines})
                        except Exception:
                            pass
                        SESSION["end_logged"] = True
                self._send(200, body, "application/json")
            elif url.path == "/api/solver/mirror":
                """Resolve the live trainer's config -> game + model."""
                length = int(self.headers.get("Content-Length", 0))
                inst = {}
                try:
                    inst = find_training_procs()[0]
                except Exception:
                    pass
                width = int(inst.get("width") or 9)
                height = int(inst.get("height") or 9)
                mines = int(inst.get("mines") or 10)
                model_file = None
                if inst.get("model"):
                    cand = os.path.join(MODELS_DIR, inst["model"])
                    model_file = inst["model"] if os.path.isfile(cand) \
                        else None
                if not model_file:
                    ckpt = find_matching_checkpoint(width, height, mines)
                    model_file = os.path.basename(ckpt) if ckpt else None
                self._send(200, json.dumps({
                    "width": width, "height": height, "mines": mines,
                    "model": model_file, "pid": inst.get("pid"),
                    "phase": inst.get("phase"),
                }).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:
            try:
                self._send(500, json.dumps({"error": str(e)}).encode(),
                           "application/json")
            except Exception:
                pass


def line_chart(canvas_id, series, colors, title, fmt=lambda v: f"{v:.3f}"):
    return (
        f"drawChart('{canvas_id}', {json.dumps(series)}, "
        f"{json.dumps(colors)}, '{title}', {json.dumps(fmt('%v'))});"
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Minesweeper RL Dashboard</title>
<style>
  :root { --bg:#0f1420; --panel:#171e2e; --ink:#dfe6f2; --dim:#8b96ab;
          --accent:#4fc3f7; --good:#66bb6a; --bad:#ef5350; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--ink);
         font-family:'Segoe UI',system-ui,sans-serif; padding:18px; }
  h1 { font-size:20px; font-weight:600; margin-bottom:4px; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:16px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; align-items:center;
         margin-bottom:14px; }
  select, button { background:var(--panel); color:var(--ink);
    border:1px solid #2a3550; border-radius:6px; padding:6px 10px;
    font-size:13px; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
  .card { background:var(--panel); border-radius:10px; padding:12px 18px;
          min-width:150px; border:1px solid #232c44; }
  .card .k { color:var(--dim); font-size:11px; text-transform:uppercase;
             letter-spacing:.08em; }
  .card .v { font-size:26px; font-weight:600; margin-top:2px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:1000px) { .grid { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border-radius:10px; padding:12px;
           border:1px solid #232c44; min-width:0; }
  .panel h2 { font-size:13px; font-weight:600; color:var(--dim);
              text-transform:uppercase; letter-spacing:.06em;
              margin-bottom:8px; }
  canvas { width:100%; height:220px; display:block; }
  .boards { display:flex; gap:10px; overflow-x:auto; padding-bottom:6px; }
  .bwrap { flex:0 0 auto; text-align:center; }
  .board { display:grid; gap:1px; background:#0a0e18; padding:2px;
           border-radius:4px; width:max-content; }
  .cell { width:15px; height:15px; font-size:10px; line-height:15px;
          text-align:center; font-weight:700; }
  .c0 { background:#2c3654; } .c10 { background:#8d5b16; color:#fff; }
  .c11 { background:#c62828; color:#fff; }
  .c1{background:#102436;color:#64b5f6;} .c2{background:#12301f;color:#81c784;}
  .c3{background:#3a1414;color:#e57373;} .c4{background:#1a1a46;color:#9575cd;}
  .c5{background:#46280f;color:#ffb74d;} .c6{background:#0e3232;color:#4dd0e1;}
  .c7{background:#22242a;color:#eeeeee;} .c8{background:#3c3c50;color:#bdbdbd;}
  .tag { font-size:11px; margin-top:3px; color:var(--dim); }
  .win { color:var(--good); } .loss { color:var(--bad); }
  #status { font-size:12px; color:var(--dim); }
  #mail { font-size:12px; margin-left:8px; }
  .mailbadge { display:inline-block; padding:1px 7px; border-radius:8px;
               margin-left:5px; font-size:11px; background:#2c3654;
               color:#8b96ab; }
  .mailbadge.hot { background:#4a1d1d; color:#ff8a80; font-weight:700; }
</style>
</head>
<body>
<h1>Minesweeper RL - Training Dashboard</h1>
<div class="sub">BC + DAgger pipeline live metrics &middot; ui v2</div>
<div class="row">
  <label>Run: <select id="runSel"></select></label>
  <label style="font-size:13px"><input id="followChk" type="checkbox"
         checked> Follow training</label>
  <button onclick="loadAll()">Refresh</button>
  <span id="status"></span>
  <span id="mail"></span>
</div>
<div class="cards">
  <div class="card"><div class="k">Latest eval win rate</div><div class="v" id="cWin">-</div></div>
  <div class="card"><div class="k">Top-1 agreement</div><div class="v" id="cTop1">-</div></div>
  <div class="card"><div class="k">NLL loss</div><div class="v" id="cNll">-</div></div>
  <div class="card"><div class="k">Dataset size</div><div class="v" id="cData">-</div></div>
  <div class="card"><div class="k">DAgger rounds</div><div class="v" id="cRounds">-</div></div>
</div>
<div class="grid">
  <div class="panel"><h2>Learning curve (top-1 / NLL per epoch)</h2>
    <canvas id="chLearn"></canvas></div>
  <div class="panel"><h2>Eval win rate over checkpoints</h2>
    <canvas id="chEval"></canvas></div>
  <div class="panel"><h2>Transitions per dataset</h2>
    <canvas id="chData"></canvas></div>
  <div class="panel"><h2>Recent games played by the agent</h2>
    <div class="boards" id="boards"></div></div>
</div>
<script>
let curRun = null;

function drawChart(id, seriesList, colors, title, yfmt) {
  const cv = document.getElementById(id);
  const ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth * devicePixelRatio;
  const H = cv.height = cv.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const padL = 46*devicePixelRatio, padB = 20*devicePixelRatio,
        padT = 10*devicePixelRatio, padR = 10*devicePixelRatio;
  let pts = seriesList.flat();
  if (!pts.length) {
    ctx.fillStyle = '#8b96ab';
    ctx.font = `${12*devicePixelRatio}px sans-serif`;
    ctx.fillText('no data yet', padL, H/2);
    return;
  }
  let xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  let x0=Math.min(...xs), x1=Math.max(...xs),
      y0=Math.min(...ys), y1=Math.max(...ys);
  if (x1-x0 < 1e-9) x1 = x0+1;
  if (y1-y0 < 1e-9) y1 = y0+1.1*y1+0.1;
  const px = v => padL + (v-x0)/(x1-x0)*(W-padL-padR);
  const py = v => H-padB - (v-y0)/(y1-y0)*(H-padT-padB);
  ctx.strokeStyle = '#2a3550'; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++) {
    const yy = padT + i*(H-padT-padB)/4;
    ctx.beginPath(); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); ctx.stroke();
    const val = y1 - i*(y1-y0)/4;
    ctx.fillStyle='#8b96ab'; ctx.textAlign='right';
    ctx.font = `${10*devicePixelRatio}px sans-serif`;
    ctx.fillText(yfmt(val), padL-4, yy+3);
  }
  seriesList.forEach((s,i)=>{
    if (!s.length) return;
    ctx.strokeStyle = colors[i]; ctx.lineWidth = 2*devicePixelRatio;
    ctx.beginPath();
    s.forEach((p,j)=>{ j?ctx.lineTo(px(p[0]),py(p[1])):ctx.moveTo(px(p[0]),py(p[1])); });
    ctx.stroke();
  });
}

function renderBoards(boards) {
  const el = document.getElementById('boards');
  el.innerHTML = '';
  boards.slice(-14).reverse().forEach(b => {
    const wrap = document.createElement('div');
    wrap.className = 'bwrap';
    const bd = document.createElement('div');
    bd.className = 'board';
    bd.style.gridTemplateColumns = `repeat(${b.width}, 15px)`;
    b.cells.forEach(row => row.forEach(v => {
      const c = document.createElement('div');
      c.className = 'cell c' + Math.min(v, 11);
      if (v >= 1 && v <= 9) {
        if (v > 1) c.textContent = String(v - 1);
      } else if (v === 10) c.textContent = '\u2691';
      else if (v === 11) c.textContent = '\u2739';
      bd.appendChild(c);
    }));
    const tag = document.createElement('div');
    tag.className = 'tag ' + (b.result === 'win' ? 'win' : 'loss');
    tag.textContent = (b.result==='win'?'WIN ':'LOSS ') + b.moves + ' moves' +
                      (b.mines ? ' \u00b7 ' + b.mines + ' mines' : '');
    wrap.appendChild(bd); wrap.appendChild(tag); el.appendChild(wrap);
  });
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function loadMail() {
  const el = document.getElementById('mail');
  try {
    const m = await (await fetch('/api/mail')).json();
    let html = '';
    for (const box of Object.keys(m)) {
      const b = m[box];
      const open = b.open_requests.length;
      if (!b.unread && !open) continue;
      const hot = open > 0;
      const title = open
        ? `${open} unanswered request(s): ` +
          b.open_requests.map(r=>'#'+r.id+' ('+r.from+'): '+r.body).join(' | ')
        : `${b.unread} unread of ${b.total}`;
      html += `<span class="mailbadge${hot?' hot':''}" title="${esc(title)}">` +
              `${box}: ${open ? open + ' open req' : b.unread + ' unread'}` +
              `</span>`;
    }
    el.innerHTML = html ||
      '<span class="mailbadge" title="both inboxes read">mail: clear</span>';
  } catch(e) { el.textContent = ''; }
}

async function loadAll() {
  const runs = await (await fetch('/api/runs')).json();
  const sel = document.getElementById('runSel');
  sel.innerHTML = runs.map(r=>`<option>${esc(r)}</option>`).join('');

  // follow active training instance
  let banner = '';
  let procs = [];
  try {
    procs = await (await fetch('/api/training')).json();
  } catch(e) {}
  const following = document.getElementById('followChk').checked;
  if (following && procs.length) {
    const inst = procs[0];
    const target = inst.effective_run || inst.run;
    if (target && runs.includes(target)) {
      if (curRun !== target) curRun = target;
      banner = ` | TRAINING pid=${inst.pid} ${inst.script} ` +
               `${inst.phase}${inst.mines?' '+inst.width+'x'+inst.height+
               '/'+inst.mines+'m':''} -> run '${target}'` +
               (inst.run ? '' : ' (default bucket: add --run for a named run)');
    } else if (target) {
      banner = ` | TRAINING pid=${inst.pid} writes run '${target}' ` +
               '(no records yet)';
    }
  }

  curRun = (curRun && runs.includes(curRun)) ? curRun : runs[0];
  if (curRun) sel.value = curRun;
  if (!curRun) {
    document.getElementById('status').textContent =
      'no runs found' + banner;
    await loadMail();
    return;
  }
  const data = await (await fetch('/api/data?run='+curRun)).json();
  const m = data.metrics, bs = data.boards;

  const trainPts = m.filter(r=>r.phase==='train')
                    .map((r,i)=>[i, r.top1 ?? null]).filter(p=>p[1]!==null);
  const nllPts = m.filter(r=>r.phase==='train')
                  .map((r,i)=>[i, r.nll ?? null]).filter(p=>p[1]!==null);
  drawChart('chLearn', [trainPts, nllPts], ['#4fc3f7','#ef5350'],
            '', v=>v.toFixed(2));
  setHint('chLearn', trainPts.length,
          'no epoch metrics in this run - the trainer must pass --run NAME');

  const evalIdx = m.map(r=>r.phase==='eval'?r:null).filter(Boolean);
  const evalPts = evalIdx.map((r,i)=>[i, r.win_rate]);
  drawChart('chEval', [evalPts], ['#66bb6a'], '', v=>(100*v).toFixed(0)+'%');

  const genRecs = {};
  m.filter(r=>r.phase==='data').forEach(r=>{
    genRecs[r.label||r.ts] = r.transitions;
  });
  const dataPts = Object.entries(genRecs).map(([k,v],i)=>[i,v]);
  drawChart('chData', [dataPts], ['#ffb74d'], '',
            v=>(v>=1e6?(v/1e6).toFixed(1)+'M':(v/1000).toFixed(0)+'k'));
  setHint('chData', dataPts.length,
          'no dataset records in this run - gen/dagger phases log these');

  const lastTrain = [...m].reverse().find(r=>r.phase==='train');
  const lastEval = [...m].reverse().find(r=>r.phase==='eval');
  const lastData = [...m].reverse().find(r=>r.phase==='data');
  document.getElementById('cWin').textContent =
    lastEval ? (100*lastEval.win_rate).toFixed(1)+'%' : '-';
  document.getElementById('cTop1').textContent =
    lastTrain ? (100*lastTrain.top1).toFixed(1)+'%' : '-';
  document.getElementById('cNll').textContent =
    lastTrain ? lastTrain.nll.toFixed(3) : '-';
  document.getElementById('cData').textContent =
    lastData ? lastData.transitions.toLocaleString() : '-';
  document.getElementById('cRounds').textContent =
    new Set(m.filter(r=>r.phase==='data').map(r=>r.label)).size || '-';

  renderBoards(bs);
  setHint('boards', bs.length,
          'no logged games in this run - dagger/eval with --run NAME logs boards');

  const age = m.length ? Math.floor(Date.now()/1000 - m[m.length-1].ts) : null;
  document.getElementById('status').textContent =
    `run '${curRun}' | ${m.length} records` +
    (age!==null ? ` | last update ${age}s ago` : '') +
    (age!==null && age < 120 ? ' \u25cf LIVE' : '') + banner;
  await loadMail();
}

function setHint(id, count, msg) {
  const cv = document.getElementById(id);
  let hint;
  if (id === 'boards') {
    hint = cv.parentElement.querySelector('.hint');
    if (!count && !hint) {
      hint = document.createElement('div');
      hint.className = 'hint';
      cv.parentElement.appendChild(hint);
    }
  } else {
    hint = cv.parentElement.querySelector('.hint');
    if (!count && !hint) {
      hint = document.createElement('div');
      hint.className = 'hint';
      cv.parentElement.appendChild(hint);
    }
  }
  if (hint) {
    if (count) { hint.remove(); }
    else {
      hint.style.color = '#8b96ab';
      hint.style.fontSize = '11px';
      hint.style.marginTop = '6px';
      hint.textContent = msg;
    }
  }
}

setInterval(() => loadAll().catch(e=>{
  document.getElementById('status').textContent = 'refresh error: ' + e;
}), 3000);
document.getElementById('runSel').onchange = () => {
  curRun = document.getElementById('runSel').value;
  document.getElementById('followChk').checked = false;
  loadAll();
};
document.getElementById('followChk').onchange = () => {
  if (document.getElementById('followChk').checked) loadAll();
};
loadAll();
</script>
</body>
</html>
"""


SOLVER_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Live Solver - Minesweeper RL</title>
<style>
  :root { --bg:#0f1420; --panel:#171e2e; --ink:#dfe6f2; --dim:#8b96ab; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--ink);
         font-family:'Segoe UI',system-ui,sans-serif; padding:18px; }
  h1 { font-size:20px; font-weight:600; margin-bottom:4px; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
         margin-bottom:14px; }
  select, button, input { background:var(--panel); color:var(--ink);
    border:1px solid #2a3550; border-radius:6px; padding:6px 10px;
    font-size:13px; }
  button.primary { background:#1c4d70; }
  button:hover { border-color:#4fc3f7; }
  input[type=number] { width:70px; }
  #board { display:grid; gap:2px; background:#0a0e18; padding:3px;
           width:max-content; border-radius:6px;
           transition: opacity .2s; }
  .cell { width:26px; height:26px; font-size:14px; line-height:26px;
          text-align:center; font-weight:700; border-radius:2px;
          position:relative; }
  .cell.last::after { content:''; position:absolute; inset:0;
                      outline:2px solid #4fc3f7; outline-offset:-2px;
                      border-radius:2px; }
  .cell.boom::after { outline-color:#ef5350; }
  .c0 { background:#2c3654; } .c10 { background:#8d5b16; color:#fff; }
  .c11 { background:#c62828; color:#fff; }
  .c1{background:#102436;color:#64b5f6;} .c2{background:#12301f;color:#81c784;}
  .c3{background:#3a1414;color:#e57373;} .c4{background:#1a1a46;color:#9575cd;}
  .c5{background:#46280f;color:#ffb74d;} .c6{background:#0e3232;color:#4dd0e1;}
  .c7{background:#22242a;color:#eeeeee;} .c8{background:#3c3c50;color:#bdbdbd;}
  .boards { display:flex; gap:10px; overflow-x:auto; padding-bottom:6px;
            max-width:100%; }
  .bwrap { flex:0 0 auto; text-align:center; }
  .board { display:grid; gap:1px; background:#0a0e18; padding:2px;
           border-radius:4px; width:max-content; }
  .tag { font-size:11px; margin-top:3px; color:var(--dim); }
  .win { color:var(--good); } .loss { color:var(--bad); }
  .cell.mini { width:11px; height:11px; font-size:8px; line-height:11px; }
  #mirrorInfo { font-size:12px; color:#8b96ab; }
  .stats { display:flex; gap:22px; margin-bottom:12px; }
  .stats div span { color:var(--dim); font-size:11px;
                    text-transform:uppercase; letter-spacing:.07em;
                    display:block; }
  .stats div b { font-size:20px; }
  #statusTag { padding:3px 10px; border-radius:5px; font-size:13px;
               background:var(--panel); }
  .win { background:#1b3a24 !important; color:#81c784 !important; }
  .loss { background:#46201f !important; color:#ef9a9a !important; }
  a { color:#4fc3f7; text-decoration:none; }
</style>
</head>
<body>
<h1>Live Solver</h1>
<div class="sub">watch the constraint solver (or a trained model) play,
one move at a time &nbsp;&middot;&nbsp; <a href="/">training dashboard</a></div>
<div class="row">
  <label>Presets:
    <select id="preset">
      <option value="9x9x10">Beginner 9x9 / 10</option>
    <option value="16x16x40">Intermediate 16x16 / 40</option>
      <option value="16x30x99">Expert 16x30 / 99</option>
    <option value="6x6x4">Small 6x6 / 4</option>
      <option value="custom">Custom</option>
    </select></label>
  <label>W <input id="w" type="number" value="9" min="4" max="30"></label>
  <label>H <input id="h" type="number" value="9" min="4" max="30"></label>
  <label>Mines <input id="m" type="number" value="10" min="1"></label>
  <label>Agent:
    <select id="modelSel"><option value="">Rule solver</option></select></label>
  <label>Speed
    <input id="speed" type="range" min="0" max="1000" value="250"
           style="width:120px">
    <input id="msInput" type="number" min="0" max="5000" step="1"
           value="250" style="width:80px"> ms/move
    <span id="msLabel"></span></label>
  <button class="primary" id="btnNew">New game</button>
  <button id="btnPause">⏸ Pause</button>
  <button id="btnMirror">▶ Mirror live training</button>
  <label style="font-size:13px"><input id="autoNext" type="checkbox"
         checked> Auto-play next game</label>
  <span id="statusTag">idle</span>
  <span id="mirrorInfo" style="font-size:12px;color:#8b96ab"></span>
</div>
<div class="stats">
  <div><span>Moves</span><b id="stMoves">0</b></div>
  <div><span>Flags</span><b id="stFlags">0</b></div>
  <div><span>Safe left</span><b id="stSafe">-</b></div>
  <div><span>Time</span><b id="stTime">-</b></div>
  <div><span>Session record</span><b id="stTally">-</b></div>
</div>
<div id="board"></div>
<div class="panel" style="margin-top:14px"><h2>Past games (this session)</h2>
  <div class="boards" id="miniGallery"></div></div>

<script>
let timer = null, paused = false, nextTimer = null, errStreak = 0;
let tally = {win: 0, loss: 0};
let mirror = false;
let gameStart = null, finalMs = null;

const $ = id => document.getElementById(id);

function fmtTime(ms) {
  if (ms === null) return '-';
  if (ms < 1000) return Math.round(ms) + ' ms';
  const s = ms / 1000;
  if (s < 90) return s.toFixed(1) + ' s';
  const m = Math.floor(s / 60);
  return m + 'm ' + Math.round(s % 60) + 's';
}

function cellHtml(v, isLast, boom) {
  const cls = 'cell c' + Math.min(v, 11) +
              (isLast ? ' last' : '') + (boom ? ' boom' : '');
  let t = '';
  if (v >= 2 && v <= 9) t = String(v - 1);
  else if (v === 10) t = '⚑';
  else if (v === 11) t = '✹';
  return `<div class="${cls}">${t}</div>`;
}

function render(s) {
  const bd = $('board');
  bd.style.gridTemplateColumns = `repeat(${s.width}, 26px)`;
  let html = '';
  const la = s.last || {};
  s.cells.forEach((row, r) => row.forEach((v, c) => {
    const isLast = la.r === r && la.c === c;
    const boom = isLast && la.mine;
    html += cellHtml(v, isLast, boom);
  }));
  bd.innerHTML = html;
  $('stMoves').textContent = s.moves ?? '-';
  $('stFlags').textContent = s.flags ?? '-';
  $('stSafe').textContent = s.safe_left ?? '-';
  if (s.status === 'playing' && s.moves === 0) gameStart = performance.now();
  if (gameStart !== null && finalMs === null) {
    const el = (s.status === 'playing')
      ? performance.now() - gameStart
      : (performance.now() - gameStart);
    if (s.status !== 'playing') finalMs = el;
    $('stTime').textContent = fmtTime(el);
  }
  if (s.status === 'error') { tag('solver error - restarting', 'loss'); return; }
  const tag2 = $('statusTag');
  tag2.className = '';
  tag2.textContent = s.status;
  if (s.status === 'win') tag2.classList.add('win');
  if (s.status === 'loss') tag2.classList.add('loss');
  if (['win','loss'].includes(s.status)) {
    stopTimer();
    tally[s.status]++;
    const tot = tally.win + tally.loss;
    $('stTally').textContent = tally.win + 'W / ' + tally.loss + 'L (' +
      Math.round(100 * tally.win / tot) + '%)';
    refreshGallery();
    if ($('autoNext').checked && !paused) {
      clearTimeout(nextTimer);
      nextTimer = setTimeout(() => { nextTimer = null; newGame(); }, 1600);
    }
  }
}

async function refreshGallery() {
  try {
    const d = await (await fetch('/api/data?run=mirror')).json();
    const el = document.getElementById('miniGallery');
    el.innerHTML = '';
    d.boards.slice(-8).reverse().forEach(b => {
      const wrap = document.createElement('div');
      wrap.className = 'bwrap';
      const bd = document.createElement('div');
      bd.className = 'board';
      bd.style.gridTemplateColumns = `repeat(${b.width}, 11px)`;
      b.cells.forEach(row => row.forEach(v => {
        const c = document.createElement('div');
        c.className = 'cell mini c' + Math.min(v, 11);
        bd.appendChild(c);
      }));
      const tagEl = document.createElement('div');
      tagEl.className = 'tag ' + (b.result === 'win' ? 'win' : 'loss');
      tagEl.textContent = (b.result === 'win' ? 'W' : 'L') + ' ' + b.moves + 'm';
      wrap.appendChild(bd); wrap.appendChild(tagEl); el.appendChild(wrap);
    });
  } catch (e) {}
}

async function post(url, body) {
  const r = await fetch(url, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body||{})});
  return r.json();
}

async function newGame() {
  stopTimer();
  clearTimeout(nextTimer);
  errStreak = 0;
  gameStart = null; finalMs = null;
  $('stTime').textContent = '-';
  const body = {width:+$('w').value, height:+$('h').value,
                mines:+$('m').value, model:$('modelSel').value,
                log: mirror};
  if (mirror) {
    try {
      const m = await post('/api/solver/mirror', {});
      if (m.pid) {
        body.width = m.width; body.height = m.height; body.mines = m.mines;
        $('w').value = m.width; $('h').value = m.height;
        $('m').value = m.mines;
        if (m.model) {
          body.model = m.model;
          const sel = $('modelSel');
          if (![...sel.options].some(o => o.value === m.model)) {
            const o = document.createElement('option');
            o.value = m.model; o.textContent = 'Model: ' + m.model;
            sel.appendChild(o);
          }
          sel.value = m.model;
        } else { body.model=''; $('modelSel').value=''; }
        mirrorInfo('mirroring pid ' + m.pid + ' (' + m.phase + ') -> ' +
                   (m.model || 'rule solver'));
      } else {
        mirrorInfo('no live training detected - using current settings');
      }
    } catch (e) {}
  }
  render(await post('/api/solver/new', body));
  if (!paused) startTimer();
}

function mirrorInfo(txt) {
  const el = document.getElementById('mirrorInfo');
  if (el) el.textContent = txt;
}

function startTimer() {
  stopTimer();
  const ms = Math.max(0, +$('msInput').value || 0);
  const turbo = ms === 0;
  const delay = turbo ? 0 : Math.max(ms, 30);
  const steps = turbo ? 40 : 1;
  updateSpeedLabel(ms);
  const tick = async () => {
    if (document.hidden) { timer = setTimeout(tick, delay); return; }
    try {
      const s = await post('/api/solver/step', {steps});
      if (s.error) throw new Error(s.error);
      errStreak = 0;
      render(s);
      if (s.status === 'idle') { await newGame(); return; }
    } catch (e) {
      errStreak++;
      tag('reconnecting x' + errStreak);
      if (errStreak >= 3) { await newGame(); return; }
    }
    if (timer !== null) timer = setTimeout(tick, delay);
  };
  timer = setTimeout(tick, delay);
}
function stopTimer() { if (timer) { clearTimeout(timer); timer = null; } }

function tag(txt, cls) {
  const t = $('statusTag');
  t.className = cls || '';
  t.textContent = txt;
}

$('btnNew').onclick = () => { tally = {win:0, loss:0};
                              $('stTally').textContent='-'; newGame(); };
$('btnMirror').onclick = () => {
  mirror = !mirror;
  $('btnMirror').textContent = mirror ? '◇ Mirror: ON'
                                      : '▶ Mirror live training';
  $('btnMirror').classList.toggle('primary', mirror);
  tally = {win:0, loss:0}; $('stTally').textContent='-';
  if (mirror) newGame();
};
$('btnPause').onclick = () => {
  paused = !paused;
  $('btnPause').textContent = paused ? '▶ Resume' : '⏸ Pause';
  if (paused) { stopTimer(); clearTimeout(nextTimer); }
  else startTimer();
};
$('speed').oninput = () => {
  $('msInput').value = $('speed').value;
  applySpeed(+$('speed').value);
};
$('msInput').onchange = () => {
  let v = Math.max(0, Math.round(+$('msInput').value || 0));
  $('msInput').value = v;
  $('speed').value = Math.min(v, 1000);
  applySpeed(v);
};

function applySpeed(ms) {
  updateSpeedLabel(ms);
  if (timer) startTimer();
}

function updateSpeedLabel(ms) {
  if (ms === 0) { $('msLabel').textContent = 'TURBO - as fast as possible'; }
  else { $('msLabel').textContent = ms + ' ms/move'; }
}
$('preset').onchange = () => {
  const v = $('preset').value.split('x');
  if (v.length === 3) {
    $('w').value = v[0]; $('h').value = v[1]; $('m').value = v[2];
    newGame();
  }
};

(async () => {
  try {
    const models = await (await fetch('/api/models')).json();
    const sel = $('modelSel');
    models.forEach(mo => {
      const o = document.createElement('option');
      o.value = mo; o.textContent = 'Model: ' + mo;
      sel.appendChild(o);
    });
  } catch(e) {}
  newGame();
})();
</script>
</body>
</html>
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--root", default=None,
                    help="runs directory (default: <project>/runs)")
    return ap.parse_args()


def main():
    global ROOT
    args = parse_args()
    if args.root:
        ROOT = args.root if os.path.isabs(args.root) \
            else os.path.join(BASE_DIR, args.root)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard on http://localhost:{args.port} "
          f"(serving runs from {ROOT}/)")
    srv.serve_forever()


if __name__ == "__main__":
    main()

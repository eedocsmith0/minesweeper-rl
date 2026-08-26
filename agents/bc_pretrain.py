"""Behavioral cloning: pretrain the policy on rule-solver moves.

Phase 1 (gen): the RuleSolver plays many games; every transition is stored
as (observation, action_mask, action).
Phase 2 (train): negative log-likelihood of the taken actions under the
MaskablePPO policy distribution, optimized with the model's own optimizer,
so the checkpoint is directly usable by agents/train.py --model.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch as th

from agents.metrics import MetricLogger
from env.minesweeper_env import MinesweeperEnv
from agents.rule_solver import RuleSolver


class TransitionBuffer:
    """Memory-efficient growing buffer for observations (uint8-coded).

    Encoding is lossless: channels 0-10 are binary; channels 11-12 hold
    k/8 values stored as k*32 (exact in uint8).
    """

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.cap = 0
        self.n = 0
        self.obs = None
        self.masks = []
        self.acts = []
        self.certain = []
        self.prio = []

    def add(self, obs, mask, act, certain=None, prio=None):
        if obs.dtype != np.uint8:
            obs = encode_obs_u8(obs)
        if self.n == self.cap:
            new_cap = max(100_000, self.cap * 2)
            new_obs = np.zeros((new_cap, 13, self.h, self.w), np.uint8)
            if self.obs is not None:
                new_obs[:self.n] = self.obs
            self.obs = new_obs
            self.cap = new_cap
        self.obs[self.n] = obs
        self.masks.append(mask)
        self.acts.append(act)
        if certain is None:
            certain = np.zeros(2 * self.h * self.w, dtype=np.int8)
        self.certain.append(certain)
        self.prio.append(1.0 if prio is None else float(prio))
        self.n += 1

    def set_last_prio(self, prio):
        """Set the weight of the most recently added transition."""
        self.prio[-1] = float(prio)

    def finalize(self):
        return (self.obs[:self.n], np.stack(self.masks),
                np.asarray(self.acts, dtype=np.int64),
                np.stack(self.certain),
                np.asarray(self.prio, dtype=np.float32))


def encode_obs_u8(obs):
    """float32 (13,H,W) -> lossless uint8 encoding."""
    u = np.empty(obs.shape, np.uint8)
    u[:11] = np.clip(obs[:11], 0, 1).astype(np.uint8)
    u[11:] = np.clip(np.round(obs[11:] * 32), 0, 255).astype(np.uint8)
    return u


def decode_obs_u8(u8):
    f = u8.astype(np.float32)
    if f.ndim == 4:
        f[:, 11:, :, :] /= 32.0
    else:
        f[11:] /= 32.0
    return f


def generate_dataset(width, height, num_mines, n_games, seed0, out_path,
                     logger=None):
    buf = TransitionBuffer(height, width)
    gamma = 0.99
    rets_all = []
    n_actions = 2 * height * width
    for g in range(n_games):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed0 + g)
        env.reset(seed=seed0 + g)
        solver = RuleSolver(env)
        traj_rew = []
        while True:
            mask = env.action_mask()
            if not mask.any():
                break
            act, ainfo = solver.next_action(return_info=True)
            certain = np.zeros(n_actions, dtype=np.int8)
            for a in ainfo["certain"]:
                if mask[a]:
                    certain[a] = 1
            buf.add(env._get_obs(), mask, act, certain)
            _, rew, term, trunc, _ = env.step(act)
            traj_rew.append(rew)
            if term or trunc:
                break
        returns = np.zeros(len(traj_rew), dtype=np.float32)
        acc = 0.0
        for t in reversed(range(len(traj_rew))):
            acc = traj_rew[t] + gamma * acc
            returns[t] = acc
        rets_all.extend(returns.tolist())
        del traj_rew
    obs, masks, acts, certain, prio = buf.finalize()
    rets = np.array(rets_all, dtype=np.float32)
    np.savez_compressed(out_path, obs=obs, masks=masks, acts=acts,
                        rets=rets, certain=certain, prio=prio)
    print(f"saved {len(acts)} transitions to {out_path}")
    if logger:
        logger.log(phase="data", label=os.path.basename(out_path),
                   transitions=int(len(acts)), mines=num_mines)


_INBOX_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "coordination", "to-training.jsonl")
INBOX_CHECK_SECS = 60


def check_inbox(since_ts):
    """Messages posted to the training inbox after since_ts."""
    msgs = []
    try:
        with open(_INBOX_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ts", 0) > since_ts:
                    msgs.append(r)
    except OSError:
        pass
    return msgs


def poll_inbox(started_ts, state):
    """Surface new inbox messages; honors {"action":"stop"} requests."""
    state["last"] = time.time()
    msgs = check_inbox(started_ts)
    for m in msgs:
        body = str(m.get("body", ""))[:220]
        print(f"[inbox {time.strftime('%H:%M:%S')}] "
              f"{m.get('from')}/{m.get('type')}: {body}", flush=True)
        if m.get("action") == "stop":
            print("[inbox] stop requested - wrapping up cleanly",
                  flush=True)
            state["stop"] = True
    return msgs


def inbox_due(state):
    return time.time() - state.get("last", 0) >= INBOX_CHECK_SECS


def _flush_shard(out_path, k, g0, g1, buf):
    """Atomically write a checkpoint shard covering games [g0, g1)."""
    obs, masks, acts, certain, prio = buf.finalize()
    tmp = f"{out_path}.flush{k:04d}.tmp.npz"
    final = f"{out_path}.flush{k:04d}.npz"
    np.savez_compressed(tmp, obs=obs, masks=masks, acts=acts,
                        rets=np.zeros(len(acts), np.float32),
                        certain=certain, prio=prio,
                        grange=np.array([g0, g1], dtype=np.int64))
    os.replace(tmp, final)  # atomic: readers never see partial shards
    return final


def _manifest_append(out_path, record):
    with open(out_path + ".manifest.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), **record}) + "\n")


def _recover_flush_state(out_path):
    """Return (valid_shards, done_games) from leftover flush shards.

    Stops at the first unreadable/disordered shard (dropped as corrupt).
    """
    import glob
    valid, done = [], 0
    for s in sorted(glob.glob(out_path + ".flush*.npz")):
        if ".tmp." in s:
            continue
        try:
            d = np.load(s)
            g0, g1 = (int(x) for x in d["grange"])
        except Exception:
            print(f"[resume] dropping corrupt shard {s}")
            os.remove(s)
            continue
        if g0 != done or g1 <= g0:
            print(f"[resume] dropping disordered shard {s} "
                  f"(expected start {done}, has [{g0},{g1}))")
            os.remove(s)
            continue
        # NOTE: transition count intentionally NOT validated against the
        # game range - one game yields many moves (variable per game).
        valid.append(s)
        done = g1
    return valid, done


def dagger_roll(model_path, width, height, num_mines, n_games, seed0,
                out_path, device, logger=None, tta=False,
                flush_every=0, resume=False, priority_weights=False):
    """Roll with the learned policy; label every visited state with the
    rule-solver's choice (classic DAgger)."""
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path, device=device)
    buf = TransitionBuffer(height, width)
    board_every = max(1, n_games // 50)
    n_actions = 2 * height * width

    # ---- checkpoint/resume state ------------------------------------ #
    g_start = 0
    shards = []
    manifest = out_path + ".manifest.jsonl"
    has_history = os.path.exists(manifest) or \
        bool(glob.glob(out_path + ".flush*.npz"))
    if resume and not has_history:
        print("[resume] nothing to resume - starting fresh")
    if has_history and not resume:
        raise SystemExit(
            f"[dagger] leftover checkpoint data for {out_path} exists.\n"
            f"  Add --resume to continue it, or delete "
            f"{out_path}.flush*.npz + .manifest.jsonl to discard.")
    if resume:
        recovered, done = _recover_flush_state(out_path)
        shards.extend(recovered)
        g_start = done
        last_ckpt = None
        if os.path.exists(manifest):
            with open(manifest) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("checkpoint"):
                            last_ckpt = r["checkpoint"]
                    except json.JSONDecodeError:
                        pass
        if last_ckpt and os.path.basename(last_ckpt) != \
                os.path.basename(model_path):
            print(f"[resume] WARNING: collected games came from policy "
                  f"'{os.path.basename(last_ckpt)}' but you are rolling "
                  f"from '{os.path.basename(model_path)}' - merged data "
                  f"will be mixed-distribution.")
        print(f"[resume] adopting {len(shards)} shard(s), resuming at "
              f"game index {g_start}/{n_games}")
        _manifest_append(out_path, {
            "event": "resume", "checkpoint": model_path,
            "seed0": seed0, "g_done": g_start, "target": n_games})

    if g_start >= n_games:
        merge_shards(shards, out_path, height, width, logger=logger,
                     num_mines=num_mines)
        _manifest_append(out_path, {"event": "complete"})
        print("[resume] target already reached - merged and exiting")
        return

    _manifest_append(out_path, {"event": "start", "checkpoint": model_path,
                                "seed0": seed0, "g0": g_start,
                                "g1": n_games, "flush_every": flush_every})

    flushed_through = g_start
    games_done = g_start
    flush_k = len(shards)
    started_ts = time.time()
    inbox_state = {"last": started_ts, "stop": False}

    for g in range(g_start, n_games):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed0 + g)
        env.reset(seed=seed0 + g)
        solver = RuleSolver(env)
        done = False
        moves = 0
        max_moves = 3 * width * height
        while not done and moves < max_moves:
            mask = env.action_mask()
            if not mask.any():
                break
            act, ainfo = solver.next_action(return_info=True)
            certain = np.zeros(n_actions, dtype=np.int8)
            for a in ainfo["certain"]:
                if mask[a]:
                    certain[a] = 1
            cur_obs = env._get_obs()
            buf.add(cur_obs, mask, act, certain)
            if tta and width == height:
                action = tta_pick(model, cur_obs, mask, width, height,
                                  device)
            else:
                action, _ = model.predict(
                    decode_obs_u8(buf.obs[buf.n - 1]),
                    action_masks=mask, deterministic=True)
            if priority_weights and certain.any() \
                    and not certain[int(action)]:
                # student error: greedy pick left the teacher's certain
                # safe set -> upweight this transition
                buf.set_last_prio(2.0)
            _, _, term, trunc, _ = env.step(int(action))
            done = term or trunc
            moves += 1
        if logger and g % board_every == 0:
            won = bool((env.revealed | env.mines).all())
            logger.log_board(env, "win" if won else "loss", moves,
                             extra={"mines": num_mines})
        games_done = g + 1
        if inbox_due(inbox_state):
            poll_inbox(started_ts, inbox_state)
            if inbox_state["stop"]:
                print(f"[dagger] stopping early at game {games_done} "
                      f"(collected data is checkpointed)", flush=True)
                break
        if flush_every and games_done - flushed_through >= flush_every:
            shard = _flush_shard(out_path, flush_k, flushed_through,
                                 games_done, buf)
            shards.append(shard)
            flush_k += 1
            _manifest_append(out_path, {"event": "flush",
                                        "g0": flushed_through,
                                        "g1": games_done,
                                        "transitions": int(buf.n)})
            flushed_through = games_done
            buf = TransitionBuffer(height, width)

    effective_end = max(n_games, flushed_through)
    if inbox_state["stop"]:
        effective_end = games_done
    if effective_end > flushed_through and (buf.n or not shards):
        shard = _flush_shard(out_path, flush_k, flushed_through,
                             effective_end, buf)
        shards.append(shard)

    merge_shards(shards, out_path, height, width, logger=logger,
                 num_mines=num_mines)
    _manifest_append(out_path, {"event": "complete",
                                "games_collected": effective_end - g_start
                                if not inbox_state["stop"] else
                                "stopped_early"})


# ------------------------------------------------------------------ #
# parallel rollout support (optional; serial paths above are unchanged)

_DAGGER_CTX = {}


def _dagger_init(model_path, device):
    from sb3_contrib import MaskablePPO
    import torch as _th
    _th.set_num_threads(1)
    _DAGGER_CTX["model"] = MaskablePPO.load(model_path, device=device)


def _play_game_range(model, g0, g1, seed0, width, height, num_mines,
                     tta, shard, run_name, device="cpu",
                     priority_weights=False):
    buf = TransitionBuffer(height, width)
    logger = MetricLogger(run_name) if run_name else None
    board_every = max(1, max(1, g1 - g0) // 25)
    n_actions = 2 * height * width
    for g in range(g0, g1):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed0 + g)
        env.reset(seed=seed0 + g)
        solver = RuleSolver(env)
        done = False
        moves = 0
        max_moves = 3 * width * height
        while not done and moves < max_moves:
            mask = env.action_mask()
            if not mask.any():
                break
            act, ainfo = solver.next_action(return_info=True)
            certain = np.zeros(n_actions, dtype=np.int8)
            for a in ainfo["certain"]:
                if mask[a]:
                    certain[a] = 1
            cur_obs = env._get_obs()
            buf.add(cur_obs, mask, act, certain)
            if tta and width == height:
                action = tta_pick(model, cur_obs.astype(np.float32), mask,
                                  width, height, device)
            else:
                action, _ = model.predict(
                    cur_obs.astype(np.float32), action_masks=mask,
                    deterministic=True)
            if priority_weights and certain.any() \
                    and not certain[int(action)]:
                buf.set_last_prio(2.0)
            _, _, term, trunc, _ = env.step(int(action))
            done = term or trunc
            moves += 1
        if logger and (g - g0) % board_every == 0:
            won = bool((env.revealed | env.mines).all())
            logger.log_board(env, "win" if won else "loss", moves,
                             extra={"mines": num_mines})
    obs, masks, acts, certain, prio = buf.finalize()
    np.savez_compressed(shard, obs=obs, masks=masks, acts=acts,
                        rets=np.zeros(len(acts), np.float32),
                        certain=certain, prio=prio)
    return shard, int(len(acts))


def _dagger_chunk(job):
    (g0, g1, seed0, shard, width, height, num_mines, tta, run_name,
     device, priority_weights) = job
    model = _DAGGER_CTX["model"]
    return _play_game_range(model, g0, g1, seed0, width, height,
                            num_mines, tta, shard, run_name, device,
                            priority_weights=priority_weights)


def _gen_chunk(job):
    g0, g1, seed0, shard, width, height, num_mines, run_name = job
    import torch as _th
    _th.set_num_threads(1)
    buf = TransitionBuffer(height, width)
    gamma = 0.99
    rets_all = []
    n_actions = 2 * height * width
    for g in range(g0, g1):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed0 + g)
        env.reset(seed=seed0 + g)
        solver = RuleSolver(env)
        traj_rew = []
        while True:
            mask = env.action_mask()
            if not mask.any():
                break
            act, ainfo = solver.next_action(return_info=True)
            certain = np.zeros(n_actions, dtype=np.int8)
            for a in ainfo["certain"]:
                if mask[a]:
                    certain[a] = 1
            buf.add(env._get_obs(), mask, act, certain)
            _, rew, term, trunc, _ = env.step(act)
            traj_rew.append(rew)
            if term or trunc:
                break
        returns = np.zeros(len(traj_rew), dtype=np.float32)
        acc = 0.0
        for t in reversed(range(len(traj_rew))):
            acc = traj_rew[t] + gamma * acc
            returns[t] = acc
        rets_all.extend(returns.tolist())
    obs, masks, acts, certain, prio = buf.finalize()
    rets = np.array(rets_all, dtype=np.float32)
    np.savez_compressed(shard, obs=obs, masks=masks, acts=acts,
                        rets=rets, certain=certain, prio=prio)
    return shard, int(len(acts))


def merge_shards(shards, out_path, height, width, logger=None,
                 num_mines=None):
    lens = []
    for s in shards:
        d = np.load(s)
        lens.append(len(d["acts"]))
        del d
    total = sum(lens)
    n_act = 2 * height * width
    obs = np.empty((total, 13, height, width), np.uint8)
    masks = np.empty((total, n_act), np.int8)
    acts = np.empty(total, np.int64)
    rets = np.empty(total, np.float32)
    certain = np.zeros((total, n_act), np.int8)
    prio = np.ones(total, np.float32)
    pos = 0
    for s, ln in zip(shards, lens):
        d = np.load(s)
        raw = d["obs"]
        if raw.dtype == np.uint8:
            obs[pos:pos + ln] = raw
        else:
            obs[pos:pos + ln] = encode_obs_u8(raw.astype(np.float32))
        masks[pos:pos + ln] = d["masks"]
        acts[pos:pos + ln] = d["acts"]
        r = d["rets"] if "rets" in d.files else np.zeros(ln, np.float32)
        rets[pos:pos + ln] = r
        if "certain" in d.files:
            certain[pos:pos + ln] = d["certain"]
        if "prio" in d.files:
            prio[pos:pos + ln] = d["prio"]
        pos += ln
        del d
    np.savez_compressed(out_path, obs=obs, masks=masks, acts=acts,
                        rets=rets, certain=certain, prio=prio)
    for s in shards:
        os.remove(s)
    print(f"saved {total} transitions to {out_path} "
          f"(merged {len(shards)} shards)")
    if logger:
        logger.log(phase="data", label=os.path.basename(out_path),
                   transitions=int(total), mines=num_mines)


def run_parallel_collection(kind, model_path, width, height, num_mines,
                            n_games, seed0, out_path, workers, device,
                            tta=False, run_name=None, logger=None,
                            priority_weights=False):
    """Spawn worker processes that each play a contiguous game range and
    write a shard file; shards are merged into out_path."""
    import multiprocessing as mp

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base = os.path.basename(out_path)[:-4]
    edges = [round(i * n_games / workers) for i in range(workers + 1)]
    jobs, shards = [], []
    for w in range(workers):
        g0, g1 = edges[w], edges[w + 1]
        if g1 <= g0:
            continue
        shard = os.path.join(os.path.dirname(out_path) or ".",
                             f"{base}.p{w}.npz")
        shards.append(shard)
        if kind == "dagger":
            jobs.append((g0, g1, seed0, shard, width, height, num_mines,
                         tta, run_name, device, priority_weights))
        else:
            jobs.append((g0, g1, seed0, shard, width, height, num_mines,
                         run_name))
    ctx = mp.get_context("spawn")
    if not jobs:
        raise SystemExit("[parallel] nothing to collect "
                         "(games x workers produced no chunks)")
    if kind == "dagger":
        pool = ctx.Pool(len(jobs), initializer=_dagger_init,
                        initargs=(model_path, device))
        fn = _dagger_chunk
    else:
        pool = ctx.Pool(len(jobs))
        fn = _gen_chunk
    try:
        results = pool.map(fn, jobs)
    finally:
        pool.close()
        pool.join()
    done_shards = [s for s, _n in results]
    merge_shards(done_shards, out_path, height, width, logger=logger,
                 num_mines=num_mines)


def _plan_takes(lens, max_transitions, rng):
    """Per-file transition counts that fit the budget.

    Newest files (last in the --data list) are kept in full, oldest are
    proportionally downsampled; 25% of the budget is reserved for the
    older half so early-stage knowledge is not entirely lost.
    """
    total = sum(lens)
    n = len(lens)
    if not max_transitions or total <= max_transitions:
        return list(lens), total
    budget = min(max_transitions, total)
    takes = [0] * n
    reserve = budget // 4 if n > 2 else 0
    newest_pool = budget - reserve
    remaining = newest_pool
    for i in reversed(range(n)):
        if remaining <= 0:
            break
        take = min(lens[i], remaining)
        takes[i] = take
        remaining -= take
    older = [i for i in range(n) if takes[i] == 0]
    older_total = sum(lens[i] for i in older)
    if older and reserve > 0 and older_total > 0:
        left = reserve
        for i in older:
            take = min(int(round(lens[i] * reserve / older_total)), left)
            takes[i] = take
            left -= take
            if left <= 0:
                break
    print(f"[data cap] dataset has {total} transitions; "
          f"using {sum(takes)} (newest-first full + proportional old)")
    return takes, sum(takes)


def load_datasets(paths, height, width, max_transitions=None, seed=0):
    """Concatenate npz datasets with bounded peak memory.

    When max_transitions is set and the combined datasets exceed it,
    newer files are kept in full and older files are uniformly
    downsampled to fit the budget (RAM stays bounded as history grows).
    """
    rng = np.random.default_rng(seed)
    lens = []
    for path in paths:
        d = np.load(path)
        lens.append(len(d["acts"]))
        del d
    takes, total = _plan_takes(lens, max_transitions, rng)
    n_act = 2 * height * width
    obs = np.empty((total, 13, height, width), dtype=np.uint8)
    masks = np.empty((total, n_act), dtype=np.int8)
    acts = np.empty(total, dtype=np.int64)
    rets = np.empty(total, dtype=np.float32)
    certain = np.zeros((total, n_act), dtype=np.int8)
    prio = np.ones(total, dtype=np.float32)
    pos = 0
    for path, ln, take in zip(paths, lens, takes):
        d = np.load(path)
        idx = np.arange(ln) if take == ln else \
            np.sort(rng.choice(ln, size=take, replace=False))
        raw = d["obs"][idx]
        if raw.dtype == np.uint8:
            obs[pos:pos + take] = raw
        else:  # legacy float16/float32 files -> lossless encode
            obs[pos:pos + take] = encode_obs_u8(raw.astype(np.float32))
        masks[pos:pos + take] = d["masks"][idx]
        acts[pos:pos + take] = d["acts"][idx]
        r = d["rets"] if "rets" in d.files \
            else np.zeros(ln, dtype=np.float32)
        rets[pos:pos + take] = r[idx]
        if "certain" in d.files:
            certain[pos:pos + take] = d["certain"][idx]
        if "prio" in d.files:
            prio[pos:pos + take] = d["prio"][idx]
        pos += take
        del d
    return obs, masks, acts, rets, certain, prio


_CELL_MAP_CACHE = {}


def _cell_perm(k, h, w, device):
    """Old-flat -> new-flat cell mapping under dihedral transform k."""
    key = (k, h, w)
    if key not in _CELL_MAP_CACHE:
        flat = np.arange(h * w)
        r = flat // w
        c = flat % w
        if k >= 4:
            c = w - 1 - c          # mirror left-right
        for _ in range(k % 4):
            r, c = w - 1 - c, r    # rotate 90 degrees CCW
        _CELL_MAP_CACHE[key] = th.as_tensor(
            (r * w + c).copy(), dtype=th.long, device=device)
    return _CELL_MAP_CACHE[key]


def dihedral_transform(obs, masks, acts, k, h, w, extra=None):
    """Apply the k-th dihedral transform to a batch (square boards only).

    obs: (B, C, H, W), masks: (B, 2*H*W) bool, acts: (B,) long.
    extra: optional (B, 2*H*W) tensor transformed identically (e.g. targets).
    """
    n = h * w
    x = obs
    m = masks.view(-1, 2, h, w)
    e = extra.view(-1, 2, h, w) if extra is not None else None
    if k >= 4:
        x = th.flip(x, dims=(-1,))
        m = th.flip(m, dims=(-1,))
        if e is not None:
            e = th.flip(e, dims=(-1,))
    x = th.rot90(x, k % 4, dims=(-2, -1))
    m = th.rot90(m, k % 4, dims=(-2, -1))
    if e is not None:
        e = th.rot90(e, k % 4, dims=(-2, -1))

    is_flag = acts >= n
    flat = torch_where(is_flag, acts - n, acts)
    new_flat = _cell_perm(k, h, w, acts.device)[flat]
    new_acts = torch_where(is_flag, new_flat + n, new_flat)
    new_extra = e.reshape(-1, 2 * n) if e is not None else None
    return x, m.reshape(-1, 2 * n), new_acts, new_extra


def torch_where(cond, a, b):
    return th.where(cond, a, b)


@th.no_grad()
def tta_pick(model, obs_np, mask_np, width, height, device="cpu"):
    """Symmetry-averaged greedy action selection (square boards).

    Averages the masked policy distribution over all 8 dihedral views of
    the board, then picks the argmax action. Falls back to a single view
    for non-square boards.
    """
    obs = th.as_tensor(obs_np, dtype=th.float32, device=device).unsqueeze(0)
    mask = th.as_tensor(mask_np.astype(np.bool_), device=device).unsqueeze(0)
    n = width * height
    if width != height:
        dist = model.policy.get_distribution(obs, action_masks=mask)
        return int(dist.distribution.probs.argmax(dim=-1))
    src = th.arange(2 * n, device=device)
    acc = th.zeros(2 * n, device=device)
    for k in range(8):
        o, m, _, _ = dihedral_transform(
            obs, mask, th.zeros(1, dtype=th.long, device=device), k,
            height, width)
        dist = model.policy.get_distribution(o, action_masks=m)
        p_new = dist.distribution.probs.squeeze(0)
        perm = _cell_perm(k, height, width, device)          # old -> new
        flat = torch_where(src >= n, src - n, src)
        new_flat = perm[flat]
        new_idx = torch_where(src >= n, new_flat + n, new_flat)
        acc = acc + p_new[new_idx]
    acc[~mask.squeeze(0)] = -1.0
    return int(acc.argmax())


def transfer_trunk(model, src_path):
    """Copy weights from a saved model where shapes match.

    Lets a CNN trunk trained on a smaller board initialize a larger one;
    size-dependent parts (flatten linear, action head) are skipped.
    """
    import zipfile
    import io

    with zipfile.ZipFile(src_path) as zf:
        name = [n for n in zf.namelist() if n.endswith("policy.pth")][0]
        with zf.open(name) as f:
            src_sd = th.load(io.BytesIO(f.read()), map_location="cpu",
                             weights_only=True)
    dst = model.policy.state_dict()
    copied, skipped = [], []
    for k, v in src_sd.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            copied.append(k)
        else:
            skipped.append(k)
    model.policy.load_state_dict(dst)
    print(f"transferred {len(copied)} tensors from {src_path} "
          f"(skipped {len(skipped)}: {skipped[:6]}...)")


def pretrain(width, height, num_mines, data_paths, epochs, batch_size,
             lr, out_model, device, features_dim=256, channels=64,
             layers=3, init_from=None, logger=None, eval_games=200,
             target_mode="hard", max_transitions=900_000,
             priority_weights=False):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agents.train import BoardCNN, make_env

    obs, masks, acts, rets, certain, prio = load_datasets(
        data_paths, height, width,
        max_transitions=max_transitions)
    ret_mean, ret_std = float(rets.mean()), float(rets.std()) + 1e-8
    print(f"dataset: {len(acts)} transitions | "
          f"return stats: mean={ret_mean:.2f} std={ret_std:.2f}")
    n = len(acts)
    started_ts = time.time()
    inbox_state = {"last": started_ts, "stop": False}
    if logger:
        logger.log(phase="data", label="+".join(
                       os.path.basename(p) for p in data_paths),
                   transitions=int(n), mines=num_mines)

    vec = DummyVecEnv([
        make_env(width, height, num_mines, rank, 777) for rank in range(4)
    ])
    model = MaskablePPO(
        "CnnPolicy", vec,
        n_steps=512, batch_size=512,
        policy_kwargs={
            "normalize_images": False,
            "features_extractor_class": BoardCNN,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "channels": channels,
                "layers": layers,
            },
        },
        verbose=0, device=device, seed=0,
    )
    if init_from and init_from.endswith(".trunk"):
        transfer_trunk(model, init_from[:-len(".trunk")])
    elif init_from:
        model.set_parameters(init_from)
        print(f"initialized weights from {init_from}")
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=lr)

    obs_t = th.as_tensor(obs)
    masks_t = th.as_tensor(masks.astype(np.int8)).to(th.bool)
    acts_t = th.as_tensor(acts)
    rets_t = th.as_tensor((rets - ret_mean) / ret_std)
    certain_t = th.as_tensor(certain).float()
    prio_t = th.as_tensor(prio)
    use_sym = width == height
    if priority_weights:
        n_up = int((prio > 1.0).sum())
        print(f"priority-weighted sampling ON: {n_up}/{len(prio)} "
              f"transitions at w=2 "
              f"(student-error states), rest w=1", flush=True)

    # Phase A: behavioral cloning of the policy
    for epoch in range(epochs):
        perm = th.randperm(n)
        total_loss, seen = 0.0, 0
        correct = 0
        last_flush = time.time()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_obs = obs_t[idx].to(device, non_blocking=True).float()
            b_obs[:, 11:, :, :] /= 32.0  # decode uint8 k*32 encoding
            b_masks = masks_t[idx].to(device, non_blocking=True)
            b_acts = acts_t[idx].to(device, non_blocking=True)
            b_certain = certain_t[idx].to(device, non_blocking=True)
            b_prio = prio_t[idx].to(device, non_blocking=True)
            if use_sym:
                k = int(th.randint(8, (1,), device=device))
                (b_obs, b_masks, b_acts,
                 b_certain) = dihedral_transform(
                    b_obs, b_masks, b_acts, k, height, width,
                    extra=b_certain)
                # weights are per-transition and permutation-invariant
            dist = policy.get_distribution(b_obs, action_masks=b_masks)
            if target_mode == "soft":
                # soft targets: distribute mass over ALL certain actions
                # when they exist (removes solver's arbitrary-pick label
                # noise); otherwise hard NLL on the taken action
                t = b_certain.clone()
                has = t.sum(-1) > 0
                if has.any():
                    t[has] = t[has] / t[has].sum(-1, keepdim=True)
                t[~has] = 0.0
                t[~has].scatter_(1, b_acts[~has].unsqueeze(1), 1.0)
                log_p_all = th.log_softmax(dist.distribution.logits, dim=-1)
                per_sample = -(t * log_p_all).sum(-1)
            else:
                per_sample = -dist.log_prob(b_acts)
            if priority_weights:
                loss = (b_prio * per_sample).sum() / b_prio.sum()
            else:
                loss = per_sample.mean()
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
            seen += len(idx)
            correct += int((dist.distribution.logits.argmax(dim=-1)
                            == b_acts).sum())
            if logger and time.time() - last_flush >= 15.0:
                last_flush = time.time()
                logger.log(phase="train",
                           epoch=epoch + 1 + i / max(1, n),
                           nll=total_loss / seen,
                           top1=correct / seen, partial=True)
        epoch_nll, epoch_top1 = total_loss / seen, correct / seen
        print(f"epoch {epoch + 1}/{epochs} "
              f"nll={epoch_nll:.4f} top1={epoch_top1:.3f}", flush=True)
        if logger:
            logger.log(phase="train", epoch=epoch + 1, nll=epoch_nll,
                       top1=epoch_top1)
        if time.time() - inbox_state["last"] >= INBOX_CHECK_SECS:
            poll_inbox(started_ts, inbox_state)
        if inbox_state["stop"]:
            print("[train] stop requested - skipping remaining epochs",
                  flush=True)
            break

    # Phase B: value-function fitting with the shared trunk frozen
    v_params = [p for pname, p in policy.named_parameters()
                if "value_net" in pname]
    for pname, p in policy.named_parameters():
        p.requires_grad_(False)
    for p in v_params:
        p.requires_grad_(True)
    v_opt = th.optim.Adam(v_params, lr=lr)
    print("phase B: fitting value function", flush=True)
    for epoch in range(max(4, epochs // 4)):
        perm = th.randperm(n)
        total_v = 0.0
        seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            v_in = obs_t[idx].to(device, non_blocking=True).float()
            v_in[:, 11:, :, :] /= 32.0  # decode uint8 k*32 encoding
            values = policy.predict_values(v_in).squeeze(-1)
            v_loss = th.nn.functional.mse_loss(
                values, rets_t[idx].to(device, non_blocking=True))
            v_opt.zero_grad()
            v_loss.backward()
            th.nn.utils.clip_grad_norm_(v_params, 0.5)
            v_opt.step()
            total_v += float(v_loss.detach()) * len(idx)
            seen += len(idx)
        print(f"v-epoch {epoch + 1}: mse={total_v / seen:.4f}", flush=True)
        if logger:
            logger.log(phase="vfit", epoch=epoch + 1, v_mse=total_v / seen)

    os.makedirs("models", exist_ok=True)
    model.save(out_model)
    print(f"saved BC-initialized model to {out_model}.zip")

    if logger and eval_games > 0:
        wr, boards = quick_eval(model, width, height, num_mines,
                                eval_games, device, logger)
        logger.log(phase="eval", win_rate=wr, games=eval_games,
                   model=os.path.basename(out_model))
        print(f"post-train eval: win_rate={wr:.3f} over {eval_games} games")


def quick_eval(model, width, height, num_mines, n_games, device,
               board_logger=None):
    wins = 0
    for seed in range(n_games):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed)
        obs, _ = env.reset(seed=seed)
        done = False
        moves = 0
        max_moves = 3 * width * height
        while not done and moves < max_moves:
            mask = env.action_mask()
            if not mask.any():
                break
            action, _ = model.predict(obs, action_masks=mask,
                                      deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
            moves += 1
        won = info["remaining_safe"] == 0
        wins += won
        if board_logger and (won or seed % max(1, n_games // 8) == 0):
            board_logger.log_board(env, "win" if won else "loss", moves,
                                   extra={"mines": num_mines})
    return wins / n_games, None


def evaluate_bc(model_path, width, height, num_mines, n_games, device,
                logger=None, tta=False):
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path, device=device)
    wins = 0
    for seed in range(n_games):
        env = MinesweeperEnv(width=width, height=height,
                             num_mines=num_mines, seed=seed)
        obs, info = env.reset(seed=seed)
        done = False
        moves = 0
        max_moves = 3 * width * height
        while not done and moves < max_moves:
            mask = env.action_mask()
            if not mask.any():
                break
            if tta and width == height:
                action = tta_pick(model, obs, mask, width, height, device)
            else:
                action, _ = model.predict(obs, action_masks=mask,
                                          deterministic=True)
                action = int(action)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            moves += 1
        won = info["remaining_safe"] == 0
        wins += won
        if logger and (won or seed % 100 == 0):
            logger.log_board(env, "win" if won else "loss", moves,
                             extra={"mines": num_mines})
    wr = wins / n_games
    if logger:
        logger.log(phase="eval", win_rate=wr, games=n_games,
                   model=os.path.basename(model_path))
    print(f"BC eval win rate: {wr:.3f} over {n_games} games "
          f"(tta={tta})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["gen", "train", "eval", "dagger"],
                   required=True)
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--mines", type=int, default=4)
    p.add_argument("--games", type=int, default=20_000)
    p.add_argument("--seed0", type=int, default=500_000)
    p.add_argument("--data", type=str, default="data/bc_6x6_4m.npz",
                   help="training npz; comma-separated list to combine")
    p.add_argument("--out", type=str, default="data/dagger_r1.npz")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model-out", type=str, default="models/bc_init")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--features-dim", type=int, default=256)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--init-from", type=str, default=None,
                   help="load weights before training (curriculum transfer)")
    p.add_argument("--run", type=str, default="default",
                   help="dashboard run name (writes runs/<name>/)")
    p.add_argument("--tta", action="store_true",
                   help="symmetry-averaged inference in eval")
    p.add_argument("--eval-games", type=int, default=200,
                   help="post-training quick eval games (0 disables)")
    p.add_argument("--target-mode", choices=["hard", "soft"],
                   default="hard",
                   help="hard: NLL on solver's action; soft: uniform over "
                        "certain actions (experimental)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes for gen/dagger phases "
                        "(1 = serial, unchanged behavior)")
    p.add_argument("--flush-every", type=int, default=0,
                   help="serial dagger only: write an atomic checkpoint "
                        "shard every N games (0 = single save at end)")
    p.add_argument("--resume", action="store_true",
                   help="with --flush-every: adopt leftover shards and "
                        "continue from the recorded game index")
    p.add_argument("--nice", type=int, default=10,
                   help="process priority offset applied to this whole "
                        "command (10 = yield to desktop/apps when they "
                        "need CPU, still uses all idle capacity)")
    p.add_argument("--max-transitions", type=int, default=900_000,
                   help="cap on combined dataset size for train phase; "
                        "newest files kept in full, older downsampled "
                        "(bounds RAM as history grows)")
    p.add_argument("--priority-weights", action="store_true",
                   help="dagger: record w=2 on student-error transitions "
                        "(greedy pick outside teacher certain set); "
                        "train: consume recorded weights as weighted-"
                        "mean NLL. Default off = legacy behavior.")
    args = p.parse_args()

    try:
        os.nice(args.nice)
    except PermissionError:
        print(f"[ops] could not apply nice={args.nice} "
              "(negative values need root)")

    logger = MetricLogger(args.run)

    if args.phase == "gen":
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        if args.workers > 1:
            run_parallel_collection(
                "gen", None, args.width, args.height, args.mines,
                args.games, args.seed0, args.data, args.workers,
                args.device, logger=logger)
        else:
            generate_dataset(args.width, args.height, args.mines,
                             args.games, args.seed0, args.data, logger)
    elif args.phase == "dagger":
        assert args.model is not None
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        if args.workers > 1:
            run_parallel_collection(
                "dagger", args.model, args.width, args.height, args.mines,
                args.games, args.seed0, args.out, args.workers,
                args.device, tta=args.tta, run_name=args.run,
                logger=logger, priority_weights=args.priority_weights)
        else:
            dagger_roll(args.model, args.width, args.height, args.mines,
                        args.games, args.seed0, args.out, args.device,
                        logger, tta=args.tta,
                        flush_every=args.flush_every, resume=args.resume,
                        priority_weights=args.priority_weights)
    elif args.phase == "train":
        pretrain(args.width, args.height, args.mines, args.data.split(","),
                 args.epochs, args.batch_size, args.lr, args.model_out,
                 args.device, features_dim=args.features_dim,
                 channels=args.channels, layers=args.layers,
                 init_from=args.init_from, logger=logger,
                 eval_games=args.eval_games,
                 target_mode=args.target_mode,
                 max_transitions=args.max_transitions,
                 priority_weights=args.priority_weights)
    else:
        assert args.model is not None
        evaluate_bc(args.model, args.width, args.height, args.mines,
                    1000, args.device, logger, tta=args.tta)

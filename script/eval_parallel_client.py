"""Orchestrator for parallel SIR eval: launches N eval_policy_client.py worker processes against
one already-running policy_model_server_parallel.py, each handling a disjoint slice of episodes,
then aggregates their results. Does not modify eval_policy_client.py -- each worker is an
unmodified subprocess invocation of it, parallelism comes entirely from running several at once
plus the server-side batching in policy_model_server_parallel.py.

Seed partitioning: eval_policy_client.py derives its starting seed as
`st_seed = 100000 * (1 + seed)` (see its main()), so passing --seed 0, 1, 2, ... N-1 to N workers
gives each a seed range 100000 apart -- far wider than any worker will consume even accounting for
scene-setup retries, so ranges cannot collide. Each worker's episode count comes from the
RMBENCH_EVAL_TEST_NUM env var (already supported by eval_policy_client.py, see its main()) --
100 // N per worker, with the remainder folded into the last worker.

Each worker gets a distinct ckpt_setting suffix (_w0, _w1, ...) so their eval_result/ save
directories can never collide even if two workers happen to start within the same wall-clock
second (save_dir includes only a second-resolution timestamp).

Usage:
    python script/eval_parallel_client.py --task_name swap_T --task_config demo_clean \\
        --ckpt_setting my_run --port 9998 --num_workers 4 [--use_graph 0] [--gpu_id 0]
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

RMBENCH_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", required=True)
    ap.add_argument("--ckpt_setting", required=True)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--total_episodes", type=int, default=100)
    ap.add_argument("--use_graph", type=int, default=0)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--policy_name", default="SIR")
    args = ap.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    base = args.total_episodes // args.num_workers
    per_worker = [base] * args.num_workers
    per_worker[-1] += args.total_episodes - base * args.num_workers  # remainder to the last worker

    env_common = os.environ.copy()
    env_common["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env_common["SIR_USE_GRAPH"] = str(args.use_graph)

    procs = []
    worker_settings = []
    t_start = time.monotonic()
    for w, n_ep in enumerate(per_worker):
        if n_ep <= 0:
            continue
        ckpt_setting_w = f"{args.ckpt_setting}_w{w}"
        worker_settings.append((ckpt_setting_w, n_ep))
        env = env_common.copy()
        env["RMBENCH_EVAL_TEST_NUM"] = str(n_ep)
        cmd = [
            sys.executable, "script/eval_policy_client.py",
            "--config", f"policy/{args.policy_name}/deploy_policy.yml",
            "--port", str(args.port),
            "--overrides",
            "--task_name", args.task_name,
            "--task_config", args.task_config,
            "--ckpt_setting", ckpt_setting_w,
            "--seed", str(w),
            "--policy_name", args.policy_name,
        ]
        log_path = RMBENCH_ROOT / f".eval_parallel_worker{w}.log"
        log_f = open(log_path, "w")
        print(f"[orchestrator] worker {w}: {n_ep} episodes, seed={w}, ckpt_setting={ckpt_setting_w} "
              f"-> {log_path}")
        p = subprocess.Popen(cmd, cwd=str(RMBENCH_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((w, p, log_f))

    exit_codes = {}
    for w, p, log_f in procs:
        rc = p.wait()
        log_f.close()
        exit_codes[w] = rc
        tag = "OK" if rc == 0 else f"EXIT {rc}"
        print(f"[orchestrator] worker {w} finished: {tag}")

    elapsed = time.monotonic() - t_start
    print(f"\n[orchestrator] all {len(procs)} workers finished in {elapsed:.1f}s")

    # ---- aggregate ----
    total_success = 0
    total_run = 0
    for w, (ckpt_setting_w, n_ep) in enumerate(worker_settings):
        result_dir = RMBENCH_ROOT / "eval_result" / args.task_name / args.policy_name / args.task_config / ckpt_setting_w
        candidates = sorted(glob.glob(str(result_dir / "*" / "_result.txt")))
        if not candidates:
            print(f"[orchestrator] WARNING: worker {w} ({ckpt_setting_w}) produced no _result.txt "
                  f"(exit code was {exit_codes.get(w)}) -- excluded from aggregate, treated as 0 "
                  f"successes out of its {n_ep} assigned episodes for the denominator.")
            total_run += n_ep
            continue
        latest = max(candidates, key=os.path.getmtime)
        # eval_policy_client.py's _result.txt has NO "Success Rate:" label (that's Pattern A's
        # eval_policy.py format, a different script) -- it's "Timestamp: ...\n\nInstruction
        # Type: ...\n\n<bare float>" (see that file's writer, just above the file_path.write call).
        # The bare float is the LAST non-empty line.
        lines = [l.strip() for l in open(latest) if l.strip()]
        rate = None
        if lines:
            try:
                rate = float(lines[-1])
            except ValueError:
                rate = None
        if rate is None:
            print(f"[orchestrator] WARNING: could not parse success rate from {latest}")
            total_run += n_ep
            continue
        n_success = round(rate * n_ep)
        total_success += n_success
        total_run += n_ep
        print(f"[orchestrator] worker {w}: {n_success}/{n_ep} ({rate*100:.1f}%)  [{latest}]")

    print(f"\n[orchestrator] COMBINED: {total_success}/{total_run} => "
          f"{total_success/max(1,total_run)*100:.2f}%")


if __name__ == "__main__":
    main()

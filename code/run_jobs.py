"""Run a list of training jobs on one machine with bounded parallelism.

Each line of the jobs file is an argument string for train_dm.py. A job is skipped when its checkpoint
already exists, so the queue can be restarted after an interruption.
Usage: python run_jobs.py jobs.txt --parallel 4
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ckpt_of(args: list[str]) -> Path:
    get = lambda k, d="": args[args.index(k) + 1] if k in args else d
    out = Path(get("--out", str(HERE / "runs")))
    return out / f"{get('--cond')}{get('--tag')}_s{get('--seed', '11')}.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--parallel", type=int, default=4)
    a = p.parse_args()
    jobs = [shlex.split(l) for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    todo = [j for j in jobs if not ckpt_of(j).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    running = []
    while todo or running:
        running = [(pr, j) for pr, j in running if pr.poll() is None]
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            log = open(str(ckpt_of(j).with_suffix(".log")), "w") if ckpt_of(j).parent.exists() else subprocess.DEVNULL
            pr = subprocess.Popen([sys.executable, str(HERE / "train_dm.py"), *j], stdout=log, stderr=subprocess.STDOUT)
            running.append((pr, j))
            print("started", " ".join(j), flush=True)
        time.sleep(5)
    print("ALL_JOBS_DONE", flush=True)


if __name__ == "__main__":
    main()

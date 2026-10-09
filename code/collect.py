"""Collect finished DockMut targets from the pods.

For every pod and shard log, targets with a TARGET_DONE line are downloaded (result CSV and, optionally,
the seed-1 wild-type poses as one archive). A target finished on two pods keeps the copy of the pod whose log
reports completion last; partially processed copies are never taken.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = Path(sys.argv[1])            # scratchpad with ssh key and pod connection list
PODS = json.loads(Path(sys.argv[2]).read_text())   # [{"name":..,"ip":..,"port":..,"shards":["s0"]}]
OUT = ROOT / "dockmut/results"
POSES = ROOT / "dockmut/poses"
OUT.mkdir(parents=True, exist_ok=True)
POSES.mkdir(parents=True, exist_ok=True)
SSH = ["-i", str(SP / "ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=10"]


def run(pod, cmd):
    r = subprocess.run(["ssh", "-n", *SSH, "-p", str(pod["port"]), f"root@{pod['ip']}", cmd],
                       capture_output=True, text=True, timeout=300)
    return r.stdout


def main(with_poses: bool):
    done = {}
    for pod in PODS:
        for sh in pod["shards"]:
            out = run(pod, f"grep TARGET_DONE /workspace/dm/shard_{sh}.log 2>/dev/null | awk '{{print $2}}'")
            for t in out.split():
                done[t] = pod
    print(len(done), "targets finished")
    for t, pod in sorted(done.items()):
        dst = OUT / f"{t}.csv"
        if not dst.exists():
            subprocess.run(["scp", "-q", *SSH, "-P", str(pod["port"]), f"root@{pod['ip']}:/workspace/dm/results/{t}.csv", str(dst)],
                           check=False, capture_output=True, timeout=300)
        if with_poses and not (POSES / f"{t}.tgz").exists():
            run(pod, f"cd /workspace/dm/poses && tar -czf /workspace/poses_{t}.tgz {t}")
            subprocess.run(["scp", "-q", *SSH, "-P", str(pod["port"]), f"root@{pod['ip']}:/workspace/poses_{t}.tgz",
                            str(POSES / f"{t}.tgz")], check=False, capture_output=True, timeout=600)
        print(t, "ok" if dst.exists() else "MISSING", flush=True)


if __name__ == "__main__":
    main("--poses" in sys.argv)

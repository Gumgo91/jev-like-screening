#!/bin/bash
# Environment of the Uni-Dock benchmark (scripts/dock_rate_bench.py) on a fresh RunPod GPU pod (Ubuntu 22.04).
# The benchmark of the paper ran with one RTX 4090 and NVIDIA driver 570.169.
# One micromamba environment "ud" from conda-forge: Uni-Dock 1.2.0, Open Babel 3.2.1, RDKit 2026.03.6 (pandas for scripts/prep_ligands.py).
# Afterwards: export PATH=/workspace/mamba/envs/ud/bin:$PATH
set -x
mkdir -p /workspace/bin && cd /workspace
for i in 1 2 3 4; do
  curl -sL -m 300 -o bin/micromamba https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
    && [ "$(stat -c %s bin/micromamba)" -gt 10000000 ] && break
  sleep 5
done
chmod +x bin/micromamba
export MAMBA_ROOT_PREFIX=/workspace/mamba
for i in 1 2 3; do
  ./bin/micromamba create -y -n ud -c conda-forge python=3.11 "unidock=1.2.0" "openbabel=3.2.1" "rdkit=2026.03.6" pandas \
    && [ -x /workspace/mamba/envs/ud/bin/unidock ] && break
  rm -rf /root/.cache/conda /root/.cache/mamba
  sleep 5
done
[ -x /workspace/mamba/envs/ud/bin/unidock ] || { echo "ENV_FAILED"; exit 1; }
/workspace/mamba/envs/ud/bin/unidock --version
/workspace/mamba/envs/ud/bin/obabel -V
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ENV_DONE

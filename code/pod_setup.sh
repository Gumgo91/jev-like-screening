#!/bin/bash
# Build the docking environment on a fresh RunPod GPU pod (Ubuntu 22.04, CUDA driver >= 575).
# Uni-Dock 1.2.0 from conda-forge, Open Babel and RDKit in one micromamba environment.
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
  ./bin/micromamba create -y -n ud -c conda-forge python=3.11 "unidock=1.2.0" openbabel rdkit pandas numpy scipy \
    && [ -x /workspace/mamba/envs/ud/bin/unidock ] && break
  rm -rf /root/.cache/conda /root/.cache/mamba
  sleep 5
done
/workspace/mamba/envs/ud/bin/unidock --version
echo ENV_DONE

# Cloud runtime defaults for the byf account on Host1.
# Source this file before running LLaMA-Factory commands on the cloud server.

export CLOUD_HOST="${CLOUD_HOST:-Host1}"
export CLOUD_WORKDIR="${CLOUD_WORKDIR:-/home/byf/byf/multimodal}"
export CLOUD_CONDA_PREFIX="${CLOUD_CONDA_PREFIX:-/HDDDATA/byf/conda/envs/byf}"
export CLOUD_PYTHON="${CLOUD_PYTHON:-$CLOUD_CONDA_PREFIX/bin/python}"
export CLOUD_PIP="${CLOUD_PIP:-$CLOUD_PYTHON -m pip}"
export CLOUD_LLAMAFACTORY_CLI="${CLOUD_LLAMAFACTORY_CLI:-$CLOUD_CONDA_PREFIX/bin/llamafactory-cli}"

export PATH="$CLOUD_CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-/HDDDATA/byf/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/gendynamics/_vendor"
PYTHON_BIN="${PYTHON:-python}"
STAGE_ROOT="$(mktemp -d)"
USE_JZ_MODULE="${USE_JZ_MODULE:-0}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.8.0}"

mkdir -p "${VENDOR_DIR}"
trap 'rm -rf "${STAGE_ROOT}"' EXIT

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "[vendor] --use-jz-module requested but 'module' is unavailable" >&2
        exit 1
    fi
    module purge || true
    module load "${JZ_MODULE}"
fi

drop_git_metadata() {
    local path="$1"
    find "${path}" -name '.git' -type d -prune -exec rm -rf {} +
    find "${path}" -name '.gitmodules' -type f -delete
}

vendor_repo() {
    local name="$1"
    local repo_url="$2"
    local stage_dir="${STAGE_ROOT}/${name}"
    local target_dir="${VENDOR_DIR}/${name}"

    echo "[vendor] ${name}"
    git clone --depth 1 "${repo_url}" "${stage_dir}"
    drop_git_metadata "${stage_dir}"
    rm -rf "${target_dir}"
    mv "${stage_dir}" "${target_dir}"
}

vendor_repo "DLPM" "https://github.com/hcherkaoui/DLPM"
vendor_repo "flow_matching" "https://github.com/facebookresearch/flow_matching"
vendor_repo "physicsnemo" "https://github.com/NVIDIA/physicsnemo"
vendor_repo "score_sde_pytorch" "https://github.com/yang-song/score_sde_pytorch"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/patch.vendor.py" --vendor-root "${VENDOR_DIR}"

"${PYTHON_BIN}" -m pip install \
  torchdiffeq \
  pot \
  torchquad \
  Cython \
  "requests>=2.31" \
  "fsspec<=2025.3.0" \
  "s3fs<=2025.3.0" \
  "importlib-metadata<9"
"${PYTHON_BIN}" -m pip install -e "${VENDOR_DIR}/flow_matching"

"${PYTHON_BIN}" - <<'PY'
import fsspec
import h5py
import pandas
import s3fs
import torch
import torchdiffeq
import torchquad
import torchvision
import ot

print(
    "[vendor] verify "
    f"torch={torch.__version__} "
    f"cuda={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()}"
)
print(
    "[vendor] imports "
    f"torchvision={torchvision.__version__} "
    f"pandas={pandas.__version__} "
    f"h5py={h5py.__version__} "
    f"fsspec={fsspec.__version__} "
    f"s3fs={s3fs.__version__}"
)
print(
    "[vendor] extras "
    f"torchdiffeq={getattr(torchdiffeq, '__version__', 'ok')} "
    f"ot={getattr(ot, '__version__', 'ok')} "
    f"torchquad={getattr(torchquad, '__version__', 'ok')}"
)
PY

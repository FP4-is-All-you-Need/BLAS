#!/usr/bin/env bash
# Run from an extracted package. Nothing uses a GPU without --gpu.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${1:-}" in
  --list|--check-package)
    sha256sum --check --quiet SHA256SUMS
    exec python3 Tests/GPU/run_single.py --package "$PWD" "$@"
    ;;
  --gpu) ;;
  *) echo 'Use --list, --check-package, or --gpu --max-seconds N' >&2; exit 2;;
esac
started=$(date +%s)
max_seconds=''
for ((arg_index=1; arg_index<=$#; arg_index++)); do
  if [[ "${!arg_index}" == '--max-seconds' ]]; then
    next_index=$((arg_index+1)); max_seconds="${!next_index:-}"
  fi
done
[[ "$max_seconds" =~ ^[0-9]+$ ]] && ((max_seconds>=90)) || { echo '--max-seconds >=90 required' >&2; exit 2; }
[[ ! -e results ]] || { echo 'results exists; preserve it and choose a fresh extraction' >&2; exit 2; }
sha256sum --check --quiet SHA256SUMS
export DEBIAN_FRONTEND=noninteractive
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/Lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# Ubuntu 24.04 uses Python 3.12; the wheel is included in the package.
setup() {
  if ! command -v python3 >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
    timeout 60 apt-get update
    timeout 90 apt-get install -y --no-install-recommends python3 python3-venv
  fi
  python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
  if ! python3 -m venv runtime; then
    timeout 60 apt-get install -y --no-install-recommends python3-venv
    python3 -m venv runtime
  fi
  timeout 45 runtime/bin/python -m pip install --no-index --no-deps wheels/*.whl
}
export -f setup
timeout "$((max_seconds-60))" bash -c 'set -euo pipefail; setup' > setup.log 2>&1
exec runtime/bin/python Tests/GPU/run_single.py --package "$PWD" --started-epoch "$started" "$@"

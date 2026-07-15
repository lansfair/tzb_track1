#!/usr/bin/env bash
# Collect a short, low-overhead system profile around Tianzhibei DDP training.
#
# Usage:
#   bash tools/diagnose_tianzhibei_gpu.sh CONFIG GPUS [DURATION_SECONDS] [WORK_DIR] [TRAIN_ARGS...]
#
# Example:
#   bash tools/diagnose_tianzhibei_gpu.sh \
#     configs/tianzhibei_car/mtp-vit-l-rvsa_kfiou_staged_40e.py \
#     4 600 work_dirs/mtp_kfiou_diag
#
# The script writes diagnostics/gpu_diag_*.tar.gz. Send that archive back for
# analysis. Set DURATION_SECONDS=0 to run until training exits or Ctrl-C.

set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,14p' "$0"
    exit 2
fi

CONFIG=$1
GPUS=$2
DURATION_SECONDS=${3:-600}
WORK_DIR=${4:-work_dirs/gpu_diagnostic}
shift $(( $# >= 4 ? 4 : $# ))
EXTRA_TRAIN_ARGS=("$@")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config does not exist: $CONFIG" >&2
    exit 2
fi
if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer: $GPUS" >&2
    exit 2
fi
if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "DURATION_SECONDS must be a non-negative integer: $DURATION_SECONDS" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required." >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${DIAG_OUT_DIR:-diagnostics/gpu_diag_${TIMESTAMP}}
LOG_INTERVAL=${LOG_INTERVAL:-10}
DATA_ROOT=${DATA_ROOT:-/mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train}
mkdir -p "$OUT_DIR" "$WORK_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
WORK_DIR=$(cd "$WORK_DIR" && pwd)

MONITOR_PIDS=()

start_monitor() {
    local output=$1
    shift
    if command -v "$1" >/dev/null 2>&1; then
        "$@" >"$OUT_DIR/$output" 2>&1 &
        MONITOR_PIDS+=("$!")
    else
        printf 'Command not installed: %s\n' "$1" >"$OUT_DIR/$output"
    fi
}

stop_monitors() {
    local pid
    for pid in "${MONITOR_PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${MONITOR_PIDS[@]:-}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
}

trap stop_monitors EXIT INT TERM

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_root=$REPO_ROOT"
    echo "config=$CONFIG"
    echo "gpus=$GPUS"
    echo "duration_seconds=$DURATION_SECONDS"
    echo "work_dir=$WORK_DIR"
    echo "data_root=$DATA_ROOT"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "log_interval=$LOG_INTERVAL"
    printf 'extra_train_args='; printf '%q ' "${EXTRA_TRAIN_ARGS[@]}"; echo
    echo
    echo '===== git ====='
    git rev-parse HEAD 2>&1 || true
    git status -sb 2>&1 || true
    echo
    echo '===== uname ====='
    uname -a 2>&1 || true
    echo
    echo '===== os-release ====='
    cat /etc/os-release 2>&1 || true
    echo
    echo '===== lscpu ====='
    lscpu 2>&1 || true
    echo
    echo '===== memory ====='
    free -h 2>&1 || true
    echo
    echo '===== block devices ====='
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,ROTA,MODEL 2>&1 || true
    echo
    echo '===== filesystems ====='
    df -hT 2>&1 || true
    echo
    echo '===== data mount ====='
    findmnt -T "$DATA_ROOT" 2>&1 || true
    echo
    echo '===== numa ====='
    numactl --hardware 2>&1 || true
    echo
    echo '===== GPU topology ====='
    nvidia-smi topo -m 2>&1 || true
    echo
    echo '===== environment ====='
    env | grep -E '^(CUDA|NCCL|OMP|MKL|PYTORCH|TORCH|MASTER|WORLD|RANK)' | sort || true
} >"$OUT_DIR/system_info.txt"

nvidia-smi -q >"$OUT_DIR/nvidia_smi_before.txt" 2>&1 || true
nvidia-smi -q -x >"$OUT_DIR/nvidia_smi_before.xml" 2>&1 || true
cp "$CONFIG" "$OUT_DIR/config.py"

python - <<'PY' >"$OUT_DIR/python_environment.txt" 2>&1 || true
import platform
import sys

print('python:', sys.version.replace('\n', ' '))
print('platform:', platform.platform())
for name in ('torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmrotate',
             'numpy', 'cv2', 'rasterio'):
    try:
        module = __import__(name)
        print(f'{name}:', getattr(module, '__version__', '<no __version__>'))
    except Exception as exc:
        print(f'{name}: ERROR: {exc!r}')

try:
    import torch
    print('cuda available:', torch.cuda.is_available())
    print('torch cuda:', torch.version.cuda)
    print('cudnn:', torch.backends.cudnn.version())
    print('cudnn benchmark:', torch.backends.cudnn.benchmark)
    print('tf32 matmul:', torch.backends.cuda.matmul.allow_tf32)
    print('tf32 cudnn:', torch.backends.cudnn.allow_tf32)
    print('gpu count:', torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f'gpu {index}: {props.name}, {props.total_memory / 2**30:.2f} GiB, '
              f'cc={props.major}.{props.minor}, sm={props.multi_processor_count}')
except Exception as exc:
    print('torch CUDA details: ERROR:', repr(exc))
PY

start_monitor gpu_query.csv nvidia-smi \
    --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current \
    --format=csv -l 1
start_monitor gpu_dmon.log nvidia-smi dmon -s pucvmt -d 1
start_monitor gpu_pmon.log nvidia-smi pmon -s um -d 1
start_monitor vmstat.log vmstat 1
start_monitor mpstat.log mpstat -P ALL 1
start_monitor iostat.log iostat -xz 1
start_monitor pidstat.log pidstat -dur -p ALL 1
start_monitor network.log sar -n DEV 1

TRAIN_CMD=(
    bash tools/dist_train.sh
    "$CONFIG"
    "$GPUS"
    --work-dir "$WORK_DIR"
    --cfg-options "default_hooks.logger.interval=$LOG_INTERVAL"
)
TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")

{
    printf 'command='; printf '%q ' "${TRAIN_CMD[@]}"; echo
    echo "started_at=$(date --iso-8601=seconds)"
} >"$OUT_DIR/command.txt"

echo "Diagnostics: $OUT_DIR"
echo "Training: ${TRAIN_CMD[*]}"
if [[ "$DURATION_SECONDS" -gt 0 ]]; then
    echo "The diagnostic run will stop after ${DURATION_SECONDS}s."
else
    echo "The diagnostic run will continue until training exits or Ctrl-C."
fi

set +e
if [[ "$DURATION_SECONDS" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout --signal=INT --kill-after=30s "${DURATION_SECONDS}s" \
        "${TRAIN_CMD[@]}" 2>&1 | tee "$OUT_DIR/training.log"
    TRAIN_RC=${PIPESTATUS[0]}
else
    "${TRAIN_CMD[@]}" 2>&1 | tee "$OUT_DIR/training.log"
    TRAIN_RC=${PIPESTATUS[0]}
fi
set -e

stop_monitors
MONITOR_PIDS=()

{
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "training_exit_code=$TRAIN_RC"
    if [[ "$TRAIN_RC" -eq 124 || "$TRAIN_RC" -eq 130 ]]; then
        echo 'note=diagnostic duration elapsed or training was interrupted; this is expected'
    fi
} >>"$OUT_DIR/command.txt"

nvidia-smi -q >"$OUT_DIR/nvidia_smi_after.txt" 2>&1 || true
nvidia-smi -q -x >"$OUT_DIR/nvidia_smi_after.xml" 2>&1 || true

ARCHIVE="${OUT_DIR}.tar.gz"
if command -v tar >/dev/null 2>&1; then
    tar -czf "$ARCHIVE" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
    echo "Created: $ARCHIVE"
else
    echo "tar is unavailable; send the directory instead: $OUT_DIR"
fi

echo "Training exit code: $TRAIN_RC"
echo "Send the archive (or at least training.log, gpu_query.csv, gpu_dmon.log,"
echo "mpstat.log, iostat.log, pidstat.log, and system_info.txt) for analysis."

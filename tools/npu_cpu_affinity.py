"""Apply the fixed eight-rank CPU mapping for the ModelArts Ascend 950DT node."""

import os
import subprocess
import sys


_CPU_START_BY_RANK = (144, 156, 48, 60, 72, 84, 168, 180)
_CORES_PER_RANK = 12
_DEVICE_ORDER = "0,1,2,3,4,5,6,7"


def apply_from_environment() -> None:
    enabled = os.getenv("NPU_CPU_AFFINITY_ENABLE", "false").lower()
    if enabled not in {"1", "true", "yes", "on"} or sys.platform != "linux":
        return

    if os.getenv("LOCAL_WORLD_SIZE") != "8" or os.getenv("ASCEND_RT_VISIBLE_DEVICES") != _DEVICE_ORDER:
        print("[cpu-affinity] skipped: expected 8 local ranks with NPU device order 0-7", flush=True)
        return

    try:
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        if not 0 <= local_rank < len(_CPU_START_BY_RANK):
            raise ValueError(f"invalid LOCAL_RANK={local_rank}")

        result = subprocess.run(["npu-smi", "info", "-m"], check=True, capture_output=True, text=True, timeout=10)
        chips = {
            columns[0]: columns[4]
            for columns in map(str.split, result.stdout.splitlines())
            if len(columns) == 5 and columns[0] == columns[3]
        }
        if chips.get(str(local_rank)) != "Ascend950DT":
            print(f"[cpu-affinity] skipped: rank {local_rank} is not a verified Ascend950DT", flush=True)
            return

        start = _CPU_START_BY_RANK[local_rank]
        cpus = set(range(start, start + _CORES_PER_RANK))
        if not cpus.issubset(os.sched_getaffinity(0)):
            raise ValueError(f"CPU {start}-{start + _CORES_PER_RANK - 1} are not all available for rank {local_rank}")
        os.sched_setaffinity(0, cpus)
        after = sorted(os.sched_getaffinity(0))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"[cpu-affinity] warning: cannot bind process: {error}", file=sys.stderr, flush=True)
        return

    print(f"[cpu-affinity] local_rank={local_rank} model=Ascend950DT after={after}", flush=True)

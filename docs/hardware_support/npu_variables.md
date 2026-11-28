# Ascend Environment Variables

This document describes the key environment variables that can be configured to optimize VeOmni performance on Ascend NPUs.

## VEOMNI_NPU_JIT_COMPILE

```bash
export VEOMNI_NPU_JIT_COMPILE=false
```

When set, VeOmni calls `torch.npu.set_compile_mode` during device setup. Use `false` to disable operator JIT compilation and prefer the available precompiled operator binaries, or `true` to enable operator JIT compilation. This setting does not fix input shapes and is separate from `torch.compile`.

Leave this variable unset to preserve the NPU runtime's default compile mode. It has no effect on GPU or CPU runs. Compare steady-state step time on the same workload before adopting either mode.

## NPU_CPU_AFFINITY_ENABLE

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPU_CPU_AFFINITY_ENABLE=true
bash train.sh tasks/train_vlm.py path/to/config.yaml
```

The default is `false`. This policy uses the CPU layout verified on the eight-card ModelArts Ascend 950DT node. The VLM entry applies it before importing VeOmni and torch, so runtime threads and DataLoader workers inherit the CPU mask.

The helper requires eight local ranks and the explicit device order shown above. It uses `npu-smi info -m` to verify that the rank's NPU is an Ascend950DT with the expected physical device ID. Local ranks 0–7 receive CPU masks `144-155`, `156-167`, `48-59`, `60-71`, `72-83`, `84-95`, `168-179`, and `180-191`, respectively.

Successful binding prints the actual CPU mask in a `[cpu-affinity]` log line. If device detection or binding fails, training continues with a diagnostic message. Use one CPU binding policy at a time when comparing performance.

## CPU_AFFINITY_CONF

```shell
export CPU_AFFINITY_CONF=1
```

Enable coarse-grained or fine-grained CPU core binding. This configuration helps prevent thread contention, improves cache hit rates, avoids memory access across different NUMA (Non-Uniform Memory Access) nodes, and reduces task scheduling overhead—collectively optimizing task execution efficiency.

### Parameter Settings:

* `0`: Disable the binding function. Default is `0`.
* `1`: Enable coarse-grained kernel binding.
* `2`: Enable fine-grained kernel binding.

## PYTORCH_NPU_ALLOC_CONF

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

`expandable_segments:<value>`: Enable the memory pool extension segment feature.

* `True`: This configuration instructs the cache allocator to create specific memory blocks with the capability to be extended later. This allows for more efficient handling of scenarios where the required memory size frequently changes during runtime.
* `False`: The memory pool extension segment feature is disabled, and the original memory allocation method is used. Default is `False`.

## MULTI_STREAM_MEMORY_REUSE

```bash
export MULTI_STREAM_MEMORY_REUSE=2
```

This environment variable enables memory reuse across multiple streams, which can help reduce memory fragmentation and improve memory utilization.

## Other Useful Environment Variables

For additional environment variables and their configurations, please refer to the Ascend CANN official documentation:
https://www.hiascend.com/document/detail/zh/canncommercial/900/index/index.html

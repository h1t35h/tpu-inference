---
name: kernel-pal
description: Assists with writing, updating, and optimizing Pallas kernels on TPUs. Fixes syntax errors and provides guidelines for handling complex scenarios based on JAX/Pallas distributed guidelines.
---

# Pallas Kernel Guidelines (kernel-pal)

## Overview
This skill provides guidelines and correct syntax for writing and fixing distributed Pallas kernels on TPUs using JAX. It should be used to ensure correct imports, API usage, and safe implementation of complex DMA operations.

## 1. Correct Syntax & Imports
When writing or fixing Pallas kernels, ensure the following imports and syntax are used:

### Imports
```python
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
P = jax.sharding.PartitionSpec
```

### Basic Syntax
- Use `pltpu.make_async_remote_copy` for inter-device DMAs.
- Use `pltpu.make_async_copy` for intra-device DMAs (e.g., HBM to VMEM/SMEM).
- Use `pl.when` to conditionally execute code based on device IDs (e.g., specific routing logic).
- Use `pl.run_scoped` for defining block-scoped memory.
- Use `pltpu.SemaphoreType.DMA` for creating semaphores used in DMAs.

**Example Async Remote Copy:**
```python
remote_copy_op = pltpu.make_async_remote_copy(
    src_ref=input_ref,
    dst_ref=output_ref,
    send_sem=send_sem,
    recv_sem=recv_sem,
    device_id=target_device_id,
    device_id_type=pl.DeviceIdType.MESH, # Default is MESH, can be LOGICAL
)
remote_copy_op.start() # Initiates the DMA
remote_copy_op.wait_send() # Blocks until all data is sent
remote_copy_op.wait_recv() # Blocks until all data is received
remote_copy_op.wait() # Blocks on both send and recv
```

## 2. Guidelines for Complex Scenarios

### Remote Direct Memory Access (RDMA) Model
- TPUs use a push-only RDMA model. You can push data to any device within the pod, but you can only read data stored locally.
- A DMA descriptor parameterizes both a "send" and "receive" operation.
- Always use `.start()` to initiate the copy (non-blocking).
- Call `.wait_send()` on the sender to block until data is fully sent.
- Call `.wait_recv()` on the receiver to block until the expected bytes are received.
- Use `jax.shard_map` with `pallas_call` to execute the distributed kernel. 

### DMA Semaphores
- `send_sem` and `recv_sem` must be allocated with `pltpu.SemaphoreType.DMA` in `scratch_shapes` of the `pallas_call` grid spec.
- Semaphores act as integer progress trackers. 
- Avoid waiting on semaphores without a corresponding `.start()` call from the sender, as this will hang indefinitely.

### Common Failure Modes & Prevention
- **Hangs (Indefinite Wait):** Occurs if a device calls `.wait_recv()` but no other device sends data to it, or if it expects more bytes than it was sent. Ensure matched `.start()` and `.wait_recv()` operations.
- **Crashes:** If a device is sent more bytes than it expects or if DMAs are started but semaphores are not waited on, leaving the semaphores in an invalid non-zero state. Always properly await `.wait_send()` or `.wait_recv()`.
- **Silent Data Corruption:** Two devices copying to the same destination buffer simultaneously can cause race conditions. Prevent this by coordinating non-overlapping target buffers or using barriers.

### Best Practices for Collectives & Communication
- Limit communication to neighbors (rings) when possible to minimize network contention. 
- Use looped kernels to stream data across devices (e.g. `lax.fori_loop` or `pl.program_id` loop over devices).
- Use `pl.when` on `lax.axis_index('x')` (device ID) to handle asymmetric communication patterns and asymmetric sends/recvs.
- TPUs are typically in a torus topology, and taking 1D slices produces rings, which is a great pattern for collective operations (like all-gather or ppermute).
- In a SPMD setup, where every device sends and receives, each device should typically call both `.start()` and `.wait()` on their respective descriptors.

## 3. Handling Complex Scenarios & Advanced Techniques

### Double-Buffering
Double-buffering allows computation to overlap with communication/memory transfers by using two sets of buffers.
- **Allocation**: Allocate VMEM/SMEM buffers with an extra leading dimension of size 2 (e.g., `pltpu.VMEM((2, ...))`).
- **Semaphores**: Allocate 2 semaphores for the buffers (e.g., one for each buffer index).
- **Execution Loop**: Inside your loop (over time/blocks), use modulo arithmetic `idx % 2` for the current buffer and `(idx + 1) % 2` for the next buffer.
- **Overlap**: 
  1. Start fetching the *next* block of data into the next buffer.
  2. Wait for the *current* block of data to finish fetching into the current buffer.
  3. Compute on the current buffer.
  *(Reference: Look at `tpu_inference/kernels/fused_moe/v1/kernel.py` where `bt_sem_id = bt_id % 2` is used for `b_gating_x2_vmem` and weights fetching).*

### Bidirectional Networking
TPU torus networks have bidirectional links between neighboring chips. You can double your communication bandwidth by sending data in both directions simultaneously.
- **Routing**: Instead of sending all data to `(my_id + 1) % num_devices`, split the data payload in half.
- Send the first half to the right neighbor: `(my_id + 1) % num_devices`.
- Send the second half to the left neighbor: `(my_id - 1) % num_devices`.
- Wait on both `recv_sem`s to ensure both halves have arrived.
- Ensure that the receiver also knows to expect data from both its left and right neighbors.

### Overlapping and Optimized Kernels (Reduce-Scatter Example)
When implementing optimized, memory-intensive collectives (like reduce-scatter) or handling arrays larger than available VMEM, you should chunk the HBM buffers along the loop dimension and aggressively overlap computation with memory transfers (DMAs) using nested pipelining and double buffering.

For a detailed example demonstrating these optimization techniques—including how to manage chunking, pipelined accumulation, and overlapping—refer to:
[Overlap & Large HBM Handling Reference](file:///usr/local/google/home/hitesy/workplace/bodhan/tpu-inference/.agents/skills/kernel-pal/ref/large_hbm_handling.md)

### Pipelining on TPU
Pallas provides primitives for building software pipelines that overlap memory operations (DMAs) with computation. 

For a comprehensive tutorial on using `emit_pipeline` to create software pipelines that overlap loading, computing, and storing on TPUs, refer to:
[Pipelining Reference](file:///usr/local/google/home/hitesy/workplace/bodhan/tpu-inference/.agents/skills/kernel-pal/ref/pipelining.md)

### Grids and BlockSpecs
Pallas uses `grid` to define iteration loops and `pl.BlockSpec` to chunk inputs and outputs for each kernel invocation.

For details on blocked indexing, element indexing, and TPU-specific block shape constraints, refer to:
[Grids and BlockSpecs Reference](file:///usr/local/google/home/hitesy/workplace/bodhan/tpu-inference/.agents/skills/kernel-pal/ref/grid_blockspec.md)

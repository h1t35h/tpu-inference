# TPU Pipelining Reference

This reference covers TPU-specific pipelining concerns in Pallas, including memory spaces, multiple buffering, nested pipelining, and Megacore execution.

## TPU Memory Spaces

A TPU has High-Bandwidth Memory (HBM, equivalent to DRAM), Vector Memory (VMEM, vector SRAM), and Scalar Memory (SMEM, scalar SRAM).

Pallas exposes these memory spaces through the following types:
- `pl.ANY`: HBM (usually) or VMEM (DRAM). A hint that memory space is unconstrained. To use, you must copy values into VMEM/SMEM using `pltpu.sync_copy` or `pltpu.async_copy`.
- `pltpu.VMEM`: Vector SRAM. The default memory space for pipelining.
- `pltpu.SMEM`: Scalar SRAM. Only scalar loads/stores.
- `pltpu.SEMAPHORE`: Used to allocate semaphores (SRAM) for constructing barriers or tracking asynchronous operations.

*Note:* You can manually specify `memory_space` in `pl.BlockSpec` (e.g., `memory_space=pl.ANY`) or for persistent scratch buffers via `scratch_shapes` in `pl.pallas_call`.

## Multiple Buffering

By default, the buffer count is 2 for inputs and outputs. You can specify a different count using `pipeline_mode` on `pl.BlockSpec`:

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count)
)
```

## Nested Pipelining (`pltpu.emit_pipeline`)

`pltpu.emit_pipeline` allows you to construct custom inner pipelines inside a kernel (e.g., an HBM-VMEM inner pipeline inside an outer inter-chip communication pipeline).

```python
def emit_pipeline(
    kernel: Callable,
    grid: tuple[int],
    in_specs: PyTree[BlockSpec] = None,
    out_specs: PyTree[BlockSpec] = None,
    dimension_semantics: tuple[GridDimensionSemantics] = None,
    core_axis: int | None = None,
) -> Callable:
```

### Lookahead Prefetch
Lookahead prefetch fetches the next input block as soon as a slot is available (instead of right before use). This is useful for variable compute workloads. Enable it via `pl.Buffered`:

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
)
```

### Dynamic Block Shapes
You can pipeline over blocks with dynamic but bounded shapes. Use `pl.BoundedSlice(max_size)` for the `block_shape` dimension, and `pl.ds(start, size)` inside the `index_map`. Both `start` and `size` are *element* indices.

```python
pl.BlockSpec(
   block_shape=(pl.BoundedSlice(32), 256),
   index_map=lambda i: (pl.ds(start_idx, dynamic_size), 0),
)
```

## TPUs in Megacore Configuration

Some TPU chips (like v4 and v5p) have two TensorCores that share HBM. To utilize both cores simultaneously for embarrassingly parallel dimensions, provide `dimension_semantics` to `pl.pallas_call`:

```python
pl.pallas_call(
    kernel_body,
    grid=(num_cores, ...),
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary")
    )
)
```

When using `pltpu.emit_pipeline` with Megacore, pass `core_axis` (the index of the parallel grid axis to partition) along with `dimension_semantics`.

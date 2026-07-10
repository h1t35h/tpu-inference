import functools
import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

LEFT = 0
RIGHT = 1


def local_barrier(left_neighbor, right_neighbor, axis_names, double_barrier=True):
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        device_id = tuple(
            neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
        )
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=device_id,
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:
        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor in [left_neighbor, right_neighbor]:
                device_id = tuple(
                    neighbor if ax == "model" else lax.axis_index(ax)
                    for ax in axis_names
                )
                pl.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=device_id,
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(second_barrier, 2)


def mod(x, n):
    return lax.rem(x + n, n)


def signal(left_or_right, semaphore, num_devices, axis_names):
    my_id = lax.axis_index("model")
    if left_or_right == LEFT:
        neighbor = mod(my_id - 1, num_devices)
    else:
        neighbor = mod(my_id + 1, num_devices)
    device_id = tuple(
        neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
    )
    pl.semaphore_signal(
        semaphore,
        inc=1,
        device_id=device_id,
        device_id_type=pl.DeviceIdType.MESH,
    )


def reduce_scatter_kernel(
    x,
    wg,
    wu,
    wd,
    o_ref,
    hbm_scratch,
    x_ref,
    left_recv_sem,
    left_send_sem,
    copy_sem,
    right_recv_sem,
    right_send_sem,
    left_capacity_sem,
    right_capacity_sem,
    x_scratch,
    a_scratch,
    *,
    num_devices,
    b_seq,
    b_inter,
    b_hidden,
    chunk_size,
    half_chunk,
    hidden_size,
    F_loc,
    axis_names,
):
    outer_step = pl.program_id(0)
    phase = pl.program_id(1)
    is_start = jnp.logical_and(outer_step == 0, phase == 0)
    last_iteration = outer_step == pl.num_programs(0) - 1

    working_slot = lax.rem(outer_step, 2)
    receiving_slot = 1 - working_slot
    my_id = lax.axis_index("model")
    right_neighbor = mod(my_id + 1, num_devices)
    left_neighbor = mod(my_id - 1, num_devices)

    left_copy_device = mod(my_id + outer_step + 1, num_devices)
    right_copy_device = mod(my_id - outer_step - 1, num_devices)
    left_copy_slice = pl.ds(0, half_chunk)
    right_copy_slice = pl.ds(half_chunk, half_chunk)
    current_phase_slice = pl.ds(phase * half_chunk, half_chunk)

    initial_left_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=tuple(
            left_neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )

    initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=tuple(
            right_neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )

    left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=tuple(
            left_neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )
    right_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=tuple(
            right_neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )

    def inner_compute_a(x_scratch, wg_tile, wu_tile, a_tile):
        x_val = x_scratch[...]
        wg_block = wg_tile[...].astype(x_val.dtype)
        wu_block = wu_tile[...].astype(x_val.dtype)
        wgu_block = jnp.concatenate([wg_block, wu_block], axis=1)

        gu_sram = jnp.matmul(x_val, wgu_block, preferred_element_type=jnp.float32)

        h_sram, u_sram = jnp.split(gu_sram, 2, axis=-1)
        a_sram_out = jax.nn.gelu(h_sram, approximate=True) * u_sram
        a_tile[...] = a_sram_out.astype(a_tile.dtype)

    def inner_compute_y(a_scratch, wd_tile, y_tile):
        a_val = a_scratch[...]
        wd_block = wd_tile[...].astype(a_val.dtype)
        y_sram = jnp.matmul(a_val, wd_block, preferred_element_type=jnp.float32)
        y_tile[...] = y_sram.astype(y_tile.dtype)

    def run_mlp_for_slice(device_idx, slice_ds):
        start_row = device_idx * chunk_size + slice_ds.start
        
        def loop_body(seq_idx, _):
            x_row = start_row + seq_idx * b_seq
            out_row = slice_ds.start + seq_idx * b_seq
            
            x_block = x_row // b_seq
            out_block = out_row // b_seq
            
            def load_x(x_tile, x_scratch_tile):
                x_scratch_tile[...] = x_tile[...]
            
            pl_load_x = pltpu.emit_pipeline(
                load_x,
                grid=(1,),
                in_specs=(pl.BlockSpec((b_seq, hidden_size), lambda j: (x_block, 0)),),
                out_specs=(pl.BlockSpec((b_seq, hidden_size), lambda j: (0, 0), memory_space=pltpu.VMEM),)
            )
            pl_load_x(x, x_scratch)
            
            wg_spec = pl.BlockSpec((hidden_size, b_inter), lambda i: (0, i), pipeline_mode=pl.Buffered(buffer_count=2))
            wu_spec = pl.BlockSpec((hidden_size, b_inter), lambda i: (0, i), pipeline_mode=pl.Buffered(buffer_count=2))
            a_spec = pl.BlockSpec((b_seq, b_inter), lambda i: (0, i), memory_space=pltpu.VMEM)
            
            pipeline_a = pltpu.emit_pipeline(
                functools.partial(inner_compute_a, x_scratch),
                grid=(F_loc // b_inter,),
                in_specs=(wg_spec, wu_spec),
                out_specs=a_spec,
            )
            pipeline_a(wg, wu, a_scratch)
            
            wd_spec = pl.BlockSpec((F_loc, b_hidden), lambda j: (0, j), pipeline_mode=pl.Buffered(buffer_count=2))
            y_spec = pl.BlockSpec((b_seq, b_hidden), lambda j: (out_block, j))
            
            pipeline_y = pltpu.emit_pipeline(
                functools.partial(inner_compute_y, a_scratch),
                grid=(hidden_size // b_hidden,),
                in_specs=(wd_spec,),
                out_specs=y_spec,
            )
            pipeline_y(wd, x_ref.at[device_idx, ...])
            return None
        
        jax.lax.fori_loop(0, half_chunk // b_seq, loop_body, None)

    # --- Prologue ---
    @pl.when(is_start)
    def _():
        run_mlp_for_slice(my_id, left_copy_slice)
        run_mlp_for_slice(my_id, right_copy_slice)

        local_barrier(left_neighbor, right_neighbor, axis_names)

        initial_left_copy.start()
        initial_left_copy.wait()
        initial_right_copy.start()

        signal(LEFT, right_capacity_sem, num_devices, axis_names)
        signal(RIGHT, left_capacity_sem, num_devices, axis_names)

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            pl.semaphore_wait(right_capacity_sem, 1)
            right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
            pl.semaphore_wait(left_capacity_sem, 1)
            left_copy.start()

    # --- Body ---
    def inner_kernel(input_ref, accum_ref):
        @pl.when(pl.program_id(1) == 0)
        def _():
            accum_ref[...] = jnp.zeros_like(accum_ref)

        accum_ref[...] += input_ref[...]

    inner_grid = (
        half_chunk // b_seq,
        hidden_size // b_hidden,
    )
    inner_block_spec = pl.BlockSpec(
        index_map=lambda i, j: (i, j),
        block_shape=(b_seq, b_hidden),
        memory_space=pltpu.VMEM,
    )

    accum_pipeline = pltpu.emit_pipeline(
        inner_kernel,
        in_specs=[inner_block_spec],
        out_specs=inner_block_spec,
        grid=inner_grid,
    )

    @pl.when(~last_iteration)
    def _():
        @pl.when(phase == LEFT)
        def _():
            run_mlp_for_slice(left_copy_device, left_copy_slice)
            accum_pipeline(
                x_ref.at[left_copy_device, left_copy_slice],
                hbm_scratch.at[working_slot, left_copy_slice],
            )

        @pl.when(phase == RIGHT)
        def _():
            run_mlp_for_slice(right_copy_device, right_copy_slice)
            accum_pipeline(
                x_ref.at[right_copy_device, right_copy_slice],
                hbm_scratch.at[working_slot, right_copy_slice],
            )

    # --- Epilogue ---
    @pl.when(is_start)
    def _():
        initial_right_copy.wait()

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            right_copy.wait()
            signal(LEFT, right_capacity_sem, num_devices, axis_names)

        @pl.when(phase == RIGHT)
        def _():
            left_copy.wait()
            signal(RIGHT, left_capacity_sem, num_devices, axis_names)

    # Store result on last iteration.
    @pl.when(last_iteration)
    def _():
        output_copy = pltpu.make_async_copy(
            src_ref=hbm_scratch.at[working_slot, current_phase_slice],
            dst_ref=o_ref.at[current_phase_slice],
            sem=copy_sem,
        )
        output_copy.start()
        output_copy.wait()

        @pl.when(phase == LEFT)
        def _():
            pl.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
            pl.semaphore_wait(left_capacity_sem, 1)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def apply_fused_mlp_sharded(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 256,
    b_inter: int = 128,
    b_hidden: int = 256,
) -> jax.Array:
    in_specs = (
        P(None, None),  # x
        P(None, "model"),  # wg
        P(None, "model"),  # wu
        P("model", None),  # wd
    )
    out_specs = P(None, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def local_fused_mlp(x_loc, wg_loc, wu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape
        F_loc = wg_loc.shape[1]
        num_devices = lax.axis_size("model")

        chunk_size = seq_len // num_devices
        half_chunk = chunk_size // 2

        out_shape = (
            jax.ShapeDtypeStruct((chunk_size, hidden_size), x_loc.dtype),  # o_ref
            jax.ShapeDtypeStruct(
                (2, chunk_size, hidden_size), x_loc.dtype
            ),  # hbm_scratch
            jax.ShapeDtypeStruct(
                (num_devices, chunk_size, hidden_size), x_loc.dtype
            ),  # x_ref (staging)
        )

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # x
                pl.BlockSpec(memory_space=pltpu.HBM),  # wg
                pl.BlockSpec(memory_space=pltpu.HBM),  # wu
                pl.BlockSpec(memory_space=pltpu.HBM),  # wd
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # o_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # hbm_scratch
                pl.BlockSpec(memory_space=pltpu.HBM),  # x_ref
            ],
            grid=(num_devices, 2),
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 5
                + [pltpu.SemaphoreType.REGULAR] * 2  # Capacity semaphores
                + [
                    pltpu.VMEM((b_seq, hidden_size), x_loc.dtype),
                    pltpu.VMEM((b_seq, F_loc), x_loc.dtype),
                ]
            ),
        )

        y_chunk = pl.pallas_call(
            functools.partial(
                reduce_scatter_kernel,
                num_devices=num_devices,
                b_seq=b_seq,
                b_inter=b_inter,
                b_hidden=b_hidden,
                chunk_size=chunk_size,
                half_chunk=half_chunk,
                hidden_size=hidden_size,
                F_loc=F_loc,
                axis_names=mesh.axis_names,
            ),
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(collective_id=0),
        )(x_loc, wg_loc, wu_loc, wd_loc)[0]

        # Gather all chunks from all devices to form the complete (seq_len, hidden_size) array
        return jax.lax.all_gather(y_chunk, axis_name="model", tiled=True)

    return local_fused_mlp(x, wg, wu, wd)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 256,
    b_inter: int = 128,
    b_hidden: int = 256,
) -> jax.Array:
    seq_len, hidden_size = x.shape
    num_devices = mesh.devices.size
    # Sequence length must be divisible by num_devices * 2 * b_seq
    # because of the way the reduce-scatter torus ring and internal pipelines operate.
    chunk_multiple = num_devices * 2 * b_seq
    rem = seq_len % chunk_multiple
    if rem == 0:
        return apply_fused_mlp_sharded(x, wg, wu, wd, mesh, b_seq, b_inter, b_hidden)

    pad_len = chunk_multiple - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")
    out_padded = apply_fused_mlp_sharded(
        x_padded, wg, wu, wd, mesh, b_seq, b_inter, b_hidden
    )
    return out_padded[:seq_len, :]

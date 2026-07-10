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
    send_scratch,
    a_scratch_hbm,
    left_recv_sem,
    left_send_sem,
    copy_sem,
    right_recv_sem,
    right_send_sem,
    left_capacity_sem,
    right_capacity_sem,
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
        src_ref=send_scratch.at[left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=tuple(
            left_neighbor if ax == "model" else lax.axis_index(ax) for ax in axis_names
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )

    initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=send_scratch.at[right_copy_slice],
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

    def inner_compute_a(wg_tile, wu_tile, x_tile, a_tile):
        wg_block = wg_tile[...].astype(x_tile.dtype)
        wu_block = wu_tile[...].astype(x_tile.dtype)
        wgu_block = jnp.concatenate([wg_block, wu_block], axis=1)

        gu_sram = jnp.matmul(x_tile[...], wgu_block, preferred_element_type=jnp.float32)

        h_sram, u_sram = jnp.split(gu_sram, 2, axis=-1)
        a_sram_out = jax.nn.gelu(h_sram, approximate=True) * u_sram
        a_tile[...] = a_sram_out.astype(a_tile.dtype)

    def inner_compute_y_write(wd_tile, a_tile, y_tile):
        wd_block = wd_tile[...].astype(a_tile.dtype)
        y_sram = jnp.matmul(a_tile[...], wd_block, preferred_element_type=jnp.float32)
        y_tile[...] = y_sram.astype(y_tile.dtype)

    def inner_compute_y_accum(wd_tile, a_tile, y_tile_in, y_tile_out):
        wd_block = wd_tile[...].astype(a_tile.dtype)
        y_sram = jnp.matmul(a_tile[...], wd_block, preferred_element_type=jnp.float32)
        y_tile_out[...] = y_tile_in[...] + y_sram.astype(y_tile_out.dtype)

    def run_mlp_for_slice(device_idx, slice_ds, dest_ref, accumulate):
        start_row = device_idx * chunk_size + slice_ds.start
        start_block = start_row // b_seq
        out_block_start = slice_ds.start // b_seq

        wg_spec = pl.BlockSpec((hidden_size, b_inter), lambda f, s: (0, f))
        wu_spec = pl.BlockSpec((hidden_size, b_inter), lambda f, s: (0, f))
        x_spec = pl.BlockSpec((b_seq, hidden_size), lambda f, s: (start_block + s, 0))
        a_spec = pl.BlockSpec((b_seq, b_inter), lambda f, s: (s, f))

        with jax.named_scope("pipeline_a"):
            pipeline_a = pltpu.emit_pipeline(
                inner_compute_a,
                grid=(F_loc // b_inter, half_chunk // b_seq),
                in_specs=(wg_spec, wu_spec, x_spec),
                out_specs=a_spec,
            )
            pipeline_a(wg, wu, x, a_scratch_hbm)

        wd_spec = pl.BlockSpec((F_loc, b_hidden), lambda h, s: (0, h))
        a_spec_in = pl.BlockSpec((b_seq, F_loc), lambda h, s: (s, 0))
        y_spec = pl.BlockSpec((b_seq, b_hidden), lambda h, s: (out_block_start + s, h))

        with jax.named_scope("pipeline_y"):
            if accumulate:
                pipeline_y = pltpu.emit_pipeline(
                    inner_compute_y_accum,
                    grid=(hidden_size // b_hidden, half_chunk // b_seq),
                    in_specs=(wd_spec, a_spec_in, y_spec),
                    out_specs=y_spec,
                )
                pipeline_y(wd, a_scratch_hbm, dest_ref, dest_ref)
            else:
                pipeline_y = pltpu.emit_pipeline(
                    inner_compute_y_write,
                    grid=(hidden_size // b_hidden, half_chunk // b_seq),
                    in_specs=(wd_spec, a_spec_in),
                    out_specs=y_spec,
                )
                pipeline_y(wd, a_scratch_hbm, dest_ref)

    # --- Prologue ---
    @pl.when(is_start)
    def _():
        with jax.named_scope("prologue_left_mlp"):
            run_mlp_for_slice(my_id, left_copy_slice, send_scratch, False)

        local_barrier(left_neighbor, right_neighbor, axis_names, double_barrier=False)

        with jax.named_scope("initial_left_copy_start"):
            initial_left_copy.start()

        with jax.named_scope("prologue_right_mlp"):
            run_mlp_for_slice(my_id, right_copy_slice, send_scratch, False)

        with jax.named_scope("initial_left_copy_wait"):
            initial_left_copy.wait()

        # We need a second barrier to ensure everyone has finished prologue_right_mlp?
        # Actually, double_barrier is not strictly needed here for the right copy, 
        # but let's keep the synchronization semantic same as original which had a double barrier.
        # Wait, the original had ONE double_barrier after both left and right MLP.
        # If we put a barrier after prologue_right_mlp, it acts as the second half of the double barrier.
        # But `initial_right_copy` just pushes to the right neighbor. The right neighbor is already 
        # guaranteed to be in the kernel by the first local_barrier!
        # So we can just start `initial_right_copy` immediately.
        with jax.named_scope("initial_right_copy_start"):
            initial_right_copy.start()

        signal(LEFT, right_capacity_sem, num_devices, axis_names)
        signal(RIGHT, left_capacity_sem, num_devices, axis_names)

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            pl.semaphore_wait(right_capacity_sem, 1)
            with jax.named_scope("right_copy_start"):
                right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
            pl.semaphore_wait(left_capacity_sem, 1)
            with jax.named_scope("left_copy_start"):
                left_copy.start()

    # --- Body ---
    @pl.when(~last_iteration)
    def _():
        @pl.when(phase == LEFT)
        def _():
            with jax.named_scope("body_left_mlp"):
                run_mlp_for_slice(
                    left_copy_device,
                    left_copy_slice,
                    hbm_scratch.at[working_slot, ...],
                    True,
                )

        @pl.when(phase == RIGHT)
        def _():
            with jax.named_scope("body_right_mlp"):
                run_mlp_for_slice(
                    right_copy_device,
                    right_copy_slice,
                    hbm_scratch.at[working_slot, ...],
                    True,
                )

    # --- Epilogue ---
    @pl.when(is_start)
    def _():
        with jax.named_scope("initial_right_copy_wait"):
            initial_right_copy.wait()

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            with jax.named_scope("right_copy_wait"):
                right_copy.wait()
            signal(LEFT, right_capacity_sem, num_devices, axis_names)

        @pl.when(phase == RIGHT)
        def _():
            with jax.named_scope("left_copy_wait"):
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
        with jax.named_scope("output_copy_start"):
            output_copy.start()
        with jax.named_scope("output_copy_wait"):
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
    b_inter: int = 512,
    b_hidden: int = 384,
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
                (chunk_size, hidden_size), x_loc.dtype
            ),  # send_scratch (was x_ref)
            jax.ShapeDtypeStruct((half_chunk, F_loc), x_loc.dtype),  # a_scratch_hbm
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
                pl.BlockSpec(memory_space=pltpu.HBM),  # send_scratch
                pl.BlockSpec(memory_space=pltpu.HBM),  # a_scratch_hbm
            ],
            grid=(num_devices, 2),
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 5
                + [pltpu.SemaphoreType.REGULAR] * 2  # Capacity semaphores
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
    b_inter: int = 512,
    b_hidden: int = 384,
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

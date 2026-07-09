import functools
import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

partition = P(None, "x")
num_devices = jax.local_device_count()

mesh = jax.make_mesh((num_devices,), ("x",))
sharding = jax.sharding.NamedSharding(mesh, partition)

LEFT = 0
RIGHT = 1


def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
    """Performs a barrier with neighbors on the global barrier semaphore.

    Optionally performs a second barrier, which prevents a potential race
    when reusing the same collective_id across kernel invocations.
    """
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:
        # The double-barrier prevents a race condition where one neighbor can
        # re-enter the kernel again on a subsequent call and increment the
        # barrier semaphore a second time. This would unblock the current device
        # even if the other neighbor is not ready yet.
        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor in [left_neighbor, right_neighbor]:
                pl.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=(neighbor,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(second_barrier, 2)


def mod(x, n):
    return lax.rem(x + n, n)


def signal(left_or_right, semaphore):
    my_id = lax.axis_index("x")
    if left_or_right == LEFT:
        neighbor = mod(my_id - 1, num_devices)
    else:
        neighbor = mod(my_id + 1, num_devices)
    pl.semaphore_signal(
        semaphore,
        inc=1,
        device_id=(neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )


# We pick a large outer kernel block size that we do not want to place
# in VMEM. For pedagogical purposes we use (4096, 4096), although in
# principle this can be much larger.
outer_block_size = (4096, 4096)
# We pick a smaller VMEM block size for the inner kernel.
inner_block_size = (128, 128)
input_arr = jax.random.uniform(
    jax.random.key(0),
    shape=(
        outer_block_size[0] * num_devices,
        outer_block_size[1] * num_devices,
    ),
)
input_arr = jax.device_put(input_arr, sharding)


inner_grid = (
    outer_block_size[0] // inner_block_size[0] // 2,
    outer_block_size[1] // inner_block_size[1],
)
inner_block_spec = pl.BlockSpec(
    index_map=lambda i, j: (i, j),
    block_shape=inner_block_size,
    memory_space=pltpu.TPUMemorySpace.VMEM,
)


def reduce_scatter_kernel(
    x_ref,
    o_ref,
    hbm_scratch,
    left_recv_sem,
    left_send_sem,
    copy_sem,
    right_recv_sem,
    right_send_sem,
    left_capacity_sem,
    right_capacity_sem,
):
    outer_step = pl.program_id(0)
    phase = pl.program_id(1)
    is_start = jnp.logical_and(outer_step == 0, phase == 0)
    last_iteration = outer_step == pl.num_programs(0) - 1

    working_slot = lax.rem(outer_step, 2)
    receiving_slot = 1 - working_slot
    my_id = lax.axis_index("x")
    right_neighbor = mod(my_id + 1, num_devices)
    left_neighbor = mod(my_id - 1, num_devices)

    left_copy_device = mod(my_id + outer_step + 1, num_devices)
    right_copy_device = mod(my_id - outer_step - 1, num_devices)
    left_copy_slice = pl.ds(0, outer_block_size[0] // 2)
    right_copy_slice = pl.ds(outer_block_size[0] // 2, outer_block_size[0] // 2)
    current_phase_slice = pl.ds(
        phase * (outer_block_size[0] // 2), outer_block_size[0] // 2
    )

    initial_left_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    right_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pl.DeviceIdType.MESH,
    )

    # --- Prologue ---
    @pl.when(is_start)
    def _():
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        local_barrier(left_neighbor, right_neighbor)

        initial_left_copy.start()
        initial_left_copy.wait()
        initial_right_copy.start()

        # We tell our left neighbor that it is allowed to send to the right.
        # (and vice versa for right neighbor)
        signal(LEFT, right_capacity_sem)
        signal(RIGHT, left_capacity_sem)

    @pl.when(~is_start)
    def _():
        @pl.when(phase == LEFT)
        def _():
            # We block here until our right neighbor tells use we can send to
            # the right.
            pl.semaphore_wait(right_capacity_sem, 1)
            right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
            # We block here until our left neighbor tells use we can send to
            # the left.
            pl.semaphore_wait(left_capacity_sem, 1)
            left_copy.start()

    # --- Body ---
    def inner_kernel(input_ref, accum_ref):
        @pl.when(pl.program_id(1) == 0)
        def _():
            accum_ref[...] = jnp.zeros_like(accum_ref)

        accum_ref[...] += input_ref[...]

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
            accum_pipeline(
                x_ref.at[left_copy_device, left_copy_slice],
                hbm_scratch.at[working_slot, left_copy_slice],
            )

        @pl.when(phase == RIGHT)
        def _():
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
            signal(LEFT, right_capacity_sem)

        @pl.when(phase == RIGHT)
        def _():
            left_copy.wait()
            signal(RIGHT, left_capacity_sem)

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

        # Clean up semaphores so that they exit with a value of 0.
        @pl.when(phase == LEFT)
        def _():
            pl.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
            pl.semaphore_wait(left_capacity_sem, 1)


out_shape = (
    jax.ShapeDtypeStruct((outer_block_size[0], outer_block_size[1]), jnp.float32),
    # Shape: [working/recv, block[0], block[1]]
    jax.ShapeDtypeStruct(
        (2, outer_block_size[0], outer_block_size[1]), jnp.float32
    ),  # hbm_scratch
)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=0,
    in_specs=[
        pl.BlockSpec(memory_space=pl.ANY),
    ],
    out_specs=[
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ],
    grid=(num_devices, 2),
    scratch_shapes=(
        [pltpu.SemaphoreType.DMA] * 5
        + [pltpu.SemaphoreType.REGULAR] * 2  # Capacity semaphores
    ),
)


def pallas_reduce_scatter(input_arr):
    input_arr = input_arr.reshape(num_devices, outer_block_size[0], outer_block_size[1])
    return pl.pallas_call(
        reduce_scatter_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )(input_arr)[0]


pallas_result = jax.jit(
    jax.shard_map(
        pallas_reduce_scatter,
        mesh=mesh,
        in_specs=P(None, "x"),
        out_specs=P("x", None),
        check_vma=False,
    )
)(input_arr)

pallas_result = jax.block_until_ready(pallas_result)

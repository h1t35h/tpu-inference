"""Optimized Gated MLP (SwiGLU) Pallas kernel for TPU.

Fuses the gate/up projections, the SiLU activation, and the down projection
into a single pipelined TPU kernel, completely eliminating intermediate HBM traffic.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import shard_map
from jax.sharding import PartitionSpec as P

def inner_compute_a(
    x_tile,
    wg_tile,
    wu_tile,
    a_tile,
):
    # 1. Fetch & Upcast weights
    wg_sram = wg_tile[...].astype(x_tile.dtype)
    wu_sram = wu_tile[...].astype(x_tile.dtype)

    # 2. Matmul 1 (x @ W_in)
    wgu_sram = jnp.concatenate([wg_sram, wu_sram], axis=1)
    gu_sram = jnp.matmul(x_tile[...], wgu_sram, preferred_element_type=jnp.float32)
    h_sram, u_sram = jnp.split(gu_sram, 2, axis=-1)

    # Activate
    a_sram = jax.nn.gelu(h_sram, approximate=True) * u_sram
    a_tile[...] = a_sram.astype(a_tile.dtype)


def inner_compute_y(
    a_tile,
    wd_tile,
    y_tile,
):
    wd_sram = wd_tile[...].astype(a_tile.dtype)
    
    # 3. Matmul 2 (a @ W_out)
    y_current_sram = jnp.matmul(a_tile[...], wd_sram, preferred_element_type=jnp.float32)
    y_tile[...] = y_current_sram.astype(y_tile.dtype)


def mlp_kernel_main(
    x_hbm, wg_hbm, wu_hbm, wd_hbm, y_hbm, a_scratch, *, b_seq, b_inter, b_hidden, hidden_size
):
    """Entry point for Pallas grid. Wires up HBM references to the pipeline."""
    seq_idx = pl.program_id(0)
    F_loc = wg_hbm.shape[1]
    num_inter = F_loc // b_inter
    num_hidden = hidden_size // b_hidden

    # 1. Block specs for Pipeline A
    x_spec = pl.BlockSpec((b_seq, hidden_size), lambda i_i: (seq_idx, 0))
    wg_spec = pl.BlockSpec(
        (hidden_size, b_inter),
        lambda i_i: (0, i_i),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    wu_spec = pl.BlockSpec(
        (hidden_size, b_inter),
        lambda i_i: (0, i_i),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    a_spec = pl.BlockSpec((b_seq, b_inter), lambda i_i: (0, i_i), memory_space=pltpu.VMEM)

    # 2. Emit Pipeline A (computes 'a' into VMEM)
    pipeline_a = pltpu.emit_pipeline(
        inner_compute_a,
        grid=(num_inter,),
        in_specs=(x_spec, wg_spec, wu_spec),
        out_specs=a_spec,
    )
    pipeline_a(x_hbm, wg_hbm, wu_hbm, a_scratch)

    # 3. Block specs for Pipeline Y
    a_full_spec = pl.BlockSpec((b_seq, F_loc), lambda j: (0, 0), memory_space=pltpu.VMEM)
    wd_spec = pl.BlockSpec(
        (F_loc, b_hidden),
        lambda j: (0, j),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    y_spec = pl.BlockSpec((b_seq, b_hidden), lambda j: (seq_idx, j))

    # 4. Emit Pipeline Y (computes 'y' into HBM)
    pipeline_y = pltpu.emit_pipeline(
        inner_compute_y,
        grid=(num_hidden,),
        in_specs=(a_full_spec, wd_spec),
        out_specs=y_spec,
    )
    pipeline_y(a_scratch, wd_hbm, y_hbm)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def apply_fused_mlp_sharded(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 64,
    b_inter: int = 128,
    b_hidden: int = 512,
) -> jax.Array:
    in_specs = (
        P(None, None),  # x
        P(None, "model"),  # wg (gate weight, sharded along model/tensor axis)
        P(None, "model"),  # wu (up weight, sharded along model/tensor axis)
        P("model", None),  # wd (down weight, sharded along model/tensor axis)
    )
    out_specs = P(None, None)

    @functools.partial(
        shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    def local_fused_mlp(x_loc, wg_loc, wu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape

        # 1D outer grid (parallelizing sequence length across TPU cores)
        grid = (seq_len // b_seq,)

        # Pass full tensors to the kernel main as HBM references
        pallas_in_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),  # x_loc
            pl.BlockSpec(memory_space=pltpu.HBM),  # wg_loc
            pl.BlockSpec(memory_space=pltpu.HBM),  # wu_loc
            pl.BlockSpec(memory_space=pltpu.HBM),  # wd_loc
        )
        pallas_out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        y_loc = pl.pallas_call(
            functools.partial(
                mlp_kernel_main,
                b_seq=b_seq,
                b_inter=b_inter,
                b_hidden=b_hidden,
                hidden_size=hidden_size,
            ),
            out_shape=jax.ShapeDtypeStruct((seq_len, hidden_size), x_loc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=pallas_in_specs,
                out_specs=pallas_out_specs,
                scratch_shapes=[pltpu.VMEM((b_seq, wg_loc.shape[1]), x_loc.dtype)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(x_loc, wg_loc, wu_loc, wd_loc)

        return jax.lax.psum(y_loc, axis_name="model")

    return local_fused_mlp(x, wg, wu, wd)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 64,
    b_inter: int = 128,
    b_hidden: int = 512,
) -> jax.Array:
    """Pads the input sequence length to be a multiple of b_seq if necessary."""
    seq_len, hidden_size = x.shape
    rem = seq_len % b_seq
    if rem == 0:
        return apply_fused_mlp_sharded(x, wg, wu, wd, mesh, b_seq, b_inter, b_hidden)

    pad_len = b_seq - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")
    out_padded = apply_fused_mlp_sharded(x_padded, wg, wu, wd, mesh, b_seq, b_inter, b_hidden)
    return out_padded[:seq_len, :]

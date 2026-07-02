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

def inner_compute_a(x_scratch, wg_tile, wu_tile, a_tile):
    b_seq = x_scratch.shape[0]
    hidden_size = x_scratch.shape[1]
    b_inter = wg_tile.shape[1]
    block_k = 128
    
    def loop_body(k, acc):
        x_block = x_scratch[pl.dslice(0, b_seq), pl.dslice(k * block_k, block_k)]
        wg_block = wg_tile[pl.dslice(k * block_k, block_k), pl.dslice(0, b_inter)].astype(x_block.dtype)
        wu_block = wu_tile[pl.dslice(k * block_k, block_k), pl.dslice(0, b_inter)].astype(x_block.dtype)
        wgu_block = jnp.concatenate([wg_block, wu_block], axis=1)
        return acc + jnp.matmul(x_block, wgu_block, preferred_element_type=jnp.float32)
        
    acc_init = jnp.zeros((b_seq, 2 * b_inter), dtype=jnp.float32)
    gu_sram = jax.lax.fori_loop(0, hidden_size // block_k, loop_body, acc_init)
    
    h_sram, u_sram = jnp.split(gu_sram, 2, axis=-1)
    a_sram_out = jax.nn.gelu(h_sram, approximate=True) * u_sram
    a_tile[...] = a_sram_out.astype(a_tile.dtype)


def inner_compute_y(a_scratch, wd_tile, y_tile):
    b_seq = a_scratch.shape[0]
    F_loc = a_scratch.shape[1]
    b_hidden = wd_tile.shape[1]
    block_k = 128
    
    def loop_body(k, acc):
        a_block = a_scratch[pl.dslice(0, b_seq), pl.dslice(k * block_k, block_k)]
        wd_block = wd_tile[pl.dslice(k * block_k, block_k), pl.dslice(0, b_hidden)].astype(a_block.dtype)
        return acc + jnp.matmul(a_block, wd_block, preferred_element_type=jnp.float32)
        
    acc_init = jnp.zeros((b_seq, b_hidden), dtype=jnp.float32)
    y_sram = jax.lax.fori_loop(0, F_loc // block_k, loop_body, acc_init)
    
    # Overlap compute and communication by reducing the block immediately
    y_sram = jax.lax.psum(y_sram, axis_name="model")
    
    y_tile[...] = y_sram.astype(y_tile.dtype)


def mlp_kernel_main(
    x_hbm, wg_hbm, wu_hbm, wd_hbm, y_hbm, x_scratch, a_scratch, *, b_seq, b_inter, b_hidden, hidden_size
):
    """Entry point for Pallas grid. Wires up HBM references to the pipeline."""
    seq_idx = pl.program_id(0)
    F_loc = wg_hbm.shape[1]
    num_inter = F_loc // b_inter
    num_hidden = hidden_size // b_hidden

    # 1. Pipeline to load x from HBM into VMEM
    def load_x(x_tile, x_scratch_tile):
        x_scratch_tile[...] = x_tile[...]
        
    pl_load_x = pltpu.emit_pipeline(
        load_x,
        grid=(1,),
        in_specs=(pl.BlockSpec((b_seq, hidden_size), lambda j: (seq_idx, 0)),),
        out_specs=(pl.BlockSpec((b_seq, hidden_size), lambda j: (0, 0), memory_space=pltpu.VMEM),),
    )
    pl_load_x(x_hbm, x_scratch)

    # 2. Block specs for Pipeline A
    wg_spec = pl.BlockSpec(
        (hidden_size, b_inter),
        lambda i: (0, i),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    wu_spec = pl.BlockSpec(
        (hidden_size, b_inter),
        lambda i: (0, i),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    a_spec = pl.BlockSpec((b_seq, b_inter), lambda i: (0, i), memory_space=pltpu.VMEM)

    # 3. Emit Pipeline A (computes 'a' into VMEM)
    pipeline_a = pltpu.emit_pipeline(
        functools.partial(inner_compute_a, x_scratch),
        grid=(num_inter,),
        in_specs=(wg_spec, wu_spec),
        out_specs=a_spec,
    )
    pipeline_a(wg_hbm, wu_hbm, a_scratch)

    # 4. Block specs for Pipeline Y
    wd_spec = pl.BlockSpec(
        (F_loc, b_hidden),
        lambda j: (0, j),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    y_spec = pl.BlockSpec((b_seq, b_hidden), lambda j: (seq_idx, j))

    # 5. Emit Pipeline Y (computes 'y' into HBM)
    pipeline_y = pltpu.emit_pipeline(
        functools.partial(inner_compute_y, a_scratch),
        grid=(num_hidden,),
        in_specs=(wd_spec,),
        out_specs=y_spec,
    )
    pipeline_y(wd_hbm, y_hbm)


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
                scratch_shapes=[
                    pltpu.VMEM((b_seq, hidden_size), x_loc.dtype),
                    pltpu.VMEM((b_seq, wg_loc.shape[1]), x_loc.dtype),
                ],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(x_loc, wg_loc, wu_loc, wd_loc)

        return y_loc

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
    """Pads the input sequence length to be a multiple of b_seq if necessary."""
    seq_len, hidden_size = x.shape
    rem = seq_len % b_seq
    if rem == 0:
        return apply_fused_mlp_sharded(x, wg, wu, wd, mesh, b_seq, b_inter, b_hidden)

    pad_len = b_seq - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")
    out_padded = apply_fused_mlp_sharded(x_padded, wg, wu, wd, mesh, b_seq, b_inter, b_hidden)
    return out_padded[:seq_len, :]

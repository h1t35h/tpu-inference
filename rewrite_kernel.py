import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def inner_compute_a(x_sram, x_dtype, wg_tile, wu_tile, a_tile):
    wg_sram = wg_tile[...].astype(x_dtype)
    wu_sram = wu_tile[...].astype(x_dtype)
    wgu_sram = jnp.concatenate([wg_sram, wu_sram], axis=1)
    gu_sram = jnp.matmul(x_sram, wgu_sram, preferred_element_type=jnp.float32)
    h_sram, u_sram = jnp.split(gu_sram, 2, axis=-1)
    a_sram_out = jax.nn.gelu(h_sram, approximate=True) * u_sram
    a_tile[...] = a_sram_out.astype(a_tile.dtype)

def inner_compute_y(a_full_sram, x_dtype, wd_tile, y_tile):
    wd_sram = wd_tile[...].astype(x_dtype)
    y_sram = jnp.matmul(a_full_sram, wd_sram, preferred_element_type=jnp.float32)
    y_tile[...] = y_sram.astype(y_tile.dtype)

def mlp_kernel_main(
    x_hbm, wg_hbm, wu_hbm, wd_hbm, y_hbm, a_scratch, *, b_seq, b_inter, b_hidden, hidden_size
):
    seq_idx = pl.program_id(0)
    F_loc = wg_hbm.shape[1]
    num_inter = F_loc // b_inter
    num_hidden = hidden_size // b_hidden

    # 1. Load x ONCE into SRAM
    x_sram = pl.load(x_hbm, (pl.dslice(seq_idx * b_seq, b_seq), pl.dslice(0, hidden_size)))
    x_dtype = x_sram.dtype

    # 2. Pipeline A (computes 'a' into VMEM)
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

    pipeline_a = pltpu.emit_pipeline(
        functools.partial(inner_compute_a, x_sram, x_dtype),
        grid=(num_inter,),
        in_specs=(wg_spec, wu_spec),
        out_specs=a_spec,
    )
    pipeline_a(wg_hbm, wu_hbm, a_scratch)

    # 3. Load a ONCE into SRAM (from VMEM)
    a_full_sram = pl.load(a_scratch, (pl.dslice(0, b_seq), pl.dslice(0, F_loc)))

    # 4. Pipeline Y (computes 'y' into HBM)
    wd_spec = pl.BlockSpec(
        (F_loc, b_hidden),
        lambda j: (0, j),
        pipeline_mode=pl.Buffered(buffer_count=2),
    )
    y_spec = pl.BlockSpec((b_seq, b_hidden), lambda j: (seq_idx, j))

    pipeline_y = pltpu.emit_pipeline(
        functools.partial(inner_compute_y, a_full_sram, x_dtype),
        grid=(num_hidden,),
        in_specs=(wd_spec,),
        out_specs=y_spec,
    )
    pipeline_y(wd_hbm, y_hbm)

# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel with int4 unpacking support."""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul import util
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    TunedValue, get_device_vmem_limit, get_tuned_block_sizes)
from tpu_inference.kernels.quantized_matmul.util import (get_kernel_name,
                                                         next_multiple,
                                                         unfold_args)

quantize_tensor = util.quantize_tensor
MXU_SIZE = 256


@jax.jit(static_argnames=[
    "block_size",
    "x_q_dtype",
    "tuned_value",
])
def int4_quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in // 2] packed int4 in int8
    w_scale: jax.Array,  # [n_in // block_size, 1, n_out]
    w_zp: jax.Array | None = None,  # [n_out]
    block_size: int | None = None,
    x_q_dtype: jnp.dtype | None = None,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """Quantized matmul kernel with int4 unpacking and blockwise support.

    Args:
      x: Input unquantized array.
      w_q: Weight packed int4 array. [n_output_features, n_input_features // 2]
      w_scale: Weight quantization scale. [n_input_features // block_size, 1, n_output_features]
      w_zp: Weight zero point for asymmetric quantization.
      block_size: Block size for subchannel quantization.
      x_q_dtype: Quantization type of the input. If None or if the value is the
        same as x.dtype, then no quantization is applied.
      tuned_value: Kernel tuned values for optimal performance.

    Returns:
      Quantized matmul result.
    """

    if block_size is None:
        raise ValueError("Block size was not specified.")
    if w_zp is not None:
        raise NotImplementedError("zero_point is not supported.")
        # Comment: We are only supporting symmetric configuration for now.
        # Assume zero point is zero and ignore it.

    if x_q_dtype is None:
        x_q_dtype = x.dtype
    quantize_activation = x_q_dtype != x.dtype

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, *_ = w_q.shape

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            n_batch=orig_n_batch,
            n_out=orig_n_out,
            n_in=orig_n_in,
            x_q_dtype=jnp.dtype(x_q_dtype).name,
            w_q_dtype="int8",  # Treat packed weights as int8 for tuning
        )
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    n_lane_multiplier = tuned_value.n_lane_multiplier
    block_size = tuned_value.in_block_size if block_size == orig_n_in else block_size

    # Pad the inputs to be multiple of block size.
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, (0, padded_n_out - orig_n_out))
    
    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        # Pad packed weights along axis 1
        pad_k = (padded_n_in - orig_n_in) // 2
        w_q = jnp.pad(w_q, ((0, 0), (0, pad_k)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_acc = n_in > 1
    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    acc_dtype = jnp.bfloat16
    if (quantize_activation and jnp.issubdtype(x_q_dtype, jnp.integer)):
        acc_dtype = jnp.int32

    vmem_limit_bytes = util.get_vmem_limit(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        x_dtype=x.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=jnp.int8,
        scale_dtype=jnp.float32,
        out_dtype=x.dtype,
        acc_dtype=acc_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
        upper_limit_bytes=get_device_vmem_limit(),
    )

    steps_k = in_block_size // block_size
    compute_tile_n = MXU_SIZE * n_lane_multiplier
    steps_n = max(1, out_block_size // compute_tile_n)

    def kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch):
        pid_k = pl.program_id(2)
        is_first_step = pid_k == 0
        is_last_step = pid_k == (orig_n_in // in_block_size - 1)

        def accum(is_first_step, is_last_step):
            accumulators = [None] * steps_n

            for i in range(steps_k):
                k_start, k_end = i * block_size, (i + 1) * block_size
                lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
                lhs_q, lhs_scale = util.quantize_block(lhs_sub, 1, x_q_dtype)
                lhs_scale = lhs_scale.astype(acc_dtype)

                # Load packed weights
                k_start_packed = k_start // 2
                k_end_packed = k_end // 2
                rhs_q_packed = rhs_ref[:, k_start_packed:k_end_packed]
                rhs_scale_full = w_scales_ref[i, :, :].astype(acc_dtype)

                # Unpack int4 to int8
                u8_packed = rhs_q_packed.astype(jnp.uint8)
                low = u8_packed & 0x0F
                high = (u8_packed >> 4) & 0x0F
                
                # Sign extend back to int8 (assuming values are -8 to 7)
                low = jnp.where(low > 7, low.astype(jnp.int8) - 16, low.astype(jnp.int8))
                high = jnp.where(high > 7, high.astype(jnp.int8) - 16, high.astype(jnp.int8))
                
                # Interleave them along the K dimension
                interleaved = jnp.stack([low, high], axis=-1)
                rhs_q_slice_full = interleaved.reshape(out_block_size, block_size)

                for j in range(steps_n):
                    n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n

                    rhs_q_slice = rhs_q_slice_full[n_start:n_end, :]
                    rhs_scale_slice = rhs_scale_full[:, n_start:n_end]
                    
                    if jnp.issubdtype(x_q_dtype, jnp.integer):
                        preferred_element_type = jnp.int32
                    else:
                        preferred_element_type = jnp.float32
                        
                    dot_res = jax.lax.dot_general(
                        lhs_q,
                        rhs_q_slice,
                        (((1, ), (1, )), ((), ())),
                        preferred_element_type=preferred_element_type,
                    )
                    res = dot_res.astype(acc_dtype)
                    res = res * lhs_scale
                    res = res * rhs_scale_slice
                    
                    if i == 0:
                        accumulators[j] = res
                    else:
                        accumulators[j] += res

            acc_block = jnp.concatenate(accumulators, axis=1)

            if not is_first_step:
                acc_block += acc_scratch[...]

            if is_last_step:
                out_ref[...] = acc_block.astype(out_ref.dtype)
            else:
                acc_scratch[...] = acc_block

        unfold_args((is_first_step, is_last_step), (), accum)

    kernel = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (batch_block_size, in_block_size),
                    lambda b, o, i: (b, i),
                    memory_space=pltpu.VMEM,
                ),  # x
                pl.BlockSpec(
                    (out_block_size, in_block_size // 2),
                    lambda b, o, i: (o, i),
                    memory_space=pltpu.VMEM,
                ),  # w_q (packed)
                pl.BlockSpec(
                    (steps_k, 1, out_block_size),
                    lambda _, o, i: (i, 0, o),
                    memory_space=pltpu.VMEM,
                ),
            ],  # w_scale
            out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                   lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), jnp.bfloat16)
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out),
                                       x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    tuned_value = get_tuned_block_sizes(
        n_batch=orig_n_batch,
        n_out=orig_n_out,
        n_in=orig_n_in,
        x_q_dtype=jnp.dtype(x_q_dtype).name,
        w_q_dtype="int8",
    )
    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        out = kernel(x, w_q, w_scale)

    return out[:orig_n_batch, :orig_n_out]

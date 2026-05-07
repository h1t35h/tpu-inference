# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Sequence

import jax
from jax import numpy as jnp
from jax.sharding import Mesh

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights,
    process_linear_weights,
)
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.utils import slice_sharded_tensor_for_concatenation


class Int4LinearMethod:
    """Implements the forward method for int4 linear layers."""

    def __init__(self, linear_config: QuantLinearConfig):
        self.linear_config = linear_config

    def _apply_fused(
        self,
        x: jax.Array,
        weight_jax: jax.Array,
        weight_scale_jax: jax.Array,
        bias: Optional[jax.Array],
    ) -> jax.Array:
        # For W4A16, activations are not quantized.
        # We pass x_q_dtype=x.dtype to signal no activation quantization.
        outs = sharded_quantized_matmul(
            x,
            weight_jax,
            weight_scale_jax,
            self.linear_config.weight_sharding,
            mesh=self.linear_config.mesh,
            x_q_dtype=x.dtype,
        )

        if bias is not None:
            outs += bias
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards
        )
        return jnp.concatenate(outs, axis=-1)


def pack_int4(w: jax.Array) -> jax.Array:
    """Packs two 4-bit integers into a single int8.

    Args:
        w: Unpacked weight tensor of shape (..., K) with values in range [-8, 7].

    Returns:
        Packed weight tensor of shape (..., K // 2) of type int8.
    """
    assert w.shape[-1] % 2 == 0, "K dimension must be even for packing"
    w_even = w[..., 0::2]
    w_odd = w[..., 1::2]

    # Treat as unsigned for bitwise operations
    w_even = w_even.astype(jnp.uint8) & 0x0F
    w_odd = w_odd.astype(jnp.uint8) & 0x0F

    packed = w_even | (w_odd << 4)
    return packed.astype(jnp.int8)


@jax.jit(
    static_argnames=(
        "weight_block_size",
        "requant_block_size",
        "output_sizes",
        "requant_weight_dtype",
        "fuse_matmuls",
        "n_shards",
    )
)
def process_blockwise_int4_linear_weights(
    weight: jax.Array,
    weight_scale: jax.Array,
    *,
    bias: jax.Array | None,
    weight_block_size: Sequence[int],
    requant_block_size,
    output_sizes,
    requant_weight_dtype,
    fuse_matmuls,
    n_shards,
) -> LinearWeights:
    # Zero points are ignored (assumed symmetric).
    # Comment: We are only supporting symmetric configuration for now.
    # Assume zero point is zero and ignore it.

    weights = []
    weight_scales = []
    block_n = weight_block_size[0]

    start = 0
    for output_size in output_sizes:
        end = start + output_size

        weight_slice = weight[start:end]
        scale_start = start // block_n
        scale_end = math.ceil(end / block_n)
        weight_scale_slice = weight_scale[scale_start:scale_end]

        # Assume input is unpacked and we need to pack it.
        packed_weight_slice = pack_int4(weight_slice)

        weights.append(packed_weight_slice)
        weight_scales.append(weight_scale_slice)

        start = end

    weight = jnp.concat(weights, axis=0)
    weight_scale = jnp.concat(weight_scales, axis=0)

    return process_linear_weights(
        LinearWeights(
            weight=weight,
            weight_scale=weight_scale,
            zero_point=None,
            bias=bias,
        ),
        fused=fuse_matmuls,
        output_sizes=output_sizes,
        reorder_size=n_shards,
    )

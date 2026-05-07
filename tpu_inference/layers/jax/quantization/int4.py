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
from functools import partial
from typing import Any, Literal, Optional, Sequence, cast

from flax import nnx
import jax
import jax.numpy as jnp

from tpu_inference.layers.common.process_weights.linear_weights import (
    shard_linear_weights,
)
from tpu_inference.layers.common.quantization import int4 as common_int4
from tpu_inference.layers.common.utils import cpu_mesh, cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (
    QuantizationConfig,
    QuantLinearConfig,
)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class Int4Config(QuantizationConfig):
    ACTIVATION_SCHEMES = ["dynamic", "static"]

    def __init__(self, hf_quant_config: dict):
        # Parse config as needed.
        # The config says group_size: 32.
        self.weight_block_size = [32, 32]  # Default
        config_groups = hf_quant_config.get("config_groups", {})
        for k, v in config_groups.items():
            weights_cfg = v.get("weights", {})
            group_size = weights_cfg.get("group_size")
            if group_size:
                self.weight_block_size = [32, group_size]
                break

    def get_quant_method(
        self, layer: JaxModule, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer, enable_sp=False)
            return Int4LinearMethod(self, layer, linear_config)
        return None


class Int4LinearMethod(QuantizeMethodBase, common_int4.Int4LinearMethod):
    def __init__(
        self,
        quant_config: Int4Config,
        layer: JaxEinsum,
        linear_config: QuantLinearConfig,
    ):
        common_int4.Int4LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config
        self.einsum_str = layer.einsum_str

        self.out_features = linear_config.out_features
        self.in_features = math.prod(linear_config.in_features)
        self.batch_features = linear_config.batch_features
        self.weight_sharding = linear_config.weight_sharding

        if self.batch_features:
            self.kernel_shape = layer.kernel_shape
        else:
            self.kernel_shape = (
                math.prod(self.out_features),
                self.in_features,
            )

    def create_weights_jax(
        self, layer: JaxModule, *weight_args, rngs, **extra_weight_attrs
    ):
        assert isinstance(layer, JaxEinsum)
        out_features = sum(self.linear_config.output_sizes)

        # Weights will be packed in process_weights_after_loading.
        # Initialize with unpacked shape so the loader can fill it.
        unpacked_shape = (out_features, self.in_features)

        # Initialize with dummy values, to be filled by loader.
        layer.weight = nnx.Param(
            jnp.zeros(unpacked_shape, dtype=jnp.int8), eager_sharding=False
        )
        layer.weight.set_metadata("sharding", self.weight_sharding)

        # Block-wise quantization scale
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        layer.weight_scale_inv = nnx.Param(
            jnp.ones(
                [
                    (out_features + block_n - 1) // block_n,
                    (self.in_features + block_k - 1) // block_k,
                ],
                dtype=layer.dtype,
            ),
            eager_sharding=False,
        )
        layer.weight_scale_inv.set_metadata("sharding", self.weight_sharding)

        # Force onto CPU for processing
        layer.weight.set_metadata("mesh", cpu_mesh())
        layer.weight_scale_inv.set_metadata("mesh", cpu_mesh())
        if layer.bias is not None:
            layer.bias.set_metadata("mesh", cpu_mesh())

    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        assert isinstance(layer, JaxEinsum)

        if not layer.weight.get_metadata(
            "_is_loaded", False
        ) or not layer.weight_scale_inv.get_metadata("_is_loaded", False):
            return False

        with cpu_mesh_context():
            weight = layer.weight[...]
            weight_scale_inv = layer.weight_scale_inv[...]
            bias = (
                layer.bias[...]
                if getattr(layer, "bias", None) is not None
                else None
            )
            if bias is not None:
                bias = bias.reshape(-1)

            weights = common_int4.process_blockwise_int4_linear_weights(
                weight,
                weight_scale_inv,
                bias=bias,
                weight_block_size=tuple(self.quant_config.weight_block_size),
                requant_block_size=self.linear_config.requant_block_size,
                output_sizes=tuple(self.linear_config.output_sizes),
                requant_weight_dtype=self.linear_config.requant_weight_dtype,
                fuse_matmuls=self.linear_config.fuse_matmuls,
                n_shards=self.linear_config.n_shards,
            )

            # Reshape scale for kernel: (num_blocks, 1, out_features)
            weights.weight_scale = jnp.expand_dims(
                jnp.transpose(weights.weight_scale),
                axis=1,
            )

        # Put onto device
        weights = shard_linear_weights(
            weights,
            mesh=None,
            weight_p_spec=self.linear_config.weight_sharding,
            bias_p_spec=self.linear_config.bias_sharding,
        )

        layer.weight = nnx.Param(weights.weight)
        layer.weight_scale_inv = nnx.Param(weights.weight_scale)
        if bias is not None:
            layer.bias = nnx.Param(weights.bias)

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        weight, scale = layer.weight[...], layer.weight_scale_inv[...]
        bias = layer.bias[...] if layer.bias is not None else None

        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)

        out = self._apply_fused(x, weight, scale, bias=bias)
        out = out.reshape(out.shape[:-1] + self.out_features)
        return out

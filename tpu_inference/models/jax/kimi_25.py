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

from abc import abstractmethod
from tpu_inference.logger import init_logger

logger = init_logger(__name__)
from typing import Optional, Iterable

import jax
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule
from vllm.config import VllmConfig

# Reuse the DeepSeek V3 ForCausalLM implementation for the text backbone.
# Kimi K2.5's text model is identical to DeepSeek-V2/V3 in terms of core LLM logic.
from tpu_inference.models.jax.deepseek_v3 import DeepseekV3ForCausalLM

# ==============================================================================
# Base Interfaces and Plugins for Multimodal Support
# ==============================================================================

class KimiK25VisionBackbone(JaxModule):
    """Base interface for Kimi K2.5 Vision Backbone.
    
    In the first iteration, this is a placeholder.
    TODO: Implement MoonViT3dPretrainedModel in JAX.
    """
    @abstractmethod
    def __call__(self, pixel_values: jax.Array, **kwargs) -> jax.Array:
        pass

class KimiK25Projector(JaxModule):
    """Base interface for Kimi K2.5 Multimodal Projector.
    
    In the first iteration, this is a placeholder.
    TODO: Implement KimiK25MultiModalProjector in JAX.
    """
    @abstractmethod
    def __call__(self, vision_embeddings: jax.Array, **kwargs) -> jax.Array:
        pass

# ==============================================================================
# Top-level Model for Conditional Generation
# ==============================================================================

class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM):
    """Kimi K2.5 model for conditional generation.
    
    It inherits from DeepseekV3ForCausalLM to reuse the text capabilities
    (MLA, MoE, etc.) without modification, aligning with vLLM where Kimi
    reuses DeepseekV2ForCausalLM.
    
    Vision capabilities can be added here in future iterations.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh
    ):
        # Kimi K2.5 typically has only the first layer as dense (first_k_dense_replace = 1),
        # while DeepSeek V3 defaults to 3. We override it here if it's set to default 3
        # or not specified, to align with Kimi K2.5 architecture.
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "first_k_dense_replace", 3) == 3:
            logger.info("Overriding first_k_dense_replace to 1 for Kimi K2.5")
            hf_config.first_k_dense_replace = 1
            
        # Initialize the base DeepSeek text model
        super().__init__(vllm_config, rng_key, mesh)
        
        # Plugins for Vision and Projector (Stubs for first iteration)
        self.vision_tower: Optional[KimiK25VisionBackbone] = None
        self.projector: Optional[KimiK25Projector] = None
        
        # TODO: Initialize vision_tower and projector when ready

    def __call__(
        self,
        *args,
        pixel_values: Optional[jax.Array] = None,
        image_grid_thw: Optional[jax.Array] = None,
        **kwargs
    ):
        # 1. Process Multimodal Inputs if present
        if pixel_values is not None and self.vision_tower is not None:
            # TODO: Implement multimodal embedding generation
            # vision_embeddings = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
            # mm_embeddings = self.projector(vision_embeddings)
            # Embed input_ids and merge with mm_embeddings
            # Then call super().__call__ with inputs_embeds
            pass
            
        # For text-only inference or when vision is not used,
        # we delegate directly to the base DeepSeek implementation.
        return super().__call__(*args, **kwargs)

    def load_weights(self, weights: Iterable) -> set[str]:
        def _strip_language_model_prefix(w):
            for key, tensor in w:
                if "mm_projector" in key:
                    logger.warning(f"Skipping mm_projector weight: {key}")
                    continue
                if key.startswith("language_model."):
                    yield key.removeprefix("language_model."), tensor
                else:
                    yield key, tensor
        
        return super().load_weights(_strip_language_model_prefix(weights))

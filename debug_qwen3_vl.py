import os
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
import vllm.platforms
vllm.platforms.current_platform.device_type = "cpu"
import tempfile
import jax
import numpy as np
import torch
from jax.sharding import Mesh
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
from tpu_inference.models.common import model_loader
from tpu_inference.distributed.jax_parallel_state import init_pp_distributed_environment

# Initialize JAX device mesh
devices = np.array(jax.devices()[:1])
devices = devices.reshape((1, 1, 1, -1))
mesh = Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))

# Create engine config for Qwen3-VL-8B-Instruct
engine_args = EngineArgs(model="Qwen/Qwen3-VL-8B-Instruct")
vllm_config = engine_args.create_engine_config()
vllm_config.model_config.dtype = torch.bfloat16
vllm_config.load_config.load_format = "dummy"

# Set current vLLM config
with set_current_vllm_config(vllm_config):
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{temp_file}",
        backend="gloo",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    try:
        os.remove(temp_file)
    except OSError:
        pass

# Initialize JAX PP environment
init_pp_distributed_environment(
    ip="",
    rank=0,
    world_size=1,
    device=jax.devices()[0],
    need_pp=False,
)

# Load the model with vllm implementation (without JITTED_MM_MODULE_KEYS)
os.environ["JITTED_MM_MODULE_KEYS"] = ""  # Explicitly clear
rng = jax.random.PRNGKey(42)

print("Loading get_vllm_model...")
model_interface = model_loader.get_vllm_model(vllm_config, rng, mesh)
print("Model loaded successfully!")

# Let's test calling the embed_multimodal function
embed_fn = model_interface.multimodal_fns.embed_multimodal_fn
params = model_interface.state

# Let's create dummy input args for Qwen3-VL embed_multimodal
vc = vllm_config.model_config.hf_config.vision_config
patch_input_dim = (vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size)
print(f"patch_input_dim: {patch_input_dim}")

dummy_pixel_values = torch.ones((16, patch_input_dim), dtype=torch.bfloat16)
dummy_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

print("Calling embed_multimodal...")
res = embed_fn(
    params,
    pixel_values=dummy_pixel_values,
    image_grid_thw=dummy_grid_thw,
)
print("Result type:", type(res))

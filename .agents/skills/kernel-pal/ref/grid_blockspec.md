# Grids and BlockSpecs Reference

This reference covers how to specify iteration grids and chunk inputs/outputs for Pallas kernels using `grid` and `pl.BlockSpec`.

## Basic Syntax and Example
Here is a basic example demonstrating how `grid` and `BlockSpec` work together in `pl.pallas_call`, highlighting how they are set for inputs (`in_specs`) and outputs (`out_specs`):

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# A kernel that reads an input block and adds a value based on its grid invocation index
def example_kernel(x_ref, o_ref):
  # Get the grid index for axis 0 and 1
  axis_0 = pl.program_id(0)
  axis_1 = pl.program_id(1)
  
  # Read from input block and write to output block
  o_ref[...] = x_ref[...] + (axis_0 * 10 + axis_1)

# The overall array shape is (8, 6)
x_shape = (8, 6)
input_array = jnp.zeros(x_shape, dtype=jnp.int32)

# Launch the kernel over a 2D grid of size (4, 2)
# The block shape is (2, 3). Since 4*2=8 and 2*3=6, this perfectly tiles the arrays.
# `in_specs` takes a list of BlockSpecs (one for each input argument).
# `out_specs` takes a BlockSpec (or a list if there are multiple outputs).
res = pl.pallas_call(
    example_kernel,
    out_shape=jax.ShapeDtypeStruct(x_shape, dtype=jnp.int32),
    grid=(4, 2),
    in_specs=[pl.BlockSpec(block_shape=(2, 3), index_map=lambda i, j: (i, j))],
    out_specs=pl.BlockSpec(block_shape=(2, 3), index_map=lambda i, j: (i, j))
)(input_array)
```

## `grid`
The `grid` argument in `pl.pallas_call` maps to nested loops. A grid of length `d` corresponds to `d` nested loops.
- **Example**: `grid=(n, m)` runs the kernel for `n * m` invocations (programs).
- **Accessing Indices**: Use `pl.program_id(axis)` to get the current iteration index for that axis.
- **Grid Size**: Use `pl.num_programs(axis)` to get the size of the grid along an axis.

## `BlockSpec`
`pl.BlockSpec` maps the loop iteration (invocation index) to the specific block of data to operate on. It is passed via `in_specs` and `out_specs`.

### Blocked Indexing Mode (Default)
When you provide integer dimensions (e.g., `block_shape=(10, 20)`), it uses `pl.Blocked` indexing.
- **`index_map`**: A function taking the invocation indices (e.g., `lambda i, j: (i, j)`) and returning **block indices**.
- Actual start index = `block_idx * block_size`.
- **Squeezing Dimensions**: Use `None` or `pl.Squeezed()` in `block_shape` to treat the dimension size as 1 and squeeze it out of the kernel view.

#### TPU Block Shape Constraints
- Must have rank >= 1.
- For rank >= 2, the last two dimensions must equal the respective array dimension, or be divisible by 8 and 128 respectively.
- For rank 1, the dimension must equal the array dimension, or be a multiple of 1024, or be a power of 2 and at least `128 * (32 / bitwidth(dtype))`.

### Element Indexing Mode
Instead of returning block indices, you can have `index_map` return **element indices** directly.
- **Usage**: Set `block_shape` using `pl.Element(block_size)`.
- **Example**: `block_shape=(pl.Element(2), pl.Element(3))`
- **Virtual Padding**: You can specify padding directly, e.g., `pl.Element(block_size, (low_pad, high_pad))`. (Currently supported only on TPU).

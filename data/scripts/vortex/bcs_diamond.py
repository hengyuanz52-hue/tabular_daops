import os
import sys

import xlb
from xlb.operator.boundary_condition import HalfwayBounceBackBC
import numpy as np
import jax.numpy as jnp
import jax

def generate_bcs(N, grid_shape, boundary_config, velocity_set, *, key=jax.random.PRNGKey(0)):
    x_min = boundary_config['x_min']
    y_min = boundary_config['y_min']
    x_max = boundary_config['x_max']
    y_max = boundary_config['y_max']
    # width: horizontal diagonal length, length: vertical diagonal length
    width = boundary_config['width']
    length = boundary_config['length']
    # Grid
    key, subkey = jax.random.split(key)
    x = jnp.arange(grid_shape[0])
    y = jnp.arange(grid_shape[1])
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Random centers
    x_centers = jax.random.randint(subkey, (N,), minval=x_min, maxval=x_max, dtype=jnp.int32)
    key, subkey = jax.random.split(key)
    y_centers = jax.random.randint(subkey, (N,), minval=y_min, maxval=y_max, dtype=jnp.int32)

    # Half-diagonals (a, b) define |x-x0|/a + |y-y0|/b <= 1 (diamond with 45° edges)
    a = width / 2.0
    b = length / 2.0

    bcs = []
    bc_indices = []
    for i in range(N):
        level = jnp.abs(X - x_centers[i]) / a + jnp.abs(Y - y_centers[i]) / b
        indices = jnp.where(level <= 1.0)

        diamond = [tuple(indices[j]) for j in range(velocity_set.d)]
        bc = HalfwayBounceBackBC(indices=diamond)
        bcs.append(bc)

        x_idx, y_idx = indices
        bc_indices.append(jnp.column_stack((x_idx, y_idx)))

    return bcs, jnp.array(bc_indices)
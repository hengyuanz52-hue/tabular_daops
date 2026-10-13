import os
import sys

import xlb
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
)
import numpy as np
import jax.numpy as jnp
import jax

def generate_bcs(N, grid_shape, boundary_config, velocity_set, *, key=jax.random.PRNGKey(0)):
    x_min = boundary_config['x_min']
    y_min = boundary_config['y_min']
    x_max = boundary_config['x_max']
    y_max = boundary_config['y_max']
    radius = boundary_config['radius']


    # Create a circular boundary condition
    key, subkey = jax.random.split(key)
    x = jnp.arange(grid_shape[0])
    y = jnp.arange(grid_shape[1])
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    x_centers = jax.random.randint(subkey, (N,), minval=x_min, maxval=x_max, dtype=jnp.int32)
    y_centers = jax.random.randint(subkey, (N,), minval=y_min, maxval=y_max, dtype=jnp.int32)
    bcs = []
    bc_indices = []
    for i in range(N):
        indices = jnp.where((X - x_centers[i]) ** 2 + (Y - y_centers[i]) ** 2 < radius**2)
        cyliner = [tuple(indices[j]) for j in range(velocity_set.d)]
        bc_cyliner = HalfwayBounceBackBC(indices=cyliner)
        bcs.append(bc_cyliner)
        x_idx, y_idx = indices
        bc_indices.append(jnp.column_stack((x_idx, y_idx)))

    return bcs, jnp.array(bc_indices)

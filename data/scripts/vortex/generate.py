import os
import sys
import argparse

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.distribute import distribute
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_image
import numpy as np
import jax.numpy as jnp
import jax
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
import bcs_box, bcs_cylinder, bcs_triangle, bcs_diamond
from netCDF4 import Dataset
import h5py
import xarray as xr
import pathlib

def load_config(config_file):
    # open yaml file
    import yaml
    with open(config_file, 'r') as config:
        config = yaml.safe_load(config)
    # Check if the configuration file is valid
    required_keys = ['N', 'nx', 'ny', 'T_0', 'T_max', 'dT', 'reynolds', 'shape', 'boundary_config']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuration file is missing required key: {key}")
    return config

def bc_profile(u_max, grid_shape):
    def bc_profile_jax():
        # Get the grid dimensions for the y direction
        H_y = float(grid_shape[1] - 1)  # Height in y direction

        y = jnp.arange(grid_shape[1])

        # Calculate normalized distance from center
        y_center = y - (H_y / 2.0)
        r_squared = (2.0 * y_center / H_y) ** 2.0

        # Parabolic profile for x velocity, zero for y and z
        u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
        u_y = jnp.zeros_like(u_x)

        return jnp.stack([u_x, u_y])
    return bc_profile_jax

def setup_fixed_boundaries(f_bc_profile, grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    bc_inlet = RegularizedBC("velocity", profile=f_bc_profile, indices=inlet)
    bc_walls = HalfwayBounceBackBC(indices=walls)
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    return [bc_walls, bc_inlet, bc_outlet]


def main(config, output, seed):
    N = config['N']
    nx = config['nx']
    ny = config['ny']
    T_0 = config['T_0']
    T_max = config['T_max']
    dT = config['dT']
    # u_max = config['u_max']
    length = config['length']
    reynolds = config['reynolds']
    boundary = config['boundary_config']
    shape = config['shape']

    inlet_velocity_mean = 0.1
    u_max = inlet_velocity_mean * 1.5
    visc = inlet_velocity_mean * length / reynolds
    omega = 1.0 / (3.0 * visc + 0.5)

    backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=backend)

    grid_shape = (nx, ny)
    grid = grid_factory(grid_shape, compute_backend=backend)
    xlb.init(
        velocity_set=velocity_set,
        default_backend=backend,
        default_precision_policy=precision_policy,
    )
    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    )
    key = jax.random.PRNGKey(seed)

    if shape == "box":
        bcs, bc_indices = bcs_box.generate_bcs(
            N=N,
            grid_shape=grid_shape,
            boundary_config=boundary,
            velocity_set=velocity_set,
            key=key,
        )
    elif shape == "cylinder":
        bcs, bc_indices = bcs_cylinder.generate_bcs(
            N=N,
            grid_shape=grid_shape,
            boundary_config=boundary,
            velocity_set=velocity_set,
            key=key,
        )
    elif shape == "triangle":
        bcs, bc_indices = bcs_triangle.generate_bcs(
            N=N,
            grid_shape=grid_shape,
            boundary_config=boundary,
            velocity_set=velocity_set,
            key=key,
        )
    elif shape == "diamond":
        bcs, bc_indices = bcs_diamond.generate_bcs(
            N=N,
            grid_shape=grid_shape,
            boundary_config=boundary,
            velocity_set=velocity_set,
            key=key,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    trajs = []
    for i, bc in enumerate(bcs):
        bcs_fixed = setup_fixed_boundaries(
            f_bc_profile=bc_profile(u_max, grid_shape),
            grid=grid,
            velocity_set=velocity_set,
        )        
        boundaries = [*bcs_fixed, bc]
        stepper = IncompressibleNavierStokesStepper(
            grid=grid,
            boundary_conditions=boundaries,
            collision_type='BGK',
            streaming_scheme="push"
        )
        stepper = distribute(stepper, grid, velocity_set)
        f0, f1, bc_mask, missing_mask = stepper.prepare_fields()
        traj = jnp.zeros((T_max - T_0, 2, *grid_shape), dtype=jnp.float32)
        for t in range(int(T_max * dT)):
            _, f0 = stepper(f0, f1, bc_mask, missing_mask, omega, 0)
            if t >= int(T_0 * dT) and (t % int(dT) == 0):
                vel = macro(f0)[1]
                traj = traj.at[(t - int(T_0 * dT)) // dT].set(vel)
        trajs.append(traj)

    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    trajs = np.array(trajs)
    dataset = xr.Dataset(
        {
            "u": (('batch', 'time', 'field', 'x', 'y'), trajs),
            "omega": (('batch', ), np.full((len(trajs),), omega, dtype=np.float32)),
            "reynolds": (('batch', ), np.full((len(trajs),), reynolds, dtype=np.float32)),
            "bc_indices": (('batch', 'bc', 'xy'), bc_indices),
        },
        coords={
            'x': x,
            'y': y
        },
        attrs={
            'N': N,
            'nx': nx,
            'ny': ny,
            'dT': dT,
            'T_0': T_0,
            'T_max': T_max,
            'shape': shape,
            'length': length,
            **config['boundary_config'],
        }
    )
    if output is None:
        output = f"LBM_{shape}.h5"
    elif not output.endswith('.h5'):
        output = f"{output}.h5"

    if not pathlib.Path(output).parent.exists():
        pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Problem configuration file")
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    config_file = args.config
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
    config = load_config(config_file)
    main(config, args.output, args.seed)


import optax as opt
import equinox as eqx
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
from torch.utils.data import DataLoader, ConcatDataset

from daops.data import vortex
from daops.modules.gnot import *
from daops.training.train import SupervisedGeometricTrainer, SupervisedTrainer
from daops.utils.callbacks import CheckpointCallback
from daops.utils.serial import save_model, load_model
import optax as opt
import jax.tree_util as jtu
from daops.data.data_utils import *
from pathlib import Path

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Adapt meta-trained model to a target SDF task")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=str, choices=["cylinder", "triangle", "box", "diamond"])
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()

def is_function(path, function_name):
    for p in path:
        if isinstance(p, jtu.DictKey):
            if p.key==function_name:
                return True
    return False

def all_function(tree, function_name):
    leaves = jtu.tree_leaves_with_path(tree)
    bluffs = []
    for p, l in leaves:
        if is_function(p, function_name):
            bluffs.append(l)
    return bluffs 

def sdf_loss_fn(model_diff, model_static, model_state, batch, *, key: PRNGKeyArray):
    print("AUX")
    model = eqx.combine(model_diff, model_static)
    data, masks = batch
    X = data['X']
    functions = data["functions"]
    sdf = data['aux']['sdf']
    mask_query = masks['X']
    masks_functions = masks['functions']
    _, aux = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
    # return jnp.mean(((sdf - aux) * mask_query)**2), model_state
    return jnp.mean(jnp.linalg.norm((sdf - aux) * mask_query, axis=1)**2 / jnp.linalg.norm(sdf * mask_query, axis=1)**2), model_state

def main(args):

    model_path = args.model_path
    seed = args.seed
    shape = args.shape
    steps = args.steps

    meta_model, model_state, config = load_model(model_path, eqx.nn.make_with_state(GeoGNOT), True)
    meta_model = eqx.nn.inference_mode(meta_model, False)
    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    dataset = vortex.VortexSDFDataset(f'./data/{shape}_train.h5', 2, 50, 75, indices=(0, 18))
    loader = DataLoader(dataset, 2, True, collate_fn=jax_padding_collate)

    model_path = os.path.dirname(model_path)
    model_path = os.path.join(model_path, "..", f"{shape.upper()}_ADAPTED_NMSE")
    Path(model_path).mkdir(parents=True, exist_ok=True)

    filter_spec = jtu.tree_map(lambda _: True, meta_model)
    filter_spec = eqx.tree_at(partial(all_function, function_name='velocity'), filter_spec, replace_fn=lambda _: False)
    filter_spec = eqx.tree_at(lambda m: jtu.tree_leaves(m.gnot.projection), filter_spec, replace_fn=lambda _: False)

    params, static = eqx.partition(meta_model, lambda m: eqx.is_array(m), replace=None)
    weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m), params, replace_fn=lambda l: l.ndim > 1)
    weight_decay_mask = eqx.tree_at(partial(all_function, function_name='velocity'), weight_decay_mask, replace_fn=lambda _: False)
    weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m.gnot.projection), weight_decay_mask, replace_fn=lambda _: False)
    
    optimizer = opt.adam(opt.schedules.exponential_decay(1e-4, 5, 0.9, staircase=True, end_value=1e-5))
    # optimizer = opt.sgd(opt.schedules.linear_schedule(1e-2, 1e-4, steps))
    trainer = SupervisedTrainer(meta_model, model_state, optimizer, freeze_specs=filter_spec, loss_fn=sdf_loss_fn)
    shape_model, model_state, opt_state = trainer.fit(loader, steps, key=subkey)
    save_model(os.path.join(model_path, "model_shape.eqx"), config, shape_model, model_state)

if __name__ == "__main__":
    args = parse_args()
    main(args)


import optax as opt
import equinox as eqx
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
from torch.utils.data import DataLoader, ConcatDataset

from daops.data import vortex
from daops.modules.gnot import *
from daops.training.train import ReptileMetaTrainer
from daops.utils.callbacks import CheckpointCallback
from daops.utils.serial import save_model, load_model
import optax as opt
import jax.tree_util as jtu
from daops.data.data_utils import *

def sdf_loss_fn(model_diff, model_static, model_state, batch, *, key=PRNGKeyArray):
    model = eqx.combine(model_diff, model_static)
    data, masks = batch
    X = data['X']
    functions = data["functions"]
    sdf = data['aux']['sdf']
    mask_query = masks['X']
    masks_functions = masks['functions']
    _, aux = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
    return jnp.mean(((sdf - aux) * mask_query)**2), model_state
    # return jnp.mean(jnp.linalg.norm((sdf - aux) * mask_query, axis=1)**2 / jnp.linalg.norm(sdf * mask_query, axis=1)**2), model_state

datasets_sdf = [
    vortex.VortexSDFDataset('./data/triangle_train.h5', step_size=2, T0=50, T1=100, indices=(0, 18)),
    vortex.VortexSDFDataset('./data/cylinder_train.h5', step_size=2, T0=50, T1=100, indices=(0, 18)),
    vortex.VortexSDFDataset('./data/box_train.h5', step_size=2, T0=50, T1=100, indices=(0, 18))
]
loader_sdf = [
    DataLoader(d, 2, shuffle=True, collate_fn=jax_padding_collate) for d in datasets_sdf
]
dataset_cylinder = vortex.VortexWindowDataset('./data/cylinder_train.h5', 6, step_size=2, T0=50, T1=100, 
                                              indices=(0, 18), paramed=False)

min_x, max_x = dataset_cylinder.compute_min_max_coords()
mean_y, std_y = dataset_cylinder.compute_mean_std_fields()
mean_y = jnp.tile(mean_y, [5, 1]).reshape([-1])
std_y = jnp.tile(std_y, [5, 1]).reshape([-1])
mean_sdf, std_sdf = datasets_sdf[1].compute_mean_std_sdf()

geognot_config = GeoGNOTConfig(
    input_dim=2,
    spatial_dim=2,
    output_dim=10,
    function_specs=dataset_cylinder.get_function_specs(),
    width_lifting=192,
    depth_lifting=3,
    activation_lifting='gelu',
    latent_dim=96,
    n_blocks=3,
    n_heads=8,
    dropout_rate=0.1,
    n_experts=4,
    width_moe=96,
    activation_moe='gelu',
    mean_out=mean_y,
    std_out=std_y,
    min_coords=min_x,
    max_coords=max_x,
    mean_geo=mean_sdf,
    std_geo=std_sdf,
    geo_dim=1,
    fnames_return_geo=['bluff_body']
)

key = jr.PRNGKey(0)
key, subkey = jr.split(key)
model, model_state = eqx.nn.make_with_state(GeoGNOT)(**geognot_config, key=subkey)
EPOCHS = 10000
INNER_STEPS = 5

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


filter_spec = jtu.tree_map(lambda _: True, model)
filter_spec = eqx.tree_at(partial(all_function, function_name='velocity'), filter_spec, replace_fn=lambda _: False)
filter_spec = eqx.tree_at(lambda m: jtu.tree_leaves(m.gnot.projection), filter_spec, replace_fn=lambda _: False)

params, static = eqx.partition(model, lambda m: eqx.is_array(m), replace=None)
weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m), params, replace_fn=lambda l: l.ndim > 1)
weight_decay_mask = eqx.tree_at(partial(all_function, function_name='velocity'), weight_decay_mask, replace_fn=lambda _: False)
weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m.gnot.projection), weight_decay_mask, replace_fn=lambda _: False)


# inner_optimizer = opt.chain(
#     # opt.add_decayed_weights(0.0005, weight_decay_mask),
#     opt.scale(-1e-4),
# )

# meta_optimizer = opt.adamw(opt.schedules.join_schedules(
#     [opt.schedules.linear_schedule(0, 1e-2, 200),
#      opt.schedules.exponential_decay(1e-2, 25, 0.98, staircase=True, end_value=1e-7)],
#     [200]
# ), weight_decay=0.0005, mask=(weight_decay_mask, False))

# inner_optimizer = opt.chain(
#     # opt.add_decayed_weights(0.0005, weight_decay_mask),
#     opt.scale(-1e-4),
# )

inner_optimizer = opt.sgd(1e-5)
# inner_optimizer = opt.adamw(1e-4, weight_decay=0.0005)

meta_optimizer = opt.adamw(opt.schedules.join_schedules(
    [opt.schedules.linear_schedule(0, 1e-3, 200),
     opt.schedules.exponential_decay(1e-3, 25, 0.98, staircase=True, end_value=1e-7)],
    [200]
), weight_decay=0.0005, mask=(weight_decay_mask, False))

# SAVE_PATH = "./experiments/GeoGNOT/META_REPTILE/NMSE_META_SHAPE_RANDOM_TASK_FixedDecay_ADAMw+JoinLinearExp(SGD1e-2)_K=5/"
SAVE_PATH = "./experiments/GeoGNOT_NewGate/META_REPTILE/META_SHAPE_RANDOM_TASK_FixedDecay_ADAMw+JoinLinearExp(SGD1e-5)_K=5/"

meta_trainer = ReptileMetaTrainer(model, model_state, inner_optimizer,
                                  inner_steps=INNER_STEPS,
                                  meta_optimizer=meta_optimizer,
                                  num_tasks=3,
                                  freeze_specs=filter_spec,
                                  loss_fn=sdf_loss_fn,
                                  callbacks=[CheckpointCallback(SAVE_PATH + "checkpoints", 
                                                                geognot_config, 
                                                                True, save_every=1000)])
key, subkey = jr.split(key)
model, model_state, opt_state = meta_trainer.fit(loader_sdf, EPOCHS, key=subkey)
save_model(SAVE_PATH+"model.eqx", geognot_config, model, model_state)


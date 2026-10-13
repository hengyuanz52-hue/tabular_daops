import optax as opt
import equinox as eqx
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
from torch.utils.data import DataLoader, ConcatDataset

from daops.data import vortex
from daops.modules.gnot import *
from daops.training.train import ReptileMetaGeoSimTrainer
from daops.utils.callbacks import CheckpointCallback
from daops.utils.serial import save_model, load_model
import optax as opt
import jax.tree_util as jtu
from daops.data.data_utils import *
from pathlib import Path

def is_bluff(path):
    for p in path:
        if isinstance(p, jtu.DictKey):
            if p.key=='bluff_body':
                return True
    return False

def all_bluffs(tree):
    leaves = jtu.tree_leaves_with_path(tree)
    bluffs = []
    for p, l in leaves:
        if is_bluff(p):
            bluffs.append(l)
    return bluffs

PATH = "experiments/GeoGNOT_NewGate/DOUBLE_META/3GEO_1SIM(tri)_FULLGRID_T=75_ADAMw(SGD1e-4)_K=6/META_MODEL/"
EPOCHS = 30000
INNER_STEPS = 6

datasets_sdf = [
    vortex.VortexSDFDataset('./data/triangle_train.h5', step_size=2, T0=50, T1=75, indices=(0, 18)),
    vortex.VortexSDFDataset('./data/cylinder_train.h5', step_size=2, T0=50, T1=75, indices=(0, 18)),
    vortex.VortexSDFDataset('./data/box_train.h5', step_size=2, T0=50, T1=75, indices=(0, 18))
]
loader_sdf = [
    DataLoader(d, 2, shuffle=True, collate_fn=jax_padding_collate) for d in datasets_sdf
]
# dataset_cylinder = vortex.VortexWindowDataset('./data/cylinder_train.h5', 6, step_size=2, T0=50, T1=75, 
#                                               indices=(0, 18), paramed=False)
# datasets_sim = [
#     vortex.VortexRandomSubsampledDataset('./data/triangle_train.h5', 6, step_size=2, T0=50, T1=75, indices=(0, 18), 
#                                          paramed=False, subsample_rate=0.1, random_seed=1, mask_label=False),
#     vortex.VortexRandomSubsampledDataset('./data/cylinder_train.h5', 6, step_size=2, T0=50, T1=75, indices=(0, 18), 
#                                          paramed=False, subsample_rate=0.1, random_seed=0, mask_label=False),
#     vortex.VortexRandomSubsampledDataset('./data/box_train.h5', 6, step_size=2, T0=50, T1=75, indices=(0, 18), 
#                                          paramed=False, subsample_rate=0.1, random_seed=1, mask_label=False)
# ]

datasets_sim = [
    vortex.VortexWindowDataset('./data/triangle_train.h5', 6, step_size=2, T0=50, T1=75, indices=(0, 18), 
                                         paramed=False),
    # vortex.VortexWindowDataset('./data/cylinder_train.h5', 6, step_size=2, T0=50, T1=75, indices=(0, 18), 
    #                                      paramed=False),
    # vortex.VortexWindowDataset('./data/box_train.h5', 6, step_size=2, T0=50, T1=75, indices=(0, 18), 
    #                                      paramed=False)
]

loader_sim = [
    DataLoader(d, 2, shuffle=True, collate_fn=jax_padding_collate) for d in datasets_sim
]

min_x, max_x = datasets_sim[0].compute_min_max_coords()
mean_y, std_y = datasets_sim[0].compute_mean_std_fields()
mean_y = jnp.tile(mean_y, [5, 1]).reshape([-1])
std_y = jnp.tile(std_y, [5, 1]).reshape([-1])
mean_sdf, std_sdf = datasets_sdf[1].compute_mean_std_sdf()

geognot_config = GeoGNOTConfig(
    input_dim=2,
    spatial_dim=2,
    output_dim=10,
    function_specs=datasets_sim[0].get_function_specs(),
    width_lifting=192,
    depth_lifting=3,
    activation_lifting='gelu',
    latent_dim=96,
    n_blocks=3,
    n_heads=8,
    dropout_rate=0.1,
    n_experts=5,
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

filter_spec_geo = jtu.tree_map(lambda _: True, model)
filter_spec_geo = eqx.tree_at(partial(all_function, function_name='velocity'), filter_spec_geo, replace_fn=lambda _: False)
filter_spec_geo = eqx.tree_at(lambda m: jtu.tree_leaves(m.gnot.projection), filter_spec_geo, replace_fn=lambda _: False)

filter_spec_sim = jtu.tree_map(lambda _: True, model)
# filter_spec_sim = eqx.tree_at(partial(all_function, function_name='bluff_body'), filter_spec_sim, replace_fn=lambda _: False)
filter_spec_sim = eqx.tree_at(lambda m: jtu.tree_leaves(m.geo_reconstructor), filter_spec_sim, replace_fn=lambda _: False)

params, static = eqx.partition(model, lambda m: eqx.is_array(m), replace=None)
weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m), params, replace_fn=lambda l: l.ndim > 1)
# weight_decay_mask = eqx.tree_at(partial(all_function, function_name='velocity'), weight_decay_mask, replace_fn=lambda _: False)
# weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m.gnot.projection), weight_decay_mask, replace_fn=lambda _: False)

inner_optimizer = opt.chain(opt.scale(-1e-4))
meta_optimizer = opt.adamw(opt.schedules.join_schedules(
    [opt.schedules.linear_schedule(0, 1e-3, 200),
     opt.schedules.exponential_decay(1e-3, 25, 0.99, staircase=True, end_value=1e-6)],
    [200]
), weight_decay=0.0005, mask=(weight_decay_mask, False))

meta_trainer = ReptileMetaGeoSimTrainer(model, model_state, inner_optimizer, meta_optimizer,
                                        INNER_STEPS, len(loader_sdf), len(loader_sim), 
                                        callbacks=[CheckpointCallback(PATH + "checkpoints", 
                                                                    geognot_config, 
                                                                    True, save_every=2000)],
                                                                    freeze_specs_geo=filter_spec_geo,
                                                                    freeze_specs_sim=filter_spec_sim,
                                                                    loss='nmse')

meta_model, meta_model_state, _ = meta_trainer.fit(loader_sdf, loader_sim, EPOCHS, key=jr.PRNGKey(10))
save_model(PATH + "model.eqx", geognot_config, meta_model, meta_model_state)


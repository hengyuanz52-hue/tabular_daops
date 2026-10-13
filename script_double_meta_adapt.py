# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

# %%
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from daops.data.vortex import *
from daops.utils.serial import *
from daops.modules.gnot import GNOTConfig
from daops.data.data_utils import *

# %%
# dataset_cyl = VortexWindowDataset("./data/cylinder_train.h5", 6, 2, 50, 100, indices=(0, 18))
# dataset_tri = VortexWindowDataset("./data/triangle_train.h5", 6, 2, 50, 100, indices=(0, 18))
# dataset_box = VortexWindowDataset("./data/box_train.h5", 6, 2, 50, 100, indices=(0, 20))

# sdf_dataset_cyl = VortexSDFDataset("./data/cylinder_train.h5", 2, 50, 100, indices=(0, 18))
# sdf_dataset_tri = VortexSDFDataset("./data/triangle_train.h5", 2, 50, 100, indices=(0, 18))
# sdf_dataset_box = VortexSDFDataset("./data/box_train.h5", 2, 50, 100, indices=(0, 18))

# %%
from daops.training.train import *
from daops.utils.callbacks import *
import optax as opt

# %%
def is_bluff(path):
    for p in path:
        if isinstance(p, jax.tree_util.DictKey):
            if p.key=='bluff_body':
                return True
    return False

def all_bluffs(tree):
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    bluffs = []
    for p, l in leaves:
        if is_bluff(p):
            bluffs.append(l)
    return bluffs

# %%
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

# %%
def sdf_loss_fn(model_diff, model_static, model_state, batch, *, key=PRNGKeyArray):
    print("AUX")
    model = eqx.combine(model_diff, model_static)
    data, masks = batch
    X = data['X']
    functions = data["functions"]
    sdf = data['aux']['sdf']
    mask_query = masks['X']
    masks_functions = masks['functions']
    _, aux, _ = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
    return jnp.mean(((sdf - aux) * mask_query)**2), model_state

def loss_val_fn_sim(model, model_state: PyTree, batch: DataDict, *, key=PRNGKeyArray):
    data, masks = batch
    X = data['X']
    functions = data["functions"]
    u = data['u']
    mask_query = masks['X']
    masks_functions = masks['functions']
    label_mask = data.get('label_mask', None)
    preds, _ = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
    if label_mask is not None:
        return jnp.mean(jnp.linalg.norm((preds - u) * mask_query * label_mask, axis=1)**2 / jnp.linalg.norm(u * mask_query * label_mask, axis=1)**2), model_state
    else:
        return jnp.mean(jnp.linalg.norm((preds - u) * mask_query, axis=1)**2 / jnp.linalg.norm(u * mask_query, axis=1)**2), model_state



# %%
model_meta_1, state_meta_1, config = load_model("./experiments/GeoGNOT_NewGate/DOUBLE_META/3GEO_1SIM(tri)_FULLGRID_T=75_ADAMw(SGD1e-4)_K=6/META_MODEL/model.eqx",
                          eqx.nn.make_with_state(GeoGNOT), True)
# model_meta_2, state_meta_2 = load_model("./experiments/GeoGNOT/META_REPTILE/META_SHAPE_RANDOM_TASK_ADAM+COS:0,1e-2,1e-7(SGD1e-5)_K=5/checkpoints/2025-10-16_01-44-33/model_epoch_1900.eqx",
#                           eqx.nn.make_with_state(GeoGNOT))

# %%
params, static = eqx.partition(model_meta_1, lambda m: eqx.is_array(m), replace=None)
weight_decay_mask = eqx.tree_at(lambda m: jtu.tree_leaves(m), params, replace_fn=lambda l: l.ndim > 1)

# %%
EPOCHS = 300
GAMMA = 0.5

task_sim_dataset = VortexRandomSubsampledDataset("./data/diamond_train.h5", 6, 2, 50, 75, indices=(0, 10),
                                                        paramed=False, mask_label=True, subsample_rate=0.005)
sdf_dataset_cyl = VortexSDFDataset("./data/diamond_train.h5", 2, 50, 75, indices=(0, 10))

task_sim_dataset_validation = VortexWindowDataset("./data/diamond_train.h5", 6, 2, 50, 75, indices=(18, 20),
                                                        paramed=False)

dataloader_sim = DataLoader(task_sim_dataset, 2, True, collate_fn=jax_padding_collate)
dataloader_sim_validation = DataLoader(task_sim_dataset_validation, 1, True, collate_fn=jax_padding_collate)
dataloader_sdf = DataLoader(sdf_dataset_cyl, 2, True, collate_fn=jax_padding_collate)


optimizer = opt.adamw(opt.schedules.exponential_decay(1e-4, len(dataloader_sim) * 5, decay_rate=0.98, end_value=1e-5, staircase=True), 
                      weight_decay=0.0005, mask=weight_decay_mask)
# optimizer = opt.adam(1e-5)

PATH_SIM = "experiments/GeoGNOT_NewGate/DOUBLE_META/3GEO_1SIM(tri)_FULLGRID_T=75_ADAMw(SGD1e-4)_K=6/ADAMW1E-4_N=10_SUB005_DIAMOND_DOUBLE_TRAINED_GAMMA=0.5_NMSE/"

geo_trainer = SupervisedGeometricTrainer(model_meta_1, state_meta_1, optimizer,
                                        callbacks=[ValidationCallback(dataloader_sim_validation, loss_val_fn_sim, 5),
                                                   CheckpointCallback(PATH_SIM + "checkpoint", config, True, 25)],
                                         loss='nmse')
sim_model, model_state, opt_state = geo_trainer.fit(dataloader_sim, dataloader_sdf, EPOCHS, GAMMA, key=jr.PRNGKey(1))

save_model(PATH_SIM + "model.eqx", config, sim_model, model_state)


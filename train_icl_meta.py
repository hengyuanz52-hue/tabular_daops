"""
Episodic Operator-ICL meta-training (TabICL-inspired).

Trains GeoGNOT so that sparse labeled points in ``functions['support']`` act as
in-context examples. Adaptation to a new shape is then a forward pass (see
``adapt_icl.py``), not hundreds of gradient steps.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax as opt
from torch.utils.data import DataLoader

from daops.data.data_utils import jax_padding_collate
from daops.data.vortex import VortexICLDataset, VortexSDFDataset
from daops.modules.gnot import GeoGNOT, GeoGNOTConfig
from daops.training.train import EpisodicICLTrainer
from daops.utils.callbacks import CheckpointCallback
from daops.utils.serial import save_model

# ---------------------------------------------------------------------------
# Speed-oriented defaults (full grid is ~180x81=14580 pts; self-attn dominates)
# ---------------------------------------------------------------------------
PATH = "experiments/GeoGNOT_NewGate/ICL_META/3SHAPE_K64_Q2k_T=75/"
EPOCHS = 10000          # was 10000; with faster steps this is plenty for a first run
BATCH_SIZE = 2
WINDOW = 6
STEP = 2
T0, T1 = 50, 75
K_SUPPORT = 64         # ICL context size (fixed for JIT)
N_QUERY = 2048         # train on random query subset (not full grid)
N_VELOCITY = 2048      # subsample velocity IC keys
STEPS_PER_EPOCH = 20   # ~Reptile outer-step feel; 5000*20 = 100k updates
LOG_EVERY = 10

SHAPE_PATHS = [
    "./data/triangle_train.h5",
    "./data/cylinder_train.h5",
    "./data/box_train.h5",
]

Path(PATH).mkdir(parents=True, exist_ok=True)

datasets = [
    VortexICLDataset(
        p,
        WINDOW,
        STEP,
        T0,
        T1,
        indices=(0, 18),
        paramed=False,
        k_support=K_SUPPORT,
        n_query=N_QUERY,
        n_velocity=N_VELOCITY,
        resample_support=True,
        query_on_support=False,
        random_seed=i,
    )
    for i, p in enumerate(SHAPE_PATHS)
]

loaders = [
    DataLoader(d, BATCH_SIZE, shuffle=True, collate_fn=jax_padding_collate) for d in datasets
]

min_x, max_x = datasets[0].compute_min_max_coords()
mean_y, std_y = datasets[0].compute_mean_std_fields()
mean_out = jnp.tile(mean_y, [WINDOW - 1, 1]).reshape([-1])
std_out = jnp.tile(std_y, [WINDOW - 1, 1]).reshape([-1])

sdf_ref = VortexSDFDataset("./data/cylinder_train.h5", STEP, T0, T1, indices=(0, 18))
mean_sdf, std_sdf = sdf_ref.compute_mean_std_sdf()

geognot_config = GeoGNOTConfig(
    input_dim=2,
    spatial_dim=2,
    output_dim=2 * (WINDOW - 1),
    function_specs=datasets[0].get_function_specs(),
    width_lifting=192,
    depth_lifting=3,
    activation_lifting="gelu",
    latent_dim=96,
    n_blocks=3,
    n_heads=8,
    dropout_rate=0.1,
    n_experts=5,
    width_moe=96,
    activation_moe="gelu",
    mean_out=mean_out,
    std_out=std_out,
    min_coords=min_x,
    max_coords=max_x,
    mean_geo=mean_sdf,
    std_geo=std_sdf,
    geo_dim=1,
    fnames_return_geo=["bluff_body"],
)

key = jr.PRNGKey(0)
key, subkey = jr.split(key)
model, model_state = eqx.nn.make_with_state(GeoGNOT)(**geognot_config, key=subkey)

params, _ = eqx.partition(model, lambda m: eqx.is_array(m), replace=None)
weight_decay_mask = eqx.tree_at(
    lambda m: jtu.tree_leaves(m), params, replace_fn=lambda l: l.ndim > 1
)

optimizer = opt.adamw(
    opt.schedules.join_schedules(
        [
            opt.schedules.linear_schedule(0, 1e-3, 200),
            opt.schedules.exponential_decay(
                1e-3, 25, 0.99, staircase=True, end_value=1e-6
            ),
        ],
        [200],
    ),
    weight_decay=0.0005,
    mask=weight_decay_mask,
)

trainer = EpisodicICLTrainer(
    model,
    model_state,
    optimizer,
    num_tasks=len(loaders),
    callbacks=[
        CheckpointCallback(PATH + "checkpoints", geognot_config, True, save_every=500)
    ],
    freeze_specs=None,
    loss="nmse",
    steps_per_epoch=STEPS_PER_EPOCH,
    log_every=LOG_EVERY,
)

n_full = datasets[0].X_star.shape[0]
print(
    f"Operator-ICL meta-train | shapes={len(SHAPE_PATHS)} | "
    f"K={K_SUPPORT} | n_query={N_QUERY}/{n_full} | n_vel={N_VELOCITY} | "
    f"steps/epoch={STEPS_PER_EPOCH} | epochs={EPOCHS}"
)
print("function_specs keys:", list(geognot_config["function_specs"].keys()))
print(
    f"Total updates ≈ {EPOCHS * STEPS_PER_EPOCH}. "
    f"First step JIT-compiles; expect ~5–20× faster steps vs full grid."
)

icl_model, icl_state, _ = trainer.fit(loaders, EPOCHS, key=jr.PRNGKey(10))
save_model(PATH + "model.eqx", geognot_config, icl_model, icl_state)
print(f"Saved -> {PATH}model.eqx")

"""
utilities for saving and loading models.
"""

import json
import os
import equinox as eqx
import jax
import jax.random as jr
import jax.nn as jnn
from ..modules.gnot import *
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import pandas as pd

def save_model(filename, hyperparams, model, state):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        hyperparams = jax.tree.map(lambda x: x.tolist() if isinstance(x, np.ndarray) or isinstance(x, jax.Array)else x, hyperparams)
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, (model, state))

def save_opt_state(filename, opt_state):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, opt_state)

def _list_to_array_if_numeric(x):
    """Convert JSON lists of numbers back to jnp arrays; leave string lists alone.

    ``fnames_return_geo`` is saved as ``[\"bluff_body\"]`` and must stay a Python list.
    """
    if not isinstance(x, list) or len(x) == 0:
        return x
    # Nested lists (e.g. nested numeric structures) — try array, else leave as-is.
    if isinstance(x[0], list):
        try:
            return jnp.array(x)
        except (TypeError, ValueError):
            return x
    if isinstance(x[0], (int, float, bool)):
        return jnp.array(x)
    return x


def load_model(filename, make, return_config=False):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        for key in list(hyperparams.keys()):
            if key.endswith("_numpy"):
                hyperparams[key[:-6]] = jax.numpy.array(hyperparams[key], )
                del hyperparams[key]
        hyperparams = jax.tree.map(
            _list_to_array_if_numeric,
            hyperparams,
            is_leaf=lambda x: isinstance(x, list),
        )
        (model, state) = make(key=jr.PRNGKey(0), **hyperparams)
        model, state = eqx.tree_deserialise_leaves(f, (model, state))
        model = eqx.nn.inference_mode(model, True)
        if return_config:
            return model, state, hyperparams
        return model, state

def save_history_csv(filename, history):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)


def load_history_csv(filename):
    df = pd.read_csv(filename)
    out = df.to_dict(orient='list')
    return out

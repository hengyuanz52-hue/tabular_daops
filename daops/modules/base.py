"""
Base layers and blocks for neural operators.
"""

import jax
import jax.numpy as jnp
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Float
from typing import Optional, Union, Sequence, Callable, Any
from enum import Enum

activation_dict = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "elu": jax.nn.elu,
    "identity": jax.nn.identity,
}

def normalized_linear_dot_product(
        query: Float[ArrayLike, "len_query d"],
        key: Float[ArrayLike, "len_key d"],
        value: Float[ArrayLike, "len_key d"]
) -> Float[ArrayLike, "len_query d"]:
    """
    Dot-product attention mechanism with GNOT normalization. Assumes that the key and value tensors are \
        already normalized. Based on GNOT paper.

    Args:
        query: Query tensor of shape (len_query, d).
        key: Key tensor of shape (len_key, d).
        value: Value tensor of shape (len_key, d).
    Returns:
        Attention output tensor of shape (len_query, d).
    """

    alpha = jnp.sum(key, axis=0, keepdims=True) # (1, d)
    alpha = alpha * query # (len_query, d)
    alpha = jnp.sum(alpha, axis=1, keepdims=True) # (len_query, 1)
    return query @ key.T @ value / alpha # (len_query, len_key) @ (len_key, d) -> (len_query, d)

class Normalization(eqx.Module):
    """
    Normalization layer for input spatial coordinates of INRs.
    """

    mean: ArrayLike
    std: ArrayLike

    def __init__(self, mean: ArrayLike, std: ArrayLike):
        if jnp.ndim(mean) == 0:
            mean = jnp.array([mean])
        if jnp.ndim(std) == 0:
            std = jnp.array([std])
        self.mean = mean
        self.std = std

    def __call__(self, x: ArrayLike, *, key=None) -> ArrayLike:
        return (x - jax.lax.stop_gradient(self.mean)) / (jax.lax.stop_gradient(self.std) + 1e-8)

class Denormalization(eqx.Module):
    """
    Denormalization layer for the output vector fields of INRs. The outputs are multiplied by the standard deviation and added to the mean.
    """

    mean: ArrayLike
    std: ArrayLike

    def __init__(self, mean: ArrayLike, std: ArrayLike):
        if jnp.ndim(mean) == 0:
            mean = jnp.array([mean])
        if jnp.ndim(std) == 0:
            std = jnp.array([std])
        self.mean = mean
        self.std = std

    def __call__(self, x: ArrayLike, *, key=None) -> ArrayLike:
        return x * jax.lax.stop_gradient(self.std) + jax.lax.stop_gradient(self.mean)
    
class Standardization(eqx.Module):
    """
    Standardization layer for the input spatial coordinates of INRs.
    """

    min: ArrayLike
    max: ArrayLike

    def __init__(self, min: ArrayLike, max: ArrayLike):
        if jnp.ndim(min) == 0:
            min = jnp.array([min])
        if jnp.ndim(max) == 0:
            max = jnp.array([max])
        self.min = min
        self.max = max

    def __call__(self, x: ArrayLike, *, key=None) -> ArrayLike:
        return 2 * (x - jax.lax.stop_gradient(self.min)) / (jax.lax.stop_gradient(self.max) - jax.lax.stop_gradient(self.min)) - 1
    
class MixedFunctionTransform(eqx.Module):

    coord_stder: Standardization
    function_normer: Normalization
    spatial_dim: int = eqx.field(static=True)

    def __init__(self, min_x, max_x, mean_f, std_f):
        self.coord_stder = Standardization(min_x, max_x)
        self.function_normer = Normalization(mean_f, std_f)
        self.spatial_dim = self.coord_stder.min.shape[0]

    def __call__(self, x: ArrayLike):
        x = x.at[:self.spatial_dim].set(self.coord_stder(x[:self.spatial_dim]))
        x = x.at[self.spatial_dim:].set(self.function_normer(x[self.spatial_dim:]))
        return x


# """
# GNOT Equinox implementation based on GNOT paper: https://arxiv.org/abs/2302.14376
# """

# import jax
# import jax.numpy as jnp
# import jax.random as jr
# import equinox as eqx
# from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Float, Bool
# from typing import Optional, Union, Sequence, Callable, Any, Dict, Tuple, TypedDict
# from enum import Enum
# from functools import partial
# from dataclasses import dataclass

# from .base import *
# from ..data.data_utils import FunctionSpec

# class GNOTConfig(TypedDict):
#     # Data specifics
#     input_dim: int
#     spatial_dim: int
#     output_dim: int
#     function_specs: Dict[str, FunctionSpec]
#     # normalization stats
#     min_coords: Float[ArrayLike, "D"]
#     max_coords: Float[ArrayLike, "D"]
#     mean_out: Float[ArrayLike, "C"]
#     std_out: Float[ArrayLike, "C"]

#     # Lifting and projection
#     width_lifting: int
#     depth_lifting: int
#     activation_lifting: str

#     # Attention specifics
#     latent_dim: int
#     n_blocks: int
#     n_heads: int
#     dropout_rate: float

#     # Gating specifics
#     n_experts: int
#     width_moe: int
#     activation_moe: str
    
# class GNOTCrossAttentionBlock(eqx.Module):
#     """
#     GNOT multi-head attention block with normalized linear attention, dropout, and reisudal connections. 

#     """

#     n_heads: int = eqx.field(static=True)
#     latent_dim: int = eqx.field(static=True)
#     input_functions: Sequence[str] = eqx.field(static=True)
#     dropout_rate: float = eqx.field(static=True)
#     f_query: eqx.Module
#     fs_key: Dict[str, eqx.Module]
#     fs_value: Dict[str, eqx.Module]
#     f_proj: eqx.Module
    
#     def __init__(self, n_heads: int, latent_dim: int, input_functions: Sequence[str], dropout_rate: float = 0.0, *, \
#                  key: PRNGKeyArray):
        
#         assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"

#         self.n_heads = n_heads
#         self.latent_dim = latent_dim
#         self.input_functions = input_functions 
#         self.dropout_rate = dropout_rate
        
#         key, subkey = jr.split(key)
#         self.f_query = eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkey)
        
#         key, *subkeys = jr.split(key, len(input_functions) + 1)
#         self.fs_key = {
#             input_functions[i]: eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkeys[i]) 
#             for i in range(len(input_functions))
#         }
        
#         key, *subkeys = jr.split(key, len(input_functions) + 1)
#         self.fs_value = {
#             input_functions[i]: eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkeys[i]) 
#             for i in range(len(input_functions))
#         }

#         key, subkey = jr.split(key)
#         self.f_proj = eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkey)

#     def __call__(self, x, y, mask_query, masks_functions):
#         """
#         Forward pass of the GNOT attention block. Pass the same 
    
#         Args:
#             x: Input query tensor of shape (len_query, d).
#             y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).
#         Returns:
#             Attention output tensor of shape (len_query, d).
#         """
        
#         q = eqx.filter_vmap(self.f_query)(x) # (len_query, d)
#         if not mask_query is None:
#             q = jnp.where(mask_query, q, -jnp.inf)
#         q = jnp.reshape(q, (q.shape[0], self.n_heads, self.latent_dim // self.n_heads)) # (len_query, n_heads, d // n_heads)
#         q = jnp.transpose(q, (1, 0, 2)) # (n_heads, len_query, d // n_heads)
#         q = jax.nn.softmax(q, axis=-1)
#         q = jnp.nan_to_num(q)
#         out = q
#         for fname in y:
#             k = eqx.filter_vmap(self.fs_key[fname])(y[fname]) # (len_key, d)
#             if (not masks_functions is None) and (not masks_functions[fname] is None):
#                 k = jnp.where(masks_functions[fname], k, -jnp.inf)
#             k = jnp.reshape(k, (k.shape[0], self.n_heads, self.latent_dim // self.n_heads)) # (len_key, n_heads, d // n_heads)
#             k = jnp.transpose(k, (1, 0, 2)) # (n_heads, len_key, d // n_heads)
#             k = jax.nn.softmax(k, axis=-1)
#             k = jnp.nan_to_num(k)

#             v = eqx.filter_vmap(self.fs_value[fname])(y[fname]) # (len_key, d)
#             if (not masks_functions is None) and (not masks_functions[fname] is None):
#                 v = v * masks_functions[fname]
#             v = jnp.reshape(v, (v.shape[0], self.n_heads, self.latent_dim // self.n_heads)) # (len_key, n_heads, d // n_heads)
#             v = jnp.transpose(v, (1, 0, 2))

#             out = out + eqx.filter_vmap(normalized_linear_dot_product)(q, k, v) # (n_heads, len_query, d // n_heads)

#         out = jnp.transpose(out, (1, 0, 2)) # (len_query, n_heads, d // n_heads)
#         out = jnp.reshape(out, (out.shape[0], self.latent_dim))
#         if not mask_query is None:
#             return eqx.filter_vmap(self.f_proj)(out) * mask_query # (len_query, d)
#         else:
#             return eqx.filter_vmap(self.f_proj)(out)
    
# class GNOTSelfAttention(GNOTCrossAttentionBlock):
#     """
#     GNOT self-attention block. Uses the same input for query, key, and value.
#     """

#     def __init__(self, n_heads: int, latent_dim: int, dropout_rate: float = 0.0, *, key: PRNGKeyArray):
#         """
#         Initialize the GNOT self-attention block.
        
#         Args:
#             n_heads: Number of attention heads.
#             latent_dim: Dimensionality of the input and output tensors.
#             dropout_rate: Dropout rate for the attention mechanism.
#             key: Random key for initialization.
#         """
#         super().__init__(n_heads, latent_dim, input_functions=["latent"], dropout_rate=dropout_rate, key=key)

#     def __call__(self, x, mask):
#         """
#         Forward pass of the GNOT self-attention block.
        
#         Args:
#             x: Input tensor of shape (len_query, d).
#         Returns:
#             Attention output tensor of shape (len_query, d).
#         """
#         return super().__call__(x, {'latent': x}, mask, {'latent': mask})
    
# class GNOTGatedAttentionBlock(eqx.Module):
#     """
#     GNOT gated attention block. Applies a gating mechanism to the attention output.
#     """

#     spatial_dim: int = eqx.field(static=True)
#     input_functions: Sequence[str] = eqx.field(static=True)
#     n_heads: int = eqx.field(static=True)
#     latent_dim: int = eqx.field(static=True)
#     n_experts: int = eqx.field(static=True) 
#     width_size: int = eqx.field(static=True)
#     dropout_rate: float = eqx.field(static=True)
#     cross_attn: GNOTCrossAttentionBlock
#     self_attn: GNOTSelfAttention
#     f_gate: eqx.Module
#     f_experts_cross: Sequence[eqx.Module]
#     f_experts_self: Sequence[eqx.Module]
#     lns_ys: Dict[str, eqx.Module]
#     lns_xs: Sequence[eqx.Module]
#     dropout_cross: eqx.Module
#     dropout_self: eqx.Module

#     def __init__(self, spatial_dim: int, input_functions: Sequence[str], n_heads: int, latent_dim: int, n_experts: int, \
#                  width_size: int, dropout_rate: float = 0.0, activation: Union[str, Callable] = "relu", *, key: PRNGKeyArray):
#         """
#         Multi-head attention block with multi-head cross and self-attention, dropout, layer normalisation, 
#         and gating mechanism.

#         Args:
#             spatial_dim: Dimensionality of the spatial input.
#             input_functions: List of names of input functions.
#             n_heads: Number of attention heads.
#             latent_dim: Dimensionality of the input and output tensors.
#             n_experts: Number of experts in the gating mechanism.
#             width_size: Width of the MLP layers in the gating mechanism.
#             dropout_rate: Dropout rate for the attention mechanism.
#             activation: Activation function to use for the gating mechanism and experts.
#             key: Random key for initialization.
#         """

#         assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"
        
#         self.spatial_dim = spatial_dim
#         self.input_functions = input_functions
#         self.n_heads = n_heads
#         self.latent_dim = latent_dim
#         self.n_experts = n_experts
#         self.width_size = width_size
#         self.dropout_rate = dropout_rate
#         act = activation_dict[activation] if isinstance(activation, str) else activation

#         key, subkey = jr.split(key)
#         self.cross_attn = GNOTCrossAttentionBlock(
#             n_heads=n_heads,
#             latent_dim=latent_dim,
#             input_functions=input_functions,
#             dropout_rate=dropout_rate,
#             key=subkey
#         )
#         key, subkey = jr.split(key)
#         self.self_attn = GNOTSelfAttention(
#             n_heads=n_heads,
#             latent_dim=latent_dim,
#             dropout_rate=dropout_rate,
#             key=subkey
#         )
#         key, subkey = jr.split(key)
#         self.f_gate = eqx.nn.MLP(
#             in_size=spatial_dim,
#             out_size=n_experts,
#             activation=act,
#             final_activation=jax.nn.identity,
#             width_size=width_size,
#             depth=2,
#             key=key
#         )
#         key, *subkeys = jr.split(key, n_experts + 1)
#         self.f_experts_cross = [
#             eqx.nn.MLP(
#                 in_size=latent_dim,
#                 out_size=latent_dim,
#                 activation=act,
#                 width_size=width_size,
#                 depth=1,
#                 final_activation=jax.nn.identity,
#                 key=subkey
#             ) for subkey in subkeys
#         ]
#         key, *subkeys = jr.split(key, n_experts + 1)
#         self.f_experts_self = [
#             eqx.nn.MLP(
#                 in_size=latent_dim,
#                 out_size=latent_dim,
#                 activation=act,
#                 width_size=width_size,
#                 depth=1,
#                 final_activation=jax.nn.identity,
#                 key=subkey
#             ) for subkey in subkeys
#         ]

#         self.lns_ys = {
#             input_functions[i]: eqx.nn.LayerNorm(latent_dim) 
#             for i in range(len(input_functions))
#         }
#         self.lns_xs = [
#             eqx.nn.LayerNorm(latent_dim) for _ in range(4)
#         ]

#         self.dropout_cross = eqx.nn.Dropout(dropout_rate)
#         self.dropout_self = eqx.nn.Dropout(dropout_rate) 
        

#     def __call__(self, x: Float[ArrayLike, "len_query d"], coords: Float[ArrayLike, "len_query spatial_dim"], \
#                  y: Dict[str, Float[ArrayLike, "len_key d"]], mask_query: Bool[ArrayLike, "max_len"], \
#                     masks_functions: Dict[str, Bool[ArrayLike, "max_len"]], *, key: PRNGKeyArray=None) \
#                     -> Float[ArrayLike, "len_query d"]:
#         """
#         Forward pass of the GNOT gated attention block.
#         Args:
#             x: Input tensor of shape (len_query, d).
#             coords: Spatial coordinates tensor of shape (len_query, spatial_dim).
#             y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).

#         Returns:
#             Gated attention output tensor of shape (len_query, d).
#         """
#         subkey = None
#         gate_scores = eqx.filter_vmap(self.f_gate)(coords)
#         if not mask_query is None:
#             gate_scores = jnp.where(mask_query, gate_scores, -jnp.inf)
#         gate_scores = jax.nn.softmax(gate_scores, axis=-1) # (len_query, n_experts)
#         gate_scores = jnp.nan_to_num(gate_scores)
#         x = eqx.filter_vmap(self.lns_xs[0])(x)  # Layer normalization for x
#         for fname in y:
#             y[fname] = (eqx.filter_vmap(self.lns_ys[fname])(y[fname]))  # Layer normalization for y
        
#         if not key is None:
#             key, subkey = jr.split(key)
#         x = x + self.dropout_cross(self.cross_attn(x, y, mask_query, masks_functions), key=subkey)
#         moe_sum = jnp.zeros_like(x) # (len_query, d)
#         for i in range(self.n_experts):
#             moe_sum = moe_sum + (gate_scores[:, i:i+1] * eqx.filter_vmap(self.f_experts_cross[i])(x))
#         moe_sum = eqx.filter_vmap(self.lns_xs[1])(moe_sum) 
#         x = x + moe_sum
#         if not mask_query is None:
#             x = x * mask_query

#         x = eqx.filter_vmap(self.lns_xs[2])(x)
#         if not key is None:
#             key, subkey = jr.split(key)
#         x = x + self.dropout_self(self.self_attn(x, mask_query), key=subkey)
#         moe_sum = jnp.zeros_like(x)
#         for i in range(self.n_experts):
#             moe_sum = moe_sum + (gate_scores[:, i:i+1] * eqx.filter_vmap(self.f_experts_self[i])(x))
#         moe_sum = eqx.filter_vmap(self.lns_xs[3])(moe_sum)
#         x = x + moe_sum
#         if not mask_query is None:
#             x = x * mask_query
#         return x
    
# class GNOT(eqx.Module):

#     f_trunk: eqx.Module
#     fs_branch: Dict[str, eqx.Module]
#     coord_transform: eqx.Module
#     function_transforms: Dict[str, eqx.Module]
#     output_transform: eqx.Module
#     projection: eqx.Module
#     attn_blocks: Sequence[GNOTGatedAttentionBlock]
#     input_dim: int = eqx.field(static=True)
#     spatial_dim: int = eqx.field(static=True)
#     output_dim: int = eqx.field(static=True)
#     input_functions: Dict[str, int] = eqx.field(static=True)
#     latent_dim: int = eqx.field(static=True)
#     n_blocks: int = eqx.field(static=True)

#     def __init__(self, input_dim: int, spatial_dim: int, output_dim: int, function_specs: Dict[str, FunctionSpec], 
#                  width_lifting, depth_lifting: int, activation_lifting: str, latent_dim: int, n_blocks: int, 
#                  n_heads: int, dropout_rate: float, n_experts: int, width_moe: int, activation_moe: str, 
#                  min_coords, max_coords, mean_out, std_out, *, key: PRNGKeyArray):
#         """
#         GNOT model with a trunk network, multiple gated attention blocks, and branch networks for each input function.

#         Args:
#             input_dim: size of the input query. Should be greater than or equal to spatial_dim.
#             spatial_dim: spatial dimension of the problem.
#             output_dim: number of output channels.
#             input_functions: dictionary of input functions with their names as keys and their size as values.
#             width_lifting: width of the trunk, branch, and projection MLPs. 
#             depth_lifting: depth of the trunk, branch, and projection MLPs. 
#             activation_lifting: activation function used for trunk, branch, and projection MLPs.
#             latent_dim: internal latent dimension of the model.
#             n_blocks: number of attention blocks.
#             n_heads: number of attention heads. 
#             dropout_rate: dropout ratio.
#             n_experts: number of expert networks used in geometric gating.
#             width_moe: width of expert networks.
#             activation_moe: activation function of the expert networks.
#             key: Random key for initialization.
#         """

#         self.spatial_dim = spatial_dim
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_functions = {
#             k: function_specs[k]['size'] for k in function_specs 
#         }
#         self.latent_dim = latent_dim
#         self.n_blocks = n_blocks

#         self.coord_transform = Standardization(min_coords, max_coords)
#         self.output_transform = Denormalization(mean_out, std_out)
#         function_transforms = {}
#         for name, spec in function_specs.items():
#             if spec['has_x'] and spec['has_y']:
#                 function_transforms[name] = MixedFunctionTransform(spec['min_x'], spec['max_x'], 
#                                                                    spec['mean_y'], spec['std_y'])
#             elif spec['has_x']:
#                 function_transforms[name] = Standardization(spec['min_x'], spec['max_x'])
#             else:
#                 function_transforms[name] = Normalization(spec['mean_y'], spec['std_y'])
#         self.function_transforms = function_transforms
#         key, subkey = jr.split(key)
#         self.f_trunk = eqx.nn.MLP(
#             in_size=self.input_dim,
#             out_size=self.latent_dim,
#             width_size=width_lifting,
#             depth=depth_lifting,
#             activation=activation_dict[activation_lifting],
#             final_activation=jax.nn.identity,
#             key=subkey
#         )
#         key, *subkeys = jr.split(key, len(self.input_functions) + 1)
#         self.fs_branch = {
#             func_name: eqx.nn.MLP(
#                 in_size=self.input_functions[func_name],
#                 out_size=self.latent_dim,
#                 width_size=width_lifting,
#                 depth=depth_lifting,
#                 activation=activation_dict[activation_lifting],
#                 final_activation=jax.nn.identity,
#                 key=subkey
#             ) for func_name, subkey in zip(self.input_functions.keys(), subkeys)
#         }
#         key, subkey = jr.split(key)
#         self.projection = eqx.nn.MLP(
#             in_size=self.latent_dim,
#             out_size=self.output_dim,
#             width_size=width_lifting,
#             depth=depth_lifting,
#             activation=activation_dict[activation_lifting],
#             final_activation=jax.nn.identity,
#             key=subkey
#         )

#         key, *subkeys = jr.split(key, self.n_blocks + 1)
#         self.attn_blocks = [
#             GNOTGatedAttentionBlock(
#                 spatial_dim=self.spatial_dim,
#                 input_functions=list(self.input_functions.keys()),
#                 n_heads=n_heads,
#                 latent_dim=self.latent_dim,
#                 n_experts=n_experts,
#                 width_size=width_moe,
#                 dropout_rate=dropout_rate,
#                 activation=activation_moe,
#                 key=subkey
#             ) for subkey in subkeys
#         ]

#     def __call__(self, x: Float[ArrayLike, "len_query input_len"], y: Dict[str, Float[ArrayLike, "len_key d"]], \
#                  mask_query: Bool[ArrayLike, "max_len"]=None, masks_functions: Dict[str, Bool[ArrayLike, "max_len"]]=None, \
#                     key: PRNGKeyArray=None) -> Float[ArrayLike, "len_query output_dim"]:
#         """
#         Forward pass of the GNOT model.
#         Args:
#             x: Query tensor of input coordinates (and optionally additional query) of shape (len_query, input_len).\
#                 The first spatial_dim columns should be the spatial coordinates.
#             y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).
#         Returns:
#             Output tensor of shape (len_query, output_dim).
#         """
#         subkey = None
#         coords = x[:, :self.spatial_dim]
#         coords = self.coord_transform(coords)
#         x = eqx.filter_vmap(self.f_trunk)(x)
#         x = x
#         y_out = {}
#         for func_name in y:
#             y_normed = eqx.filter_vmap(self.function_transforms[func_name])(y[func_name])
#             y_out[func_name] = eqx.filter_vmap(self.fs_branch[func_name])(y_normed)

#         for block in self.attn_blocks:
#             if not key is None:
#                 key, subkey = jr.split(key)
#             x = block(x, coords, y_out, mask_query, masks_functions, key=subkey)
#         if not mask_query is None:  
#             return self.output_transform(eqx.filter_vmap(self.projection)(x)) * mask_query
#         else:
#             return self.output_transform(eqx.filter_vmap(self.projection)(x))            
    
# class GeoGNOTConfig(GNOTConfig):
#     geo_dim: int
#     mean_geo: ArrayLike
#     std_geo: ArrayLike

# class GeoGNOT(eqx.Module):

#     gnot: GNOT
#     geo_reconstructor: eqx.Module
#     denorm_geo: eqx.Module

#     def __init__(self, input_dim, spatial_dim, output_dim, function_specs, width_lifting, \
#                  depth_lifting, activation_lifting, latent_dim, n_blocks, n_heads, dropout_rate, \
#                     n_experts, width_moe, activation_moe, min_coords, max_coords, mean_out, std_out, \
#                         geo_dim, mean_geo, std_geo, *, key):
        

#         self.gnot = GNOT(input_dim, spatial_dim, output_dim, function_specs, width_lifting, depth_lifting, \
#                          activation_lifting, latent_dim, n_blocks, n_heads, dropout_rate, n_experts, width_moe, \
#                             activation_moe, min_coords, max_coords, mean_out, std_out, key=key)
        
#         key, subkey = jr.split(key)
#         self.geo_reconstructor = eqx.nn.MLP(
#             in_size=latent_dim,
#             out_size=geo_dim,
#             width_size=width_lifting,
#             depth=depth_lifting,
#             activation=activation_dict[activation_lifting],
#             final_activation=jax.nn.identity,
#             key=subkey
#         )
#         self.denorm_geo = Denormalization(mean_geo, std_geo)
        
#     def __call__(self, x: Float[ArrayLike, "len_query input_len"], y: Dict[str, Float[ArrayLike, "len_key d"]], \
#                  mask_query: Bool[ArrayLike, "max_len"]=None, masks_functions: Dict[str, Bool[ArrayLike, "max_len"]]=None, \
#                     key: PRNGKeyArray=None) -> Float[ArrayLike, "len_query output_dim"]:
#         """
#         Forward pass of the GNOT model.
#         Args:
#             x: Query tensor of input coordinates (and optionally additional query) of shape (len_query, input_len).\
#                 The first spatial_dim columns should be the spatial coordinates.
#             y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).
#         Returns:
#             Output tensor of shape (len_query, output_dim).
#         """
#         subkey = None
#         coords = x[:, :self.gnot.spatial_dim]
#         coords = self.gnot.coord_transform(coords)
#         x = eqx.filter_vmap(self.gnot.f_trunk)(x)
#         y_out = {}
#         for func_name in y:
#             y_normed = eqx.filter_vmap(self.gnot.function_transforms[func_name])(y[func_name])
#             y_out[func_name] = eqx.filter_vmap(self.gnot.fs_branch[func_name])(y_normed)

#         for block in self.gnot.attn_blocks:
#             if not key is None:
#                 key, subkey = jr.split(key)
#             x = block(x, coords, y_out, mask_query, masks_functions, key=subkey) 

#         out = self.gnot.output_transform(eqx.filter_vmap(self.gnot.projection)(x))
#         geo = self.denorm_geo(eqx.filter_vmap(self.geo_reconstructor)(x))
#         if not mask_query is None:
#             out = out * mask_query
#             geo = geo * mask_query
        
#         return out, geo



"""
GNOT Equinox implementation based on GNOT paper: https://arxiv.org/abs/2302.14376
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Float, Bool
from typing import Optional, Union, Sequence, Callable, Any, Dict, Tuple, TypedDict
from enum import Enum
from functools import partial
from dataclasses import dataclass

from .base import *
from ..data.data_utils import FunctionSpec

class GNOTConfig(TypedDict):
    # Data specifics
    input_dim: int
    spatial_dim: int
    output_dim: int
    function_specs: Dict[str, FunctionSpec]
    # normalization stats
    min_coords: Float[ArrayLike, "D"]
    max_coords: Float[ArrayLike, "D"]
    mean_out: Float[ArrayLike, "C"]
    std_out: Float[ArrayLike, "C"]

    # Lifting and projection
    width_lifting: int
    depth_lifting: int
    activation_lifting: str

    # Attention specifics
    latent_dim: int
    n_blocks: int
    n_heads: int
    dropout_rate: float

    # Gating specifics
    n_experts: int
    width_moe: int
    activation_moe: str
    
class GNOTCrossAttentionBlock(eqx.Module):
    """
    GNOT multi-head attention block with normalized linear attention, dropout, and reisudal connections. 

    """

    n_heads: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    input_functions: Sequence[str] = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    f_query: eqx.Module
    fs_key: Dict[str, eqx.Module]
    fs_value: Dict[str, eqx.Module]
    f_proj: eqx.Module
    
    def __init__(self, n_heads: int, latent_dim: int, input_functions: Sequence[str], dropout_rate: float = 0.0, *, \
                 key: PRNGKeyArray):
        
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.input_functions = input_functions 
        self.dropout_rate = dropout_rate
        
        key, subkey = jr.split(key)
        self.f_query = eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkey)
        
        key, *subkeys = jr.split(key, len(input_functions) + 1)
        self.fs_key = {
            input_functions[i]: eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkeys[i]) 
            for i in range(len(input_functions))
        }
        
        key, *subkeys = jr.split(key, len(input_functions) + 1)
        self.fs_value = {
            input_functions[i]: eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkeys[i]) 
            for i in range(len(input_functions))
        }

        key, subkey = jr.split(key)
        self.f_proj = eqx.nn.Linear(self.latent_dim, self.latent_dim, key=subkey)

    def __call__(self, x, y, mask_query, masks_functions):
        """
        Forward pass of the GNOT attention block. Pass the same 
    
        Args:
            x: Input query tensor of shape (len_query, d).
            y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).
        Returns:
            Attention output tensor of shape (len_query, d).
        """
        
        q = eqx.filter_vmap(self.f_query)(x) # (len_query, d)
        if not mask_query is None:
            q = jnp.where(mask_query, q, -jnp.inf)
        q = jnp.reshape(q, (q.shape[0], self.n_heads, self.latent_dim // self.n_heads)) # (len_query, n_heads, d // n_heads)
        q = jnp.transpose(q, (1, 0, 2)) # (n_heads, len_query, d // n_heads)
        q = jax.nn.softmax(q, axis=-1)
        q = jnp.nan_to_num(q)
        out = q
        geo_attn = jnp.zeros_like(out)
        for fname in y:
            k = eqx.filter_vmap(self.fs_key[fname])(y[fname]) # (len_key, d)
            if (not masks_functions is None) and (not masks_functions[fname] is None):
                k = jnp.where(masks_functions[fname], k, -jnp.inf)
            k = jnp.reshape(k, (k.shape[0], self.n_heads, self.latent_dim // self.n_heads)) # (len_key, n_heads, d // n_heads)
            k = jnp.transpose(k, (1, 0, 2)) # (n_heads, len_key, d // n_heads)
            k = jax.nn.softmax(k, axis=-1)
            k = jnp.nan_to_num(k)

            v = eqx.filter_vmap(self.fs_value[fname])(y[fname]) # (len_key, d)
            if (not masks_functions is None) and (not masks_functions[fname] is None):
                v = v * masks_functions[fname]
            v = jnp.reshape(v, (v.shape[0], self.n_heads, self.latent_dim // self.n_heads)) # (len_key, n_heads, d // n_heads)
            v = jnp.transpose(v, (1, 0, 2))

            dot_product = eqx.filter_vmap(normalized_linear_dot_product)(q, k, v) # (n_heads, len_query, d // n_heads)
            out = out + dot_product
            if fname == "bluff_body":
                # print("Bluff body attention computed.")
                geo_attn = geo_attn.at[:].set(dot_product)
                # jax.debug.print("dot prod: {x}", x=jnp.sum(dot_product))

            # out = out + eqx.filter_vmap(normalized_linear_dot_product)(q, k, v) # (n_heads, len_query, d // n_heads)

        out = jnp.transpose(out, (1, 0, 2)) # (len_query, n_heads, d // n_heads)
        out = jnp.reshape(out, (out.shape[0], self.latent_dim))
        geo_attn = jnp.transpose(geo_attn, (1, 0, 2)) # (len_query, n_heads, d // n_heads)
        geo_attn = jnp.reshape(geo_attn, (geo_attn.shape[0], self.latent_dim))
        # jax.debug.print("geo attn: {x}", x=jnp.sum(geo_attn))
        if not mask_query is None:
            return eqx.filter_vmap(self.f_proj)(out) * mask_query, geo_attn * mask_query # (len_query, d)
        else:
            return eqx.filter_vmap(self.f_proj)(out), geo_attn
    
class GNOTSelfAttention(GNOTCrossAttentionBlock):
    """
    GNOT self-attention block. Uses the same input for query, key, and value.
    """

    def __init__(self, n_heads: int, latent_dim: int, dropout_rate: float = 0.0, *, key: PRNGKeyArray):
        """
        Initialize the GNOT self-attention block.
        
        Args:
            n_heads: Number of attention heads.
            latent_dim: Dimensionality of the input and output tensors.
            dropout_rate: Dropout rate for the attention mechanism.
            key: Random key for initialization.
        """
        super().__init__(n_heads, latent_dim, input_functions=["latent"], dropout_rate=dropout_rate, key=key)

    def __call__(self, x, mask):
        """
        Forward pass of the GNOT self-attention block.
        
        Args:
            x: Input tensor of shape (len_query, d).
        Returns:
            Attention output tensor of shape (len_query, d).
        """
        return super().__call__(x, {'latent': x}, mask, {'latent': mask})
    
class GNOTGatedAttentionBlock(eqx.Module):
    """
    GNOT gated attention block. Applies a gating mechanism to the attention output.
    """

    spatial_dim: int = eqx.field(static=True)
    input_functions: Sequence[str] = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_experts: int = eqx.field(static=True) 
    width_size: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    cross_attn: GNOTCrossAttentionBlock
    self_attn: GNOTSelfAttention
    f_gate: eqx.Module
    f_experts_cross: Sequence[eqx.Module]
    f_experts_self: Sequence[eqx.Module]
    lns_ys: Dict[str, eqx.Module]
    lns_xs: Sequence[eqx.Module]
    ln_geo: eqx.Module
    dropout_cross: eqx.Module
    dropout_self: eqx.Module

    def __init__(self, spatial_dim: int, input_functions: Sequence[str], n_heads: int, latent_dim: int, n_experts: int, \
                 width_size: int, dropout_rate: float = 0.0, activation: Union[str, Callable] = "relu", *, key: PRNGKeyArray):
        """
        Multi-head attention block with multi-head cross and self-attention, dropout, layer normalisation, 
        and gating mechanism.

        Args:
            spatial_dim: Dimensionality of the spatial input.
            input_functions: List of names of input functions.
            n_heads: Number of attention heads.
            latent_dim: Dimensionality of the input and output tensors.
            n_experts: Number of experts in the gating mechanism.
            width_size: Width of the MLP layers in the gating mechanism.
            dropout_rate: Dropout rate for the attention mechanism.
            activation: Activation function to use for the gating mechanism and experts.
            key: Random key for initialization.
        """

        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"
        
        self.spatial_dim = spatial_dim
        self.input_functions = input_functions
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.width_size = width_size
        self.dropout_rate = dropout_rate
        act = activation_dict[activation] if isinstance(activation, str) else activation

        key, subkey = jr.split(key)
        self.cross_attn = GNOTCrossAttentionBlock(
            n_heads=n_heads,
            latent_dim=latent_dim,
            input_functions=input_functions,
            dropout_rate=dropout_rate,
            key=subkey
        )
        key, subkey = jr.split(key)
        self.self_attn = GNOTSelfAttention(
            n_heads=n_heads,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            key=subkey
        )
        key, subkey = jr.split(key)
        self.f_gate = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=n_experts,
            activation=act,
            final_activation=jax.nn.identity,
            width_size=width_size,
            depth=2,
            key=key
        )
        key, *subkeys = jr.split(key, n_experts + 1)
        self.f_experts_cross = [
            eqx.nn.MLP(
                in_size=latent_dim,
                out_size=latent_dim,
                activation=act,
                width_size=width_size,
                depth=1,
                final_activation=jax.nn.identity,
                key=subkey
            ) for subkey in subkeys
        ]
        key, *subkeys = jr.split(key, n_experts + 1)
        self.f_experts_self = [
            eqx.nn.MLP(
                in_size=latent_dim,
                out_size=latent_dim,
                activation=act,
                width_size=width_size,
                depth=1,
                final_activation=jax.nn.identity,
                key=subkey
            ) for subkey in subkeys
        ]

        self.lns_ys = {
            input_functions[i]: eqx.nn.LayerNorm(latent_dim) 
            for i in range(len(input_functions))
        }
        self.lns_xs = [
            eqx.nn.LayerNorm(latent_dim) for _ in range(4)
        ]
        self.ln_geo = eqx.nn.LayerNorm(latent_dim)

        self.dropout_cross = eqx.nn.Dropout(dropout_rate)
        self.dropout_self = eqx.nn.Dropout(dropout_rate) 
        

    def __call__(self, x: Float[ArrayLike, "len_query d"], coords: Float[ArrayLike, "len_query spatial_dim"], \
                 y: Dict[str, Float[ArrayLike, "len_key d"]], mask_query: Bool[ArrayLike, "max_len"], \
                    masks_functions: Dict[str, Bool[ArrayLike, "max_len"]], *, key: PRNGKeyArray=None) \
                    -> Float[ArrayLike, "len_query d"]:
        """
        Forward pass of the GNOT gated attention block.
        Args:
            x: Input tensor of shape (len_query, d).
            coords: Spatial coordinates tensor of shape (len_query, spatial_dim).
            y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).

        Returns:
            Gated attention output tensor of shape (len_query, d).
        """
        subkey = None
        # gate_scores = eqx.filter_vmap(self.f_gate)(coords)
        # if not mask_query is None:
        #     gate_scores = jnp.where(mask_query, gate_scores, -jnp.inf)
        # gate_scores = jax.nn.softmax(gate_scores, axis=-1) # (len_query, n_experts)
        # gate_scores = jnp.nan_to_num(gate_scores)
        x = eqx.filter_vmap(self.lns_xs[0])(x)  # Layer normalization for x
        for fname in y:
            y[fname] = (eqx.filter_vmap(self.lns_ys[fname])(y[fname]))  # Layer normalization for y
        
        if not key is None:
            key, subkey = jr.split(key)
        # x = x + self.dropout_cross(self.cross_attn(x, y, mask_query, masks_functions), key=subkey)
        attn_out, geo_attn = self.cross_attn(x, y, mask_query, masks_functions)
        geo_attn = jax.lax.stop_gradient(geo_attn)
        # geo_attn = eqx.filter_vmap(self.ln_geo)(geo_attn)
        x = x + self.dropout_cross(attn_out, key=subkey)
        gate_scores = eqx.filter_vmap(self.f_gate)(geo_attn)
        if not mask_query is None:
            gate_scores = jnp.where(mask_query, gate_scores, -jnp.inf)
        gate_scores = jax.nn.softmax(gate_scores, axis=-1) # (len_query, n_experts)
        gate_scores = jnp.nan_to_num(gate_scores)

        moe_sum = jnp.zeros_like(x) # (len_query, d)
        for i in range(self.n_experts):
            moe_sum = moe_sum + (gate_scores[:, i:i+1] * eqx.filter_vmap(self.f_experts_cross[i])(x))
        moe_sum = eqx.filter_vmap(self.lns_xs[1])(moe_sum) 
        x = x + moe_sum
        if not mask_query is None:
            x = x * mask_query

        x = eqx.filter_vmap(self.lns_xs[2])(x)
        if not key is None:
            key, subkey = jr.split(key)
        # x = x + self.dropout_self(self.self_attn(x, mask_query), key=subkey)
        attn_out, _ = self.self_attn(x, mask_query)
        x = x + self.dropout_self(attn_out, key=subkey)

        moe_sum = jnp.zeros_like(x)
        for i in range(self.n_experts):
            moe_sum = moe_sum + (gate_scores[:, i:i+1] * eqx.filter_vmap(self.f_experts_self[i])(x))
        moe_sum = eqx.filter_vmap(self.lns_xs[3])(moe_sum)
        x = x + moe_sum
        if not mask_query is None:
            x = x * mask_query
        return x, gate_scores
    
class GNOT(eqx.Module):

    f_trunk: eqx.Module
    fs_branch: Dict[str, eqx.Module]
    coord_transform: eqx.Module
    function_transforms: Dict[str, eqx.Module]
    output_transform: eqx.Module
    projection: eqx.Module
    attn_blocks: Sequence[GNOTGatedAttentionBlock]
    input_dim: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    input_functions: Dict[str, int] = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)

    def __init__(self, input_dim: int, spatial_dim: int, output_dim: int, function_specs: Dict[str, FunctionSpec], 
                 width_lifting, depth_lifting: int, activation_lifting: str, latent_dim: int, n_blocks: int, 
                 n_heads: int, dropout_rate: float, n_experts: int, width_moe: int, activation_moe: str, 
                 min_coords, max_coords, mean_out, std_out, *, key: PRNGKeyArray):
        """
        GNOT model with a trunk network, multiple gated attention blocks, and branch networks for each input function.

        Args:
            input_dim: size of the input query. Should be greater than or equal to spatial_dim.
            spatial_dim: spatial dimension of the problem.
            output_dim: number of output channels.
            input_functions: dictionary of input functions with their names as keys and their size as values.
            width_lifting: width of the trunk, branch, and projection MLPs. 
            depth_lifting: depth of the trunk, branch, and projection MLPs. 
            activation_lifting: activation function used for trunk, branch, and projection MLPs.
            latent_dim: internal latent dimension of the model.
            n_blocks: number of attention blocks.
            n_heads: number of attention heads. 
            dropout_rate: dropout ratio.
            n_experts: number of expert networks used in geometric gating.
            width_moe: width of expert networks.
            activation_moe: activation function of the expert networks.
            key: Random key for initialization.
        """

        self.spatial_dim = spatial_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_functions = {
            k: function_specs[k]['size'] for k in function_specs 
        }
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks

        self.coord_transform = Standardization(min_coords, max_coords)
        self.output_transform = Denormalization(mean_out, std_out)
        function_transforms = {}
        for name, spec in function_specs.items():
            if spec['has_x'] and spec['has_y']:
                function_transforms[name] = MixedFunctionTransform(spec['min_x'], spec['max_x'], 
                                                                   spec['mean_y'], spec['std_y'])
            elif spec['has_x']:
                function_transforms[name] = Standardization(spec['min_x'], spec['max_x'])
            else:
                function_transforms[name] = Normalization(spec['mean_y'], spec['std_y'])
        self.function_transforms = function_transforms
        key, subkey = jr.split(key)
        self.f_trunk = eqx.nn.MLP(
            in_size=self.input_dim,
            out_size=self.latent_dim,
            width_size=width_lifting,
            depth=depth_lifting,
            activation=activation_dict[activation_lifting],
            final_activation=jax.nn.identity,
            key=subkey
        )
        key, *subkeys = jr.split(key, len(self.input_functions) + 1)
        self.fs_branch = {
            func_name: eqx.nn.MLP(
                in_size=self.input_functions[func_name],
                out_size=self.latent_dim,
                width_size=width_lifting,
                depth=depth_lifting,
                activation=activation_dict[activation_lifting],
                final_activation=jax.nn.identity,
                key=subkey
            ) for func_name, subkey in zip(self.input_functions.keys(), subkeys)
        }
        key, subkey = jr.split(key)
        self.projection = eqx.nn.MLP(
            in_size=self.latent_dim,
            out_size=self.output_dim,
            width_size=width_lifting,
            depth=depth_lifting,
            activation=activation_dict[activation_lifting],
            final_activation=jax.nn.identity,
            key=subkey
        )

        key, *subkeys = jr.split(key, self.n_blocks + 1)
        self.attn_blocks = [
            GNOTGatedAttentionBlock(
                spatial_dim=self.spatial_dim,
                input_functions=list(self.input_functions.keys()),
                n_heads=n_heads,
                latent_dim=self.latent_dim,
                n_experts=n_experts,
                width_size=width_moe,
                dropout_rate=dropout_rate,
                activation=activation_moe,
                key=subkey
            ) for subkey in subkeys
        ]

    def __call__(self, x: Float[ArrayLike, "len_query input_len"], y: Dict[str, Float[ArrayLike, "len_key d"]], \
                 mask_query: Bool[ArrayLike, "max_len"]=None, masks_functions: Dict[str, Bool[ArrayLike, "max_len"]]=None, \
                    key: PRNGKeyArray=None) -> Float[ArrayLike, "len_query output_dim"]:
        """
        Forward pass of the GNOT model.
        Args:
            x: Query tensor of input coordinates (and optionally additional query) of shape (len_query, input_len).\
                The first spatial_dim columns should be the spatial coordinates.
            y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).
        Returns:
            Output tensor of shape (len_query, output_dim).
        """
        subkey = None
        coords = x[:, :self.spatial_dim]
        coords = self.coord_transform(coords)
        x = eqx.filter_vmap(self.f_trunk)(x)
        x = x
        y_out = {}
        for func_name in y:
            y_normed = eqx.filter_vmap(self.function_transforms[func_name])(y[func_name])
            y_out[func_name] = eqx.filter_vmap(self.fs_branch[func_name])(y_normed)

        gate_scores = []
        for block in self.attn_blocks:
            if not key is None:
                key, subkey = jr.split(key)
            x, gate_score = block(x, coords, y_out, mask_query, masks_functions, key=subkey)
            gate_scores.append(gate_score)
        if not mask_query is None:  
            return self.output_transform(eqx.filter_vmap(self.projection)(x)) * mask_query, gate_scores
        else:
            return self.output_transform(eqx.filter_vmap(self.projection)(x)), gate_scores

class GeoGNOTConfig(GNOTConfig):
    geo_dim: int
    mean_geo: ArrayLike
    std_geo: ArrayLike

class GeoGNOT(eqx.Module):

    gnot: GNOT
    geo_reconstructor: eqx.Module
    denorm_geo: eqx.Module

    def __init__(self, input_dim, spatial_dim, output_dim, function_specs, width_lifting, \
                 depth_lifting, activation_lifting, latent_dim, n_blocks, n_heads, dropout_rate, \
                    n_experts, width_moe, activation_moe, min_coords, max_coords, mean_out, std_out, \
                        geo_dim, mean_geo, std_geo, *, key):
        

        self.gnot = GNOT(input_dim, spatial_dim, output_dim, function_specs, width_lifting, depth_lifting, \
                         activation_lifting, latent_dim, n_blocks, n_heads, dropout_rate, n_experts, width_moe, \
                            activation_moe, min_coords, max_coords, mean_out, std_out, key=key)
        
        key, subkey = jr.split(key)
        self.geo_reconstructor = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=geo_dim,
            width_size=width_lifting,
            depth=depth_lifting,
            activation=activation_dict[activation_lifting],
            final_activation=jax.nn.identity,
            key=subkey
        )
        self.denorm_geo = Denormalization(mean_geo, std_geo)
        
    def __call__(self, x: Float[ArrayLike, "len_query input_len"], y: Dict[str, Float[ArrayLike, "len_key d"]], \
                 mask_query: Bool[ArrayLike, "max_len"]=None, masks_functions: Dict[str, Bool[ArrayLike, "max_len"]]=None, \
                    key: PRNGKeyArray=None) -> Float[ArrayLike, "len_query output_dim"]:
        """
        Forward pass of the GNOT model.
        Args:
            x: Query tensor of input coordinates (and optionally additional query) of shape (len_query, input_len).\
                The first spatial_dim columns should be the spatial coordinates.
            y: Dictionary of n_input items, each an input function tensor of shape (len_key, d).
        Returns:
            Output tensor of shape (len_query, output_dim).
        """
        subkey = None
        coords = x[:, :self.gnot.spatial_dim]
        coords = self.gnot.coord_transform(coords)
        x = eqx.filter_vmap(self.gnot.f_trunk)(x)
        y_out = {}
        for func_name in y:
            y_normed = eqx.filter_vmap(self.gnot.function_transforms[func_name])(y[func_name])
            y_out[func_name] = eqx.filter_vmap(self.gnot.fs_branch[func_name])(y_normed)

        # gate_scores = []
        for block in self.gnot.attn_blocks:
            if not key is None:
                key, subkey = jr.split(key)
            x, gate_score = block(x, coords, y_out, mask_query, masks_functions, key=subkey)
            # gate_scores.append(gate_score)

        out = self.gnot.output_transform(eqx.filter_vmap(self.gnot.projection)(x))
        geo = self.denorm_geo(eqx.filter_vmap(self.geo_reconstructor)(x))
        if not mask_query is None:
            out = out * mask_query
            geo = geo * mask_query

        return out, geo


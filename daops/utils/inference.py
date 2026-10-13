"""
低内存推理：逐样本循环替代 eqx.filter_vmap，避免 OOM。
"""

import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map


def predict_batch_loop(model, model_state, X, functions, mask_query, masks_functions, key):
    """
    按样本循环推理，避免整批 vmap 导致 OOM。
    返回的 predictions 形状与原来 vmap 一致。

    Args:
        model: 模型（可为 eqx.nn.make_with_state 返回的带 state 的模型）
        model_state: 模型 state（本函数内未使用，保留接口一致）
        X, functions, mask_query, masks_functions: 与 DataLoader 返回的 batch 一致
        key: jax random key

    Returns:
        predictions: shape (batch_size, len_query, output_dim)，与 eqx.filter_vmap(model,...)(...) 一致
    """
    batch_size = X.shape[0]
    key, *subkeys = jr.split(key, batch_size + 1)
    out_list = []
    for i in range(batch_size):
        x_i = X[i]
        y_i = tree_map(lambda arr: arr[i], functions)
        mq_i = mask_query[i]
        mf_i = tree_map(lambda arr: arr[i], masks_functions)
        out_i = model(x_i, y_i, mq_i, mf_i, key=subkeys[i])
        if isinstance(out_i, (list, tuple)):
            out_i = out_i[0]
        out_list.append(out_i)
    return jnp.stack(out_list, axis=0)

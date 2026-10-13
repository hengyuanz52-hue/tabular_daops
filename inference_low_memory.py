"""
低内存推理示例：用逐样本循环替代 eqx.filter_vmap，避免 OOM。

用法：在加载好 model、model_state、dataloader 后，先导入 predict_batch_loop，再调用即可。

    from daops.utils.inference import predict_batch_loop
    predictions = predict_batch_loop(model, model_state, X, functions, mask_query, masks_functions, key)
"""

import jax.random as jr
import equinox as eqx

from daops.utils.inference import predict_batch_loop


# ========== 在你的脚本里这样用 ==========
# 1. 获取一个样本（可先把 batch 改小，例如 batch_size=1）
# batch_data, masks = next(iter(dataloader))
# X = batch_data['X']
# functions = batch_data['functions']
# mask_query = masks['X']
# masks_functions = masks['functions']

# 2. 低内存推理（替代原来的 eqx.filter_vmap）
# key = jr.PRNGKey(0)
# predictions = predict_batch_loop(model, model_state, X, functions, mask_query, masks_functions, key)

# print(f"预测形状: {predictions.shape}")
# print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")


# ========== 完整示例：加载 vortex 模型 + 低内存预测 ==========
def run_low_memory_inference_example():
    """完整流程：加载数据、加载模型、用逐样本循环预测，避免 OOM。"""
    import numpy as np
    from torch.utils.data import DataLoader

    from daops.data import vortex
    from daops.data.data_utils import jax_padding_collate
    from daops.modules.gnot import GNOT
    from daops.utils.serial import load_model

    # 1. 加载数据
    dataset = vortex.VortexWindowDataset(
        './data/box.h5',
        window_size=6,
        step_size=2,
        T0=50,
        T1=100,
        indices=(0, 18),
        paramed=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=jax_padding_collate
    )

    # 2. 加载训练好的模型（与 train_vortex 保存的路径一致）
    model_path = "experiments/GNOT-F_VELOCITY-denormal_masking/vortex/model.eqx"
    model, model_state, _ = load_model(
        model_path, eqx.nn.make_with_state(GNOT), return_config=True
    )

    # 3. 取一个 batch
    batch_data, masks = next(iter(dataloader))
    X = batch_data['X']
    functions = batch_data['functions']
    mask_query = masks['X']
    masks_functions = masks['functions']

    # 4. 低内存推理（逐样本循环，替代 eqx.filter_vmap）
    key = jr.PRNGKey(0)
    predictions = predict_batch_loop(
        model, model_state, X, functions, mask_query, masks_functions, key
    )

    print(f"预测形状: {predictions.shape}")
    print(f"预测值范围: [{np.asarray(predictions.min()):.4f}, {np.asarray(predictions.max()):.4f}]")
    return predictions


if __name__ == "__main__":
    run_low_memory_inference_example()

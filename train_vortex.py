
import optax as opt
import equinox as eqx
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
from torch.utils.data import DataLoader

from daops.data.data_utils import numpy_collate, jax_padding_collate
from daops.data import vortex
from daops.modules.gnot import GNOT, GNOTConfig
from daops.training.train import SupervisedTrainer
from daops.utils.callbacks import CheckpointCallback
from daops.utils.serial import save_model

dataset = vortex.VortexWindowDataset('./data/box.h5', 6, step_size=2, T0=50, T1=100, indices=(0, 18),
                                     paramed=False)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=jax_padding_collate)
min_x, max_x = dataset.compute_min_max_coords()
mean_y, std_y = dataset.compute_mean_std_fields()
mean_y = jnp.tile(mean_y, [5, 1]).reshape([-1])
std_y = jnp.tile(std_y, [5, 1]).reshape([-1])

gnot_config = GNOTConfig(
    input_dim=2,
    spatial_dim=2,
    output_dim=10,
    function_specs=dataset.get_function_specs(),
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
    max_coords=max_x
)

key = jr.PRNGKey(0)

model, model_state = eqx.nn.make_with_state(GNOT)(**gnot_config, key=key)

optimizer = opt.adamw(opt.schedules.exponential_decay(init_value=1e-3, transition_steps=len(dataloader), decay_rate=0.98, staircase=True, end_value=1e-6),
                      weight_decay=0.0005)

trainer = SupervisedTrainer(model, model_state, optimizer, 
                            callbacks=[CheckpointCallback("experiments/GNOT-F_VELOCITY-denormal_masking/vortex/checkpoints", 
                                                gnot_config, 
                                                True, save_every=20)])

model, model_state, _ = trainer.fit(dataloader, 400, key=jr.PRNGKey(10))
save_model("experiments/GNOT-F_VELOCITY-denormal_masking/vortex/model.eqx", gnot_config, model, model_state)

# 可视化训练 loss 曲线
def plot_training_history(history, save_path=None):
    """绘制并可选保存 loss 曲线。"""
    import matplotlib.pyplot as plt
    import os
    losses = [float(np.asarray(x)) for x in history['loss']]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (train_vortex)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Loss 曲线已保存到: {save_path}")
        # 自动打开图片，确保能看到
        _path = os.path.abspath(save_path)
        if os.path.isfile(_path):
            import subprocess
            import sys
            if sys.platform == 'win32':
                os.startfile(_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', _path])
            else:
                subprocess.run(['xdg-open', _path])
    plt.show()_save_path = "experiments/GNOT-F_VELOCITY-denormal_masking/vortex/training_loss.png"
try:
    plot_training_history(trainer.history, save_path=_save_path)
except NameError:
    # trainer 未定义时（如单独运行可视化部分），从 checkpoint CSV 加载
    from daops.utils.serial import load_history_csv
    import os
    _csv = "experiments/GNOT-F_VELOCITY-denormal_masking/vortex/checkpoints/history_epoch.csv"
    if os.path.isfile(_csv):
        plot_training_history(load_history_csv(_csv), save_path=_save_path)
    else:
        print("无法绘图：请先完整运行训练，或使用 python visualize_training.py <checkpoint目录>")
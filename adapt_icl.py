"""
Zero-shot (gradient-free) few-shot adaptation via Operator-ICL.

Loads an ICL-meta-trained GeoGNOT and predicts on a held-out shape by packing
sparse labels into ``functions['support']`` — no fine-tuning by default.
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from torch.utils.data import DataLoader

from daops.data.data_utils import jax_padding_collate
from daops.data.vortex import VortexICLDataset
from daops.modules.gnot import GeoGNOT
from daops.utils.serial import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot Operator-ICL adaptation")
    p.add_argument(
        "--model_path",
        type=str,
        default="experiments/GeoGNOT_NewGate/ICL_META/3SHAPE_K64-256_T=75/model.eqx",
    )
    p.add_argument(
        "--shape",
        type=str,
        default="diamond",
        choices=["cylinder", "triangle", "box", "diamond"],
    )
    p.add_argument("--support_rate", type=float, default=0.005, help="Label budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eval_indices",
        type=int,
        nargs=2,
        default=(18, 20),
        help="Held-out trajectories for evaluation",
    )
    p.add_argument(
        "--velocity_subsample_rate",
        type=float,
        default=None,
        help="Optional: also sparsify velocity IC (match old adapt script)",
    )
    return p.parse_args()


def nmse(pred, target, mask):
    num = jnp.linalg.norm((pred - target) * mask, axis=-1) ** 2
    den = jnp.linalg.norm(target * mask, axis=-1) ** 2
    return jnp.mean(num / jnp.maximum(den, 1e-8))


def main():
    args = parse_args()
    data_path = f"./data/{args.shape}_train.h5"

    model, model_state, config = load_model(
        args.model_path, eqx.nn.make_with_state(GeoGNOT), True
    )
    if "support" not in config["function_specs"]:
        raise ValueError(
            "Checkpoint has no 'support' function — train with train_icl_meta.py first."
        )

    # Full-field eval: on each held-out trajectory, pack support_rate labels as
    # context and score NMSE on the remaining query points (true few-shot ICL).
    eval_ds = VortexICLDataset(
        data_path,
        window_size=6,
        step_size=2,
        T0=50,
        T1=75,
        indices=tuple(args.eval_indices),
        paramed=False,
        support_rate=args.support_rate,
        resample_support=False,
        query_on_support=False,
        velocity_subsample_rate=args.velocity_subsample_rate,
        random_seed=args.seed,
    )

    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=jax_padding_collate)

    @eqx.filter_jit
    def predict(model, X, functions, mask_query, masks_functions, key):
        out = model(X, functions, mask_query, masks_functions, key)
        return out[0] if isinstance(out, tuple) else out

    losses = []
    key = jr.PRNGKey(args.seed)
    for batch in loader:
        data, masks = batch
        key, subkey = jr.split(key)
        # batch dim = 1
        pred = predict(
            model,
            data["X"][0],
            {k: v[0] for k, v in data["functions"].items()},
            masks["X"][0],
            {k: v[0] for k, v in masks["functions"].items()},
            subkey,
        )
        weight = masks["X"][0]
        if data.get("label_mask") is not None:
            weight = weight * data["label_mask"][0]
        losses.append(float(nmse(pred, data["u"][0], weight)))

    mean_nmse = float(np.mean(losses))
    k_support = eval_ds.k_support_min
    print(
        f"[ICL-0] shape={args.shape} | support_rate={args.support_rate} "
        f"(K≈{k_support}) | eval NMSE={mean_nmse:.6e} | n_batches={len(losses)}"
    )

    out_dir = Path(args.model_path).parent / f"ADAPT_ICL0_{args.shape.upper()}_SUB{args.support_rate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eval_nmse.npy", np.array(losses))
    print(f"Wrote {out_dir / 'eval_nmse.npy'}")


if __name__ == "__main__":
    main()

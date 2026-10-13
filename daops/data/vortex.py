from .data_utils import *
import numpy as np
import h5py as h5
from enum import Enum
import jax.numpy as jnp
from jaxtyping import Float, ArrayLike, Array

FUNCTIONS = {
    "bluff_body": FunctionSpec(size=2, has_x=True, has_y=False),
    "velocity": FunctionSpec(size=4, has_x=True, has_y=True)
}

PARAMETERIZED_FUNCTIONS = {
    "bluff_body": FunctionSpec(size=2, has_x=True, has_y=False),
    "velocity": FunctionSpec(size=4, has_x=True, has_y=True),
    "omega": FunctionSpec(size=1, has_x=False, has_y=True)
}

def flatten(u):
    # T * C * X * Y
    u = jnp.reshape(u, (u.shape[0], u.shape[1], -1))
    u = jnp.transpose(u, [2, 0, 1])
    u = jnp.reshape(u, (-1, u.shape[1] * u.shape[2]))
    return u

def unflatten(u, nx, ny, t):
    # XY * TC
    u = jnp.reshape(u, (-1, t, 2))
    u = jnp.transpose(u, [1, 2, 0])
    u = jnp.reshape(u, (u.shape[0], u.shape[1], nx, ny))
    return u

class VortexWindowDataset(BaseDataset):
    def __init__(self, path, window_size: int, step_size: int, T0: int = 0, T1: int = -1,
                 indices=None, paramed=False, domain: int=0):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size
        self.T0 = T0
        self.T1 = T1 if T1 != -1 else self.data['u'].shape[0]
        self.paramed = paramed

        with h5.File(path, 'r') as f:
            u = f['u'][:] 
            x = f['x'][:]
            y = f['y'][:]
            omega = f['omega'][:]
            bc_indices = f['bc_indices'][:]

        if indices is not None:
            if len(indices) == 2:
                u = u[indices[0]:indices[1]]
                omega = omega[indices[0]:indices[1]]    
                bc_indices = bc_indices[indices[0]:indices[1]]
            else:
                u = u[indices]
                omega = omega[indices]
                bc_indices = bc_indices[indices]
        self.u = u[:, T0:T1:step_size]
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.X_star = np.asarray(np.stack([self.X.flatten(), self.Y.flatten()], axis=-1), dtype=np.float32)
        self.omega = omega
        self.bc_indices = np.asarray(bc_indices[:], dtype=np.float32)
        self.max_start_time = self.u.shape[1] - window_size
        self.domain = domain

    def __len__(self):
        return (self.max_start_time + 1) * self.u.shape[0]

    def __getitem__(self, idx):
        traj_idx, start_idx = divmod(idx, self.max_start_time + 1)
        u = self.u[traj_idx, start_idx:start_idx + self.window_size].reshape(self.window_size, 2, -1) # T x 2 x nx*ny
        u = np.transpose(u, [2, 0, 1]) # (coords, time, channels)
        vel, u = u[:, 0], u[:, 1:]
        u = u.reshape(-1, 2 * (self.window_size - 1)) # 
        return DataDict(
            domain=self.domain,
            X=self.X_star,
            u=u,
            functions = {
                "omega": self.omega[traj_idx],
                "bluff_body": np.asarray(self.bc_indices[traj_idx], dtype=np.float32),
                "velocity": np.concat([self.X_star, vel], axis=-1)
            } if self.paramed else {
                "bluff_body": np.asarray(self.bc_indices[traj_idx], dtype=np.float32),
                "velocity": np.concat([self.X_star, vel], axis=-1)
            }
        )
    
    def compute_mean_std_fields(self):
        return np.mean(self.u, axis=(0, 1, 3, 4)), np.std(self.u, axis=(0, 1, 3, 4))
    
    def compute_mean_std_coords(self):
        return np.array([np.mean(self.x), np.mean(self.y)]), np.array([np.std(self.x), np.std(self.y)])
    
    def compute_min_max_coords(self):
        return np.array([np.min(self.x), np.min(self.y)]), np.array([np.max(self.x), np.max(self.y)])
    
    def get_function_specs(self):
        min_x, max_x = self.compute_min_max_coords()
        mean_y, std_y = self.compute_mean_std_fields()
        velocity_specs = FunctionSpec(size=4, has_x=True, has_y=True, min_x=jnp.array(min_x), max_x=jnp.array(max_x),
                                      mean_y=jnp.array(mean_y), std_y=jnp.array(std_y))
        bluff_specs = FunctionSpec(size=2, has_x=True, has_y=False, min_x=jnp.array(min_x), max_x=jnp.array(max_x))
        functions = {
            "velocity": velocity_specs,
            "bluff_body": bluff_specs
        }
        if self.paramed:
            functions['omega'] = FunctionSpec(size=1, has_x=False, has_y=True, mean_y=jnp.mean(self.omega),
                                              std_y=jnp.std(self.omega))
        return functions
    
class VortexRandomSubsampledDataset(VortexWindowDataset):

    def __init__(self, path, window_size, step_size, T0 = 0, T1 = -1, indices=None, paramed=False, domain = 0,
                 subsample_rate=0.1, random_seed=0, mask_label=False):
        super().__init__(path, window_size, step_size, T0, T1, indices, paramed, domain)
        self.subsample_rate = subsample_rate
        self.n_subsampled = int(self.X_star.shape[0] * subsample_rate)
        np.random.seed(random_seed)
        self.mask_label = mask_label

    def __getitem__(self, *args, **kwds):
        data = super().__getitem__(*args, **kwds)
        # randomly sample som coordinates of the 'velocity' function
        idxs = np.arange(self.X_star.shape[0])
        np.random.shuffle(idxs)
        idxs = idxs[:self.n_subsampled]
        # print(data['functions']['velocity'].shape)
        # print(data['functions']['velocity'][idxs])
        # print(idxs)
        data['functions']['velocity'] = data['functions']['velocity'][idxs]
        if self.mask_label:
            label_mask = np.zeros((self.X_star.shape[0], 1), dtype=bool)
            label_mask[idxs] = True
            data['label_mask'] = label_mask
        return data
    
class VortexFixedSubsampledDataset(VortexWindowDataset):

    def __init__(self, path, window_size, step_size, T0 = 0, T1 = -1, indices=None, paramed=False, domain = 0,
                 subsample_rate=0.1, random_seed=0, mask_label=False):
        super().__init__(path, window_size, step_size, T0, T1, indices, paramed, domain)
        self.subsample_rate = subsample_rate
        self.n_subsampled = int(self.X_star.shape[0] * subsample_rate)
        np.random.seed(random_seed)
        idxs = np.arange(self.X_star.shape[0])
        np.random.shuffle(idxs)
        self.idxs = idxs[:self.n_subsampled]
        self.mask_label = mask_label
        if self.mask_label:
            self.label_mask = np.zeros((self.X_star.shape[0], 1), dtype=bool)
            self.label_mask[self.idxs] = True

    def __getitem__(self, *args, **kwds):
        data = super().__getitem__(*args, **kwds)
        # randomly sample som coordinates of the 'velocity' function
        data['functions']['velocity'] = data['functions']['velocity'][self.idxs]
        if self.mask_label:
            data['label_mask'] = self.label_mask
        return data
    
class VortexRegularSubsampledDataset(VortexWindowDataset):
    
    def __init__(self, path, window_size, step_size, T0 = 0, T1 = -1, indices=None, paramed=False, domain = 0,
                 subsample_rate=0.1, mask_label=False):
        super().__init__(path, window_size, step_size, T0, T1, indices, paramed, domain)
        self.subsample_rate = subsample_rate
        self.n_subsampled = int(self.X_star.shape[0] * subsample_rate)
        step = int(1 / subsample_rate)
        self.idxs = np.arange(0, self.X_star.shape[0], step)[:self.n_subsampled]
        self.mask_label = mask_label
        if self.mask_label:
            self.label_mask = np.zeros((self.X_star.shape[0], 1), dtype=bool)
            self.label_mask[self.idxs] = True

    def __getitem__(self, *args, **kwds):
        data = super().__getitem__(*args, **kwds)
        # randomly sample som coordinates of the 'velocity' function
        data['functions']['velocity'] = data['functions']['velocity'][self.idxs]
        if self.mask_label:
            data['label_mask'] = self.label_mask
        return data

class VortexSDFDataset(BaseDataset):
    def __init__(self, path, step_size: int, T0: int = 0, T1: int = -1, indices=None, domain: int=0):
        super().__init__()

        with h5.File(path, 'r') as f:
            u = f['u'][:] 
            x = f['x'][:]
            y = f['y'][:]
            bc_indices = f['bc_indices'][:]

        if indices is not None:
            if len(indices) == 2:
                u = u[indices[0]:indices[1]]
                bc_indices = bc_indices[indices[0]:indices[1]]
            else:
                u = u[indices]
                bc_indices = bc_indices[indices]
        self.u = u[:, T0:T1:step_size]
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.X_star = np.asarray(np.stack([self.X.flatten(), self.Y.flatten()], axis=-1), dtype=np.float32)
        self.bc_indices = np.asarray(bc_indices[:], dtype=np.float32)
        self.domain = domain

        self.sdf_data = []
        for i in range(len(self.bc_indices)):
            sdf = create_sdf_from_coords(bc_indices[i], self.X.shape, ((self.x[0], self.x[-1]), (self.y[0], self.y[-1])), clip_zero=True) # (nx x ny)
            self.sdf_data.append(sdf)
            
    def __len__(self):
        return self.bc_indices.shape[0]

    def __getitem__(self, idx):
        return DataDict(
            domain=self.domain,
            X=self.X_star,
            functions = {
                "bluff_body": np.asarray(self.bc_indices[idx], dtype=np.float32),
            },
            aux={
                'sdf': self.sdf_data[idx].reshape(-1, 1)
            }
        )
    
    def compute_mean_std_fields(self):
        return np.mean(self.u, axis=(0, 1, 3, 4)), np.std(self.u, axis=(0, 1, 3, 4))
    
    def compute_mean_std_coords(self):
        return np.array([np.mean(self.x), np.mean(self.y)]), np.array([np.std(self.x), np.std(self.y)])
    
    def compute_min_max_coords(self):
        return np.array([np.min(self.x), np.min(self.y)]), np.array([np.max(self.x), np.max(self.y)])
    
    def get_function_specs(self):
        min_x, max_x = self.compute_min_max_coords()
        mean_y, std_y = self.compute_mean_std_fields()
        velocity_specs = FunctionSpec(size=4, has_x=True, has_y=True, min_x=jnp.array(min_x), max_x=jnp.array(max_x),
                                      mean_y=jnp.array(mean_y), std_y=jnp.array(std_y))
        bluff_specs = FunctionSpec(size=2, has_x=True, has_y=False, min_x=jnp.array(min_x), max_x=jnp.array(max_x))
        functions = {
            "velocity": velocity_specs,
            "bluff_body": bluff_specs
        }
        if self.paramed:
            functions['omega'] = FunctionSpec(size=1, has_x=False, has_y=True, mean_y=jnp.mean(self.omega),
                                              std_y=jnp.std(self.omega))
        return functions
    
    def compute_mean_std_sdf(self):
        return np.mean(self.sdf_data), np.std(self.sdf_data)


class VortexICLDataset(VortexWindowDataset):
    """Episodic in-context dataset: sparse labeled points become a ``support`` function.

    Each item keeps query coordinates ``X`` / target ``u``, and adds
    ``functions['support'] = concat([x, u])`` at K support locations (TabICL-style
    context). ``label_mask`` is True on query points so the loss ignores support
    by default.

    For fast training, set ``n_query`` / ``n_velocity`` to fixed counts so JAX
    keeps a static shape while avoiding full-grid self-attention (N≈15k).
    """

    def __init__(
        self,
        path,
        window_size: int,
        step_size: int,
        T0: int = 0,
        T1: int = -1,
        indices=None,
        paramed=False,
        domain: int = 0,
        k_support: int = 64,
        k_support_min: int = None,
        k_support_max: int = None,
        support_rate: float = None,
        query_on_support: bool = False,
        resample_support: bool = True,
        n_query: int = None,
        n_velocity: int = None,
        velocity_subsample_rate: float = None,
        random_seed: int = 0,
    ):
        super().__init__(path, window_size, step_size, T0, T1, indices, paramed, domain)
        self.query_on_support = query_on_support
        self.resample_support = resample_support
        self.n_query = n_query
        self.n_velocity = n_velocity
        self.velocity_subsample_rate = velocity_subsample_rate
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        n_points = self.X_star.shape[0]
        if support_rate is not None:
            k = max(1, int(n_points * support_rate))
            self.k_support_min = k
            self.k_support_max = k
        elif k_support_min is not None or k_support_max is not None:
            self.k_support_min = k_support_min if k_support_min is not None else k_support
            self.k_support_max = k_support_max if k_support_max is not None else k_support
        else:
            self.k_support_min = k_support
            self.k_support_max = k_support

        self.k_support_min = int(max(1, min(self.k_support_min, n_points - 1)))
        self.k_support_max = int(max(self.k_support_min, min(self.k_support_max, n_points - 1)))

        if self.n_query is not None:
            self.n_query = int(min(self.n_query, n_points))
        if self.n_velocity is not None:
            self.n_velocity = int(min(self.n_velocity, n_points))
        elif self.velocity_subsample_rate is not None:
            self.n_velocity = max(1, int(n_points * self.velocity_subsample_rate))

        # Fixed index set used when resample_support=False (e.g. zero-shot adapt).
        self._fixed_support_idx = self._rng.choice(
            n_points, size=self.k_support_max, replace=False
        )

    def _sample_k(self) -> int:
        if self.k_support_min == self.k_support_max:
            return self.k_support_min
        return int(self._rng.integers(self.k_support_min, self.k_support_max + 1))

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        n_points = data["X"].shape[0]
        k = self._sample_k()
        if self.resample_support:
            support_idx = self._rng.choice(n_points, size=k, replace=False)
        else:
            support_idx = self._fixed_support_idx[:k]

        # Support is always taken from the full field (ICL context).
        support = np.concatenate(
            [data["X"][support_idx], data["u"][support_idx]], axis=-1
        ).astype(np.float32)
        data["functions"]["support"] = support

        # Optional fixed-size velocity keys (static shape for JIT).
        if self.n_velocity is not None:
            vel_idx = self._rng.choice(n_points, size=self.n_velocity, replace=False)
            data["functions"]["velocity"] = data["functions"]["velocity"][vel_idx]

        # Optional fixed-size query subsample — dominant speedup (self-attn ~ O(N^2)).
        if self.n_query is not None:
            query_idx = self._rng.choice(n_points, size=self.n_query, replace=False)
            data["X"] = data["X"][query_idx]
            data["u"] = data["u"][query_idx]
            if self.query_on_support:
                label_mask = np.ones((self.n_query, 1), dtype=bool)
            else:
                label_mask = (~np.isin(query_idx, support_idx))[:, None]
            data["label_mask"] = label_mask
        else:
            label_mask = np.ones((n_points, 1), dtype=bool)
            if not self.query_on_support:
                label_mask[support_idx] = False
            data["label_mask"] = label_mask

        return data

    def get_function_specs(self):
        functions = super().get_function_specs()
        min_x, max_x = self.compute_min_max_coords()
        mean_y, std_y = self.compute_mean_std_fields()
        n_out = 2 * (self.window_size - 1)
        mean_out = np.tile(mean_y, [self.window_size - 1, 1]).reshape([-1])
        std_out = np.tile(std_y, [self.window_size - 1, 1]).reshape([-1])
        assert mean_out.shape[0] == n_out
        functions["support"] = FunctionSpec(
            size=2 + n_out,
            has_x=True,
            has_y=True,
            min_x=jnp.array(min_x),
            max_x=jnp.array(max_x),
            mean_y=jnp.array(mean_out),
            std_y=jnp.array(std_out),
        )
        return functions

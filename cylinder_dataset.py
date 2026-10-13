import sys
import os

import jax
import jax.numpy as jnp
from torch.utils.data import Dataset
import numpy as np
import h5py as h5
import torch

class CylinderDatasetTorch(Dataset):

    def __init__(self, path, start_time, num_time_steps, histpry_steps=1, future_steps=0, indices=None, paramed=False):
        self.num_time_steps = num_time_steps
        self.history_steps = histpry_steps
        self.future_steps = future_steps
        self.paramed = paramed
        self.max_start = self.num_time_steps - self.history_steps - self.future_steps
        with h5.File(path) as f:
            u = f['u'][:]
            x = f['x'][:]
            y = f['y'][:]
            omega = f['omega'][:]
            self.cylinder_diameter = f.attrs['length'][:]
            self.inner_steps = f.attrs['dT'][:]
            self.bc_indices = f['bc_indices'][:]

        if indices is not None:
            if len(indices) == 2:
                print(indices)
                u = u[indices[0]:indices[1]]
                omega = omega[indices[0]:indices[1]]    
            else:
                u = u[indices]
                omega = omega[indices]
        
        u = u[:, start_time:start_time + num_time_steps]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")
        X = X.flatten()
        Y = Y.flatten()
        self.dt = 1.0
        t = (np.arange(0, num_time_steps) + start_time) * self.dt * self.inner_steps
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.u = torch.tensor(u)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.omega = torch.tensor(omega)
        self.t = torch.tensor(t)
        self.dt = torch.tensor(self.dt)
        self.dx = torch.tensor(x[1] - x[0])
        self.length = (x[-1], y[-1])
        self.bc_indices = torch.tensor(self.bc_indices)
        if paramed:
            self.node_args = self.omega

    def compute_mean_std_fields(self):
        return torch.mean(self.u, dim=(0, 1, 3, 4)).numpy(), torch.std(self.u, dim=(0, 1, 3, 4)).numpy()
    
    def compute_mean_std_coords(self):
        return torch.tensor([torch.mean(self.x), torch.mean(self.y)]).numpy(), torch.tensor([torch.std(self.x), torch.std(self.y)]).numpy()
    
    def compute_min_max_coords(self):
        return torch.tensor([torch.min(self.x), torch.min(self.y)]).numpy(), torch.tensor([torch.max(self.x), torch.max(self.y)]).numpy()
    
    def get_coordinates(self):
        return self.x, self.y, self.t
    
    def get_trajectory(self, idx):
        return self.u[idx]
        
    def __len__(self):
        return (self.max_start + 1) * len(self.u)

    def __getitem__(self, idx):
        traj_idx = idx // (self.max_start + 1)
        start_idx = idx % (self.max_start + 1)
        history = self.u[traj_idx, start_idx:start_idx + self.history_steps]
        if self.history_steps == 1:
            history = history.squeeze(dim=0)
        return {
            "data": history,
            "t": self.t[start_idx:start_idx + self.history_steps].squeeze(dim=0),
            "time_idx": start_idx,
            "coords": self.X,
            "dt": self.dt,
            "dx": self.dx,
            "idx": traj_idx,
            "solver_args": [self.omega[traj_idx]],
            "node_args": self.omega[traj_idx]
        } if self.paramed else {
            "data": history,
            "t": self.t[start_idx:start_idx + self.history_steps].squeeze(dim=0),
            "time_idx": start_idx,
            "coords": self.X,
            "dt": self.dt,
            "dx": self.dx,
            "idx": traj_idx,
            "solver_args": [self.omega[traj_idx]],        
        }
    
class CylinderTrajDatasetTorch(Dataset):

    def __init__(self, path, start_time, num_time_steps, indices=None, paramed=False):
        self.num_time_steps = num_time_steps
        self.paramed = paramed
        with h5.File(path) as f:
            u = f['u'][:]
            x = f['x'][:]
            y = f['y'][:]
            omega = f['omega'][:]
            # reynold = f['reynold'][:]
            self.cylinder_diameter = f.attrs['length'][:]
            self.inner_steps = f.attrs['dT'][:]
            self.bc_indices = f['bc_indices'][:]

        if indices is not None:
            if len(indices) == 2:
                u = u[indices[0]:indices[1]]
                omega = omega[indices[0]:indices[1]]    
            else:
                u = u[indices]
                omega = omega[indices]        
        u = u[:, start_time:start_time + num_time_steps]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")
        X = X.flatten()
        Y = Y.flatten()
        self.dt = 1.0
        t = (np.arange(0, num_time_steps) + start_time) * self.dt * self.inner_steps
        self.X = torch.tensor(np.stack([X, Y], axis=-1))
        self.u = torch.tensor(u)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.omega = torch.tensor(omega)
        self.t = torch.tensor(t)
        self.dt = torch.tensor(self.dt)
        self.dx = torch.tensor(x[1] - x[0])
        self.length = (x[-1], y[-1])
        self.bc_indices = torch.tensor(self.bc_indices)
        if paramed:
            self.node_args = self.omega

    def compute_mean_std_fields(self):
        return torch.mean(self.u, dim=(0, 1, 3, 4)).numpy(), torch.std(self.u, dim=(0, 1, 3, 4)).numpy()
    
    def compute_mean_std_coords(self):
        return torch.tensor([torch.mean(self.x), torch.mean(self.y)]).numpy(), torch.tensor([torch.std(self.x), torch.std(self.y)]).numpy()
    
    def compute_min_max_coords(self):
        return torch.tensor([torch.min(self.x), torch.min(self.y)]).numpy(), torch.tensor([torch.max(self.x), torch.max(self.y)]).numpy()
    
    def get_coordinates(self):
        return self.x, self.y, self.t
    
    def get_trajectory(self, idx):
        return self.u[idx]
        
    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            "data": self.u[idx],
            "t": self.t,
            "coords": self.X,
            "dt": self.dt,
            "dx": self.dx,
            "idx": idx,
            "solver_args": [self.omega[idx]],
            "node_args": self.omega[idx]
        } if self.paramed else {
            "data": self.u[idx],
            "t": self.t,
            "coords": self.X,
            "dt": self.dt,
            "dx": self.dx,
            "idx": idx,
            "solver_args": [self.omega[idx]],
        }
    
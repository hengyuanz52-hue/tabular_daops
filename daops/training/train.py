import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu
import optax as opt
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Float
from typing import Callable, Optional, Sequence, Any, TypedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.data_utils import DataDict, iterator_sharded_prefetch
from ..utils.callbacks import Callback

class SupervisedTrainer:
    model: eqx.Module
    model_state: PyTree
    optimizer: opt.GradientTransformation
    opt_state: opt.OptState = None
    loss: str = "mse"
    callbacks: Sequence[Callback] = []
    history: dict
    freeze_specs: PyTree
    loss_fn: Callable
    
    def __init__(self,
                 model: eqx.Module,
                 model_state: PyTree,
                 optimizer: opt.GradientTransformation,
                 callbacks: Sequence[Callable] = None,
                 opt_state: opt.OptState = None,
                 freeze_specs: PyTree=None,
                 loss_fn: Callable=None,
                 loss: str="mse"):
        self.model = model
        self.model_state = model_state
        self.optimizer = optimizer
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks
        if opt_state is None:
            opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.opt_state = opt_state
        self.history = {'loss': []}
        self._freeze_specs = freeze_specs if freeze_specs else jtu.tree_map(lambda _: True, model)
        self._loss_fn = loss_fn if loss_fn else self.default_loss
        self.loss = loss

    @staticmethod
    def default_loss(model_diff: eqx.Module, model_static:eqx.Module, model_state: PyTree, batch: DataDict, *, key=PRNGKeyArray):
        model = eqx.combine(model_diff, model_static)
        data, masks = batch
        X = data['X']
        functions = data["functions"]
        u = data['u']
        mask_query = masks['X']
        masks_functions = masks['functions']
        label_mask = data.get('label_mask', None)
        preds = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
        if label_mask is not None:
            return jnp.mean(((preds - u) * mask_query * label_mask)**2), model_state
        else:
            return jnp.mean(((preds - u) * mask_query)**2), model_state
    
    def _step(self, model: eqx.Module, model_state: PyTree, optimizer: opt.GradientTransformation, opt_state: opt.OptState, 
              batch: DataDict, sharding_model: jax.sharding.Sharding, sharding_data: jax.sharding.Sharding, *, key: PRNGKeyArray):
        model, model_state, opt_state = eqx.filter_shard((model, model_state, opt_state), sharding_model)
        batch = jax.tree_util.tree_map(lambda x: eqx.filter_shard(x, sharding_data), batch)
        model_diff, model_static = eqx.partition(model, self._freeze_specs)
        loss_grad_fn = eqx.filter_value_and_grad(self._loss_fn, has_aux=True)
        (loss_value, model_state), grads = loss_grad_fn(model_diff, model_static, model_state, batch, key=key)
        updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, model_state, opt_state, loss_value
    
    def fit(self, dataloader_train: DataLoader, epochs: int, *, key: PRNGKeyArray, **kwargs):
        devices = jax.devices()
        num_batches = len(dataloader_train)
        num_devices = len(devices)
        assert dataloader_train.batch_size % num_devices == 0, "Batch size should be divisible by number of devices"
        mesh = jax.make_mesh((num_devices, ), ('shard', ), devices=devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(('shard', ))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)

        jitted_step = eqx.filter_jit(self._step, donate="all")
        model = self.model
        model_state = self.model_state
        optimizer_state = self.opt_state

        model, model_state, optimizer_state = eqx.filter_shard(
            (model, model_state, optimizer_state), sharding_model
        )

        for epoch in range(epochs):
            model = eqx.nn.inference_mode(model, False)
            mean_loss = 0.0
            prefetcher = iterator_sharded_prefetch(iter(dataloader_train), 2, sharding_data)
            for batch in prefetcher:
                key, subkey = jr.split(key)
                model, model_state, optimizer_state, loss_value = jitted_step(model, model_state, self.optimizer, \
                                                                              optimizer_state, batch, sharding_model,\
                                                                                sharding_data, key=subkey)
                mean_loss += loss_value
            mean_loss /= num_batches
            self.history['loss'].append(mean_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss}")
            for callback in self.callbacks:
                model, model_state, optimizer_state, self.history = callback(model, model_state, \
                                                                        optimizer_state, self.history, epoch)
            self.model = model
            self.model_state = model_state
            self.opt_state = optimizer_state
        
        return self.model, self.model_state, self.opt_state

class ReptileMetaTrainer(SupervisedTrainer):
    inner_steps: int
    num_tasks: int
    meta_optimizer: opt.GradientTransformation
    meta_opt_state: opt.OptState = None

    def __init__(self, model, model_state, optimizer, meta_optimizer, inner_steps, \
                 num_tasks, callbacks = None, opt_state = None, meta_opt_state=None, \
                    freeze_specs = None, loss_fn = None):
        super().__init__(model, model_state, optimizer, callbacks, opt_state, freeze_specs, loss_fn)
        self.inner_steps = inner_steps
        self.num_tasks = num_tasks
        self.meta_optimizer = meta_optimizer
        if meta_opt_state is None:
            meta_opt_state = self.meta_optimizer.init(eqx.filter((model, model_state), eqx.is_array))
        self.meta_opt_state = meta_opt_state
 
    def meta_update(self, meta_model, meta_state, task_model, task_state, meta_opt_state):

        meta, static = eqx.partition((meta_model, meta_state), eqx.is_array)
        task, _ = eqx.partition((task_model, task_state), eqx.is_array)
        meta_grad = jtu.tree_map(lambda a, b: a - b, meta, task)
        updates, meta_opt_state = self.meta_optimizer.update(meta_grad, \
                                                             meta_opt_state, params=eqx.filter(meta, eqx.is_array))
        meta = eqx.apply_updates(meta, updates)
        meta_model, meta_state = eqx.combine(meta, static)

        return meta_model, meta_state, meta_opt_state

    def fit(self, dataloaders, epochs, *, key, **kwargs):
        """
        Fit the model using Reptile meta-learning across multiple tasks.
        Args:
            dataloaders (list of DataLoader): List of DataLoader objects, each corresponding to a different task.
            epochs (int): Number of epochs to train.
            key (PRNGKeyArray): JAX random key for randomness.
        """

        devices = jax.devices()
        num_devices = len(devices)
        mesh = jax.make_mesh((num_devices, ), ('shard', ), devices=devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(('shard', ))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
        jitted_step = eqx.filter_jit(self._step, donate="none")
        jitted_meta_update = eqx.filter_jit(self.meta_update, donate="all")

        meta_model = self.model
        meta_model_state = self.model_state
        meta_opt_state = self.meta_opt_state
        inner_opt_state = self.opt_state

        meta_model, meta_model_state, meta_opt_state, inner_opt_state = eqx.filter_shard(
            (meta_model, meta_model_state, meta_opt_state, inner_opt_state), sharding_model)
        prefetchers = [iterator_sharded_prefetch(iter(dataloader), 2, sharding_data) for dataloader in dataloaders]
        
        for epoch in range(epochs):
            key, subkey = jr.split(key)
            # task = epoch % self.num_tasks
            task = int(jr.randint(subkey, (), 0, self.num_tasks))
            task_model, task_model_state, task_opt_state = meta_model, meta_model_state, inner_opt_state
            task_model = eqx.nn.inference_mode(task_model, False)
            mean_loss = 0.0
            prefetcher_task = prefetchers[task]        
            for _ in range(self.inner_steps):
                try:
                    batch = next(prefetcher_task)
                except StopIteration:
                    prefetchers[task] = iterator_sharded_prefetch(iter(dataloaders[task]), 2, sharding_data)
                    prefetcher_task = prefetchers[task]
                    batch = next(prefetcher_task)
                key, subkey = jr.split(key)
                task_model, task_model_state, task_opt_state, loss_value = jitted_step(task_model,
                                                                                              task_model_state, 
                                                                                              self.optimizer, 
                                                                                              task_opt_state, 
                                                                                              batch, 
                                                                                              sharding_model, 
                                                                                              sharding_data, 
                                                                                              key=subkey)
                # mean_loss += loss_value
                

            meta_model, meta_model_state, meta_opt_state = jitted_meta_update(meta_model, meta_model_state, task_model, task_model_state, \
                                                                         meta_opt_state)
            mean_loss = loss_value
            self.history['loss'].append(mean_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Task {task + 1}/{self.num_tasks}, Loss: {mean_loss}")
            for callback in self.callbacks:
                meta_model, meta_model_state, meta_opt_state, self.history = callback(meta_model, meta_model_state,
                                                                        meta_opt_state, self.history, epoch)
            self.model = meta_model
            self.model_state = meta_model_state
            self.meta_opt_state = meta_opt_state

        return self.model, self.model_state, self.meta_opt_state
  
class SupervisedGeometricTrainer(SupervisedTrainer):

    def __init__(self, model, model_state, optimizer, callbacks = None, opt_state = None, 
                 freeze_specs_sim = None, freeze_specs_geo = None, loss='mse',
                 loss_fn_sim = None, loss_fn_geo = None):
        super().__init__(model, model_state, optimizer, callbacks, opt_state)
        self._freeze_specs_sim = freeze_specs_sim if freeze_specs_sim else jtu.tree_map(lambda _: True, model)
        self._freeze_specs_geo = freeze_specs_geo if freeze_specs_geo else jtu.tree_map(lambda _: True, model)
        self._loss_fn_sim = loss_fn_sim if loss_fn_sim else self.loss_fn_sim
        self._loss_fn_geo = loss_fn_geo if loss_fn_geo else self.loss_fn_geo
        self.history = {'loss_sim': [], 'loss_geo': []}
        self.loss = loss

    def loss_fn_geo(self, model_diff: eqx.Module, model_static:eqx.Module, model_state: PyTree, batch: DataDict, *, key=PRNGKeyArray):
        model = eqx.combine(model_diff, model_static)
        data, masks = batch
        X = data['X']
        functions = data["functions"]
        sdf = data['aux']['sdf']
        mask_query = masks['X']
        masks_functions = masks['functions']
        _, aux = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
        if self.loss == 'mse':
            return jnp.mean(((sdf - aux) * mask_query)**2), model_state
        elif self.loss == 'nmse':
            print("nmse geo")
            return jnp.mean(jnp.linalg.norm((sdf - aux) * mask_query, axis=1)**2 / jnp.linalg.norm(sdf * mask_query, axis=1)**2), model_state

    def loss_fn_sim(self, model_diff: eqx.Module, model_static:eqx.Module, model_state: PyTree, batch: DataDict, *, key=PRNGKeyArray):
        model = eqx.combine(model_diff, model_static)
        data, masks = batch
        X = data['X']
        functions = data["functions"]
        u = data['u']
        mask_query = masks['X']
        masks_functions = masks['functions']
        label_mask = data.get('label_mask', None)
        preds, _ = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(X, functions, mask_query, masks_functions, key)
        if self.loss == 'mse':
            if label_mask is not None:
                print("masked label")
                return jnp.mean(((preds - u) * mask_query * label_mask)**2), model_state
            else:
                return jnp.mean(((preds - u) * mask_query)**2), model_state
        elif self.loss == 'nmse':
            print("nmse sim")
            if label_mask is not None:
                print("masked label")
                return jnp.mean(jnp.linalg.norm((preds - u) * mask_query * label_mask, axis=1)**2 / jnp.linalg.norm(u * mask_query * label_mask, axis=1)**2), model_state
            else:
                return jnp.mean(jnp.linalg.norm((preds - u) * mask_query, axis=1)**2 / jnp.linalg.norm(u * mask_query, axis=1)**2), model_state

    def _step(self, model: eqx.Module, model_state: PyTree, opt_state: opt.OptState, batch_sim: DataDict, batch_geo: DataDict,
              gamma: float, sharding_model: jax.sharding.Sharding, sharding_data: jax.sharding.Sharding, *, key: PRNGKeyArray):
        
        model, model_state, opt_state = eqx.filter_shard((model, model_state, opt_state), sharding_model)
        batch_sim = jax.tree_util.tree_map(lambda x: eqx.filter_shard(x, sharding_data), batch_sim)
        batch_geo = jax.tree_util.tree_map(lambda x: eqx.filter_shard(x, sharding_data), batch_geo)

        model_diff_geo, model_static_geo = eqx.partition(model, self._freeze_specs_geo)
        model_diff_sim, model_static_sim = eqx.partition(model, self._freeze_specs_sim)
        
        loss_grad_fn_sim = eqx.filter_value_and_grad(self._loss_fn_sim, has_aux=True)
        loss_grad_fn_geo = eqx.filter_value_and_grad(self._loss_fn_geo, has_aux=True)

        (loss_value_sim, model_state), grads_sim = loss_grad_fn_sim(model_diff_sim, model_static_sim, model_state, batch_sim, key=key)
        (loss_value_geo, model_state), grads_geo = loss_grad_fn_geo(model_diff_geo, model_static_geo, model_state, batch_geo, key=key)
        grads = jax.tree.map(lambda a, b: (1. - gamma) * a + gamma * b, grads_sim, grads_geo)
        updates, opt_state = self.optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))

        model = eqx.apply_updates(model, updates)
        return model, model_state, opt_state, {"loss_sim": loss_value_sim, "loss_geo": loss_value_geo} 
    
    def fit(self, dataloader_sim: DataLoader, dataloader_geo: DataLoader, epochs: int, gamma: float, *, key: PRNGKeyArray, **kwargs):
        devices = jax.devices()
        num_batches_sim = len(dataloader_sim)
        num_batches_geo = len(dataloader_geo)
        num_devices = len(devices)
        assert dataloader_sim.batch_size % num_devices == 0, "Batch size should be divisible by number of devices"
        assert dataloader_geo.batch_size % num_devices == 0, "Batch size should be divisible by number of devices"
        mesh = jax.make_mesh((num_devices, ), ('shard', ), devices=devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(('shard', ))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)

        jitted_step = eqx.filter_jit(self._step, donate="all")
        model = self.model
        model_state = self.model_state
        optimizer_state = self.opt_state

        model, model_state, optimizer_state = eqx.filter_shard(
            (model, model_state, optimizer_state), sharding_model
        )

        prefetcher_geo = iterator_sharded_prefetch(iter(dataloader_geo), 2, sharding_data)
        for epoch in range(epochs):
            model = eqx.nn.inference_mode(model, False)
            mean_loss_sim = 0.0
            mean_loss_geo = 0.0
            prefetcher_sim = iterator_sharded_prefetch(iter(dataloader_sim), 2, sharding_data)
            for batch_sim in prefetcher_sim:
                try:
                    batch_geo = next(prefetcher_geo)
                except StopIteration:
                    prefetcher_geo = iterator_sharded_prefetch(iter(dataloader_geo), 2, sharding_data)
                    batch_geo = next(prefetcher_geo)
                key, subkey = jr.split(key)
                model, model_state, optimizer_state, loss_values = jitted_step(model, model_state,
                                                                              optimizer_state, batch_sim, batch_geo,
                                                                              gamma, sharding_model,
                                                                              sharding_data, key=subkey)
                mean_loss_sim += loss_values["loss_sim"]
                mean_loss_geo += loss_values["loss_geo"]
            mean_loss_sim /= num_batches_sim
            mean_loss_geo /= num_batches_geo
            self.history['loss_sim'].append(mean_loss_sim)
            self.history['loss_geo'].append(mean_loss_geo)
            print(f"Epoch {epoch + 1}/{epochs}, Loss Sim: {mean_loss_sim}, Loss Geo: {mean_loss_geo}")
            for callback in self.callbacks:
                model, model_state, optimizer_state, self.history = callback(model, model_state,
                                                                        optimizer_state, self.history, epoch)
            self.model = model
            self.model_state = model_state
            self.opt_state = optimizer_state    

        return self.model, self.model_state, self.opt_state

class ReptileMetaGeoSimTrainer(SupervisedGeometricTrainer):
    inner_steps: int
    num_geo_tasks: int
    num_sim_tasks: int
    
    def __init__(self, model, model_state, optimizer, meta_optimizer, inner_steps, \
                 num_geo_tasks, num_sim_tasks, callbacks = None, opt_state = None, meta_opt_state=None, \
                 freeze_specs_sim = None, freeze_specs_geo = None, loss='mse',):
        super().__init__(model, model_state, optimizer, callbacks, opt_state, 
                         freeze_specs_sim, freeze_specs_geo, loss)
        
        self.inner_steps = inner_steps
        self.num_geo_tasks = num_geo_tasks
        self.num_sim_tasks = num_sim_tasks
        self.meta_optimizer = meta_optimizer
        if meta_opt_state is None:
            meta_opt_state = self.meta_optimizer.init(eqx.filter((model, model_state), eqx.is_array))
        self.meta_opt_state = meta_opt_state

    # Meta update function remains the same as in ReptileMetaTrainer
    def meta_update(self, meta_model, meta_state, task_model, task_state, meta_opt_state):

        meta, static = eqx.partition((meta_model, meta_state), eqx.is_array)
        task, _ = eqx.partition((task_model, task_state), eqx.is_array)
        meta_grad = jtu.tree_map(lambda a, b: a - b, meta, task)
        updates, meta_opt_state = self.meta_optimizer.update(meta_grad, \
                                                             meta_opt_state, params=eqx.filter(meta, eqx.is_array))
        meta = eqx.apply_updates(meta, updates)
        meta_model, meta_state = eqx.combine(meta, static)
        return meta_model, meta_state, meta_opt_state
    
    def geo_step(self, model, model_state, opt_state, batch_geo, sharding_model, sharding_data, *, key):
        model, model_state, opt_state = eqx.filter_shard((model, model_state, opt_state), sharding_model)
        batch_geo = jax.tree_util.tree_map(lambda x: eqx.filter_shard(x, sharding_data), batch_geo)
        model_diff_geo, model_static_geo = eqx.partition(model, self._freeze_specs_geo)
        loss_grad_fn_geo = eqx.filter_value_and_grad(self._loss_fn_geo, has_aux=True)
        (loss_value_geo, model_state), grads_geo = loss_grad_fn_geo(model_diff_geo, model_static_geo, model_state, batch_geo, key=key)
        updates, opt_state = self.optimizer.update(grads_geo, opt_state, params=eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, model_state, opt_state, loss_value_geo
    
    def sim_step(self, model, model_state, opt_state, batch_sim, sharding_model, sharding_data, *, key):
        model, model_state, opt_state = eqx.filter_shard((model, model_state, opt_state), sharding_model)
        batch_sim = jax.tree_util.tree_map(lambda x: eqx.filter_shard(x, sharding_data), batch_sim)
        model_diff_sim, model_static_sim = eqx.partition(model, self._freeze_specs_sim)
        loss_grad_fn_sim = eqx.filter_value_and_grad(self._loss_fn_sim, has_aux=True)
        (loss_value_sim, model_state), grads_sim = loss_grad_fn_sim(model_diff_sim, model_static_sim, model_state, batch_sim, key=key)
        updates, opt_state = self.optimizer.update(grads_sim, opt_state, params=eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, model_state, opt_state, loss_value_sim
    
    def fit(self, dataloaders_geo: Sequence[DataLoader], dataloaders_sim: Sequence[DataLoader], epochs: int, *, key: PRNGKeyArray, **kwargs):
        devices = jax.devices()
        num_devices = len(devices)
        mesh = jax.make_mesh((num_devices, ), ('shard', ), devices=devices)
        pspec_model = jax.sharding.PartitionSpec()
        pspec_data = jax.sharding.PartitionSpec(('shard', ))
        sharding_model = jax.sharding.NamedSharding(mesh, pspec_model)
        sharding_data = jax.sharding.NamedSharding(mesh, pspec_data)
        jitted_geo_step = eqx.filter_jit(self.geo_step, donate="none")
        jitted_sim_step = eqx.filter_jit(self.sim_step, donate="none")
        jitted_meta_update = eqx.filter_jit(self.meta_update, donate="all")

        meta_model = self.model
        meta_model_state = self.model_state
        meta_opt_state = self.meta_opt_state
        inner_opt_state = self.opt_state

        meta_model, meta_model_state, meta_opt_state, inner_opt_state = eqx.filter_shard(
            (meta_model, meta_model_state, meta_opt_state, inner_opt_state), sharding_model)
        
        prefetchers_geo = [iterator_sharded_prefetch(iter(dataloader), 2, sharding_data) for dataloader in dataloaders_geo]
        prefetchers_sim = [iterator_sharded_prefetch(iter(dataloader), 2, sharding_data) for dataloader in dataloaders_sim]

        for epoch in range(epochs):
            key, subkey = jr.split(key)
            task = int(jr.randint(subkey, (), 0, self.num_geo_tasks + self.num_sim_tasks))
            task_model, task_model_state, task_opt_state = meta_model, meta_model_state, inner_opt_state
            task_model = eqx.nn.inference_mode(task_model, False)
            mean_loss_geo = 0.0
            mean_loss_sim = 0.0
            for _ in range(self.inner_steps):
                if task < self.num_geo_tasks:
                    try:
                        batch_geo = next(prefetchers_geo[task])
                    except StopIteration:
                        prefetchers_geo[task] = iterator_sharded_prefetch(iter(dataloaders_geo[task]), 2, sharding_data)
                        batch_geo = next(prefetchers_geo[task])
                    key, subkey = jr.split(key)
                    task_model, task_model_state, task_opt_state, loss_value_geo = jitted_geo_step(task_model,
                                                                                  task_model_state, 
                                                                                  task_opt_state, 
                                                                                  batch_geo, 
                                                                                  sharding_model, 
                                                                                  sharding_data, 
                                                                                  key=subkey)
                    mean_loss_geo = loss_value_geo
                else:
                    task_sim = task - self.num_geo_tasks
                    try:
                        batch_sim = next(prefetchers_sim[task_sim])
                    except StopIteration:
                        prefetchers_sim[task_sim] = iterator_sharded_prefetch(iter(dataloaders_sim[task_sim]), 2, sharding_data)
                        batch_sim = next(prefetchers_sim[task_sim])
                    key, subkey = jr.split(key)
                    task_model, task_model_state, task_opt_state, loss_value_sim = jitted_sim_step(task_model,
                                                                                  task_model_state, 
                                                                                  task_opt_state, 
                                                                                  batch_sim, 
                                                                                  sharding_model, 
                                                                                  sharding_data, 
                                                                                  key=subkey)
                    mean_loss_sim = loss_value_sim
        
            meta_model, meta_model_state, meta_opt_state = jitted_meta_update(meta_model, meta_model_state, task_model, task_model_state, \
                                                                         meta_opt_state)
            if task < self.num_geo_tasks:
                self.history['loss_geo'].append(mean_loss_geo)
                self.history['loss_sim'].append(jnp.nan)
                print(f"Epoch {epoch + 1}/{epochs}, Geo Task {task + 1}/{self.num_geo_tasks}, Loss Geo: {mean_loss_geo}")
            else:
                self.history['loss_sim'].append(mean_loss_sim)
                self.history['loss_geo'].append(jnp.nan)
                print(f"Epoch {epoch + 1}/{epochs}, Sim Task {task_sim + 1}/{self.num_sim_tasks}, Loss Sim: {mean_loss_sim}")
                
            for callback in self.callbacks:
                meta_model, meta_model_state, meta_opt_state, self.history = callback(meta_model, meta_model_state,
                                                                        meta_opt_state, self.history, epoch)
            self.model = meta_model
            self.model_state = meta_model_state
            self.meta_opt_state = meta_opt_state    
        return self.model, self.model_state, self.meta_opt_state


class EpisodicICLTrainer(SupervisedTrainer):
    """Multi-task episodic trainer for operator in-context learning.

    Each step samples a task dataloader, runs a forward pass with
    ``functions['support']`` as labeled context, and updates shared weights.
    No Reptile inner loop — adaptation is entirely in-context.
    """

    num_tasks: int

    def __init__(
        self,
        model,
        model_state,
        optimizer,
        num_tasks,
        callbacks=None,
        opt_state=None,
        freeze_specs=None,
        loss_fn=None,
        loss="nmse",
        steps_per_epoch: int = None,
        log_every: int = 1,
    ):
        super().__init__(
            model,
            model_state,
            optimizer,
            callbacks,
            opt_state,
            freeze_specs,
            loss_fn,
            loss,
        )
        self.num_tasks = num_tasks
        self.steps_per_epoch = steps_per_epoch
        self.log_every = log_every
        if loss_fn is None:
            self._loss_fn = self.icl_loss_fn

    def icl_loss_fn(
        self,
        model_diff: eqx.Module,
        model_static: eqx.Module,
        model_state: PyTree,
        batch: DataDict,
        *,
        key=PRNGKeyArray,
    ):
        model = eqx.combine(model_diff, model_static)
        data, masks = batch
        X = data["X"]
        functions = data["functions"]
        u = data["u"]
        mask_query = masks["X"]
        masks_functions = masks["functions"]
        label_mask = data.get("label_mask", None)
        preds = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None))(
            X, functions, mask_query, masks_functions, key
        )
        # GeoGNOT returns (u_hat, geo); plain GNOT returns u_hat only.
        if isinstance(preds, tuple):
            preds = preds[0]

        weight = mask_query
        if label_mask is not None:
            weight = weight * label_mask

        if self.loss == "mse":
            return jnp.mean(((preds - u) * weight) ** 2), model_state
        # nmse
        num = jnp.linalg.norm((preds - u) * weight, axis=1) ** 2
        den = jnp.linalg.norm(u * weight, axis=1) ** 2
        return jnp.mean(num / jnp.maximum(den, 1e-8)), model_state

    def _step(
        self,
        model: eqx.Module,
        model_state: PyTree,
        optimizer: opt.GradientTransformation,
        opt_state: opt.OptState,
        batch: DataDict,
        sharding_model: jax.sharding.Sharding,
        sharding_data: jax.sharding.Sharding,
        *,
        key: PRNGKeyArray,
    ):
        """Like SupervisedTrainer._step, but zero-fills frozen/None grads for optax."""
        model, model_state, opt_state = eqx.filter_shard(
            (model, model_state, opt_state), sharding_model
        )
        batch = jax.tree_util.tree_map(lambda x: eqx.filter_shard(x, sharding_data), batch)
        model_diff, model_static = eqx.partition(model, self._freeze_specs)
        loss_grad_fn = eqx.filter_value_and_grad(self._loss_fn, has_aux=True)
        (loss_value, model_state), grads = loss_grad_fn(
            model_diff, model_static, model_state, batch, key=key
        )
        # Frozen / non-array leaves are None. Only zero-fill where a real param exists.
        params = eqx.filter(model, eqx.is_array)
        grads = jtu.tree_map(
            lambda g, p: (
                None if p is None else (jnp.zeros_like(p) if g is None else g)
            ),
            grads,
            params,
            is_leaf=lambda x: x is None,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        model = eqx.apply_updates(model, updates)
        return model, model_state, opt_state, loss_value

    def fit(self, dataloaders: Sequence[DataLoader], epochs: int, *, key: PRNGKeyArray, **kwargs):
        devices = jax.devices()
        num_devices = len(devices)
        assert all(
            dl.batch_size % num_devices == 0 for dl in dataloaders
        ), "Batch size should be divisible by number of devices"

        mesh = jax.make_mesh((num_devices,), ("shard",), devices=devices)
        sharding_model = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        sharding_data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("shard",)))

        jitted_step = eqx.filter_jit(self._step, donate="all")
        model = self.model
        model_state = self.model_state
        optimizer_state = self.opt_state
        model, model_state, optimizer_state = eqx.filter_shard(
            (model, model_state, optimizer_state), sharding_model
        )

        prefetchers = [
            iterator_sharded_prefetch(iter(dl), 2, sharding_data) for dl in dataloaders
        ]
        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = max(len(dl) for dl in dataloaders)

        for epoch in range(epochs):
            model = eqx.nn.inference_mode(model, False)
            mean_loss = 0.0
            for step in range(steps_per_epoch):
                key, subkey = jr.split(key)
                task = int(jr.randint(subkey, (), 0, self.num_tasks))
                try:
                    batch = next(prefetchers[task])
                except StopIteration:
                    prefetchers[task] = iterator_sharded_prefetch(
                        iter(dataloaders[task]), 2, sharding_data
                    )
                    batch = next(prefetchers[task])
                key, subkey = jr.split(key)
                if epoch == 0 and step == 0:
                    print("JIT compiling first step...", flush=True)
                model, model_state, optimizer_state, loss_value = jitted_step(
                    model,
                    model_state,
                    self.optimizer,
                    optimizer_state,
                    batch,
                    sharding_model,
                    sharding_data,
                    key=subkey,
                )
                mean_loss += loss_value
                # Avoid per-step device_get (GPU sync); only sync when logging.
                if self.log_every and (step + 1) % self.log_every == 0:
                    print(
                        f"  epoch {epoch + 1} step {step + 1}/{steps_per_epoch} "
                        f"task={task} loss={float(jax.device_get(loss_value)):.6e}",
                        flush=True,
                    )
            mean_loss = float(jax.device_get(mean_loss)) / steps_per_epoch
            self.history["loss"].append(mean_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss}", flush=True)
            for callback in self.callbacks:
                model, model_state, optimizer_state, self.history = callback(
                    model, model_state, optimizer_state, self.history, epoch
                )
            self.model = model
            self.model_state = model_state
            self.opt_state = optimizer_state

        return self.model, self.model_state, self.opt_state


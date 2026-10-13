"""
Callback functions for training Equinox models. Each callback function should have the following signature:
    Args:
    - model: eqx.Module, the model to be trained
    - model_state: PyTree, the state of the model
    - opt_state: optax.OptState, the optimizer state
    - history: Dict[str, ArrayLike], the history of the training process
    - epoch: int, the current epoch number
    Returns:
    - Tuple containing the updated model, optimizer state, history, and training configuration
"""

import os, sys
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from typing import Callable, Dict, Tuple, Any
from jaxtyping import ArrayLike, PRNGKeyArray, Array, PyTree
from .serial import *
from pathlib import Path
from datetime import datetime

class Callback:
    """
    Base class for all callbacks. Child classes should implement the `call` method.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

    def call(self, model: eqx.Module, model_state: PyTree, opt_state: optax.OptState, history: Dict[str, ArrayLike],
              epoch: int) \
        -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        The main method to be implemented in the child class.

        Args:
        - model: eqx.Module, the model to be trained
        - opt_state: optax.OptState, the optimizer state
        - history: Dict[str, ArrayLike], the history of the training process
        - training_config: Dict[str, ArrayLike], the configuration of the training process
        - epoch: int, the current epoch number

        Returns:
        - Tuple containing the updated model, optimizer state, history, and training configuration
        """
        raise NotImplementedError("The `call` method should be implemented in the child class.")

class CheckpointCallback(Callback):
    """
    A callback class for saving the model and optimizer state at specified intervals.
    """
    def __init__(self, checkpoint_dir: str, hyper_params: Dict, add_time: bool=False, save_every: int=50):
        """
        Initializes the CheckpointCallback instance.

        Args:
        - checkpoint_dir: str, the directory to save the checkpoint.
        - hyper_params: Dict, hyperparameters to be saved along with the model.
        - save_every: int, the frequency of saving the checkpoint. Default is 10 (save every 10 epochs).
        """
        self.hyper_params = hyper_params
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        if add_time:
            now = datetime.now()
            self.checkpoint_dir = os.path.join(checkpoint_dir, now.strftime("%Y-%m-%d_%H-%M-%S"))
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def call(self, model: eqx.Module, model_state: PyTree, opt_state: optax.OptState, history: Dict[str, ArrayLike],
             epoch: int) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        Saves the model and optimizer state at the specified interval and returns the inputs unchanged.

        Args:
        - model: eqx.Module, the model being trained.
        - opt_state: optax.OptState, the current optimizer state.
        - history: Dict[str, ArrayLike], a dictionary to store training history.
        - training_config: Dict[str, ArrayLike], a dictionary containing training configurations.
        - epoch: int, the current epoch count.

        Returns:
        - Tuple containing the model, optimizer state, history, and training configuration unchanged.
        """
        if epoch % self.save_every == 0:
            path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.eqx")
            path_opt = os.path.join(self.checkpoint_dir, f"opt_state_epoch_{epoch}.opx")
            path_hist = os.path.join(self.checkpoint_dir, f"history_epoch.csv")
            save_model(path, self.hyper_params, model, model_state)
            # save_opt_state(path_opt, opt_state)
            save_history_csv(path_hist, history)
            print(f"Checkpoint saved at {path}")
        return model, model_state, opt_state, history

class ValidationCallback(Callback):
    """
    A callback class for performing validation at specified intervals.
    """
    def __init__(self, val_dataloader: Any, loss_fn: Callable, validate_every: int=10, history_key: str='val_loss'):
        """
        Initializes the ValidationCallback instance.

        Args:
        - val_dataloader: DataLoader, the dataloader for the validation dataset.
        - loss_fn: Callable, the loss function to be used for validation.
        - validate_every: int, the frequency of validation. Default is 10 (validate every 10 epochs).
        """
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.validate_every = validate_every
        self.history_key = history_key

    def call(self, model: eqx.Module, model_state: PyTree, opt_state: optax.OptState, history: Dict[str, ArrayLike],
             epoch: int) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        Performs validation at the specified interval and updates the history with validation loss.

        Args:
        - model: eqx.Module, the model being trained.
        - opt_state: optax.OptState, the current optimizer state.
        - history: Dict[str, ArrayLike], a dictionary to store training history.
        - training_config: Dict[str, ArrayLike], a dictionary containing training configurations.
        - epoch: int, the current epoch count.

        Returns:
        - Tuple containing the model, optimizer state, history (with updated validation loss), and training configuration unchanged.
        """
        if epoch % self.validate_every == 0:
            model = eqx.nn.inference_mode(model, True)
            total_loss = 0.0
            count = 0
            for batch in self.val_dataloader:
                loss, _ = self.loss_fn(model, model_state, batch=batch, key=jax.random.PRNGKey(0))
                total_loss += loss
                count += 1
            avg_loss = total_loss / count
            history.setdefault(self.history_key, []).append(avg_loss)
            print(f"Epoch {epoch}: Validation Loss ({self.history_key}) = {avg_loss}")
            model = eqx.nn.inference_mode(model, False)
        else:
            history.setdefault(self.history_key, []).append(jnp.nan)
        return model, model_state, opt_state, history
    

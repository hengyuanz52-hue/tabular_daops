from abc import ABC, abstractmethod
import h5py as h5
import numpy as np
from typing import Tuple, Union, Sequence, TypedDict
from jaxtyping import Float, Array, ArrayLike, Bool
from jax.tree_util import tree_map
import jax.numpy as jnp
from flax.jax_utils import prefetch_to_device
import itertools
import equinox as eqx
import jax
import collections
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt

class FunctionSpec(TypedDict):
    size: int
    has_x: bool
    has_y: bool
    min_x: Float[ArrayLike, "D"] = None
    max_x: Float[ArrayLike, "D"] = None
    mean_y: Float[ArrayLike, "F"] = None
    std_y: Float[ArrayLike, "F"] = None

class DataDict(TypedDict):
    domain: int=0
    X: Float[ArrayLike, "N D"]
    functions: dict[str, Union[Float[ArrayLike, "M F"], Float[ArrayLike, "F"]]]={}
    u: Float[ArrayLike, "N C"] = None
    aux: dict[str, Union[Float[ArrayLike, "N C"], Float[ArrayLike, "N T C"]]]={}
    label_mask: Bool[ArrayLike, "N 1"] = None

class BaseDataset(Dataset, ABC):

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> DataDict:
        raise NotImplementedError

def iterator_sharded_prefetch(iterator, size: int, sharding):

    queue = collections.deque()

    def _prefetch(xs):
        return eqx.filter_shard(xs, sharding)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)

def numpy_collate(batch):
    """
    Custom collate function that uses jax.tree_util.tree_map to handle
    arbitrarily nested pytrees.
    
    Args:
        batch: A list of pytrees (e.g., list of nested dictionaries).

    Returns:
        A single pytree with all leaves stacked into NumPy arrays.
    """
    # The `*batch` unpacks the list of dictionaries into separate arguments for tree_map.
    # The lambda function `lambda *leaves: np.stack(leaves)` then receives the
    # corresponding leaf from each dictionary in the batch as a separate argument.
    # For example, it will be called like: np.stack([img_1, img_2, img_3, ...])
    # and then: np.stack([label_1, label_2, ...]), and so on for all leaves.
    return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *batch)

def pad_and_mask(array: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pads a NumPy array with zeros at the end of the first axis and creates
    a 1D boolean mask indicating the padded rows.

    Args:
        array (NDArray): The input NumPy array.
        target_length (int): The desired length of the first axis after padding.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing:
            - The padded array.
            - A 1D boolean mask of shape (target_length,), where `False`
              indicates a row is a padded row, and `True` indicates it
              was part of the original array.
              
    Raises:
        ValueError: If the target_length is smaller than the array's current
                    length on the first axis.
    """
    # Get the current length of the first axis
    original_length = array.shape[0]

    # Check if padding is necessary or if the target length is invalid
    if target_length < original_length:
        raise ValueError(
            "target_length must be greater than or equal to the "
            "array's current length on the first axis."
        )
    
    if target_length == original_length:
        # No padding needed, so the mask indicates no padded rows
        return array, np.ones(original_length, dtype=bool)[:, np.newaxis]

    # --- 1. Pad the Array ---
    # Calculate how much padding is needed
    pad_amount = target_length - original_length
    
    # Create the padding configuration for np.pad
    pad_width = [(0, pad_amount)] + [(0, 0)] * (array.ndim - 1)
    
    padded_array = np.pad(
        array, 
        pad_width, 
        mode='constant', 
        constant_values=0
    )

    # --- 2. Create the 1D Mask ---
    # Create a 1D mask of False's with the target length
    mask = np.ones(target_length, dtype=bool)
    
    # Set the elements corresponding to the padded rows to True
    mask[original_length:] = False

    return padded_array, mask[:, np.newaxis]

def jax_padding_collate(batch):
    """
    Correctly pads pytrees and returns the data and masks as two separate,
    parallel pytrees using a single-pass flatten/unflatten pattern.
    """
    # If the batch is empty, return an empty tuple
    if not batch:
        return ()

    # Get the pytree structure from the first sample
    treedef = jax.tree_util.tree_structure(batch[0])
    
    # Flatten each sample in the batch into a list of leaves
    flat_samples = [jax.tree_util.tree_leaves(sample) for sample in batch]
    
    # Transpose the list of lists: from (num_samples, num_leaves) to (num_leaves, num_samples)
    # Now, `transposed_leaves` is an iterable where each item is a tuple of corresponding leaves
    # from all samples in the batch (e.g., (X_sample1, X_sample2, ...)).
    transposed_leaves = zip(*flat_samples)
    
    # Initialize lists to hold the final batched leaves for each tree
    padded_data_leaves = []
    mask_leaves = []
    
    # Process each group of corresponding leaves
    for leaves in transposed_leaves:
        # Check if padding is applicable for this group of leaves
        is_paddable = all(isinstance(leaf, np.ndarray) and leaf.ndim > 0 for leaf in leaves)
        
        if not is_paddable:
            # For non-paddable items (like scalars), just stack them.
            batched_data = np.stack(leaves)
            # The mask is all False, as no padding was applied.
            batched_mask = np.zeros(batched_data.shape[0], dtype=bool)
        else:
            # Find the max length for this group of leaves
            max_len = max(leaf.shape[0] for leaf in leaves)
            
            # Pad each leaf and collect the results
            temp_padded_list = []
            temp_mask_list = []
            for leaf in leaves:
                padded_leaf, mask = pad_and_mask(leaf, max_len)
                temp_padded_list.append(padded_leaf)
                temp_mask_list.append(mask)
            
            # Stack the collected leaves to form the final batch for this leaf position
            batched_data = np.stack(temp_padded_list)
            batched_mask = np.stack(temp_mask_list)
        
        padded_data_leaves.append(batched_data)
        mask_leaves.append(batched_mask)
        
    # Reconstruct the two pytrees from the lists of batched leaves
    data_batch = jax.tree_util.tree_unflatten(treedef, padded_data_leaves)
    mask_batch = jax.tree_util.tree_unflatten(treedef, mask_leaves)
    
    return data_batch, mask_batch

def create_sdf_from_coords(body_coords, grid_shape, grid_extents, clip_zero=False):
    """
    Generates a Signed Distance Field from a list of body coordinates.

    Args:
        body_coords (np.ndarray): An (N, 2) array of coordinates for points inside the body.
        grid_shape (tuple): The resolution of the grid (nx, ny).
        grid_extents (tuple): The physical domain size ((xmin, xmax), (ymin, ymax)).

    Returns:
        np.ndarray: The calculated SDF on the grid.
    """
    nx, ny = grid_shape
    (xmin, xmax), (ymin, ymax) = grid_extents

    # 1. Create a boolean mask from the coordinates
    mask = np.full(grid_shape, False, dtype=bool)

    # Convert world coordinates to grid indices
    # We need to handle potential out-of-bounds coordinates
    ix = np.round((body_coords[:, 0] - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
    # ix2 = body_coords[:, 0].astype(int)
    # print(ix - ix2)
    iy = np.round((body_coords[:, 1] - ymin) / (ymax - ymin) * (ny - 1)).astype(int)
    # iy = body_coords[:, 1].astype(int)

    # Filter indices that are within the grid bounds
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    mask[ix[valid], iy[valid]] = True

    # 2. Calculate distance to the nearest True point (for outside)
    # The `~` inverts the mask (False becomes True and vice-versa)
    dist_outside = distance_transform_edt(~mask)

    # 3. Calculate distance to the nearest False point (for inside)
    dist_inside = distance_transform_edt(mask)

    # 4. Combine to create the SDF
    # The result is positive outside, negative inside
    sdf = dist_outside - dist_inside
    
    # Optional: Scale distances to match world coordinates if needed
    # This makes the SDF values correspond to physical distance units
    pixel_width = (xmax - xmin) / (nx - 1)
    sdf *= pixel_width

    if clip_zero:
        sdf = np.where(sdf > 0, sdf, 0)

    return sdf


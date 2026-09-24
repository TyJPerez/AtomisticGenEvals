
import numpy as np
from torch.utils.data import Subset

def get_qm9(root="./tmp/qm9"):
    # qm9 is a dataset of small organic molecules
    try:
        from torch_geometric.datasets import QM9
    except ImportError as e:
        raise ImportError("torch_geometric is required to use get_qm9") from e
    # get the full qm9 dataset
    return QM9(root=root)

def get_mp20(split = 'all'):
    # mp20 is a dataset of periodic materials structures
    try:
        from StructureCloud.Datasets import MP20_dataset
    except ImportError as e:
        raise ImportError("StructureCloud is required to use get_mp20") from e
    # get the full mp20 dataset
    return MP20_dataset(split=split, preprocess=True)



def make_splits(dataset, fold_size, seed=42, include_remainder=False, max_splits=None):
    # split the dataset in one sitting set, and as many folds as can be created given the fold size
    # folds are simply non-overlapping slices of the remaining dataset that is not in the fit set
    '''
    args:
        dataset (Dataset object): The dataset to split.
        fold_size  (Int or list of ints): The size of each fold. or if list given, the sizes of each fold.
        include_remainder (Bool): Whether to include the remainder of the dataset in the last fold.
        seed: The random seed for reproducibility.

       
        max_splits (Int): The maximum number of splits to create. if None, all possible splits are created.
            if an int (n) is provided, then the first n splits are created, and the remaining data is given 
            in the last fold if include_remainder is True.
    '''
    n = len(dataset)
    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)

    #make fold splits
    all_fold_indices = indices
    folds = []
    if isinstance(fold_size, list):
        if max_splits is None:
            fold_sizes = fold_size
        else:
            if max_splits <= 0:
                print("WARNING: Invalid max_splits. Returning empty list.")
                return []
            fold_sizes = fold_size[:max_splits]
            if include_remainder and len(fold_size) > max_splits and fold_sizes:
                fold_sizes[-1] = max(0, n - sum(fold_sizes[:-1]))
    else:
        if max_splits is None:
            fold_sizes = [fold_size] * (len(all_fold_indices) // fold_size)
            if include_remainder and len(all_fold_indices) % fold_size:
                fold_sizes.append(len(all_fold_indices) % fold_size)
        else:
            if max_splits <= 0:
                print("WARNING: Invalid max_splits. Returning empty list.")
                return []
            if include_remainder:
                split_count = min(max_splits, (len(all_fold_indices) + fold_size - 1) // fold_size)
                fold_sizes = [fold_size] * (split_count - 1)
                if split_count:
                    fold_sizes.append(len(all_fold_indices) - (fold_size * (split_count - 1)))
            else:
                split_count = min(max_splits, len(all_fold_indices) // fold_size)
                fold_sizes = [fold_size] * split_count

    start = 0
    for size in fold_sizes:
        end = start + size
        fold_indices = all_fold_indices[start:end]
        folds.append(Subset(dataset, fold_indices))
        start = end

    return folds

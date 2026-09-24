'''
Optional dataset classes to help with loading reference datasets

'''

def get_qm9():
    try:
        from StructureCloud.Datasets import TGDataset as tgd
    except ImportError as e:
        raise ImportError("StructureCloud is required to use get_qm9") from e
    # qm9 is a dataset of small organic molecules
    label_fmt = lambda x : x['object']['SMILES'] #set ['labels] = SMILES string in dataset
    tg_dataset = tgd('QM9', split='all', label_fmt=label_fmt)

    return tg_dataset

def get_mp20(split = 'all'):
    # mp20 is a dataset of periodic materials structures
    try:
        from StructureCloud.Datasets import MP20_dataset
    except ImportError as e:
        raise ImportError("StructureCloud is required to use get_mp20") from e
    # get the full mp20 dataset
    return MP20_dataset(split=split, preprocess=True)
    


def get_geom(subsample_name='GEOM10', split='drugs'):
    try:
        from StructureCloud.Datasets import GEOM
    except ImportError as e:
        raise ImportError("StructureCloud is required to use get_geom") from e

    dataset= GEOM(
            subsample_name=subsample_name,
            preprocess=False,
            allow_multiprocessing=True,
            split=split,
        )

    print(f"Dataset length: {len(dataset)}")

    return dataset
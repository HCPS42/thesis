from our_datasets.base import BaseDataset


dataset_map: dict[str, BaseDataset] = {
    "eval": BaseDataset("data/eval/eval.csv"),
}

def get_dataset(dataset_name: str):
    return dataset_map[dataset_name]

__all__ = ["BaseDataset", "get_dataset", "dataset_map"]
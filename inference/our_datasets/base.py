from typing import Any
import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path)
        index = df.to_dict(orient="records")

        self._assert_index_is_valid(index)
        self.index = index

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]]):
        assert isinstance(index, list)

        for entry in index:
            assert isinstance(entry, dict)
            assert "problem" in entry, "No `problem` field"
            assert "answer" in entry, "No `answer` field"
            assert "id" in entry, "No `id` field"

            assert isinstance(entry["answer"], int), "Answer should be a single integer"
            assert isinstance(entry["id"], int), "ID should be an integer"

    def __getitem__(self, idx):
        return self.index[idx]

    def __len__(self):
        return len(self.index)
    
    def filter(self, filter_fn):
        self.index = [entry for entry in self.index if filter_fn(entry)]
        return self
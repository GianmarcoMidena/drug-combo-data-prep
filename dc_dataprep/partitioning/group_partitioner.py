from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from dc_dataprep.partitioning.partitioner import Partitioner


class GroupPartitioner(Partitioner, ABC):
    def _split(self, combinations: pd.DataFrame, n_partitions: int, seed: Optional[int] = None):
        groups = self._groups(combinations)

        for train_indices, test_indices \
                in GroupKFold(n_splits=n_partitions).split(groups, groups=groups):
            yield test_indices

    @abstractmethod
    def _groups(self, combinations: pd.DataFrame) -> np.ndarray:
        ...

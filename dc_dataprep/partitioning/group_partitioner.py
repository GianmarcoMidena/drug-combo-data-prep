from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from dc_dataprep.partitioning.partitioner import Partitioner


class GroupPartitioner(Partitioner, ABC):
    """Group partitioner

    Splits a drug combination dataset into k parts with non-overlapping groups.

    The same group will not appear in two different partitions (the number of
    distinct groups has to be at least equal to the number of partitions).

    The partitions are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.
    """

    def _split(self, combinations: pd.DataFrame, n_partitions: int, seed: Optional[int] = None):
        groups = self._groups(combinations)

        for train_indices, test_indices \
                in GroupKFold(n_splits=n_partitions).split(groups, groups=groups):
            yield test_indices

    @abstractmethod
    def _groups(self, combinations: pd.DataFrame) -> np.ndarray:
        ...

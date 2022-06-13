from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from dc_dataprep.partitioning.partitioner import Partitioner


class GroupPartitioner(Partitioner, ABC):
    """Split a drug combination dataset into n parts with non-overlapping groups.

    The same group will not appear in two different partitions (the number of
    distinct groups has to be at least equal to the number of partitions).

    The partitions are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.
    """

    def _split(self, combinations: pd.DataFrame):
        groups = self._groups(combinations)

        for _, indices in GroupKFold(n_splits=self._n_splits).split(combinations, groups=groups):
            yield indices

    @abstractmethod
    def _groups(self, combinations: pd.DataFrame) -> np.ndarray:
        ...

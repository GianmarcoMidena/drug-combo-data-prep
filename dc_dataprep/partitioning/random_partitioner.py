import pandas as pd
from sklearn.model_selection import KFold

from dc_dataprep.partitioning.partitioner import Partitioner


class RandomPartitioner(Partitioner):
    """Split a drug combination dataset into n parts."""

    def _split(self, combinations: pd.DataFrame):
        for _, indices in KFold(n_splits=self._n_splits, shuffle=False).split(combinations):
            yield indices

from typing import Optional

import pandas as pd
from sklearn.model_selection import KFold

from dc_dataprep.partitioning.partitioner import Partitioner


class RandomPartitioner(Partitioner):
    def _split(self, combinations: pd.DataFrame, n_partitions: int, seed: Optional[int] = None):
        for train_indices, test_indices \
                in KFold(n_splits=n_partitions, shuffle=False).split(combinations):
            yield test_indices

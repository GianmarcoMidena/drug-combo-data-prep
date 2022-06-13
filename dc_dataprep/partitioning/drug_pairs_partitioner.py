import numpy as np
import pandas as pd

from dc_dataprep.partitioning.group_partitioner import GroupPartitioner


class DrugPairsPartitioner(GroupPartitioner):
    """Split a drug combination dataset into n parts with non-overlapping drug pairs.

    The same drug pairs will not appear in two different partitions (the number of
    distinct drug pairs has to be at least equal to the number of partitions).

    The partitions are approximately balanced in the sense that the number of
    distinct drug pairs is approximately the same in each fold.
    """

    def _groups(self, combinations: pd.DataFrame) -> np.ndarray:
        drug_label_min = np.minimum(combinations[self._drug_a_id_label],
                                    combinations[self._drug_b_id_label])
        drug_label_max = np.maximum(combinations[self._drug_a_id_label],
                                    combinations[self._drug_b_id_label])
        groups = pd.DataFrame([drug_label_min, drug_label_max]).T.apply(tuple, axis=1)
        return pd.Categorical(groups).codes

import numpy as np
import pandas as pd

from dc_dataprep.partitioning.group_partitioner import GroupPartitioner


class DrugPairsPartitioner(GroupPartitioner):
    def _groups(self, combinations: pd.DataFrame) -> np.ndarray:
        drug_label_min = np.minimum(combinations[self._drug_a_id_label],
                                    combinations[self._drug_b_id_label])
        drug_label_max = np.maximum(combinations[self._drug_a_id_label],
                                    combinations[self._drug_b_id_label])
        groups = pd.DataFrame([drug_label_min, drug_label_max]).T.apply(tuple, axis=1)
        return pd.Categorical(groups).codes

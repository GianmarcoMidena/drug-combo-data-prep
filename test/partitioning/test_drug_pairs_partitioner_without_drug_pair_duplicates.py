from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import pytest

from dc_dataprep.partitioning.drug_pairs_partitioner import DrugPairsPartitioner
from dc_dataprep.partitioning.partitioner import Partitioner

_N_COMBINATIONS = 50
_N_PARTITIONS = 10
_NON_CONSTANT_DRUG_FEATURES = (2, 3, 1, 5, 16, 10, 21, 18, 15)
_NON_CONSTANT_GENES = (4, 7, 3, 2, 14, 8, 9)
_DRUG_A_ID_LABEL = "drug a smiles"
_DRUG_B_ID_LABEL = "drug b smiles"
_CELL_LINE_ID_LABEL = "cell line name"
_RESPONSE_LABEL = "synergy"
_META_COLUMNS = (_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL, _CELL_LINE_ID_LABEL, _RESPONSE_LABEL)
_FEATURE_COLUMNS = (
    *(f"drug_a_{d1}" for d1 in sorted(_NON_CONSTANT_DRUG_FEATURES)),
    *(f"drug_b_{d2}" for d2 in sorted(_NON_CONSTANT_DRUG_FEATURES)),
    *(f"gene_{g}" for g in sorted(_NON_CONSTANT_GENES)),
    "synergy",
)
_SEED = 3


@pytest.fixture
def partitioner() -> DrugPairsPartitioner:
    yield DrugPairsPartitioner(_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL, _CELL_LINE_ID_LABEL, _RESPONSE_LABEL)


@pytest.fixture
def drug_a_ids() -> pd.Series:
    yield pd.concat(
        (
            pd.Series((1, 3)),
            pd.Series(np.arange(1, _N_COMBINATIONS * 2 + 1, 2)),
            pd.Series((9, 11)),
        ), axis=0, ignore_index=True)


@pytest.fixture
def drug_b_ids(drug_a_ids: pd.Series) -> pd.Series:
    yield pd.concat(
        (
            pd.Series((1, 3)),
            pd.Series(np.arange(_N_COMBINATIONS * 2 + 1, _N_COMBINATIONS * 4 + 1, 2)),
            pd.Series((9, 11)),
        ), axis=0, ignore_index=True)


@pytest.fixture
def cell_line_ids() -> pd.Series:
    yield pd.Series(np.arange(1, _N_COMBINATIONS * 2 + 1, 2))


@pytest.fixture
def synergies() -> pd.Series:
    np.random.seed(_SEED)
    yield pd.Series(np.random.rand(_N_COMBINATIONS))


@pytest.fixture
def original_combinations_without_drug_pair_duplicates(drug_a_ids: pd.Series, drug_b_ids: pd.Series,
                                                       cell_line_ids: pd.Series, synergies: pd.Series) -> pd.DataFrame:
    combinations = pd.DataFrame(
        {
            _DRUG_A_ID_LABEL: drug_a_ids,
            _DRUG_B_ID_LABEL: drug_b_ids,
            _CELL_LINE_ID_LABEL: cell_line_ids,
            _RESPONSE_LABEL: synergies,
        })
    yield combinations.loc[:, _META_COLUMNS]


@pytest.fixture
def drug_features(drug_a_ids: pd.Series, drug_b_ids: pd.Series) -> pd.DataFrame:
    np.random.seed(_SEED)
    upper_drug_id = max(drug_a_ids.max(), drug_b_ids.max()) - 2
    features = pd.DataFrame(np.random.sample((upper_drug_id, len(_NON_CONSTANT_DRUG_FEATURES))),
                            index=range(1, upper_drug_id + 1), columns=_NON_CONSTANT_DRUG_FEATURES)
    for c in set(range(1, np.max(_NON_CONSTANT_DRUG_FEATURES))).difference(_NON_CONSTANT_DRUG_FEATURES):
        features[c] = c
    yield features.sort_index(axis=1)


@pytest.fixture
def cell_line_features(cell_line_ids: pd.Series) -> pd.DataFrame:
    np.random.seed(_SEED)
    upper_cell_line_id = cell_line_ids.max() - 2
    features = pd.DataFrame(np.random.sample((upper_cell_line_id, len(_NON_CONSTANT_GENES))),
                            index=range(1, upper_cell_line_id + 1), columns=_NON_CONSTANT_GENES)
    for c in set(range(1, np.max(_NON_CONSTANT_GENES))).difference(_NON_CONSTANT_GENES):
        features[c] = c
    yield features.sort_index(axis=1)
    
    
@pytest.fixture
def output_dir(tmp_path) -> Path:
    yield tmp_path.joinpath("outputs")


def test_shuffling(partitioner: DrugPairsPartitioner, original_combinations_without_drug_pair_duplicates: pd.DataFrame,
                   drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_without_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    good_counter = total_counter = 0

    for partition_path in _search_meta(output_dir):
        partition = pd.read_csv(partition_path)
        partition_indices = []
        for _, r in partition.iterrows():
            index = original_combinations_without_drug_pair_duplicates[
                np.isclose(original_combinations_without_drug_pair_duplicates, r).all(axis=1)].index[0]
            partition_indices.append(index)
        partition_indices = sorted(partition_indices)
        for i in range(0, len(partition_indices) - 1):
            if partition_indices[i + 1] > partition_indices[i] + 1:
                good_counter += 1
        total_counter += len(partition_indices) - 1

        assert good_counter / total_counter > 0.8


def test_monotherapies_absence(partitioner: DrugPairsPartitioner,
                               original_combinations_without_drug_pair_duplicates: pd.DataFrame,
                               drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_without_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    for partition_path in _search_meta(output_dir):
        meta_p = pd.read_csv(partition_path)
        assert not (meta_p[_DRUG_A_ID_LABEL] == meta_p[_DRUG_B_ID_LABEL]).any()


def _partition(partitioner: Partitioner, combinations: pd.DataFrame, drug_features: pd.DataFrame,
               cell_line_features: pd.DataFrame, output_dir: Path, reverse_drug_pairs: bool = False,
               max_n_cell_lines: Optional[int] = None):
    partitioner.partition(combinations=combinations,
                          drug_features=drug_features,
                          cell_line_features=cell_line_features,
                          n_partitions=_N_PARTITIONS,
                          output_dir=output_dir,
                          reverse_drug_pairs=reverse_drug_pairs,
                          max_n_cell_lines=max_n_cell_lines,
                          seed=_SEED)


def _search_meta(output_dir: Path) -> Generator[Path, None, None]:
    return output_dir.glob("meta_p*.csv")

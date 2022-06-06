from math import floor, ceil
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import pytest

from dc_dataprep.partitioning.partitioner import Partitioner
from dc_dataprep.partitioning.random_partitioner import RandomPartitioner

_N_COMBINATIONS = 50
_N_PARTITIONS = 10
_NON_CONSTANT_DRUG_FEATURES = (1, 2, 3, 5, 10, 14, 16, 18, 21)
_NON_CONSTANT_GENES = (2, 3, 4, 7, 8, 9, 14)
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
def partitioner() -> RandomPartitioner:
    yield RandomPartitioner(_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL, _CELL_LINE_ID_LABEL, _RESPONSE_LABEL)


@pytest.fixture
def drug_a_ids() -> pd.Series:
    yield pd.concat(
        (
            pd.Series((1, 3)),
            pd.Series(np.arange(1, _N_COMBINATIONS + 1, 2)),
            pd.Series((5, 6)),
            pd.Series(np.arange(1, _N_COMBINATIONS + 1, 2)),
            pd.Series((9, 11)),
        ), axis=0, ignore_index=True)


@pytest.fixture
def drug_b_ids() -> pd.Series:
    yield pd.concat(
        (
            pd.Series((1, 3)),
            pd.Series(np.arange(1, _N_COMBINATIONS + 1, 2))[::-1],
            pd.Series((5, 6)),
            pd.Series(np.arange(1, _N_COMBINATIONS + 1, 2))[::-1],
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
def original_combinations_with_drug_pair_duplicates(drug_a_ids: pd.Series, drug_b_ids: pd.Series,
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


def test_nr_output_files(partitioner: RandomPartitioner, original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                         drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    assert len(list(output_dir.iterdir())) == _N_PARTITIONS * 2


def test_output_file_names(partitioner: RandomPartitioner,
                           original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                           drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    assert set(map(lambda x: x.name, output_dir.iterdir())) == \
           {
               *{f"features_p{i}.csv" for i in range(1, _N_PARTITIONS + 1)},
               *{f"meta_p{i}.csv" for i in range(1, _N_PARTITIONS + 1)},
           }


def test_nr_records_per_split(partitioner: RandomPartitioner,
                              original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                              drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    filtered_original_combinations_with_drug_pair_duplicates = _filter_combinations(
        original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features)

    sample_size = filtered_original_combinations_with_drug_pair_duplicates.shape[0]

    for partition_path in _search_meta(output_dir):
        partition_size = pd.read_csv(partition_path).shape[0]
        assert partition_size >= floor(sample_size / _N_PARTITIONS)
        assert partition_size <= ceil(sample_size / _N_PARTITIONS)


def test_nr_records_per_split_with_inverse_drug_pairs(partitioner: RandomPartitioner,
                                                      original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                                                      drug_features: pd.DataFrame, cell_line_features: pd.DataFrame,
                                                      output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir, reverse_drug_pairs=True)

    filtered_original_combinations_with_drug_pair_duplicates = _filter_combinations(
        original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features)

    sample_size = filtered_original_combinations_with_drug_pair_duplicates.shape[0]

    for partition_path in _search_meta(output_dir):
        partition_size = pd.read_csv(partition_path).shape[0]
        assert partition_size >= 2 * floor(sample_size / _N_PARTITIONS)
        assert partition_size <= 2 * ceil(sample_size / _N_PARTITIONS)


def test_feature_names_by_partition(partitioner: RandomPartitioner,
                                    original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                                    drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    for partition_path in _search_features(output_dir):
        assert tuple(pd.read_csv(partition_path).columns.values) == _FEATURE_COLUMNS


def test_meta_columns_by_partition(partitioner: RandomPartitioner,
                                   original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                                   drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    for partition_path in _search_meta(output_dir):
        assert tuple(pd.read_csv(partition_path).columns.values) == _META_COLUMNS


def test_metadata_reconstruction(partitioner: RandomPartitioner,
                                 original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                                 drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    reconstructed_combinations = _reconstruct_combinations(output_dir)

    filtered_original_combinations_with_drug_pair_duplicates = _filter_combinations(
        original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features)

    pd.testing.assert_frame_equal(
        reconstructed_combinations.sort_values(by=list(reconstructed_combinations.columns))
            .reset_index(drop=True),
        filtered_original_combinations_with_drug_pair_duplicates
            .sort_values(by=list(filtered_original_combinations_with_drug_pair_duplicates.columns))
            .reset_index(drop=True))


def test_shuffling(partitioner: RandomPartitioner, original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                   drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    good_counter = total_counter = 0

    for partition_path in _search_meta(output_dir):
        partition = pd.read_csv(partition_path)
        partition_indices = []
        for _, r in partition.iterrows():
            index = original_combinations_with_drug_pair_duplicates[
                np.isclose(original_combinations_with_drug_pair_duplicates, r).all(axis=1)].index[0]
            partition_indices.append(index)
        partition_indices = sorted(partition_indices)
        for i in range(0, len(partition_indices) - 1):
            if partition_indices[i + 1] > partition_indices[i] + 1:
                good_counter += 1
        total_counter += len(partition_indices) - 1

        assert good_counter / total_counter > 0.85


def test_overlapping_groups(partitioner: RandomPartitioner,
                            original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                            drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    seen_groups = set()
    for partition_path in _search_meta(output_dir):
        partition = pd.read_csv(partition_path)
        partition_groups = set(pd.concat([partition[[_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL]].drop_duplicates()
                                         .T.reset_index(drop=True).T,
                                          partition[[_DRUG_B_ID_LABEL, _DRUG_A_ID_LABEL]].drop_duplicates()
                                         .T.reset_index(drop=True).T])
                               .drop_duplicates().apply(tuple, axis=1).values)

        if len(seen_groups.intersection(partition_groups)) > 0:
            return

        seen_groups = seen_groups.union(partition_groups)

    assert False


def test_n_cell_lines_equals_ten(partitioner: RandomPartitioner,
                                 original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                                 drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    max_n_cell_lines = 10

    _test_n_cell_lines_equals_max(cell_line_features, drug_features, max_n_cell_lines,
                                  original_combinations_with_drug_pair_duplicates, output_dir, partitioner)


def test_top_ten_most_frequent_cell_line_ids(partitioner: RandomPartitioner,
                                             original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                                             drug_features: pd.DataFrame, cell_line_features: pd.DataFrame,
                                             output_dir: Path):
    max_n_cell_lines = 10

    _test_top_n_most_frequent_cell_lines(cell_line_features, drug_features, max_n_cell_lines,
                                         original_combinations_with_drug_pair_duplicates, output_dir, partitioner)


def test_monotherapies_absence(partitioner: RandomPartitioner,
                               original_combinations_with_drug_pair_duplicates: pd.DataFrame,
                               drug_features: pd.DataFrame, cell_line_features: pd.DataFrame, output_dir: Path):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir)

    for partition_path in _search_meta(output_dir):
        meta_p = pd.read_csv(partition_path)
        assert not (meta_p[_DRUG_A_ID_LABEL] == meta_p[_DRUG_B_ID_LABEL]).any()


def _test_n_cell_lines_equals_max(cell_line_features, drug_features, max_n_cell_lines,
                                  original_combinations_with_drug_pair_duplicates, output_dir, partitioner):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir, max_n_cell_lines=max_n_cell_lines)
    reconstructed_combinations = _reconstruct_combinations(output_dir)
    assert len(reconstructed_combinations[_CELL_LINE_ID_LABEL].unique()) == max_n_cell_lines


def _test_top_n_most_frequent_cell_lines(cell_line_features, drug_features, max_n_cell_lines,
                                         original_combinations_with_drug_pair_duplicates, output_dir, partitioner):
    _partition(partitioner, original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features,
               output_dir, max_n_cell_lines=max_n_cell_lines)
    filtered_original_combinations_with_drug_pair_duplicates = _filter_combinations(
        original_combinations_with_drug_pair_duplicates, drug_features, cell_line_features)
    top_n_most_frequent_cell_line_ids = filtered_original_combinations_with_drug_pair_duplicates[_CELL_LINE_ID_LABEL] \
                                            .value_counts(ascending=False) \
                                            .to_frame("freq") \
                                            .rename_axis(index=_CELL_LINE_ID_LABEL) \
                                            .reset_index(drop=False) \
                                            .sort_values(by=["freq", _CELL_LINE_ID_LABEL], ascending=(False, True)) \
                                            .iloc[:max_n_cell_lines][_CELL_LINE_ID_LABEL] \
        .tolist()
    reconstructed_cell_line_ids = _reconstruct_combinations(output_dir)[_CELL_LINE_ID_LABEL] \
        .drop_duplicates() \
        .tolist()
    assert sorted(reconstructed_cell_line_ids) == sorted(top_n_most_frequent_cell_line_ids)


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


def _search_features(output_dir: Path) -> Generator[Path, None, None]:
    return output_dir.glob("features_p*.csv")


def _search_meta(output_dir: Path) -> Generator[Path, None, None]:
    return output_dir.glob("meta_p*.csv")


def _filter_combinations(combinations_original, drug_features, cell_line_features):
    combinations_filtered = _drop_monotherapies(combinations_original)
    combinations_filtered = combinations_filtered[
        combinations_filtered[_DRUG_A_ID_LABEL].isin(drug_features.index.values) &
        combinations_filtered[_DRUG_B_ID_LABEL].isin(drug_features.index.values) &
        combinations_filtered[_CELL_LINE_ID_LABEL].isin(cell_line_features.index.values)
        ]
    return combinations_filtered


def _drop_monotherapies(combinations: pd.DataFrame):
    return combinations[combinations[_DRUG_A_ID_LABEL] != combinations[_DRUG_B_ID_LABEL]]


def _reconstruct_combinations(output_dir):
    reconstructed_data = pd.DataFrame()
    for partition_path in _search_meta(output_dir):
        partition = pd.read_csv(partition_path)
        reconstructed_data = pd.concat([reconstructed_data, partition])
    return reconstructed_data

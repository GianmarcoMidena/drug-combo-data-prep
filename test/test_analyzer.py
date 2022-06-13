from typing import Tuple, List

import pytest
import pandas as pd

from dc_dataprep.analyzer import Analyzer
from dc_dataprep.filter import Filter

_DRUG_A_ID_LABEL = "drug_a_id"
_DRUG_B_ID_LABEL = "drug_b_id"
_CELL_LINE_ID_LABEL = "cell_line_id"


@pytest.fixture
def filter() -> Filter:
    yield Filter(_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL)


@pytest.fixture
def analyzer(filter: Filter) -> Analyzer:
    yield Analyzer(filter, _DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL, _CELL_LINE_ID_LABEL)


def test_search_top_three_most_freq_cell_lines_empty(analyzer: Analyzer):
    combinations = pd.DataFrame()
    expected_output = []

    top_n_most_freq_cell_lines = analyzer.search_top_n_most_frequent_cell_lines(3, combinations)

    assert top_n_most_freq_cell_lines == expected_output


def test_search_non_constant_features_empty(analyzer: Analyzer):
    combinations = pd.DataFrame()
    drug_features = pd.DataFrame()
    cell_line_features = pd.DataFrame()
    expected_output = []

    top_n_most_freq_cell_lines = analyzer.search_non_constant_features(combinations,
                                                                       drug_features,
                                                                       cell_line_features)

    assert top_n_most_freq_cell_lines == expected_output


@pytest.mark.parametrize(("cell_line_ids", "expected_output"),
                         (
                                 ((), []),
                                 ((1, 1), [1]),
                                 ((1, 2, 2), [2, 1]),
                                 ((1, 1, 2, 1, 3, 3), [1, 3, 2]),
                                 ((1, 1, 3, 2, 3, 2), [1, 2, 3]),
                                 ((3, 1, 3, 1, 2, 4, 1, 3, 3, 4), [3, 1, 4]),
                         ))
def test_search_top_three_most_freq_cell_lines(analyzer: Analyzer, cell_line_ids: Tuple, expected_output: List):
    combinations = pd.DataFrame({
        _CELL_LINE_ID_LABEL: cell_line_ids
    })

    top_n_most_freq_cell_lines = analyzer.search_top_n_most_frequent_cell_lines(3, combinations)

    assert top_n_most_freq_cell_lines == expected_output


def test_search_non_constant_features_one(analyzer: Analyzer):
    combinations = pd.DataFrame({
        _DRUG_A_ID_LABEL: (1,),
        _DRUG_B_ID_LABEL: (2,),
        _CELL_LINE_ID_LABEL: (3,),
    })
    drug_features = pd.DataFrame({
        "id": (1, 2),
        "f1": (4, 5),
        "f2": (6, 7),
    }).set_index("id")
    cell_line_features = pd.DataFrame({
        "id": (3,),
        "f3": (8,),
        "f4": (10,),
    }).set_index("id")

    expected_non_constant_features = []

    actual_non_constant_features = analyzer.search_non_constant_features(combinations, drug_features, cell_line_features)

    assert actual_non_constant_features == expected_non_constant_features


def test_search_non_constant_features_two(analyzer: Analyzer):
    combinations = pd.DataFrame({
        _DRUG_A_ID_LABEL: (1, 2),
        _DRUG_B_ID_LABEL: (3, 4),
        _CELL_LINE_ID_LABEL: (5, 6),
    })
    drug_features = pd.DataFrame({
        "f1": (4, 5, 6, 7),
        "f2": (8, 9, 10, 11),
    }, index=(1, 2, 3, 4))
    cell_line_features = pd.DataFrame({
        "f3": (12, 13),
        "f4": (14, 15),
        "f5": (16, 17),
    }, index=(5, 6))

    expected_non_constant_features = ["drug_a_f1", "drug_a_f2",
                                      "drug_b_f1", "drug_b_f2",
                                      "gene_f3", "gene_f4", "gene_f5"]

    actual_non_constant_features = analyzer.search_non_constant_features(combinations, drug_features, cell_line_features)

    assert actual_non_constant_features == expected_non_constant_features


def test_search_non_constant_features_three(analyzer: Analyzer):
    combinations = pd.DataFrame({
        _DRUG_A_ID_LABEL: (1, 2, 3),
        _DRUG_B_ID_LABEL: (4, 5, 6),
        _CELL_LINE_ID_LABEL: (7, 8, 9),
    })
    drug_features = pd.DataFrame({
        "f1": (10, 11, 12, 13, 14, 15),
        "f2": (16, 17, 18, 19, 20, 21),
    }, index=(1, 2, 3, 4, 5, 6))
    cell_line_features = pd.DataFrame({
        "f3": (22, 23, 24),
        "f4": (25, 26, 27),
        "f5": (28, 29, 30),
    }, index=(7, 8, 9))

    expected_non_constant_features = ["drug_a_f1", "drug_a_f2",
                                      "drug_b_f1", "drug_b_f2",
                                      "gene_f3", "gene_f4", "gene_f5"]

    actual_non_constant_features = analyzer.search_non_constant_features(combinations, drug_features, cell_line_features)

    assert actual_non_constant_features == expected_non_constant_features


def test_search_non_constant_features_three_extra(analyzer: Analyzer):
    combinations = pd.DataFrame({
        _DRUG_A_ID_LABEL: (1, 2, 7, 3),
        _DRUG_B_ID_LABEL: (4, 5, 8, 6),
        _CELL_LINE_ID_LABEL: (7, 8, 9, 10),
    })
    drug_features = pd.DataFrame({
        "f1": (10, 11, 12, 13, 14, 15),
        "f2": (16, 17, 18, 19, 20, 21),
    }, index=(1, 2, 3, 4, 5, 6))
    cell_line_features = pd.DataFrame({
        "f3": (22, 23, 24),
        "f4": (25, 26, 27),
        "f5": (28, 29, 30),
    }, index=(7, 8, 9))

    expected_non_constant_features = ["drug_a_f1", "drug_a_f2",
                                      "drug_b_f1", "drug_b_f2",
                                      "gene_f3", "gene_f4", "gene_f5"]

    actual_non_constant_features = analyzer.search_non_constant_features(combinations, drug_features, cell_line_features)

    assert actual_non_constant_features == expected_non_constant_features


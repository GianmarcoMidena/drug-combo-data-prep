from typing import Tuple, Dict

import pytest
import pandas as pd

from dc_dataprep.filter import Filter

_DRUG_A_ID_LABEL = "drug_a_id"
_DRUG_B_ID_LABEL = "drug_b_id"


@pytest.fixture
def filter() -> Filter:
    yield Filter(_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL)


def test_drop_monotherapies_empty(filter: Filter):
    data = expected_output = pd.DataFrame()

    actual_output = filter.drop_monotherapies(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)


def test_drop_constant_features_empty(filter: Filter):
    data = expected_output = pd.DataFrame()

    actual_output = filter.drop_constant_features(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    ("drug_a_ids", "drug_b_ids"),
    (
            ((1,), (2,)),
            ((1, 2), (3, 4)),
            ((32, 9, 4), (5, 12, 8))
     )
)
def test_drop_no_monotherapies(filter: Filter, drug_a_ids: Tuple[int], drug_b_ids: Tuple[int]):
    data = expected_output = pd.DataFrame({
        _DRUG_A_ID_LABEL: drug_a_ids,
        _DRUG_B_ID_LABEL: drug_b_ids,
    })

    actual_output = filter.drop_monotherapies(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    ("drug_a_ids_in", "drug_b_ids_in", "drug_a_ids_out", "drug_b_ids_out"),
    (
            ((1,), (1,), (), (), ),
            ((1, 2), (1, 4), (2,), (4,), ),
            ((32, 9, 4), (5, 9, 8), (32, 4), (5, 8))
     )
)
def test_drop_monotherapies(filter: Filter, drug_a_ids_in: Tuple[int], drug_b_ids_in: Tuple[int],
                            drug_a_ids_out: Tuple[int], drug_b_ids_out: Tuple[int]):
    data = pd.DataFrame({
        _DRUG_A_ID_LABEL: drug_a_ids_in,
        _DRUG_B_ID_LABEL: drug_b_ids_in,
    })
    expected_output = pd.DataFrame({
        _DRUG_A_ID_LABEL: drug_a_ids_out,
        _DRUG_B_ID_LABEL: drug_b_ids_out,
    })

    actual_output = filter.drop_monotherapies(data)

    pd.testing.assert_frame_equal(actual_output.reset_index(drop=True),
                                  expected_output.reset_index(drop=True),
                                  check_dtype=False)


@pytest.mark.parametrize(
    ("f1", "f2", "f3"),
    (
            ((1, 2), (3, 4), (5, 6)),
            ((32, 9, 4), (5, 12, 8), (98, 34, 56))
     )
)
def test_drop_no_constant_features(filter: Filter, f1: Tuple[int], f2: Tuple[int], f3: Tuple[int]):
    data = expected_output = pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
    })

    actual_output = filter.drop_constant_features(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    ("input", "output"),
    (
            (
                    {"f1": (1,), "f2": (2,), "f3": (3,)},
                    {},
            ),
            (
                    {"f1": (1, 1), "f2": (3, 4), "f3": (5, 6)},
                    {"f2": (3, 4), "f3": (5, 6)},
            ),
            (
                    {"f1": (32, 9, 4), "f2": (5, 5, 5), "f3": (98, 34, 56)},
                    {"f1": (32, 9, 4), "f3": (98, 34, 56)},
            ),
     )
)
def test_drop_constant_features(filter: Filter, input: Dict, output: Dict):
    data = pd.DataFrame(input)

    expected_output = pd.DataFrame(output)

    actual_output = filter.drop_constant_features(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)

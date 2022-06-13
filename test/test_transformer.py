from typing import Tuple

import pytest
import pandas as pd

from dc_dataprep.transformer import Transformer

_DRUG_A_ID_LABEL = "drug_a_id"
_DRUG_B_ID_LABEL = "drug_b_id"


@pytest.fixture
def tx() -> Transformer:
    yield Transformer(_DRUG_A_ID_LABEL, _DRUG_B_ID_LABEL)


def test_reverse_zero_records(tx: Transformer):
    data = expected_output = pd.DataFrame()

    actual_output = tx.reverse_drug_pairs(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    ("drug_a_ids", "drug_b_ids"),
    (
            ((1,), (2,)),
            ((1, 2), (3, 4)),
            ((32, 9, 4), (5, 12, 8))
     )
)
def test_reverse_records(tx: Transformer, drug_a_ids: Tuple[int], drug_b_ids: Tuple[int]):
    data = pd.DataFrame({
        _DRUG_A_ID_LABEL: drug_a_ids,
        _DRUG_B_ID_LABEL: drug_b_ids,
    })

    expected_output = pd.DataFrame({
        _DRUG_A_ID_LABEL: drug_b_ids,
        _DRUG_B_ID_LABEL: drug_a_ids,
    })

    actual_output = tx.reverse_drug_pairs(data)

    pd.testing.assert_frame_equal(actual_output, expected_output)

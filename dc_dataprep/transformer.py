import pandas as pd


class Transformer:
    """Transform drug combination data."""

    def __init__(self, drug_a_id_label: str, drug_b_id_label: str):
        """
        Parameters
        ----------
        drug_a_id_label : str
            Name of the column containing the identifiers for drug `a`s.

        drug_b_id_label : str
            Name of the column containing the identifiers for drug `b`s.
        """
        self._drug_a_id_label = drug_a_id_label
        self._drug_b_id_label = drug_b_id_label

    def reverse_drug_pairs(self, combinations: pd.DataFrame) -> pd.DataFrame:
        """Reverse drug pairs in a set of combinations.

        Parameters
        ----------
        combinations: pd.DataFrame
            Records including two drug ids, a cell line id and a response.

        Returns
        -------
        reverse_drug_pairs_combinations: pd.DataFrame
            The input set of combinations with reversed drug pairs.
        """
        combinations = combinations.copy()
        if not combinations.empty:
            combinations[[self._drug_a_id_label, self._drug_b_id_label]] = \
                combinations[[self._drug_b_id_label, self._drug_a_id_label]]
        return combinations

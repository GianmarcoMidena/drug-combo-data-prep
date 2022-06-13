from typing import List

import pandas as pd

from dc_dataprep.filter import Filter


class Analyzer:
    """Analize drug combination data."""

    _MAX_DATA_PER_BATCH = 5000

    def __init__(self, filter: Filter, drug_a_id_label: str, drug_b_id_label: str, cell_line_id_label: str):
        """
        Parameters
        ----------
        filter : Filter
            Data filtering service

        drug_a_id_label : str
            Name of the column containing the identifiers for drug `a`s.

        drug_b_id_label : str
            Name of the column containing the identifiers for drug `b`s.

        cell_line_id_label : str
            Name of the column containing the identifiers for cell lines.
        """
        self._filter = filter
        self._drug_a_id_label = drug_a_id_label
        self._drug_b_id_label = drug_b_id_label
        self._cell_line_id_label = cell_line_id_label

    def search_top_n_most_frequent_cell_lines(self, n: int, combinations: pd.DataFrame) -> List:
        """Search the top n most frequent cell lines w.r.t their occurrence in combinations.

        Parameters
        ----------
        n: int
            Limit to the number of cell lines to consider.

        combinations: pd.DataFrame
            Records including two drug ids, a cell line id and a response.

        Returns
        -------
        top_n_most_frequent_cell_lines: List
            Top n most frequent cell lines, according to their number of combinations.
        """
        if combinations.empty:
            return []
        return combinations[self._cell_line_id_label].value_counts(ascending=False) \
                   .to_frame("freq") \
                   .rename_axis(index=self._cell_line_id_label) \
                   .reset_index(drop=False) \
                   .sort_values(by=["freq", self._cell_line_id_label], ascending=(False, True)) \
                   .iloc[:n][self._cell_line_id_label].values.tolist()

    def search_non_constant_features(self, combinations: pd.DataFrame,
                                     drug_features: pd.DataFrame,
                                     cell_line_features: pd.DataFrame) -> List[str]:
        """Looks for names of features varying across data.

        A feature is non-constant if it assumes at least two different values across the smple.

        Parameters
        ----------
        combinations: pd.DataFrame
            Records including two drug ids, a cell line id and a response.

        drug_features : pd.DataFrame
            Records of features indexed by drug ids.

        cell_line_features : pd.DataFrame
            Records of features indexed by cell line ids.

        Returns
        -------
        non_constant_features: List[str]
            Names of all features varying across data.

        """
        if combinations.empty or drug_features.empty or cell_line_features.empty:
            return []

        drug_a_ids = combinations[self._drug_a_id_label].unique()
        drug_a_ids = drug_features.index[drug_features.index.isin(drug_a_ids)]
        drug_a_features = drug_features.loc[drug_a_ids] \
            .reset_index(drop=True) \
            .add_prefix("drug_a_")
        drug_a_features = self._filter.drop_constant_features(drug_a_features)

        drug_b_ids = combinations[self._drug_b_id_label].unique()
        drug_b_ids = drug_features.index[drug_features.index.isin(drug_b_ids)]
        drug_b_features = drug_features.loc[drug_b_ids] \
            .reset_index(drop=True) \
            .add_prefix("drug_b_")
        drug_b_features = self._filter.drop_constant_features(drug_b_features)

        cell_line_ids = combinations[self._cell_line_id_label].unique()
        cell_line_ids = cell_line_features.index[cell_line_features.index.isin(cell_line_ids)]
        cell_line_features = cell_line_features.loc[cell_line_ids] \
            .reset_index(drop=True) \
            .add_prefix("gene_")
        cell_line_features = self._filter.drop_constant_features(cell_line_features)

        non_constant_feature_names = set(drug_a_features) \
            .union(drug_b_features) \
            .union(cell_line_features)

        return sorted(list(non_constant_feature_names))

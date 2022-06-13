import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, List

import pandas as pd

from dc_dataprep.analyzer import Analyzer
from dc_dataprep.filter import Filter
from dc_dataprep.transformer import Transformer


class Partitioner(ABC):
    """Split a drug combination dataset into n parts."""

    def __init__(self, filter: Filter, analyzer: Analyzer, tx: Transformer,
                 drug_a_id_label: str, drug_b_id_label: str, cell_line_id_label: str,
                 response_label: str, n_splits: int = 5, seed: Optional[int] = None):
        """
        Parameters
        ----------
        filter : Filter
            Data filtering service

        analyzer : Analyzer
            Data analysis service

        tx: transformer
            Data transformation service

        drug_a_id_label : str
            Name of the column containing the identifiers for drug `a`s.

        drug_b_id_label : str
            Name of the column containing the identifiers for drug `b`s.

        cell_line_id_label : str
            Name of the column containing the identifiers for cell lines.

        response_label : str
            Name of the column containing the responses.

        n_splits : int, default=5
            Number of parts. Must be at least 2.

        seed : int, default=None
            Controls the randomness of each partition.
        """
        self._filter = filter
        self._analyzer = analyzer
        self._tx = tx
        self._drug_a_id_label = drug_a_id_label
        self._drug_b_id_label = drug_b_id_label
        self._cell_line_id_label = cell_line_id_label
        self._response_label = response_label
        self._n_splits = n_splits
        self._seed = seed

    def partition(self, combinations: pd.DataFrame, drug_features: pd.DataFrame, cell_line_features: pd.DataFrame,
                  output_dir: Path, reverse_drug_pairs: bool = False, max_n_cell_lines: Optional[int] = None):
        """Split data.

        Parameters
        ----------
        combinations: pd.DataFrame
            Records including two drug ids, a cell line id and a response.

        drug_features : pd.DataFrame
            Records of features indexed by drug ids.

        cell_line_features : pd.DataFrame
            Records of features indexed by cell line ids.

        output_dir : Path
            Folder to save the parts.

        reverse_drug_pairs : bool, default=False
            Request to duplicate records with reversed drug pairs.

        max_n_cell_lines : int, default=None
            Limit to the number of cell lines to consider.
        """

        combinations = self._filter.drop_monotherapies(combinations)

        drug_features = self._filter.drop_constant_features(drug_features)
        cell_line_features = self._filter.drop_constant_features(cell_line_features)

        combinations = combinations[
            combinations[self._drug_a_id_label].isin(drug_features.index.values) &
            combinations[self._drug_b_id_label].isin(drug_features.index.values) &
            combinations[self._cell_line_id_label].isin(cell_line_features.index.values)
        ]

        drug_ids = pd.concat([combinations[self._drug_a_id_label].drop_duplicates(),
                              combinations[self._drug_b_id_label].drop_duplicates()]).unique()
        drug_features = drug_features.loc[drug_ids]

        cell_line_ids = combinations[self._cell_line_id_label].unique()
        cell_line_features = cell_line_features.loc[cell_line_ids]

        if max_n_cell_lines:
            top_n_most_frequent_cell_lines = self._analyzer.search_top_n_most_frequent_cell_lines(max_n_cell_lines,
                                                                                                combinations)
            combinations = combinations[
                combinations[self._cell_line_id_label].isin(top_n_most_frequent_cell_lines)
            ]

        features_names = self._analyzer.search_non_constant_features(combinations, drug_features, cell_line_features)

        combinations = combinations.sample(frac=1, replace=False, random_state=self._seed).reset_index(drop=True)

        for p, indices in enumerate(self._split(combinations), start=1):
            logging.info(f"Building partition {p}...")

            combinations_p = combinations.iloc[indices].reset_index(drop=True)

            drug_ids_p = pd.concat([combinations_p[self._drug_a_id_label].drop_duplicates(),
                                    combinations_p[self._drug_b_id_label].drop_duplicates()]).unique()
            drug_features_p = drug_features.loc[drug_ids_p]

            cell_line_ids_p = combinations[self._cell_line_id_label].unique()
            cell_line_features_p = cell_line_features.loc[cell_line_ids_p]

            if reverse_drug_pairs:
                combinations_p = pd.concat([combinations_p, self._tx.reverse_drug_pairs(combinations_p)],
                                           axis=0, ignore_index=True)

            combinations_p = combinations_p.sample(frac=1, replace=False, random_state=self._seed)

            drug_a_features_p = drug_features_p.loc[combinations_p[self._drug_a_id_label]] \
                .reset_index(drop=True) \
                .add_prefix("drug_a_")
            drug_a_features_p = drug_a_features_p[[c for c in drug_a_features_p.columns if c in features_names]]

            drug_b_features_p = drug_features_p.loc[combinations_p[self._drug_b_id_label]] \
                .reset_index(drop=True) \
                .add_prefix("drug_b_")
            drug_b_features_p = drug_b_features_p[[c for c in drug_b_features_p.columns if c in features_names]]

            cell_line_features_p = cell_line_features_p.loc[combinations_p[self._cell_line_id_label]] \
                .reset_index(drop=True) \
                .add_prefix("gene_")
            cell_line_features_p = cell_line_features_p[
                [c for c in cell_line_features_p.columns if c in features_names]
            ]

            response_p = combinations_p[self._response_label].reset_index(drop=True)

            features_p = pd.concat([
                drug_a_features_p,
                drug_b_features_p,
                cell_line_features_p,
                response_p,
            ], axis=1)

            output_dir.mkdir(exist_ok=True)
            features_p.to_csv(output_dir.joinpath(f"features_p{p}.csv"), index=False)
            combinations_p.to_csv(output_dir.joinpath(f"meta_p{p}.csv"), index=False)

    @abstractmethod
    def _split(self, combinations: pd.DataFrame):
        ...

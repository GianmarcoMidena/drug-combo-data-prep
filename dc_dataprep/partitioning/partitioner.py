import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, List

import pandas as pd


class Partitioner(ABC):
    """Partitioner

    Splits a drug combination dataset into k parts.
    """

    def __init__(self, drug_a_id_label: str, drug_b_id_label: str, cell_line_id_label: str, response_label: str):
        self._drug_a_id_label = drug_a_id_label
        self._drug_b_id_label = drug_b_id_label
        self._cell_line_id_label = cell_line_id_label
        self._response_label = response_label

    def partition(self, combinations: pd.DataFrame, drug_features: pd.DataFrame, cell_line_features: pd.DataFrame,
                  n_partitions: int, output_dir: Path, reverse_drug_pairs: bool = False,
                  max_n_cell_lines: Optional[int] = None, seed: Optional[int] = None):

        combinations = self._drop_monotherapies(combinations)

        drug_features = self._drop_constant_features(drug_features)
        cell_line_features = self._drop_constant_features(cell_line_features)

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
            nth_most_frequent_cell_lines = self._search_nth_most_frequent_cell_lines(max_n_cell_lines, combinations)
            combinations = combinations[
                combinations[self._cell_line_id_label].isin(nth_most_frequent_cell_lines)
            ]

        features_names = self._search_non_constant_features(combinations, drug_features, cell_line_features,
                                                            n_partitions, seed)

        combinations = combinations.sample(frac=1, replace=False, random_state=seed).reset_index(drop=True)

        for p, indices in enumerate(self._split(combinations, n_partitions, seed), start=1):
            logging.info(f"Building partition {p}...")

            combinations_p = combinations.iloc[indices].reset_index(drop=True)

            drug_ids_p = pd.concat([combinations_p[self._drug_a_id_label].drop_duplicates(),
                                    combinations_p[self._drug_b_id_label].drop_duplicates()]).unique()
            drug_features_p = drug_features.loc[drug_ids_p]

            cell_line_ids_p = combinations[self._cell_line_id_label].unique()
            cell_line_features_p = cell_line_features.loc[cell_line_ids_p]

            if reverse_drug_pairs:
                combinations_p = pd.concat([combinations_p, self._reverse_drug_pairs(combinations_p)],
                                           axis=0, ignore_index=True)

            combinations_p = combinations_p.sample(frac=1, replace=False, random_state=seed)

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
    def _split(self, combinations: pd.DataFrame, n_partitions: int, seed: Optional[int] = None):
        ...

    def _drop_monotherapies(self, combinations: pd.DataFrame) -> pd.DataFrame:
        return combinations[combinations[self._drug_a_id_label] != combinations[self._drug_b_id_label]]

    def _reverse_drug_pairs(self, combinations: pd.DataFrame) -> pd.DataFrame:
        combinations = combinations.copy()
        combinations[[self._drug_a_id_label, self._drug_b_id_label]] = \
            combinations[[self._drug_b_id_label, self._drug_a_id_label]]
        return combinations

    def _search_nth_most_frequent_cell_lines(self, n: int, combinations: pd.DataFrame):
        return combinations[self._cell_line_id_label].value_counts(ascending=False) \
                   .to_frame("freq") \
                   .rename_axis(index=self._cell_line_id_label) \
                   .reset_index(drop=False) \
                   .sort_values(by=["freq", self._cell_line_id_label], ascending=(False, True)) \
                   .iloc[:n][self._cell_line_id_label].values

    def _search_non_constant_features(self, combinations: pd.DataFrame, drug_features: pd.DataFrame,
                                      cell_line_features: pd.DataFrame, n_partitions: int, seed: Optional[int] = None) \
            -> List[str]:

        non_constant_feature_names = set()

        for p, indices in enumerate(self._split(combinations, n_partitions, seed), start=1):

            combinations_p = combinations.iloc[indices].reset_index(drop=True)

            drug_a_features_p = drug_features.loc[combinations_p[self._drug_a_id_label]] \
                .reset_index(drop=True) \
                .add_prefix("drug_a_")
            drug_a_features_p = self._drop_constant_features(drug_a_features_p)

            drug_b_features_p = drug_features.loc[combinations_p[self._drug_b_id_label]] \
                .reset_index(drop=True) \
                .add_prefix("drug_b_")
            drug_b_features_p = self._drop_constant_features(drug_b_features_p)

            cell_line_features_p = cell_line_features.loc[combinations_p[self._cell_line_id_label]] \
                .reset_index(drop=True) \
                .add_prefix("gene_")
            cell_line_features_p = self._drop_constant_features(cell_line_features_p)

            non_constant_feature_names = non_constant_feature_names \
                .union(drug_a_features_p) \
                .union(drug_b_features_p) \
                .union(cell_line_features_p)

        return sorted(list(non_constant_feature_names))

    @staticmethod
    def _drop_constant_features(features: pd.DataFrame) -> pd.DataFrame:
        return features.loc[:, (features != features.iloc[0]).any()]

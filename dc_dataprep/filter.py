import pandas as pd


class Filter:
    """Filter drug combination data."""

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

    def drop_monotherapies(self, combinations: pd.DataFrame) -> pd.DataFrame:
        """Remove all the combinations of a drug with itself.

        Parameters
        ----------
        combinations: pd.DataFrame
            Records including two drug ids, a cell line id and a response.

        Returns
        -------

        """
        if not combinations.empty:
            combinations = combinations[combinations[self._drug_a_id_label] !=
                                        combinations[self._drug_b_id_label]]
        return combinations

    def drop_constant_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove features that does not vary among different examples.

        Parameters
        ----------
        features : pd.DataFrame
            Records of features.

        Returns
        -------
        variable_features : pd.DataFrame
            Records of all features that vary among different examples.
        """
        if not features.empty:
            features = features.loc[:, (features != features.iloc[0]).any()]
            if features.empty:
                features = pd.DataFrame()
        return features

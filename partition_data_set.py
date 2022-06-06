from argparse import ArgumentParser
import math
from pathlib import Path

import pandas as pd
import logging


def invert_drugs(data):
    data = data.copy()
    data = data[data["drug_a_smiles"] != data["drug_b_smiles"]]
    data[["drug_a_smiles", "drug_b_smiles"]] = data[["drug_b_smiles", "drug_a_smiles"]]
    return data


def _drop_constant_features(data: pd.DataFrame):
    return data.loc[:, (data != data.iloc[0]).any()]


def _search_non_constant_features(partition_size, seed):
    logging.info("Searching non constant features...")

    non_constant_features = set()
    for p, j in enumerate(range(0, quadruples.shape[0], partition_size), start=1):

        quadruples_p = quadruples.iloc[j:min(j + partition_size, quadruples.shape[0])]
        quadruples_p = quadruples_p.append(invert_drugs(quadruples_p)) \
            .reset_index(drop=True) \
            .sample(frac=1, replace=False, random_state=seed)
        smiles_p = set(quadruples_p["drug_a_smiles"].unique()).union(quadruples_p["drug_b_smiles"].unique())
        drug_features_p = drug_features.loc[smiles_p]
        cell_line_names_p = quadruples_p["cell_line_name"].unique()
        cell_line_features_p = cell_line_features.loc[cell_line_names_p]

        data_a = drug_features_p.add_prefix("drug_a_").loc[quadruples_p["drug_a_smiles"].values] \
            .reset_index(drop=True)
        data_a = _drop_constant_features(data_a)
        non_constant_features = non_constant_features.union(data_a.columns)

        data_b = drug_features_p.add_prefix("drug_b_").loc[quadruples_p["drug_b_smiles"].values] \
            .reset_index(drop=True)
        data_b = _drop_constant_features(data_b)
        non_constant_features = non_constant_features.union(data_b.columns)

        if len(cell_line_names_p) > 1:
            data_c = cell_line_features_p.add_prefix("gene_").loc[quadruples_p["cell_line_name"].values] \
                .reset_index(drop=True)
            data_c = _drop_constant_features(data_c)
            non_constant_features = non_constant_features.union(data_c.columns)
    return sorted(list(non_constant_features))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--drug_combinations_path", type=Path, required=True)
    arg_parser.add_argument("--physicochemical_features_path", type=Path, required=True)
    arg_parser.add_argument("--cell_line_features_path", type=Path, required=True)
    arg_parser.add_argument("--output_dir", type=Path, required=True)
    arg_parser.add_argument("--max_drugs", type=int)
    arg_parser.add_argument("--max_cell_lines", type=int)
    arg_parser.add_argument("--n_partitions", type=int)
    arg_parser.add_argument("--seed", type=int, default=3)
    # arg_parser.add_argument("--partition_size", type=int, default=100)
    # arg_parser.add_argument("--max_partitions", type=int)
    args = arg_parser.parse_args()

    quadruples = pd.read_csv(args.drug_combinations_path)
    no_monotherapies = quadruples["drug_a_smiles"] != quadruples["drug_b_smiles"]
    quadruples = quadruples[no_monotherapies]

    if args.max_cell_lines:
        most_frequent_cell_lines = quadruples["cell_line_name"] \
                                       .value_counts(ascending=False) \
                                       .iloc[:args.max_cell_lines].index.values
        quadruples = quadruples[quadruples["cell_line_name"].isin(most_frequent_cell_lines)]

    if args.max_drugs:
        most_frequent_drugs = quadruples[["drug_a_smiles", "drug_b_smiles"]] \
            .unstack() \
            .value_counts(ascending=False) \
            .iloc[:args.max_drugs].index.values
        quadruples = quadruples[quadruples["drug_a_smiles"].isin(most_frequent_drugs) &
                                quadruples["drug_b_smiles"].isin(most_frequent_drugs)]

    # quadruples = quadruples.append(invert_drugs(quadruples)).reset_index(drop=True)
    quadruples = quadruples.sample(frac=1, replace=False, random_state=args.seed)

    drug_features = pd.read_csv(args.physicochemical_features_path).set_index("smiles")
    drug_features = _drop_constant_features(drug_features)
    cell_line_features = pd.read_csv(args.cell_line_features_path).set_index("name")
    cell_line_features = _drop_constant_features(cell_line_features)

    partition_size = math.ceil(quadruples.shape[0] / args.n_partitions)

    feature_names = _search_non_constant_features(partition_size, args.seed)

    # for p, j in enumerate(range(args.partition_size // 2, quadruples.shape[0], args.partition_size // 2), start=1):
    for p, j in enumerate(range(0, quadruples.shape[0], partition_size), start=1):
        # if p > args.max_partitions:
        #     break
        logging.info(f"Building partition {p}...")

        quadruples_p = quadruples.iloc[j:min(j + partition_size, quadruples.shape[0])]
        quadruples_p = quadruples_p.append(invert_drugs(quadruples_p))\
                                   .reset_index(drop=True)\
                                   .sample(frac=1, replace=False, random_state=args.seed)
        smiles_p = set(quadruples_p["drug_a_smiles"].unique()).union(quadruples_p["drug_b_smiles"].unique())
        drug_features_p = drug_features.loc[smiles_p]
        cell_line_names_p = quadruples_p["cell_line_name"].unique()
        cell_line_features_p = cell_line_features.loc[cell_line_names_p]

        logging.info("Adding drug a features...")
        data_a = drug_features_p.add_prefix("drug_a_").loc[quadruples_p["drug_a_smiles"].values] \
            .reset_index(drop=True)

        logging.info("Adding drug b features...")
        data_b = drug_features_p.add_prefix("drug_b_").loc[quadruples_p["drug_b_smiles"].values] \
            .reset_index(drop=True)

        if len(cell_line_names_p) > 1:
            logging.info("Adding cell line features...")
            data_c = cell_line_features_p.add_prefix("gene_").loc[quadruples_p["cell_line_name"].values] \
                .reset_index(drop=True)
        else:
            data_c = pd.DataFrame()

        synergy = quadruples_p[["synergy_loewe"]].reset_index(drop=True)

        args.output_dir.mkdir(exist_ok=True)
        features_p = pd.concat([data_a, data_b, data_c, synergy], axis=1)
        features_p = features_p[feature_names + ["synergy_loewe"]]
        features_p.to_csv(args.output_dir.joinpath(f"dataset_p{p}.csv"), index=False)

        quadruples_p.to_csv(args.output_dir.joinpath(f"dataset_p{p}_meta.csv"), index=False)

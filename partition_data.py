import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from dc_dataprep.partitioning.random_partitioner import RandomPartitioner
from dc_dataprep.partitioning.drug_pairs_partitioner import DrugPairsPartitioner

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--n_partitions", type=int, required=True)
    arg_parser.add_argument("--combinations_path", type=Path, required=True)
    arg_parser.add_argument("--drug_features_path", type=Path, required=True)
    arg_parser.add_argument("--cell_line_features_path", type=Path, required=True)
    arg_parser.add_argument("--output_dir", type=Path, required=True)
    arg_parser.add_argument("--max_cell_lines", type=int)
    arg_parser.add_argument("--reverse_drug_pairs", action='store_true', default=False)
    arg_parser.add_argument("--drug_a_id_label", type=str, required=True)
    arg_parser.add_argument("--drug_b_id_label", type=str, required=True)
    arg_parser.add_argument("--cell_line_id_label", type=str, required=True)
    arg_parser.add_argument("--response_label", type=str, required=True)
    arg_parser.add_argument("--group", type=str, choices=["drug_pairs"], default=None)
    arg_parser.add_argument("--seed", type=int)
    args = arg_parser.parse_args()

    combinations = pd.read_csv(args.combinations_path)
    drug_features = pd.read_csv(args.drug_features_path).set_index("smiles")
    cell_line_features = pd.read_csv(args.cell_line_features_path).set_index("name")

    if args.group and args.group == "drug_pairs":
        partitioner_class = DrugPairsPartitioner
    else:
        partitioner_class = RandomPartitioner

    partitioner_class(drug_a_id_label=args.drug_a_id_label,
                      drug_b_id_label=args.drug_b_id_label,
                      cell_line_id_label=args.cell_line_id_label,
                      response_label=args.response_label) \
        .partition(combinations=combinations,
                   drug_features=drug_features,
                   cell_line_features=cell_line_features,
                   n_partitions=args.n_partitions,
                   output_dir=args.output_dir,
                   max_n_cell_lines=args.max_cell_lines,
                   reverse_drug_pairs=args.reverse_drug_pairs,
                   seed=args.seed)

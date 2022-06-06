python partition_data_set.py \
  --drug_combinations_path "../DrugComb/drug_comb_mean_filtered.csv" \
  --physicochemical_features_path "../DrugComb/psycochemical_features_filtered.csv" \
  --cell_line_features_path "../DrugComb/cell_line_features_filtered.csv" \
  --output_dir "../DrugComb" \
  --max_cell_lines 5 \
  --n_partitions 10
#  --partition_size 2000 \

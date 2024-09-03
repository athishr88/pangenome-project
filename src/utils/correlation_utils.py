import random
from collections import defaultdict
import pandas as pd
import os

def find_pearson_correlation(cfg):
    dataset_text_file = cfg.file_paths.best_features_dataset.dataset_text_file
    df = pd.read_csv(dataset_text_file, sep=", ", index_col=0)
    df = df.drop(columns=['serotype_encoded'])

    corr_matrix = df.corr(method='pearson')
    # Restrict decimal points to 3
    corr_matrix = corr_matrix.round(3)
    out_file = cfg.file_paths.corr_matrix.out_file

    corr_out_folder = os.path.dirname(out_file)
    os.makedirs(corr_out_folder, exist_ok=True)

    corr_matrix.to_csv(out_file)

def find_correlated_groups(correlated_pairs):
    """
    Groups correlated indices together.
    """
    groups = defaultdict(set)

    def find_root(index, parent):
        while parent[index] != index:
            index = parent[index]
        return index

    parent = {}
    for a, b in correlated_pairs:
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        root_a = find_root(a, parent)
        root_b = find_root(b, parent)
        if root_a != root_b:
            parent[root_b] = root_a

    for index in parent:
        root = find_root(index, parent)
        groups[root].add(index)

    return groups

def select_representatives(groups):
    """
    Randomly selects one representative from each group of correlated indices.
    """
    representatives = []
    for group in groups.values():
        representative = random.choice(list(group))
        print(f"Chose {representative} from {group}")
        representatives.append(representative)
        
    representatives = [random.choice(list(group)) for group in groups.values()]
    return representatives

def select_from_correlated_indices(cfg):
    corr_matrix_file = cfg.file_paths.corr_matrix.out_file
    df = pd.read_csv(corr_matrix_file, index_col=0)

    correlation_threshold = cfg.file_paths.corr_matrix.correlation_threshold
    target_length = cfg.file_paths.corr_matrix.target_length
    final_indices = []

    stuck = 0

    while len(final_indices) != target_length:
        stuck += 1
        if stuck > 50:
            print("Stuck in loop")
            break
        print(f"Correlation threshold: {correlation_threshold}")
        correlated_pairs = []
        col_names = df.columns
        for i in range(len(df.columns)):
            for j in range(i):
                if abs(df.iloc[i, j]) > correlation_threshold:
                    correlated_pairs.append((col_names[i], col_names[j]))

        groups = find_correlated_groups(correlated_pairs)
        representatives = select_representatives(groups)
        all_indices = set(df.columns)
        correlated_indices = set(index for group in groups.values() for index in group)
        non_correlated_indices = all_indices - correlated_indices

        final_indices = list(non_correlated_indices) + representatives
        final_indices = sorted([int(index) for index in final_indices])

        if len(final_indices) < target_length:
            print(f"Number of indices less than target length: {len(final_indices)}")
            correlation_threshold += 0.001  # Increase threshold to get more indices
        elif len(final_indices) > target_length:
            print(f"Number of indices greater than target length: {len(final_indices)}")
            correlation_threshold -= 0.001  # Descrease threshold to get fewer indices

    out_file = cfg.file_paths.corr_matrix.filtered_indices_file

    if len(final_indices) == target_length:
        with open(out_file, 'w') as f:
            for idx in final_indices:
                f.write("%s\n" % idx)

    return final_indices

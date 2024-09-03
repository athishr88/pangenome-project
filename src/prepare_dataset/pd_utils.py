from prepare_dataset.best_indices_dataset import BestFeaturesDataclassDataset, CorrelationFilteredDataset
import matplotlib.pyplot as plt
from kneed import KneeLocator
import pandas as pd
import numpy as np
import os

def identify_best_features_ankle_point(cfg):
    fi_folder = cfg.file_paths.supporting_files.fi_folder
    files = os.listdir(fi_folder)
    files = [f for f in files if f.endswith('.csv')]

    folds = cfg.best_features_dataset.dataset.folds
    num_best_features = cfg.best_features_dataset.dataset.num_best_features
    all_indices = []
    ankle_points = {}

    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
        df = df.sort_values("score", ascending=False)
        top_500_scores = df.head(num_best_features)['score'].values.tolist()
        knee_locator = KneeLocator(range(1, num_best_features + 1), top_500_scores, curve="convex", direction="decreasing")
        ankle_point = knee_locator.knee
        num_top_choices = int(ankle_point * folds)
        top_features = df.head(num_top_choices).index.tolist()
        top_features = [int(x.split("_")[1]) for x in top_features]
        all_indices.extend(top_features)

        bottom_500_scores = df.tail(num_best_features)['score'].values.tolist()
        bottom_500_scores = bottom_500_scores[::-1]
        knee_locator = KneeLocator(range(1, num_best_features + 1), bottom_500_scores, curve="concave", direction="increasing")
        ankle_point = knee_locator.knee
        ankle_points[file+'_bottom'] = ankle_point
        num_bottom_choices = int(ankle_point * folds)
        bottom_features = df.tail(num_bottom_choices).index.tolist()
        bottom_features = [int(x.split("_")[1]) for x in bottom_features]
        all_indices.extend(bottom_features)
    
    unique_indices = sorted(list(set(all_indices)))
    out_folder = cfg.file_paths.best_features_dataset.best_features_names_out_folder
    os.makedirs(out_folder, exist_ok=True)
    filename = f"Important_Indices_fold_{folds}.txt"
    with open(os.path.join(out_folder, filename), 'w') as f:
        for idx in unique_indices:
            f.write("%s\n" % idx)

def identify_best_features_cutoff(cfg):
    fi_folder = cfg.file_paths.explanation.deeplift_fi_folder
    files = os.listdir(fi_folder)
    files = [f for f in files if f.endswith('.csv')]

    cutoff = cfg.best_features_dataset.dataset.cutoff
    all_indices = []

    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
        df = df.sort_values("score", ascending=False)
        top_features = df.head(cutoff).index.tolist()
        top_features = [int(x.split("_")[1]) for x in top_features]
        all_indices.extend(top_features)

        bottom_features = df.tail(cutoff).index.tolist()
        bottom_features = [int(x.split("_")[1]) for x in bottom_features]
        all_indices.extend(bottom_features)
    
    unique_indices = sorted(list(set(all_indices)))
    out_folder = cfg.file_paths.best_features_dataset.best_features_names_out_folder
    os.makedirs(out_folder, exist_ok=True)
    filename = f"Important_Indices_cutoff_{cutoff}.txt"
    with open(os.path.join(out_folder, filename), 'w') as f:
        for idx in unique_indices:
            f.write("%s\n" % idx)

def identify_best_features_combined(cfg):
    # Finds out the indices where there is even a miniscule of affect on the prediction. Avoids 0s
    fi_folder = cfg.file_paths.supporting_files.fi_folder
    files = os.listdir(fi_folder)
    files = [f for f in files if f.endswith('.csv')]

    df_combined = pd.DataFrame()
    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
        df_combined = pd.concat([df_combined, df], axis=0)
    df_combined = df_combined.sort_values(by="score", ascending=False)
    top_features = df_combined.head(28000).index.tolist()
    bottom_features = df_combined.tail(28000).index.tolist()
    

    top_features = list(set([int(x.split("_")[1]) for x in top_features]))
    bottom_features = list(set([int(x.split("_")[1]) for x in bottom_features]))
    best_features = top_features + bottom_features
    print(len(top_features), len(bottom_features))
    print(len(best_features))
    
    # print(df_combined)

def identify_best_features_non_coding(cfg):
    non_coding_region_file = cfg.file_paths.explanation.non_coding_regions_file
    df_non_coding = pd.read_csv(non_coding_region_file, sep=', ', header=None, index_col=0)
    non_coding_regions = df_non_coding.index.tolist()
    non_coding_regions = (np.array(non_coding_regions)-1).tolist()

    fi_folder = cfg.file_paths.explanation.deeplift_fi_folder
    
    files = os.listdir(fi_folder)
    files = [f for f in files if f.endswith('.csv')]

    cutoff = cfg.best_features_dataset.dataset.cutoff
    all_indices = []

    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
        df = df.sort_values("score", ascending=False)
        all_features = df.index.tolist()
        all_features = [int(x.split("_")[1]) for x in all_features]

        total_added = 0
        for k in range(len(all_features)):
            if all_features[k] in non_coding_regions:
                all_indices.append(all_features[k])
                total_added += 1
                if total_added == cutoff:
                    break
        
        total_added = 0
        for k in range(len(all_features)):
            if all_features[-k] in non_coding_regions:
                all_indices.append(all_features[-k])
                total_added += 1
                if total_added == cutoff:
                    break
    
    unique_indices = sorted(list(set(all_indices)))
    out_folder = cfg.file_paths.best_features_dataset.best_features_names_out_folder
    os.makedirs(out_folder, exist_ok=True)
    filename = f"Important_Indices_cutoff_{cutoff}.txt"
    with open(os.path.join(out_folder, filename), 'w') as f:
        for idx in unique_indices:
            f.write("%s\n" % idx)

def identify_best_features_oh_excluded(cfg):
    oh_region_file = cfg.file_paths.explanation.oh_regions_file
    df_oh = pd.read_csv(oh_region_file, sep=', ', header=None)
    oh_regions = df_oh[0].values

    fi_folder = cfg.file_paths.explanation.deeplift_fi_folder
    
    files = os.listdir(fi_folder)
    files = [f for f in files if f.endswith('.csv')]

    cutoff = cfg.best_features_dataset.dataset.cutoff
    all_indices = []

    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
        df = df.sort_values("score", ascending=False)
        all_features = df.index.tolist()
        all_features = [int(x.split("_")[1]) for x in all_features]

        total_added = 0
        for k in range(len(all_features)):
            if all_features[k] not in oh_regions:
                all_indices.append(all_features[k])
                total_added += 1
                if total_added == cutoff:
                    break
        
        total_added = 0
        for k in range(len(all_features)):
            if all_features[-k] not in oh_regions:
                all_indices.append(all_features[-k])
                total_added += 1
                if total_added == cutoff:
                    break
    
    unique_indices = sorted(list(set(all_indices)))
    out_folder = cfg.file_paths.best_features_dataset.best_features_names_out_folder
    os.makedirs(out_folder, exist_ok=True)
    filename = f"Important_Indices_cutoff_{cutoff}.txt"
    with open(os.path.join(out_folder, filename), 'w') as f:
        for idx in unique_indices:
            f.write("%s\n" % idx)

def identify_best_features_coding(cfg):
    coding_region_file = cfg.file_paths.explanation.coding_regions_file
    df_non_coding = pd.read_csv(coding_region_file, sep=', ', header=None, index_col=0)
    coding_regions = df_non_coding.index.tolist()
    coding_regions = (np.array(coding_regions)-1).tolist()

    fi_folder = cfg.file_paths.explanation.deeplift_fi_folder
    
    files = os.listdir(fi_folder)
    files = [f for f in files if f.endswith('.csv')]

    cutoff = cfg.best_features_dataset.dataset.cutoff
    all_indices = []

    for i, file in enumerate(files):
        print(file)
        df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
        df = df.sort_values("score", ascending=False)
        all_features = df.index.tolist()
        all_features = [int(x.split("_")[1]) for x in all_features]

        total_added = 0
        for k in range(len(all_features)):
            if all_features[k] in coding_regions:
                all_indices.append(all_features[k])
                total_added += 1
                if total_added == cutoff:
                    break
        
        total_added = 0
        for k in range(len(all_features)):
            if all_features[-k] in coding_regions:
                all_indices.append(all_features[-k])
                total_added += 1
                if total_added == cutoff:
                    break
    
    unique_indices = sorted(list(set(all_indices)))
    out_folder = cfg.file_paths.best_features_dataset.best_features_names_out_folder
    os.makedirs(out_folder, exist_ok=True)
    filename = f"Important_Indices_cutoff_{cutoff}.txt"
    with open(os.path.join(out_folder, filename), 'w') as f:
        for idx in unique_indices:
            f.write("%s\n" % idx)



def create_best_features_dataset(cfg):
    num_top_serotypes = cfg.preprocessing.dataset.top_n
    dataset = BestFeaturesDataclassDataset(cfg, num_top_serotypes)
    dataset.generate_dataset()

def create_best_features_dataset_from_corr_vals(cfg):
    num_top_serotypes = cfg.preprocessing.dataset.top_n
    dataset = CorrelationFilteredDataset(cfg, num_top_serotypes)
    dataset.generate_dataset()

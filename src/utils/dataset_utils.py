from dataclasses import dataclass
from typing import List
import pandas as pd
import zipfile
import pickle
import os

@dataclass
class Sample:
    name: str
    indices: List[int]
    sparse_vals: List[int]
    serotype: str

def create_pickled_dataset(cfg):
    tfr_samples_folder = cfg.utils.locations.tfr_samples_folder
    serotype_file = cfg.utils.locations.serotype_file
    out_folder = cfg.utils.locations.out_folder

    os.makedirs(out_folder, exist_ok=True)

    serotype_df = pd.read_csv(serotype_file, index_col=0)
    i = 0
    total = 530000

    zip_files = [f for f in os.listdir(tfr_samples_folder) if f.endswith('.zip')]
    for zip_file in zip_files:
        with zipfile.ZipFile(os.path.join(tfr_samples_folder, zip_file), 'r') as zip_ref:
            file_names = zip_ref.namelist()
            for file_name in file_names:
                with zip_ref.open(file_name) as file:
                    tfr_indices = file.readline().decode('utf-8').strip().split(',')
                    tfr_indices = [int(i) for i in tfr_indices]
                    sparse_vals = file.readline().decode('utf-8').strip().split(',')
                    sparse_vals = [int(i) for i in sparse_vals]
                    
                    serotype = serotype_df.loc[file_name.split('.')[0], 'Serotype']
                    sample = Sample(file_name, tfr_indices, sparse_vals, serotype)
                i += 1
                if i % 1000 == 0:
                    print(f"{i}/{total}")

                with open(os.path.join(out_folder, file_name.split('.')[0] + '.pkl'), 'wb') as f:
                    pickle.dump(sample, f)


def filter_samples(cfg):
    norm_data_filename = cfg.file_paths.best_features_dataset.dataset_text_file
    df = pd.read_csv(norm_data_filename, sep=", ", index_col=0)
    unique_serotypes = df['serotype_encoded'].unique()

    for serotype in unique_serotypes:
        df_serotype = df[df['serotype_encoded'] == serotype]
    pass
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
import zipfile
import pickle
import os

def create_serotype_mapping(cfg):
    serotype_file_path = cfg.preprocessing.dataset.serotype_file_path
    serotype_df = pd.read_csv(serotype_file_path)
    serotype_counts = serotype_df['Serotype'].value_counts()
    serotype_descending = serotype_counts.index.tolist()
    
    if '0' in serotype_descending:
        serotype_descending.remove('0')
    
    serotype_mapping = {serotype: i for i, serotype in enumerate(serotype_descending)}
    
    # Write dict to file as txt
    mapping_file_location = cfg.utils.prepare_dataset.serotype_mapping_file
    mapping_dir = os.path.dirname(mapping_file_location)
    os.makedirs(mapping_dir, exist_ok=True)
    
    with open(mapping_file_location, 'w') as f:
        for serotype, i in serotype_mapping.items():
            f.write(f"{serotype},{i}\n")
    
    # Write dict to file as pickle
    mapping_pickle_location = cfg.utils.prepare_dataset.serotype_mapping_pickle
    with open(mapping_pickle_location, 'wb') as f:
        pickle.dump(serotype_mapping, f)

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

@dataclass
class Sample:
    name: str
    indices: np.ndarray
    sparse_vals: np.ndarray
    serotype: int

def create_pickled_dataset(cfg):
    tfr_samples_folder = cfg.preprocessing.dataset.zip_path
    serotype_file = cfg.preprocessing.dataset.serotype_file_path
    out_folder = cfg.utils.prepare_dataset.dataclass_out_folder_full
    serotype_mapping_dict_file = cfg.utils.prepare_dataset.serotype_mapping_pickle

    os.makedirs(out_folder, exist_ok=True)

    # if serotype_mapping_dict_file = cfg.utils.prepare_dataset.serotype_mapping_pickle path exists run line

    if not os.path.exists(serotype_mapping_dict_file):
        create_serotype_mapping(cfg)
    
    with open(serotype_mapping_dict_file, 'rb') as f:
        serotype_mapping = pickle.load(f)

    serotype_df = pd.read_csv(serotype_file, index_col=0)
    

    i = 0
    total = 530000

    zip_files = [f for f in os.listdir(tfr_samples_folder) if f.endswith('.zip')]
    for zip_file in zip_files:
        with zipfile.ZipFile(os.path.join(tfr_samples_folder, zip_file), 'r') as zip_ref:
            file_names = zip_ref.namelist()
            for file_name in file_names:
                i += 1
                sample_name = file_name.split('.')[0]
                out_filename = os.path.join(out_folder, sample_name + '.pkl')
                
                if os.path.exists(out_filename):
                    continue
                
                with zip_ref.open(file_name) as file:
                    tfr_indices = file.readline().decode('utf-8').strip().split(',')
                    tfr_indices = [int(i) for i in tfr_indices]
                    tfr_indices = np.array(tfr_indices, dtype='uint32')

                    sparse_vals = file.readline().decode('utf-8').strip().split(',')
                    sparse_vals = [int(i) for i in sparse_vals]
                    sparse_vals = np.array(sparse_vals, dtype='uint8')
                    
                    try:
                        serotype = serotype_df.loc[sample_name, 'Serotype']
                        serotype_encoded = serotype_mapping[serotype]
                    except:
                        serotype = serotype_df.loc[sample_name, 'Serotype'].values.tolist()[0]
                        serotype_encoded = serotype_mapping[serotype]

                    sample = Sample(sample_name, tfr_indices, sparse_vals, serotype_encoded)

                    with open(os.path.join(out_filename), 'wb') as f:
                        pickle.dump(sample, f)
                    serotype = serotype_df.loc[sample_name, 'Serotype']

                if i % 1000 == 0:
                    print(f"{i}/{total}")
                

                
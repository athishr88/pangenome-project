from dataclasses import dataclass
import time
import zipfile
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import os
from dataclasses import dataclass

@dataclass
class Sample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

class BestFeaturesDataclassDataset:
    def __init__(self, cfg: dict, num_top_serotypes: int):
        self.cfg = cfg
        self.num_top_serotypes = num_top_serotypes
        self.tfr_sample_ys_df = self._get_ys_df()
        self._best_features = self.get_best_features() #[93242, 34775, ...(~5000 or less)]
        
        self.sample_names_to_zipfile_map = {} #{SRR12345.csv: Sra10k_01.zip, SRR67890.csv: Sra10k_02.zip, ...}
        
        self.tfr_sample_names = self._get_tfr_sample_names() #[SRR12345.csv, SRR67890.csv, ... (~300000 or more)]
        
        self.write_to_file = False #TODO Change to True

        self.create_headers()

    def _update_classes(self, classes):
        self.cfg.preprocessing.dataset.classes = classes

    def _get_ys_df(self):
        serotype_file_path = self.cfg.preprocessing.dataset.serotype_file_path
        serotype_df = pd.read_csv(serotype_file_path)
        serotype_counts = serotype_df['Serotype'].value_counts()
        serotype_descending = serotype_counts.index.tolist()
        
        if '0' in serotype_descending:
            serotype_descending.remove('0')
        
        top_serotypes = serotype_descending[:self.num_top_serotypes]


        self._update_classes(top_serotypes)
        df_filtered = serotype_df[serotype_df['Serotype'].isin(top_serotypes)]
        return df_filtered

    def create_headers(self):
        out_folder = self.cfg.best_features_dataset.dataset.dataclass_out_folder
        os.makedirs(out_folder, exist_ok=True)

        with open(os.path.join(out_folder, f'best_{self.num_top_serotypes}_dataset.txt'), 'w') as f:
            f.write(f'SampleID, {str(self.best_features)[1:-1]}, serotype_encoded\n')

    @property
    def best_features(self):
        # Will be used to filter and position the best features in the dataset
        return self._best_features

    def get_best_features(self):
        best_features_file = self.cfg.best_features_dataset.dataset.best_features_in_file
        with open(best_features_file, 'r') as f:
            tfr_indices = f.readlines()
            tfr_indices = [int(i.rstrip()) for i in tfr_indices]
        return tfr_indices
    
    def _get_tfr_sample_names(self):
        df = self.tfr_sample_ys_df
        tfr_filenames_stem = df['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.csv' for name in tfr_filenames_stem]
        filenames_numbers = df['Src_file'].values
        filenames = [f'Sra10k_{num:02d}.zip' for num in filenames_numbers]
        for tfr_filename, filename in zip(tfr_filenames, filenames):
            self.sample_names_to_zipfile_map[tfr_filename] = filename

        return tfr_filenames
    
    def _get_full_path(self, filename):
        """Searches which zip file contains the given filename
        from the mapping and returns the full path to the file.
        """
        zip_file = self.sample_names_to_zipfile_map[filename]
        return os.path.join(self.cfg.preprocessing.dataset.zip_path, zip_file)
    
    def _get_dataclass_full_path(self, filename):
        dataclass_out_folder = self.cfg.utils.prepare_dataset.dataclass_out_folder_full
        filename_pkl = filename.split('.')[0] + '.pkl'
        return os.path.join(dataclass_out_folder, filename_pkl)
    
    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector of dim=n from indices and sparse values."""
        feature_vector_len = self.cfg.preprocessing.dataset.input_size #230000
        X = np.zeros(feature_vector_len) #[1, 230000]

        indices = np.array(indices, dtype=int)
        sparse_values = np.array(sparse_values, dtype=float)
        X[indices] = sparse_values

        X_filtered = X[self.best_features]

        return X_filtered 
    
    def _get_y(self, idx):
        filename = self.tfr_sample_names[idx].split('.')[0]
        serotype_mapping_file = self.cfg.utils.prepare_dataset.serotype_mapping_pickle
        with open(serotype_mapping_file, 'rb') as f:
            serotype_mapping = pickle.load(f)
        serotype = self.tfr_sample_ys_df[self.tfr_sample_ys_df[
            'SRA_ACCESSION_NUMBER'] == filename]['Serotype'].values[0]
        y = serotype_mapping[serotype]
        return y
    
    def _indices_and_sparse_vals_DC(self, idx):
        filename = self.tfr_sample_names[idx]
        full_path = self._get_dataclass_full_path(filename)
        with open(full_path, 'rb') as f:
            sample = pickle.load(f)

        return sample.indices, sample.sparse_vals
    
    def _indices_and_sparse_vals(self, idx):
        filename = self.tfr_sample_names[idx]
        full_path = self._get_full_path(filename)
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            with zip_ref.open(filename) as file:
                X = file.readline().decode('utf-8').strip().split(',')
                sparse_vals = file.readline().decode('utf-8').strip().split(',')
        return X, sparse_vals
    
    def generate_dataset(self):
        out_folder = self.cfg.best_features_dataset.dataset.dataclass_out_folder
        st = time.time()

        for idx, sample_id in enumerate(self.tfr_sample_names):
            if idx % 1000 == 1:
                
                # Calculate remaining time
                elapsed = time.time() - st
                samples_remaining = len(self.tfr_sample_names) - idx
                time_remaining = elapsed / idx * samples_remaining / 60
                print(f"Processing {idx}th sample, {time_remaining:.2f} minutes remaining.")
            
            X, sparse_vals = self._indices_and_sparse_vals_DC(idx)
            X_filtered = self._get_X(X, sparse_vals)
            y = self._get_y(idx)
            if self.write_to_file:
                with open(os.path.join(out_folder, f'best_{self.num_top_serotypes}_dataset_faster.txt'), 'a') as f:
                    f.write(f'{sample_id}, ')
                    for val in X_filtered:
                        f.write(f'{val}, ')
                    f.write(f'{y}\n')
            else:
                X_filtered = np.array(X_filtered, dtype='uint8')
                sample = Sample(sample_id, X_filtered, y)
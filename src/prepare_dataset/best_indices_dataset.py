from dataclasses import dataclass
from utils.logger import Logger
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
    indices: np.ndarray
    sparse_vals: np.ndarray
    serotype: int

@dataclass
class FilteredSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

class BestFeaturesDataclassDataset:
    def __init__(self, cfg: dict, num_top_serotypes: int):
        self.cfg = cfg
        self.num_top_serotypes = num_top_serotypes
        self.tfr_sample_ys_df = self._get_ys_df()
        self._best_features = self.get_best_features() #[93242, 34775, ...(~5000 or less)]
        
        self.tfr_sample_names = self._get_tfr_sample_names() #[SRR12345.csv, SRR67890.csv, ... (~300000 or more)]
        self.create_headers()
        self.logger = Logger(cfg)

    def create_headers(self):
        out_folder = self.cfg.file_paths.best_features_dataset.out_folder
        os.makedirs(out_folder, exist_ok=True)

        with open(os.path.join(out_folder, f'best_{self.num_top_serotypes}_dataset.txt'), 'w') as f:
            f.write(f'SampleID, {str(self.best_features)[1:-1]}, serotype_encoded\n')

    def _update_classes(self, classes):
        self.cfg.preprocessing.dataset.classes = classes

    def _get_ys_df(self):
        serotype_file_path = self.cfg.file_paths.supporting_files.serotype_file_path
        serotype_df = pd.read_csv(serotype_file_path)
        
        y_file = self.cfg.file_paths.supporting_files.serotype_mapping_file_path
        df = pd.read_csv(y_file, sep=', ', header=None)
        # Remove the rows in which the Serotype value is '-'
        classes = df[0].values.tolist()
        classes.remove('-')
        top_n = self.cfg.preprocessing.dataset.top_n
        top_serotypes = classes[:top_n]


        self._update_classes(top_serotypes)
        df_filtered = serotype_df[serotype_df['Serotype'].isin(top_serotypes)]
        return df_filtered

    @property
    def best_features(self):
        # Will be used to filter and position the best features in the dataset
        return self._best_features

    def get_best_features(self):
        best_features_folder = self.cfg.file_paths.best_features_dataset.best_features_names_out_folder
        cutoff = self.cfg.best_features_dataset.dataset.cutoff
        # filename = f'Important_Indices_fold_{cutoff}.txt'
        filename = f'Important_Indices_cutoff_{cutoff}.txt'
        best_features_file = os.path.join(best_features_folder, filename)

        with open(best_features_file, 'r') as f:
            tfr_indices = f.readlines()
            tfr_indices = [int(i.rstrip()) for i in tfr_indices]

        print(f"Number of best features: {len(tfr_indices)}")
        return tfr_indices
    
    def _get_tfr_sample_names(self):
        df = self.tfr_sample_ys_df
        tfr_filenames_stem = df['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.pkl' for name in tfr_filenames_stem]
        return tfr_filenames

    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector of dim=n from indices and sparse values."""
        feature_vector_len = 236071 #230000
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
    
    def _indices_and_sparse_vals(self, idx):
        dataclass_folder = self.cfg.file_paths.full_dataset.dataclass_in_folder
        filename = self.tfr_sample_names[idx]
        with open(os.path.join(dataclass_folder, filename), 'rb') as f:
            sample = pickle.load(f)

        return sample.indices, sample.sparse_vals, sample.serotype
    
    def save_dataset_as_txt(self, sample_id, X_filtered, y, out_folder):
        with open(os.path.join(out_folder, f'best_{self.num_top_serotypes}_dataset.txt'), 'a') as f:
            f.write(f'{sample_id}, ')
            for val in X_filtered:
                f.write(f'{val}, ')
            f.write(f'{y}\n')

    def save_dataset_as_dataclass(self, sample_id, X_filtered, y, out_folder):
        filtered_sample = FilteredSample(name=sample_id.split(".")[0], sparse_vals=X_filtered, serotype=y)
        with open(os.path.join(out_folder, f'{sample_id}'), 'wb') as f:
            pickle.dump(filtered_sample, f)

    def calculate_remaining_time(self, idx, st):
        elapsed = time.time() - st
        samples_remaining = len(self.tfr_sample_names) - idx
        time_remaining = elapsed / idx * samples_remaining / 60
        print(f"Processing {idx}th sample, {time_remaining:.2f} minutes remaining.")

    def generate_dataset(self):
        out_folder = self.cfg.file_paths.best_features_dataset.out_folder
        save_txt = self.cfg.best_features_dataset.dataset.save_txt
        save_dataclass = self.cfg.best_features_dataset.dataset.save_dataclass

        dataclass_folder = out_folder + '/dataclass'
        metadata_folder = out_folder + '/metadata'
        os.makedirs(dataclass_folder, exist_ok=True)
        os.makedirs(metadata_folder, exist_ok=True)

        st = time.time()

        for idx, sample_id in enumerate(self.tfr_sample_names):
            X, sparse_vals, y = self._indices_and_sparse_vals(idx)
            X_filtered = self._get_X(X, sparse_vals)

            if save_txt:
                self.save_dataset_as_txt(sample_id, X_filtered, y, out_folder)

            if save_dataclass:
                self.save_dataset_as_dataclass(sample_id, X_filtered, y, dataclass_folder)

            if idx % 10000 == 1:
                self.calculate_remaining_time(idx, st)

class CorrelationFilteredDataset(BestFeaturesDataclassDataset):
    def __init__(self, cfg: dict, num_top_serotypes: int):
        super().__init__(cfg, num_top_serotypes)

    def get_best_features(self):
        print("Getting best features for correlation filtered dataset")
        filtered_indices_file = self.cfg.file_paths.corr_matrix.filtered_indices_file
        with open(filtered_indices_file, 'r') as f:
            tfr_indices = f.readlines()
            tfr_indices = [int(i.rstrip()) for i in tfr_indices]
import os
import torch
import pickle
import zipfile
import numpy as np
import pandas as pd
from utils.logger import Logger
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class BestSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

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

class SingleClassDataset(Dataset):
    def __init__(self, cfg, serotype) -> None:
        super().__init__()
        self.cfg = cfg
        self._y_map = None
        self.serotype = serotype
        self.tfr_sample_names = self._get_tfr_sample_names()

    @property
    def y_map(self):
        all_y_map_file = "cache/dataclass/metadata/serotype_mapping.pkl"
        with open(all_y_map_file, 'rb') as f:
            self._y_map = pickle.load(f)
        
        return self._y_map

    def _get_tfr_sample_names(self):
        y_file = self.cfg.file_paths.supporting_files.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype']==self.serotype]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.pkl' for name in tfr_filenames_stem]

        return tfr_filenames
    
    def _get_full_path(self, filename):
        """Searches which zip file contains the given filename
        from the mapping and returns the full path to the file.
        """
        dataclass_folder = self.cfg.file_paths.full_dataset.dataclass_in_folder
        return os.path.join(dataclass_folder, filename)
    
    def _indices_and_sparse_vals(self, idx):
        filename = self.tfr_sample_names[idx]
        full_path = self._get_full_path(filename)
        with open(full_path, 'rb') as f:
            sample = pickle.load(f)
        X = sample.indices
        sparse_vals = sample.sparse_vals
        return X, sparse_vals
    
    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector from indices and sparse values."""
        feature_vector_len = self.cfg.preprocessing.dataset.input_size
        X = np.zeros(feature_vector_len)
        indices = np.array(indices, dtype=int)
        if self.cfg.preprocessing.dataset.sparse_vals_used:
            sparse_values = np.array(sparse_values, dtype=float)
            X[indices] = sparse_values
        else:
            X[indices] = 1.0
        return X
    
    def _get_y(self):
        y = self.y_map[self.serotype]
        return y

    def __len__(self):
        return len(self.tfr_sample_names)

    def __getitem__(self, idx):
        indices, sparse_values = self._indices_and_sparse_vals(idx)
        X = self._get_X(indices, sparse_values)
        y = self._get_y()
        return X, y
    
class SingleClassFiltered(SingleClassDataset):
    def __init__(self, cfg, serotype):
        super().__init__(cfg, serotype)
        self.excluded_indices = self._get_excluded_indices()

    def _get_excluded_indices(self):
        filename = self.cfg.file_paths.excluded_indices.excluded_indices_file
        df = pd.read_csv(filename, header=None)
        return df[0].values
    
    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector from indices and sparse values."""
        fixed_feature_len = self.cfg.preprocessing.dataset.fixed_feature_len
        X = np.zeros(fixed_feature_len)
        indices = np.array(indices, dtype=int)
        if self.cfg.preprocessing.dataset.sparse_vals_used:
            sparse_values = np.array(sparse_values, dtype=float)
            X[indices] = sparse_values
        else:
            X[indices] = 1.0

        # Drop the columns corresponding to the excluded indices
        X_out = np.delete(X, self.excluded_indices)
        return X_out

class BestSingleClassDatasetCorrFiltered(Dataset):
    def __init__(self, cfg, serotype) -> None:
        super().__init__()
        self.cfg = cfg
        self._y_map = None
        self.serotype = serotype
        self.tfr_sample_names = self._get_tfr_sample_names()
        self.corr_included_indices = self._get_corr_included_indices()
        self.cfg.preprocessing.dataset.input_size = len(self.corr_included_indices)


    @property
    def y_map(self):
        all_y_map_file = "cache/dataclass/metadata/serotype_mapping.pkl"
        with open(all_y_map_file, 'rb') as f:
            self._y_map = pickle.load(f)
        
        return self._y_map

    def _get_tfr_sample_names(self):
        y_file = self.cfg.file_paths.supporting_files.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype']==self.serotype]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.pkl' for name in tfr_filenames_stem]

        return tfr_filenames
    
    def _get_corr_included_indices(self):
        # Removes all the indices which are correlated and selects
        # a random index from the correlated indices
        filename = self.cfg.file_paths.corr_matrix.filtered_indices_file

        with open(filename, "r") as f:
            features = f.readlines()
            features = [int(i.rstrip()) for i in features]

        # Rest of the code is to get the "indices" of the features from the best features
        full_features_file = self.cfg.file_paths.best_features_dataset.best_features_names_out_folder
        cutoff = self.cfg.best_features_dataset.dataset.cutoff
        filename = f'Important_Indices_cutoff_{cutoff}.txt'
        with open(os.path.join(full_features_file, filename), 'r') as f:
            tfr_indices = f.readlines()
            tfr_indices = [int(i.rstrip()) for i in tfr_indices]

        excluded_features_index_nums = [tfr_indices.index(i) for i in features]
        return excluded_features_index_nums

    def __len__(self):
        return len(self.tfr_sample_names)

    def __getitem__(self, idx):
        sample_name = self.tfr_sample_names[idx]

        # dataclass_folder = self.cfg.utils.prepare_dataset.dataclass_out_folder_full
        dataclass_folder = self.cfg.file_paths.best_features_dataset.dataclass_in_folder
        
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = sample.sparse_vals #np.uint8
        # Convert all values above 0 to 1
        # X = np.where(X > 0, 1, 0) # TODO uncomment for MLP
        X[X == 99] = 21
        X = X[self.corr_included_indices]
        y = sample.serotype

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        return X, y
   
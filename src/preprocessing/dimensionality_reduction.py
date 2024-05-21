from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import zipfile
import os

class TFRSubsetPartialDataset(Dataset):
    data_initialized = False

    @classmethod
    def initialize_data(cls, cfg):
        if not cls.data_initialized:
            cls.cfg = cfg
            cls.classes = cfg.preprocessing.dataset.classes #TODO remove
            cls.sample_names_to_zipfile_map = {}
            cls.tfr_sample_ys_df = None
            cls.tfr_sample_names = cls._get_tfr_sample_names()
            cls.train_indices, cls.val_indices, cls.test_indices = cls._get_train_val_test_indices()
            cls.data_initialized = True

    @classmethod
    def _get_tfr_sample_names(cls):
        y_file = cls.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.cfg.preprocessing.dataset.classes)]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.csv' for name in tfr_filenames_stem]
        filenames_numbers = df_filtered['Src_file'].values
        filenames = [f'Sra10k_{num:02d}.zip' for num in filenames_numbers]
        for tfr_filename, filename in zip(tfr_filenames, filenames):
            cls.sample_names_to_zipfile_map[tfr_filename] = filename

        le = LabelEncoder()
        df_filtered['serotype_encoded'] = le.fit_transform(df_filtered['Serotype'])
        cls.tfr_sample_ys_df = df_filtered
        return tfr_filenames
    
    @classmethod
    def _get_train_val_test_indices(self):
        val_size = self.cfg.preprocessing.dataset.val_size
        test_size = self.cfg.preprocessing.dataset.test_size
        random_state = self.cfg.preprocessing.dataset.random_state
        indices = list(range(len(self.tfr_sample_names)))
        temp_train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            temp_train_indices, test_size=val_size_adjusted, random_state=random_state
        )

        return train_indices, val_indices, test_indices
    
    @classmethod
    def from_split(cls, split, top_n):
        instance = cls(split, top_n)
        return instance

    def __init__(self, split, top_n):
        """
        @param split: str, {'train', 'val' or 'test'}
        """
        if not self.data_initialized:
            raise ValueError("Data has not been initialized. Call 'initialize_data' first.")
        super().__init__()
        self.top_n = top_n
        self.split = split
        self.top_indices = self._get_top_and_bottom_n_indices()

    def _get_explanation_filename(self, serotype):
        explanation_dir = self.cfg.explanation.deeplift.explanations_folder
        filename = f"{serotype}.csv"
        filename = os.path.join(explanation_dir, filename)
        return filename
    
    def _get_top_and_bottom_n_indices(self):
        top_n_indices = self._get_top_n_indices()
        bottom_n_indices = self._get_bottom_n_indices()
        indices = top_n_indices + bottom_n_indices
        return indices
    
    def _get_bottom_n_indices(self):
        serotypes = self.cfg.preprocessing.dataset.classes
        bottom_indices_from_all_serotypes = []
        for serotype in serotypes:
            filename = self._get_explanation_filename(serotype)
            df = pd.read_csv(filename, index_col=0)
            bottom_n_indices = df.iloc[-self.top_n:].index
            bottom_n_indices = [int(idx.split('_')[1]) for idx in bottom_n_indices]
            bottom_indices_from_all_serotypes.extend(bottom_n_indices)
        return bottom_indices_from_all_serotypes

    def _get_top_n_indices(self):
        serotypes = self.cfg.preprocessing.dataset.classes
        top_indices_from_all_serotypes = []
        for serotype in serotypes:
            filename = self._get_explanation_filename(serotype)
            df = pd.read_csv(filename, index_col=0)
            top_n_indices = df.iloc[:self.top_n].index
            top_n_indices = [int(idx.split('_')[1]) for idx in top_n_indices]
            top_indices_from_all_serotypes.extend(top_n_indices)
        return top_indices_from_all_serotypes


    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector of dim=n from indices and sparse values."""
        feature_vector_len = self.cfg.preprocessing.dataset.input_size
        # top_indices = self._get_top_and_bottom_n_indices() #TODO change if necessary
        X = np.zeros(feature_vector_len)

        indices = np.array(indices, dtype=int)
        if self.cfg.preprocessing.dataset.sparse_vals_used:
            sparse_values = np.array(sparse_values, dtype=float)
            X[indices] = sparse_values
            raise NotImplementedError("Sparse values not implemented.")
        else:
            X[indices] = 1.0

        X_filtered = X[self.top_indices]

        return X_filtered 
    
    def _get_y(self, idx):
        filename = self.tfr_sample_names[idx].split('.')[0]
        y = self.tfr_sample_ys_df[self.tfr_sample_ys_df[
            'SRA_ACCESSION_NUMBER'] == filename]['serotype_encoded'].values[0]
        return y
    
    def _get_full_path(self, filename):
        """Searches which zip file contains the given filename
        from the mapping and returns the full path to the file.
        """
        zip_file = self.sample_names_to_zipfile_map[filename]
        return os.path.join(self.cfg.preprocessing.dataset.zip_path, zip_file)
    
    
    def _indices_and_sparse_vals(self, idx):
        filename = self.tfr_sample_names[idx]
        full_path = self._get_full_path(filename)
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            with zip_ref.open(filename) as file:
                X = file.readline().decode('utf-8').strip().split(',')
                sparse_vals = file.readline().decode('utf-8').strip().split(',')
        return X, sparse_vals
    
    def __len__(self):
        if self.split == 'train':
            return len(self.train_indices)
        elif self.split == 'val':
            return len(self.val_indices)
        else:  # 'test'
            return len(self.test_indices)
        
    def __getitem__(self, idx):
        if self.split == 'train':
            actual_idx = self.train_indices[idx]
        elif self.split == 'val':
            actual_idx = self.val_indices[idx]
        else:  # 'test'
            actual_idx = self.test_indices[idx]

        indices, sparse_values = self._indices_and_sparse_vals(actual_idx)
        X = self._get_X(indices, sparse_values)
        y = self._get_y(actual_idx)
        return X, y
        
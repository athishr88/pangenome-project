import os
import zipfile
import numpy as np
import pandas as pd
from utils.logger import Logger
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class TFRecordsDataset(Dataset):
    data_initialized = False

    @classmethod
    def initialize_data(cls, cfg):
        if not cls.data_initialized:
            cls.cfg = cfg
            cls.sample_names_to_zipfile_map = {}
            cls.tfr_sample_names = cls._get_tfr_sample_names()
            cls.tfr_sample_ys_df = cls._get_tfr_sample_ys()
            cls.train_indices, cls.val_indices, cls.test_indices = cls._get_train_val_test_indices()
            cls.data_initialized = True
    
    @classmethod
    def from_split(cls, split):
        instance = cls(split)
        return instance
    
    def __init__(self, split):
        """
        @param cfg: Config object
        @param split: str, {'train', 'val' or 'test'}
        """
        if not self.data_initialized:
            raise ValueError("Data has not been initialized. Call 'initialize_data' first.")
        super().__init__()
        self.split = split

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
    def _get_tfr_sample_ys(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        le = LabelEncoder()
        df['serotype_encoded'] = le.fit_transform(df['Serotype'])
        return df
    
    @classmethod
    def _get_tfr_sample_names(self):
        zip_dir = self.cfg.preprocessing.dataset.zip_path
        all_files_list = []

        for filename in os.listdir(zip_dir):
            if filename.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(zip_dir, filename), 'r') as zip_ref:
                    tfr_filenames = [name for name in zip_ref.namelist() if name.endswith('.csv')]
                    all_files_list.extend(tfr_filenames)
                    for tfr_filename in tfr_filenames:
                        self.sample_names_to_zipfile_map[tfr_filename] = filename

        return all_files_list
    
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

    
    def _get_y(self, idx):
        filename = self.tfr_sample_names[idx].split('.')[0]
        y = self.tfr_sample_ys_df[self.tfr_sample_ys_df[
            'SRA_ACCESSION_NUMBER'] == filename]['serotype_encoded'].values[0]
        return y

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
    

class TFRecordsPartialDataset(Dataset):
    data_initialized = False

    @classmethod
    def initialize_data(cls, cfg):
        if not cls.data_initialized:
            cls.cfg = cfg
            cls.classes = cfg.preprocessing.dataset.classes
            cls.sample_names_to_zipfile_map = {}
            cls.tfr_sample_ys_df = None
            cls.tfr_sample_names = cls._get_tfr_sample_names()
            cls.train_indices, cls.val_indices, cls.test_indices = cls._get_train_val_test_indices()
            cls.data_initialized = True

    @classmethod
    def from_split(cls, split):
        instance = cls(split)
        return instance
    
    def __init__(self, split):
        """
        @param cfg: Config object
        @param split: str, {'train', 'val' or 'test'}
        """
        if not self.data_initialized:
            raise ValueError("Data has not been initialized. Call 'initialize_data' first.")
        super().__init__()
        self.split = split

    @classmethod
    def _get_tfr_sample_names(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(self.classes)]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.csv' for name in tfr_filenames_stem]
        filenames_numbers = df_filtered['Src_file'].values
        filenames = [f'Sra10k_{num:02d}.zip' for num in filenames_numbers]
        for tfr_filename, filename in zip(tfr_filenames, filenames):
            self.sample_names_to_zipfile_map[tfr_filename] = filename

        le = LabelEncoder()
        df_filtered['serotype_encoded'] = le.fit_transform(df_filtered['Serotype'])
        self.tfr_sample_ys_df = df_filtered
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

    
    def _get_y(self, idx):
        filename = self.tfr_sample_names[idx].split('.')[0]
        y = self.tfr_sample_ys_df[self.tfr_sample_ys_df[
            'SRA_ACCESSION_NUMBER'] == filename]['serotype_encoded'].values[0]
        return y

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
    
class SingleClassDataset(Dataset):
    def __init__(self, cfg, serotype) -> None:
        super().__init__()
        self.cfg = cfg
        self._y_map = None
        self.serotype = serotype
        self.sample_names_to_zipfile_map = {}
        self.tfr_sample_names = self._get_tfr_sample_names()
        self.indices = self._get_indices()

    @property
    def y_map(self):
        return self._y_map

    def _get_tfr_sample_names(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype']==self.serotype]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.csv' for name in tfr_filenames_stem]
        filenames_numbers = df_filtered['Src_file'].values
        filenames = [f'Sra10k_{num:02d}.zip' for num in filenames_numbers]
        for tfr_filename, filename in zip(tfr_filenames, filenames):
            self.sample_names_to_zipfile_map[tfr_filename] = filename

        le = LabelEncoder()
        df_filtered['serotype_encoded'] = le.fit_transform(df_filtered['Serotype'])
        self.tfr_sample_ys_df = df_filtered
        self._y_map = {class_label:index for index, class_label in enumerate(le.classes_)}
        return tfr_filenames
    
    def _get_indices(self):
        indices = list(range(len(self.tfr_sample_names)))
        return indices
    
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

    
    def _get_y(self, idx):
        filename = self.tfr_sample_names[idx].split('.')[0]
        y = self.tfr_sample_ys_df[self.tfr_sample_ys_df[
            'SRA_ACCESSION_NUMBER'] == filename]['serotype_encoded'].values[0]
        return y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        indices, sparse_values = self._indices_and_sparse_vals(actual_idx)
        X = self._get_X(indices, sparse_values)
        y = self._get_y(actual_idx)
        return X, y
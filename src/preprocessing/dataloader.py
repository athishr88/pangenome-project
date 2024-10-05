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

# @dataclass
# class Sample:
#     name: str
#     sparse_vals: np.ndarray
#     serotype: int

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

class TFRecordsPartialDatasetDataclass(Dataset):
    data_initialized = False

    @classmethod
    def initialize_data(cls, cfg):
        if not cls.data_initialized:
            cls.logger = Logger(cfg)

            cls.cfg = cfg
            cls.classes = cls._get_classes()
            cls.train_samples, cls.val_samples, cls.test_samples = cls._get_train_val_test_samples()
            cls.class_weights = cls._calculate_class_weights()
            cls.data_initialized = True

    @classmethod
    def _calculate_class_weights(cls):
        y_file = cls.cfg.file_paths.supporting_files.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.classes)]
        class_weights = df_filtered['Serotype'].value_counts(normalize=True).sort_index()
        class_weights = class_weights.sort_values(ascending=False)
        class_weights = 1/class_weights
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.values
        return class_weights

    @classmethod
    def _get_classes(self):
        y_file = self.cfg.file_paths.supporting_files.serotype_mapping_file_path
        df = pd.read_csv(y_file, sep=', ', header=None)
        # Remove the rows in which the Serotype value is '-'
        classes = df[0].values.tolist()
        classes.remove('-')
        top_n = self.cfg.preprocessing.dataset.top_n
        top_serotypes = classes[:top_n]
        self.logger.log(f"Top {top_n} serotypes: {top_serotypes}")
        return top_serotypes

    @classmethod
    def _get_train_val_test_samples(cls):
        y_file = cls.cfg.file_paths.supporting_files.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.classes)]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.pkl' for name in tfr_filenames_stem]

        val_size = cls.cfg.preprocessing.dataset.val_size
        test_size = cls.cfg.preprocessing.dataset.test_size
        random_state = cls.cfg.preprocessing.dataset.random_state

        temp_train_samples, test_samples = train_test_split(
            tfr_filenames, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_samples, val_samples = train_test_split(
            temp_train_samples, test_size=val_size_adjusted, random_state=random_state
        )
        return train_samples, val_samples, test_samples
    
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


    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector from indices and sparse values."""
        feature_vector_len = self.cfg.preprocessing.dataset.input_size
        X = np.zeros(feature_vector_len)
        if self.cfg.preprocessing.dataset.sparse_vals_used:
            sparse_values = np.array(sparse_values, dtype=float)
            X[indices] = sparse_values
        else:
            X[indices] = 1.0
        return X

    def __len__(self):
        if self.split == 'train':
            return len(self.train_samples)
        elif self.split == 'val':
            return len(self.val_samples)
        else:
            return len(self.test_samples)
        
    def __getitem__(self, idx):
        if self.split == 'train':
            sample_name = self.train_samples[idx]
        elif self.split == 'val':
            sample_name = self.val_samples[idx]
        else:
            sample_name = self.test_samples[idx]

        dataclass_folder = self.cfg.file_paths.full_dataset.dataclass_in_folder
        
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = self._get_X(sample.indices, sample.sparse_vals)
        y = sample.serotype
        # Adjustment for '-'
        if y > 116:
            y -= 1
        return X, y

class TFRBestFeaturesDataclass(Dataset):
    # Dataset comprising best features of top 97 serotypes
    data_initialized = False

    @classmethod
    def initialize_data(cls, cfg):
        if not cls.data_initialized:
            cls.logger = Logger(cfg)

            cls.cfg = cfg
            cls._config_files_prepare()
            cls.classes = cfg.preprocessing.dataset.classes
            cls.train_samples, cls.val_samples, cls.test_samples = cls._get_train_val_test_samples()
            cls.class_weights = cls._calculate_class_weights()
            cls.data_initialized = True

    @classmethod
    def _config_files_prepare(self):
        y_file = self.cfg.file_paths.supporting_files.serotype_mapping_file_path
        df = pd.read_csv(y_file, sep=', ', header=None)
        # Remove the rows in which the Serotype value is '-'
        classes = df[0].values.tolist()
        classes.remove('-')
        top_n = self.cfg.preprocessing.dataset.top_n
        top_serotypes = classes[:top_n]
        self.cfg.preprocessing.dataset.classes = top_serotypes
        self.logger.log(f"Top {top_n} serotypes: {top_serotypes}")


    @classmethod
    def _calculate_class_weights(cls):
        y_file = cls.cfg.file_paths.supporting_files.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.classes)]
        class_weights = df_filtered['Serotype'].value_counts(normalize=True).sort_index()
        class_weights = class_weights.sort_values(ascending=False)
        class_weights = 1/class_weights
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.values
        return class_weights

    @classmethod
    def _get_train_val_test_samples(cls):
        y_file = cls.cfg.file_paths.supporting_files.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.classes)]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.pkl' for name in tfr_filenames_stem]

        val_size = cls.cfg.preprocessing.dataset.val_size
        test_size = cls.cfg.preprocessing.dataset.test_size
        random_state = cls.cfg.preprocessing.dataset.random_state

        temp_train_samples, test_samples = train_test_split(
            tfr_filenames, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_samples, val_samples = train_test_split(
            temp_train_samples, test_size=val_size_adjusted, random_state=random_state
        )
        return train_samples, val_samples, test_samples
    
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

    def __len__(self):
        if self.split == 'train':
            return len(self.train_samples)
        elif self.split == 'val':
            return len(self.val_samples)
        else:
            return len(self.test_samples)
        
    def __getitem__(self, idx):
        if self.split == 'train':
            sample_name = self.train_samples[idx]
        elif self.split == 'val':
            sample_name = self.val_samples[idx]
        else:
            sample_name = self.test_samples[idx]

        # dataclass_folder = self.cfg.utils.prepare_dataset.dataclass_out_folder_full
        dataclass_folder = self.cfg.file_paths.best_features_dataset.dataclass_in_folder
        
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = sample.sparse_vals #np.uint8
        # Convert all values above 0 to 1
        X = np.where(X > 0, 1, 0)
        y = sample.serotype
        # Adjustment for '-'
        if y > 116:
            y -= 1

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

class CorrFilteredDataset(TFRBestFeaturesDataclass):
    data_initialized = False

    def __init__(self, split):
        super().__init__(split)
        self.corr_included_indices = self._get_corr_included_indices()
        self.cfg.preprocessing.dataset.input_size = len(self.corr_included_indices)

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
        self.logger.log(f"Following indices used {features}")
        return excluded_features_index_nums
    
    def __getitem__(self, idx):
        if self.split == 'train':
            sample_name = self.train_samples[idx]
        elif self.split == 'val':
            sample_name = self.val_samples[idx]
        else:
            sample_name = self.test_samples[idx]

        # dataclass_folder = self.cfg.utils.prepare_dataset.dataclass_out_folder_full
        dataclass_folder = self.cfg.file_paths.best_features_dataset.dataclass_in_folder
        
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = sample.sparse_vals #np.uint8
        # Convert all values above 0 to 1
        # X = np.where(X > 0, 1, 0) # TODO uncomment for MLP
        vocab_size = self.cfg.preprocessing.dataset.vocab_size
        X[X == 99] = 21
        X = X[self.corr_included_indices]
        X = np.eye(vocab_size)[np.array(X, dtype=int)]
        y = sample.serotype
        # Adjustment for '-'
        if y > 116:
            y -= 1

        # X = torch.tensor(X, dtype=torch.float)
        # y = torch.tensor(y, dtype=torch.long)
        return X, y

class CorrFilteredDatasetTR(TFRBestFeaturesDataclass):
    data_initialized = False

    def __init__(self, split):
        super().__init__(split)
        self.corr_included_indices = self._get_corr_included_indices()
        
        self.cfg.preprocessing.dataset.input_size = len(self.corr_included_indices)

        self.normed_indices = (torch.tensor(self.corr_included_indices)-1)/max(self.corr_included_indices)
        self.normed_indices = torch.tensor(self.normed_indices, dtype=torch.float)

        self.fixed_indices = torch.arange(len(self.corr_included_indices), dtype=torch.long)


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
        self.logger.log(f"Following indices used {features}")
        return excluded_features_index_nums
    
    def __getitem__(self, idx):
        if self.split == 'train':
            sample_name = self.train_samples[idx]
        elif self.split == 'val':
            sample_name = self.val_samples[idx]
        else:
            sample_name = self.test_samples[idx]

        # dataclass_folder = self.cfg.utils.prepare_dataset.dataclass_out_folder_full
        dataclass_folder = self.cfg.file_paths.best_features_dataset.dataclass_in_folder
        
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = sample.sparse_vals #np.uint8
        # Convert all values above 0 to 1
        # X = np.where(X > 0, 1, 0) # TODO uncomment for MLP
        vocab_size = self.cfg.preprocessing.dataset.vocab_size
        X[X == 99] = 21
        X = X[self.corr_included_indices]
        # X = np.eye(vocab_size)[np.array(X, dtype=int)]
        y = sample.serotype
        # Adjustment for '-'
        if y > 116:
            y -= 1

        # X = torch.tensor(X, dtype=torch.float)
        # y = torch.tensor(y, dtype=torch.long)
        return self.fixed_indices, self.normed_indices, X, y

class TFRFilteredFeaturesDataclass(TFRecordsPartialDatasetDataclass):
    data_initialized = False

    def _get_X(self, indices, sparse_values):
        """Produces fixed length feature vector from indices and sparse values."""
        feature_vector_len = self.cfg.preprocessing.dataset.fixed_feature_len
        X = np.zeros(feature_vector_len)
        if self.cfg.preprocessing.dataset.sparse_vals_used:
            sparse_values = np.array(sparse_values, dtype=float)
            X[indices] = sparse_values
        else:
            X[indices] = 1.0

        # Drop the columns corresponding to the excluded indices
        X_out = np.delete(X, self.excluded_indices)
        return X_out
    
    def __init__(self, split):
        super().__init__(split)
        self.excluded_indices = self._get_excluded_indices() # array([11,12,13, ..]

    def _get_excluded_indices(self):
        filename = self.cfg.file_paths.excluded_indices.excluded_indices_file
        df = pd.read_csv(filename, header=None)
        return df[0].values

class TFRTransformerDataset(TFRBestFeaturesDataclass):
    @classmethod
    def initialize_transformer_dataset(self, cfg):
        super().initialize_data(cfg)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample_name = self.train_samples[idx]
        elif self.split == 'val':
            sample_name = self.val_samples[idx]
        else:
            sample_name = self.test_samples[idx]

        dataclass_folder = self.cfg.file_paths.best_features_dataset.dataclass_in_folder
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = sample.sparse_vals #np.uint8
        # If the value is 99, change it to 21
        X = np.where(X == 99, 21, X)
        y = sample.serotype
        # Adjustment for '-'
        if y > 116:
            y -= 1
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return X, y
    
class TFRTransformerDatasetVocab3(TFRBestFeaturesDataclass):
    @classmethod
    def initialize_transformer_dataset(self, cfg):
        super().initialize_data(cfg)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample_name = self.train_samples[idx]
        elif self.split == 'val':
            sample_name = self.val_samples[idx]
        else:
            sample_name = self.test_samples[idx]

        dataclass_folder = self.cfg.file_paths.best_features_dataset.dataclass_in_folder
        
        with open(os.path.join(dataclass_folder, sample_name), 'rb') as f:
            sample = pickle.load(f)

        X = sample.sparse_vals #np.uint8
        # If the value is 99, change it to 21
        X = np.where(X == 99, 0, X)
        # Rest of the values are 1
        X = np.where(X > 0, 1, X)
        y = sample.serotype
        # Adjustment for '-'
        if y > 116:
            y -= 1
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return X, y
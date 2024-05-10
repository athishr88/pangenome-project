import os
import zipfile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class TFRecordsDataset(Dataset):
    def __init__(self, cfg, split):
        """
        @param cfg: Config object
        @param split: str, {'train', 'val' or 'test'}
        """
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.sample_names_to_zipfile_map = {}
        self.tfr_sample_names = self._get_tfr_sample_names()
        self.tfr_sample_ys_df = self._get_tfr_sample_ys()
        self.train_indices, self.val_indices, self.test_indices = self._get_train_val_test_indices()

    def _get_train_val_test_indices(self):
        val_size = self.cfg.training.transformers.val_size
        test_size = self.cfg.training.transformers.test_size
        random_state = self.cfg.training.transformers.random_state
        indices = list(range(len(self.tfr_sample_names)))
        temp_train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            temp_train_indices, test_size=val_size_adjusted, random_state=random_state
        )

        return train_indices, val_indices, test_indices

    def _get_tfr_sample_ys(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        le = LabelEncoder()
        df['serotype_encoded'] = le.fit_transform(df['Serotype'])
        return df

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
        feature_vector_len = self.cfg.preprocessing.dataset.highest_index+1
        X = np.zeros(feature_vector_len)
        indices = np.array(indices, dtype=int)
        sparse_values = np.array(sparse_values, dtype=float)
        X[indices] = sparse_values
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

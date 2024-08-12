from torch.utils.data import Dataset
import os
import json
import numpy as np
import pandas as pd
from utils.logger import Logger
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class TFRWindowedDataset(Dataset):
    data_initialized = False

    @classmethod
    def initialize_data(cls, cfg):
        if not cls.data_initialized:
            cls.logger = Logger(cfg)

            cls.cfg = cfg
            cls.classes = cfg.preprocessing.dataset.classes
            cls.train_samples, cls.val_samples, cls.test_samples = cls._get_train_val_test_samples_temp()
            cls.class_weights = cls._calculate_class_weights()
            cls.data_initialized = True

    @classmethod
    def _calculate_class_weights(cls):
        y_file = cls.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.classes)]
        class_weights = df_filtered['Serotype'].value_counts(normalize=True).sort_index()
        class_weights = class_weights.sort_values(ascending=False)
        class_weights = 1/class_weights
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.values
        return class_weights
    
    @classmethod
    def _get_train_val_test_samples_temp(cls):
        windowed_data_dir = cls.cfg.preprocessing.dataset.windowed_data_dir
        tfr_filenames = os.listdir(windowed_data_dir)

        val_size = cls.cfg.preprocessing.dataset.val_size
        test_size = cls.cfg.preprocessing.dataset.test_size
        random_state = cls.cfg.preprocessing.dataset.random_state

        temp_train_samples, test_samples = train_test_split(tfr_filenames, test_size=test_size, random_state=random_state)
        val_size_adjusted = val_size / (1 - test_size)
        train_samples, val_samples = train_test_split(temp_train_samples, test_size=val_size_adjusted, random_state=random_state)
        return train_samples, val_samples, test_samples
    
    @classmethod
    def _get_train_val_test_samples(cls):
        y_file = cls.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(cls.classes)]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        num_windows = cls.cfg.best_features_dataset.windowed.num_windows

        # for stem in tfr_filenames_stem:
        #     for window_num in range(num_windows):
        #         tfr_filenames = f'{stem}_{window_num}.json'

        # windowed_data_dir = cls.cfg.preprocessing.dataset.windowed_data_dir
        # tfr_filenames = os.listdir(windowed_data_dir)
        # tfr_filenames = [file for file in tfr_filenames if file.endswith('.json')]

        val_size = cls.cfg.preprocessing.dataset.val_size
        test_size = cls.cfg.preprocessing.dataset.test_size
        random_state = cls.cfg.preprocessing.dataset.random_state

        temp_train_samples, test_samples = train_test_split(
            tfr_filenames_stem, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_samples, val_samples = train_test_split(
            temp_train_samples, test_size=val_size_adjusted, random_state=random_state
        )

        train_sample_windows = []
        for train_sample in train_samples:
            for window_num in range(num_windows):
                train_sample_window = f'{train_sample}_{window_num+1}.json'
                train_sample_windows.append(train_sample_window)
        
        val_sample_windows = []
        for val_sample in val_samples:
            for window_num in range(num_windows):
                val_sample_window = f'{val_sample}_{window_num+1}.json'
                val_sample_windows.append(val_sample_window)

        test_sample_windows = []
        for test_sample in test_samples:
            for window_num in range(num_windows):
                test_sample_window = f'{test_sample}_{window_num+1}.json'
                test_sample_windows.append(test_sample_window)
        return train_sample_windows, val_sample_windows, test_sample_windows

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

        windowed_data_dir = self.cfg.preprocessing.dataset.windowed_data_dir
        # print(f'Lookinf in {windowed_data_dir} for {sample_name}')
        with open(os.path.join(windowed_data_dir, sample_name), 'r') as f:
            window = json.load(f)

        indices = np.array(window['indices'])
        sparse_vals = np.array(window['sparse_vals'])
        y = window['serotype']
        return indices, sparse_vals, y
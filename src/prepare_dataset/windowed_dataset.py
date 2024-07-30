import pandas as pd
import os
import pickle
import numpy as np
from dataclasses import dataclass, asdict
import json

@dataclass
class Sample:
    name: str
    indices: np.ndarray
    sparse_vals: np.ndarray
    serotype: int

@dataclass
class Window:
    name: str
    indices: np.ndarray
    sparse_vals: np.ndarray
    serotype: int

class WindowedDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prepare_directories()
        self._configs_file_prepare()
        self.classes = cfg.preprocessing.dataset.classes
        self.sample_names = self._get_sample_names()
        self.selected_windows = self.selected_windows()

    def selected_windows(self):
        selected_windows_file = self.cfg.best_features_dataset.windowed.selected_windows_file
        selected_windows = []
        with open(selected_windows_file, 'r') as f:
            for line in f:
                window = line.strip()[1:-1].split(", ")
                window = [int(w) for w in window]
                selected_windows.append(window)
        return selected_windows

    def prepare_directories(self):
        out_folder = self.cfg.best_features_dataset.windowed.out_folder
        os.makedirs(out_folder, exist_ok=True)

    def _configs_file_prepare(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        # Remove the rows in which the Serotype value is 0
        df = df[df['Serotype'] != '0']
        df = df[df['Serotype'] != '---']
        top_n = self.cfg.preprocessing.dataset.top_n
        top_serotypes = df['Serotype'].value_counts().head(top_n).index.tolist()
        self.cfg.preprocessing.dataset.classes = top_serotypes

    def _get_sample_names(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(self.classes)]
        tfr_filenames_stem = df_filtered['SRA_ACCESSION_NUMBER'].values
        tfr_filenames = [f'{name}.pkl' for name in tfr_filenames_stem]
        return tfr_filenames
    
    def save_windowed_sample(self, windowed_sample, i):
        # Save as a JSON file
        out_folder = self.cfg.best_features_dataset.windowed.out_folder
        window_num = i+1
        out_file = os.path.join(out_folder, f'{windowed_sample.name}_{window_num}.json')
        with open(out_file, 'w') as f:
            json.dump(asdict(windowed_sample), f)
    
    def get_windows_in_sample(self, sample):
        window_size = self.cfg.best_features_dataset.windowed.window_size

        indices = sample.indices
        sparse_vals = sample.sparse_vals
        serotype = sample.serotype
        feature_vector_len = self.cfg.preprocessing.dataset.input_size
        X = np.zeros(feature_vector_len)
        indices = np.array(indices, dtype=int)
        sparse_values = np.array(sparse_vals, dtype=float)
        X[indices] = sparse_values

        for i, window in enumerate(self.selected_windows):
            window_sparse = X[window]
            # If a val is 99, replace it with 21
            window_sparse[window_sparse == 99] = 21
            # Padd the window to the window_size with value 22
            if len(window_sparse) < window_size:
                window_sparse = np.pad(window_sparse, (0, window_size - len(window_sparse)), 'constant', constant_values=(22))
            if len(window) < window_size:
                # Pad with -1
                window = np.pad(window, (0, window_size - len(window)), 'constant', constant_values=(-1))

            # Make all the values serializable for JSON
            try:
                window = window.tolist()
            except:
                pass
            window_sparse = window_sparse.tolist()
            serotype = int(serotype)

            windowed_sample = Window(sample.name, window, window_sparse, serotype)
            self.save_windowed_sample(windowed_sample, i)

    def create_windowed_dataset(self):
        dataclass_folder = self.cfg.utils.prepare_dataset.dataclass_out_folder_full
        for i, sample_name in enumerate(self.sample_names):
            if i % 1000 == 0:
                print(f'Processing sample {i+1}/{len(self.sample_names)}')
            sample_file = os.path.join(dataclass_folder, sample_name)
            with open(sample_file, 'rb') as f:
                sample = pickle.load(f)
                self.get_windows_in_sample(sample)
            
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import zipfile
import time
import os

class BestFeaturesDataset:
    def __init__(self, cfg: dict, best_n: int):
        self.cfg = cfg
        self.best_n = best_n
        self._best_features = self.get_best_features() #[93242, 34775, ...(~5000 or less)]
        self.sample_names_to_zipfile_map = {}
        self.tfr_sample_ys_df = None
        self.tfr_sample_names = self._get_tfr_sample_names() #[SRR12345.csv, SRR67890.csv, ... (~300000 or more)]
        self.create_headers()
    
    def create_headers(self):
        out_folder = self.cfg.best_features_dataset.dataset.out_folder
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        with open(os.path.join(out_folder, f'best_{self.best_n}_dataset.txt'), 'w') as f:
            f.write(f'SampleID, {str(self.best_features)[1:-1]}, serotype_encoded\n')

    @property
    def best_features(self):
        # Will be used to filter and position the best features in the dataset
        return self._best_features

    def _get_best_features_from(self, serotype):
        feat_imps_folder = self.cfg.explanation.deeplift.explanations_folder
        filename = f"{serotype}.csv"
        path = os.path.join(feat_imps_folder, filename)
        df = pd.read_csv(path, index_col=0)
        best_n_features = df.index.values.tolist()[:self.best_n]
        best_n_features = [int(x.split("_")[1]) for x in best_n_features]
        return set(best_n_features)

    def get_best_features(self):
        serotypes = self.cfg.preprocessing.dataset.classes
        best_features = set()
        for serotype in serotypes:
            best_features = best_features.union(self._get_best_features_from(serotype))

        return list(best_features)
    
    def _get_tfr_sample_names(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        df_filtered = df[df['Serotype'].isin(self.cfg.preprocessing.dataset.classes)]
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
    
    def _get_full_path(self, filename):
        """Searches which zip file contains the given filename
        from the mapping and returns the full path to the file.
        """
        zip_file = self.sample_names_to_zipfile_map[filename]
        return os.path.join(self.cfg.preprocessing.dataset.zip_path, zip_file)
    
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
        y = self.tfr_sample_ys_df[self.tfr_sample_ys_df[
            'SRA_ACCESSION_NUMBER'] == filename]['serotype_encoded'].values[0]
        return y
    
    def _indices_and_sparse_vals(self, idx):
        filename = self.tfr_sample_names[idx]
        full_path = self._get_full_path(filename)
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            with zip_ref.open(filename) as file:
                X = file.readline().decode('utf-8').strip().split(',')
                sparse_vals = file.readline().decode('utf-8').strip().split(',')
        return X, sparse_vals
    
    def generate_dataset(self):
        out_folder = self.cfg.best_features_dataset.dataset.out_folder
        st = time.time()

        for idx, sample_id in enumerate(self.tfr_sample_names):
            if idx % 1000 == 1:
                
                # Calculate remaining time
                elapsed = time.time() - st
                samples_remaining = len(self.tfr_sample_names) - idx
                time_remaining = elapsed / idx * samples_remaining / 60
                print(f"Processing {idx}th sample, {time_remaining:.2f} minutes remaining.")
            X, sparse_vals = self._indices_and_sparse_vals(idx)
            X_filtered = self._get_X(X, sparse_vals)
            y = self._get_y(idx)
            with open(os.path.join(out_folder, f'best_{self.best_n}_dataset.txt'), 'a') as f:
                f.write(f'{sample_id}, ')
                for val in X_filtered:
                    f.write(f'{val}, ')
                f.write(f'{y}\n')
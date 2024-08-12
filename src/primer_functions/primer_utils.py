import pickle
import pandas as pd
import numpy as np
import os

class PrimerSegmentSearch:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        pass

    def _find_row(self, data, target):
        def binary_search(data, target):
            low, high = 0, len(data) - 1
            while low <= high:
                mid = (low + high) // 2
                start, end = data[mid][0], data[mid][1]
                if start <= target <= end:
                    return data[mid]
                elif target < start:
                    high = mid - 1
                else:
                    low = mid + 1
            return None

        return binary_search(data, target)

    def _import_primer_info(self, primer_info_file):
        with open(primer_info_file, 'rb') as f:
            primer_info = pickle.load(f)
        
        return np.array(primer_info, dtype=np.int32)

    def get_primer_site_for_tfr_indices(self, tfr_indices: list, primer_info_file: str):
        primer_info = self._import_primer_info(primer_info_file)
        primer_sites = []
        for tfr_index in tfr_indices:
            primer_site = self._find_row(primer_info, tfr_index)
            primer_sites.append(primer_site)
        return primer_sites
    
    def top_selection_choice(self, df_ref, n): #TODO change
        top_tfr_indices_for_serotype = df_ref.index.values.tolist()[:n]
        bottom_tfr_indices_for_serotype = df_ref.index.values.tolist()[-n:]
        return top_tfr_indices_for_serotype + bottom_tfr_indices_for_serotype

    def get_primer_site_for_serotype(self, serotype):
        """Generates a csv file primer_site_{serotype}.csv of header: TFRIndex, fin150_30_3, fin150_40_3, **"""
        explanation_folder = self.cfg.explanation.deeplift.explanations_folder
        exp_filename_for_serotype = os.path.join(explanation_folder, f'{serotype}.csv')
        df_exp = pd.read_csv(exp_filename_for_serotype, index_col=0)
        
        primer_info_files = list(dict(self.cfg.primer_functions.primer_info).keys())
        num_top_features = self.cfg.primer_functions.hyperparams.top_n

        top_tfr_indices_for_serotype = self.top_selection_choice(df_exp, num_top_features)
        top_tfr_indices_for_serotype = [int(tfr_index.split('_')[1]) for tfr_index in top_tfr_indices_for_serotype]

        #create df_combined with index as top_tfr_indices
        df_combined = pd.DataFrame(index=top_tfr_indices_for_serotype)
        
        for primer_info_file_key in primer_info_files:
            primer_info_file = self.cfg.primer_functions.primer_info[primer_info_file_key]
            print(primer_info_file)
            primer_sites = self.get_primer_site_for_tfr_indices(top_tfr_indices_for_serotype, primer_info_file)
            df_combined[primer_info_file_key] = primer_sites
        
        out_dir = self.cfg.primer_functions.hyperparams.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_filename = os.path.join(out_dir, f'primer_site_{serotype}.csv')
        df_combined.to_csv(out_filename)
    
    def get_primer_site_for_all_serotypes(self):
        serotypes = self.cfg.preprocessing.dataset.classes
        for serotype in serotypes:
            self.get_primer_site_for_serotype(serotype)
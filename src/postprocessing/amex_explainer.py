from preprocessing.single_class_datasets import BestSingleClassDatasetCorrFiltered
from models.transformer import PangenomeTransformerModel
from torch.utils.data import DataLoader
from utils.logger import Logger
import os

import pandas as pd
import torch

device = torch.device("cpu")

class AMEXExplainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._config_files_prepare()
        self.dataset = BestSingleClassDatasetCorrFiltered
        self.model = self.load_model()
        self.logger = Logger(cfg)

    def load_model(self):
        model = PangenomeTransformerModel(self.cfg)
        saved_model_path = self.cfg.file_paths.model.model_path
        model.load_state_dict(torch.load(saved_model_path))
        model.eval()
        model.to(device)
        return model

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
        
    def _explain(self, loader, serotype, features):
        self.logger.log(f"Explaining serotype {serotype}")
        total_attributions =  0
        for i, (x, y) in enumerate(loader):
            x = x.long().to(torch.device("cpu"))
            if i % 100 == 0:
                self.logger.log(f"Explaining batch {i}/{len(loader)}")
            attributions = self.model.get_attention_matrix(x)
            # Shape is torch.Size([1, 32, 71, 71]), average the values to make it [71, 71]
            attributions = attributions.mean(dim=1).mean(dim=0)
            total_attributions += attributions
        
        mean_attributions = total_attributions / (i+1)
        mean_attributions = mean_attributions.detach().numpy()
        explain_df = pd.DataFrame(mean_attributions, columns=features, index=features)
        out_folder = self.cfg.file_paths.explanation.attention_matrix_out_folder
        os.makedirs(out_folder, exist_ok=True)
        filename = f"{serotype}_attention_matrix.csv"
        explain_df.to_csv(os.path.join(out_folder, filename))
    
    def get_col_names(self):
        col_names_file = self.cfg.file_paths.corr_matrix.filtered_indices_file
        with open(col_names_file, 'r') as f:
            col_names = f.readlines()
            col_names = [i.rstrip() for i in col_names]
        return col_names

    def explain(self):
        self.logger.log("Starting explanation")
        classes = self.cfg.preprocessing.dataset.classes
        
        for serotype in classes:
            self.logger.log(f"Explaining serotype {serotype}")
            dataset = self.dataset(self.cfg, serotype)
            features = self.get_col_names()
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            self._explain(loader, serotype, features)
            
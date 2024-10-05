from preprocessing.single_class_datasets import SingleClassDataset
from torch.utils.data import DataLoader
from models.mlp import MLPModel
from captum.attr import DeepLift
from utils.logger import Logger
import pandas as pd
import numpy as np
import pickle
import torch
import os

class DeepLiftExplainerTop97:
    def __init__(self, cfg, model_path=None):
        self.logger = Logger(cfg)
        self.cfg = cfg
        self._config_files_prepare()
        self.model = self._load_model(model_path)
        self.explainer = DeepLift(self.model)
        self.num_iters = 100 # Number of times the test_loader is called
        self.y_map = None

    def _load_model(self, model_path=None):
        if model_path is None:
            model_path = self.cfg.file_paths.model.model_path

        self.logger.log(f"Model loading from {model_path}")
        model = MLPModel(self.cfg)
        model.load_state_dict(torch.load(model_path))
        
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
    
    def _explain(self, loader, serotype):
        self.logger.log(f"Explaining serotype {serotype}")
        total_attributions =  0
        for i, (x, y) in enumerate(loader):
            # if i == self.num_iters:
            #     i -= 1
            #     break
            x = x.float().to(torch.device("cpu"))
            if i % 100 == 0:
                self.logger.log(f"Explaining batch {i}/{len(loader)}")
            y_val = int(self.y_map[serotype])
            attributions = self.explainer.attribute(x, target=y_val)
            # Shape torch.Size([32, 236071]), Take mean over all samples
            attributions = attributions.mean(dim=0).detach().numpy()
            # self.logger.log(f"Attributions for batch {i} saved")
            total_attributions += attributions
        
        mean_attributions = total_attributions / (i+1)

        explain_df = pd.DataFrame(mean_attributions)
        
        # Save to file
        fixed_feature_len = self.cfg.preprocessing.dataset.fixed_feature_len
        all_rows = [i for i in range(fixed_feature_len)]
        # rows = [f"sequence_{i}" for i in all_rows if i not in self.excluded_indices]
        rows = [f"sequence_{i}" for i in all_rows]
        # rows = [f"sequence_{i}" for i in range(len(mean_attributions))]
        explain_df.index = rows
        explain_df = explain_df.sort_values(by=0, ascending=False)
        deeplift_out_folder = self.cfg.file_paths.explanation.deeplift_fi_folder
        os.makedirs(deeplift_out_folder, exist_ok=True)
        explain_df.to_csv(f"{deeplift_out_folder}/{serotype}.csv")
    
    def explain_test(self):
        self.logger.log("Starting explanation")
        classes = self.cfg.preprocessing.dataset.classes
        for serotype in classes:
            # dataset = SingleClassDataset(self.cfg, serotype)
            dataset = SingleClassDataset(self.cfg, serotype)
            self.y_map = dataset.y_map
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            self._explain(loader, serotype)

class DeepLiftExplainerFiltered(DeepLiftExplainerTop97):
    def __init__(self, cfg, model_path=None):
        super().__init__(cfg, model_path)
        
        
    def explain_test(self):

        self.logger.log("Starting explanation")
        classes = self.cfg.preprocessing.dataset.classes
        for serotype in classes:
            dataset = SingleClassFiltered(self.cfg, serotype)
            self.excluded_indices = dataset.excluded_indices
            self.y_map = dataset.y_map
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            self._explain(loader, serotype)
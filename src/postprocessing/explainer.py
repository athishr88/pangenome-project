from preprocessing.dataloader import SingleClassDataset, SingleClassDatasetDataclass
from torch.utils.data import DataLoader
from models.mlp import MLPModel
from captum.attr import DeepLift
from utils.logger import Logger
import pandas as pd
import numpy as np
import pickle
import torch
import os

class DeepLiftExplainerPartialDataset:
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
            model_path = self.cfg.explanation.deeplift.model_path
        model = MLPModel(self.cfg)
        model.load_state_dict(torch.load(model_path))
        self.logger.log(f"Model loaded from {model_path}")
        return model
    
    def _config_files_prepare(self):
        y_file = self.cfg.preprocessing.dataset.serotype_file_path
        df = pd.read_csv(y_file)
        # Remove the rows in which the Serotype value is 0
        df = df[df['Serotype'] != '0']
        top_n = self.cfg.preprocessing.dataset.top_n
        top_serotypes = df['Serotype'].value_counts().head(top_n).index.tolist()
        self.cfg.preprocessing.dataset.classes = top_serotypes
        self.logger.log(f"Top {top_n} serotypes: {top_serotypes}")
    
    def _explain(self, loader, serotype):
        self.logger.log(f"Explaining serotype {serotype}")
        mean_attributions = []
        for i, (x, y) in enumerate(loader):
            # if i == self.num_iters:
            #     break
            x = x.float().to(torch.device("cpu"))
            if i % 100 == 0:
                self.logger.log(f"Explaining batch {i}/{len(loader)}")
            y_val = int(self.y_map[serotype])
            attributions = self.explainer.attribute(x, target=y_val)
            # Shape torch.Size([32, 236071]), Take mean over all samples
            attributions = attributions.mean(dim=0).detach().numpy()
            # self.logger.log(f"Attributions for batch {i} saved")
            mean_attributions.append(attributions)
        
        explain_array = np.array(mean_attributions)
        explain_array_mean = np.mean(explain_array, axis=0)

        explain_df = pd.DataFrame(explain_array_mean)
        
        # Save to file
        rows = [f"sequence_{i}" for i in range(mean_attributions[0].shape[0])]
        explain_df.index = rows
        explain_df = explain_df.sort_values(by=0, ascending=False)
        os.makedirs("explanations", exist_ok=True)
        explain_df.to_csv(f"explanations/{serotype}.csv")
    
    def explain_test(self):

        self.logger.log("Starting explanation")
        classes = self.cfg.preprocessing.dataset.classes
        for serotype in classes:
            # dataset = SingleClassDataset(self.cfg, serotype)
            dataset = SingleClassDatasetDataclass(self.cfg, serotype)
            self.y_map = dataset.y_map
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            self._explain(loader, serotype)
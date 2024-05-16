from preprocessing.dataloader import TempSingleClassDataset
from torch.utils.data import DataLoader
from models.mlp import MLPModel
from captum.attr import DeepLift
from utils.logger import Logger
import pandas as pd
import numpy as np
import pickle
import torch

class DeepLiftExplainerPartialDataset:
    def __init__(self, cfg):
        self.logger = Logger(cfg)
        self.cfg = cfg
        self.model = self._load_model()
        self.explainer = DeepLift(self.model)
        self.num_iters = 20
        self.y_map = {'Enteritidis': 0,
                      'I14512i-': 1,
                      'Infantis': 2,
                      'Javiana': 3,
                      'Kentucky': 4,
                      'Montevideo': 5,
                      'Newport': 6,
                      'Saintpaul': 7,
                      'Typhi': 8,
                      'Typhimurium': 9}

    def _load_model(self):
        model_path = self.cfg.explanation.deeplift.model_path
        model = MLPModel(self.cfg)
        model.load_state_dict(torch.load(model_path))
        self.logger.log(f"Model loaded from {model_path}")
        return model
    
    def _explain(self, loader, serotype):
        self.logger.log(f"Explaining serotype {serotype}")
        mean_attributions = []
        for i, (x, y) in enumerate(loader):
            if i == self.num_iters:
                break
            x = x.float().to(torch.device("cpu"))
            self.logger.log(f"Explaining batch {i}")
            y_val = self.y_map[serotype]
            attributions = self.explainer.attribute(x, target=y_val)
            # Shape torch.Size([32, 236071]), Take mean over all samples
            attributions = attributions.mean(dim=0).detach().numpy()
            self.logger.log(f"Attributions for batch {i} saved")
            mean_attributions.append(attributions)
        
        explain_array = np.array(mean_attributions)
        explain_array_mean = np.mean(explain_array, axis=0)

        explain_df = pd.DataFrame(explain_array_mean)
        print(explain_df)
        
        # Save to file
        rows = [f"sequence_{i}" for i in range(mean_attributions[0].shape[0])]
        explain_df.index = rows
        explain_df = explain_df.sort_values(by=0, ascending=False)
        explain_df.to_csv(f"explanations/{serotype}.csv")
    
    def explain_test(self):

        self.logger.log("Starting explanation")
        classes = self.cfg.preprocessing.dataset.classes
        for serotype in classes:
            dataset = TempSingleClassDataset(self.cfg, serotype)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            self._explain(loader, serotype)


        
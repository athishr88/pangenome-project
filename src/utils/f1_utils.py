from preprocessing.dataloader import CorrFilteredDataset, CorrFilteredDatasetTR
from sklearn.metrics import confusion_matrix
from models import transformer, mlp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.logger import Logger
from models import transformer
import pandas as pd
import torch
import os

device = torch.device("cpu")

class ConfusionMatrixGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.prepare_config_files(cfg)
        self.dataset = CorrFilteredDataset
        self.logger = Logger(cfg)
        self._initialize_dataset()
        self._initialize_model()


    def prepare_config_files(self, cfg):
        cutoff = cfg.best_features_dataset.dataset.cutoff
        imp_indices_folder = cfg.file_paths.best_features_dataset.best_features_names_out_folder
        imp_indices_filename = f"Important_Indices_cutoff_{cutoff}.txt"

        full_path = os.path.join(imp_indices_folder, imp_indices_filename)
        with open(full_path, 'r') as f:
            indices = f.readlines()
        
        num_indices = len(indices)
        # cfg.preprocessing.dataset.input_size = num_indices
        cfg.preprocessing.dataset.input_size = 50
        pass
               

    def _initialize_dataset(self):
        self.dataset.initialize_data(self.cfg)
        test_dataset = self.dataset.from_split('test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)

    def _initialize_model(self):
        self.model = mlp.MLPModel(self.cfg)
        saved_model_path = self.cfg.file_paths.model.model_path
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.eval()

    def dtype_batch(self, x, y):
        return x.float().to(device), y.to(device)

    def generate_confusion_matrix(self):
        preds, targets = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = self.dtype_batch(x, y)
                if i % 1000 == 0:
                    self.logger.log(f"Batch {i}")
                x, y = x.to(device), y.to(device)
                y_pred = self.model(x)
                preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        cm = confusion_matrix(targets, preds)
        self.save_to_file(cm)
    
    def save_to_file(self, cm):
        classes = self.cfg.preprocessing.dataset.classes
        df = pd.DataFrame(cm, columns=classes, index=classes)
        out_folder = self.cfg.file_paths.best_features_dataset.best_features_names_out_folder
        # cutoff = self.cfg.best_features_dataset.dataset.cutoff
        threshold = self.cfg.file_paths.corr_matrix.correlation_threshold
        out_folder = out_folder + f"/corr_threshold_{threshold}"
        os.makedirs(out_folder, exist_ok=True)
        filename = f"confusion_matrix.xlsx"
        df.to_excel(os.path.join(out_folder, filename))
        self.logger.log(f"Confusion matrix saved at {os.path.join(out_folder, filename)}")

class CMTransformer(ConfusionMatrixGenerator):
    def _initialize_model(self):
        self.model = transformer.PangenomeTransformerModel(self.cfg)
        saved_model_path = self.cfg.file_paths.model.model_path
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.eval()

    def dtype_batch(self, x, y):
        return x.long().to(device), y.to(device)
    
class CMNormal(ConfusionMatrixGenerator):

    def _initialize_model(self):
        self.model = mlp.MLPModelOH(self.cfg)
        saved_model_path = self.cfg.file_paths.model.model_path
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.eval()

class CMTransformerV2(ConfusionMatrixGenerator):
    def __init__(self, cfg):
        self.cfg = cfg
        # self.prepare_config_files(cfg)
        self.dataset = CorrFilteredDatasetTR
        self.logger = Logger(cfg)
        self._initialize_dataset()
        self._initialize_model()

    def _initialize_model(self):
        self.model = transformer.PGTransformerV2(self.cfg)
        saved_model_path = self.cfg.file_paths.model.model_path
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.eval()

    def _initialize_model(self):
        self.model = transformer.PGTransformerV2(self.cfg)
        saved_model_path = self.cfg.file_paths.model.model_path
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.eval()

    def dtype_batch(self, x, y):
        return x.long().to(device), y.to(device)
    
    def save_to_file(self, cm):
        classes = self.cfg.preprocessing.dataset.classes
        df = pd.DataFrame(cm, columns=classes, index=classes)
        out_folder = self.cfg.file_paths.best_features_dataset.best_features_names_out_folder
        # cutoff = self.cfg.best_features_dataset.dataset.cutoff
        threshold = self.cfg.file_paths.corr_matrix.correlation_threshold
        out_folder = out_folder + f"/corr_threshold_{threshold}"
        os.makedirs(out_folder, exist_ok=True)
        filename = "confusion_matrix_tr.xlsx"
        df.to_excel(os.path.join(out_folder, filename))
        self.logger.log(f"Confusion matrix saved at {os.path.join(out_folder, filename)}")

    
    def generate_confusion_matrix(self):
        preds, targets = [], []
        with torch.no_grad():
            for i, (f, n, x, y) in enumerate(self.test_loader):
                x, y = self.dtype_batch(x, y)
                f, n = f.to(device), n.to(device)
                if i % 1000 == 0:
                    self.logger.log(f"Batch {i}")
                y_pred = self.model(f, n, x)
                preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        cm = confusion_matrix(targets, preds)
        self.save_to_file(cm)
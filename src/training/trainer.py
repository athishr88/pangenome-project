from preprocessing.dataloader import TFRecordsPartialDatasetDataclass, TFRBestFeaturesDataclass
from preprocessing.dataloader import TFRTransformerDataset, TFRTransformerDatasetVocab3
from preprocessing.dataloader import TFRFilteredFeaturesDataclass, BestSample, CorrFilteredDataset
from preprocessing.windowed import TFRWindowedDataset
from models.transformer import PangenomeTransformerModel, PangenomeWindowedTransformerModel
from utils.json_logger import update_metrics
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models.mlp import MLPModel
from utils.logger import Logger
from utils.cacher import cache
import pandas as pd
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPTrainer:
    def __init__(self, cfg):
        self.logger = Logger(cfg)
        self.cfg = cfg
        self._create_directories()
        self.dataset = TFRecordsPartialDatasetDataclass
        self.architecture = MLPModel
        self.logger.log("Using partial dataset")
    
    def _create_directories(self):
        model_save_path = self.cfg.file_paths.model.model_path
        model_save_dir = os.path.dirname(model_save_path)
        os.makedirs(model_save_dir, exist_ok=True)

    def _get_dataloaders(self):
        self.dataset.initialize_data(self.cfg)
        train_dataset = self.dataset.from_split('train')
        # Set the input dimension
        self.cfg.preprocessing.dataset.input_size = train_dataset[0][0].shape[0]
        val_dataset = self.dataset.from_split('val')
        test_dataset = self.dataset.from_split('test')
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def dtype_batch(self, x, y):
        x, y = x.float().to(device), y.long().to(device)
        return x, y
        
    def evaluate(self, model, val_loader, test_loader, criterion,
                 best_val_f1):
        model.eval()
        val_preds, val_targets, total_val_loss = [], [], 0
        test_preds, test_targets, total_test_loss = [], [], 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = self.dtype_batch(x, y)
                y_pred = model(x)
                val_loss = criterion(y_pred, y)
                total_val_loss += val_loss.item()
                val_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

            val_f1 = f1_score(val_targets, val_preds, average='macro')
            val_loss = total_val_loss / len(val_loader)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.logger.log(f"Saving model with F1: {val_f1}")
                torch.save(model.state_dict(), self.cfg.file_paths.model.model_path)

            for i, (x, y) in enumerate(test_loader):
                x, y = self.dtype_batch(x, y)
                y_pred = model(x)
                test_loss = criterion(y_pred, y)
                total_test_loss += test_loss.item()
                test_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                test_targets.extend(y.cpu().numpy())
            
            test_f1 = f1_score(test_targets, test_preds, average='macro')
            test_loss = total_test_loss / len(test_loader)
        
            self.logger.log(f"Interim evaluation Val F1: {val_f1}, Test F1: {test_f1}")
            # update_metrics(epoch, train_f1, val_f1, test_f1, train_loss, val_loss, test_loss, self.cfg)
        return val_f1, test_f1, best_val_f1, val_loss, test_loss
    
    def train(self):
        self.logger.log("Initializing training")
        train_loader, val_loader, test_loader = self._get_dataloaders()
        class_weights = self.dataset.class_weights
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        model = self.architecture(self.cfg).to(device)

        # Hyperparameters
        lr = self.cfg.training.hyperparams.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = self.cfg.training.hyperparams.num_epochs

        best_val_f1 = 0
        self.logger.log(f"Starting training on device {device}")

        for epoch in range(num_epochs):
            self.logger.log(f"Epoch {epoch}")
            train_preds, train_targets, total_train_loss = [], [], 0
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = self.dtype_batch(x, y)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if i % 30 == 0:
                    val_f1, test_f1, best_val_f1, val_loss, test_loss = self.evaluate(model, val_loader, test_loader, criterion, best_val_f1)
                    self.logger.log(f"Train Batch {i}/{len(train_loader)}: Loss: {loss.item()}")

                total_train_loss += loss.item()
                loss.backward()
                train_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.cpu().numpy())
                optimizer.step()
            train_f1 = f1_score(train_targets, train_preds, average='macro')
            train_loss = total_train_loss / len(train_loader)

            val_f1, test_f1, best_val_f1, val_loss, test_loss = self.evaluate(model, val_loader, test_loader, criterion, best_val_f1)            
            self.logger.log(f"Full eval Epoch {epoch}: Train F1: {train_f1}, Val F1: {val_f1}, Test F1: {test_f1}")
            update_metrics(epoch, train_f1, val_f1, test_f1, train_loss, val_loss, test_loss, self.cfg)


class MLPTrainerBestFeatures(MLPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = TFRBestFeaturesDataclass

class MLPTrainerFilteredFeatures(MLPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = TFRFilteredFeaturesDataclass

class MLPTrainerCorrFiltered(MLPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = CorrFilteredDataset


class TransformerTrainer(MLPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = TFRTransformerDataset
        self.architecture = PangenomeTransformerModel
        self.logger.log("Using Transformer model")

    def _get_dataloaders(self):
        self.dataset.initialize_data(self.cfg)
        train_dataset = self.dataset.from_split('train')
        # Set the input dimension
        self.cfg.preprocessing.dataset.input_size = train_dataset[0][0].shape[0]
        val_dataset = self.dataset.from_split('val')
        test_dataset = self.dataset.from_split('test')
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
class TransformerTrainerCorrFiltered(MLPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = CorrFilteredDataset
        self.logger.log("Using Correlation filtered dataset")
        self.architecture = PangenomeTransformerModel
        self.logger.log("Using Transformer model")

    def dtype_batch(self, x, y):
        x, y = x.long().to(device), y.long().to(device)
        return x, y

    def _get_dataloaders(self):
        self.dataset.initialize_data(self.cfg)
        train_dataset = self.dataset.from_split('train')
        # Set the input dimension
        self.cfg.preprocessing.dataset.input_size = train_dataset[0][0].shape[0]
        val_dataset = self.dataset.from_split('val')
        test_dataset = self.dataset.from_split('test')
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
class FilteredMLP(MLPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = TFRFilteredFeaturesDataclass
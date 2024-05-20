from preprocessing.dataloader import TFRecordsDataset, TFRecordsPartialDataset
from utils.json_logger import update_metrics
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models.mlp import MLPModel
from utils.logger import Logger
from utils.cacher import cache
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPTrainer:
    def __init__(self, cfg):
        self.logger = Logger(cfg)
        self.cfg = cfg

        # Check if full or partial dataset is used
        if hasattr(cfg.preprocessing.dataset, "partial_used"):
            self.dataset = TFRecordsPartialDataset
            self.logger.log("Using partial dataset")
        else:
            self.dataset = TFRecordsDataset
            self.logger.log("Using full dataset")

    def _get_dataloaders(self):
        self.dataset.initialize_data(self.cfg)
        train_dataset = self.dataset.from_split('train')
        val_dataset = self.dataset.from_split('val')
        test_dataset = self.dataset.from_split('test')
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def train(self):
        self.logger.log("Initializing training")
        train_loader, val_loader, test_loader = self._get_dataloaders()
        model = MLPModel(self.cfg).to(device)

        # Hyperparameters
        lr = self.cfg.training.hyperparams.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = self.cfg.training.hyperparams.num_epochs

        best_val_f1 = 0
        self.logger.log(f"Starting training on device {device}")

        for epoch in range(num_epochs):
            self.logger.log(f"Epoch {epoch}")
            train_preds, train_targets, total_train_loss = [], [], 0
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.float().to(device), y.long().to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if i % 2 == 0:
                    self.logger.log(f"Train Batch {i}/{len(train_loader)}: Loss: {loss.item()}")
                total_train_loss += loss.item()
                loss.backward()
                train_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.cpu().numpy())
                optimizer.step()
            train_f1 = f1_score(train_targets, train_preds, average='macro')
            train_loss = total_train_loss / len(train_loader)

            model.eval()
            val_preds, val_targets, total_val_loss = [], [], 0
            test_preds, test_targets, total_test_loss = [], [], 0
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    x, y = x.float().to(device), y.long().to(device)
                    y_pred = model(x)
                    val_loss = criterion(y_pred, y)
                    total_val_loss += val_loss.item()
                    val_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                    val_targets.extend(y.cpu().numpy())

                val_f1 = f1_score(val_targets, val_preds, average='macro')
                val_loss = total_val_loss / len(val_loader)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), self.cfg.training.model.save_path)

                for i, (x, y) in enumerate(test_loader):
                    x, y = x.float().to(device), y.long().to(device)
                    y_pred = model(x)
                    test_loss = criterion(y_pred, y)
                    total_test_loss += test_loss.item()
                    test_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                    test_targets.extend(y.cpu().numpy())
                
                test_f1 = f1_score(test_targets, test_preds, average='macro')
                test_loss = total_test_loss / len(test_loader)
            
                self.logger.log(f"Epoch {epoch}: Train F1: {train_f1}, Val F1: {val_f1}, Test F1: {test_f1}")
                update_metrics(epoch, train_f1, val_f1, test_f1, train_loss, val_loss, test_loss, self.cfg)
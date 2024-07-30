from preprocessing.dataloader import TFRBestFeaturesDataclass
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.mlp import MLPModel
import torch

device = torch.device("cpu")

class ConfusionMatrixGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = TFRBestFeaturesDataclass
        
        self._initialize_dataset()
        self._initialize_model()
               

    def _initialize_dataset(self):
        self.dataset.initialize_data(self.cfg)
        test_dataset = self.dataset.from_split('test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.training.hyperparams.batch_size, shuffle=False)

    def _initialize_model(self):
        self.model = MLPModel(self.cfg)
        saved_model_path = self.cfg.training.model.save_path
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.eval()

    def generate_confusion_matrix(self):
        preds, targets = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                if i % 100 == 0:
                    print(f"Batch {i}")
                x, y = x.to(device), y.to(device)
                y_pred = self.model(x)
                preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        cm = confusion_matrix(targets, preds)
        # Plot the confusion matrix and save it
        # plt.matshow(cm)
        # plt.title('Confusion matrix')
        # plt.colorbar()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.savefig('confusion_matrix.png') 
        # plt.show()
        return cm


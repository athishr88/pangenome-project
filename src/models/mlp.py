import torch
from torch import nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, cfg):
        super(MLPModel, self).__init__()
        input_size = cfg.preprocessing.dataset.input_size
        print(input_size)
        hidden_size = cfg.model.model_params.hidden_size
        output_size = cfg.preprocessing.dataset.num_classes
        num_hidden_layers = cfg.model.model_params.num_hid_layers
        self.is_batch_norm = cfg.model.model_params.batch_norm
        self.dropout_val = cfg.model.model_params.dropout_val
        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_val, training=self.training)
        if self.is_batch_norm:
            x = self.bn(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_val, training=self.training)
            if self.is_batch_norm:
                x = self.bn(x)
        x = self.fc2(x)
        return x
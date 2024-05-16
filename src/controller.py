from utils.pangenome_graph_utils import get_sequence_map
from preprocessing.dataloader import TFRecordsDataset
from training.trainer import MLPTrainer

class Controller:
    """Controller contains all individual services and 
    can be used to call them"""
    def __init__(self) -> None:
        pass

    def get_sequence_map(self, config):
        return get_sequence_map(config)
    
    def get_dataset(self, cfg):
        TFRecordsDataset.initialize_data(cfg)
        train_dataset = TFRecordsDataset.from_split('train')
        val_dataset = TFRecordsDataset.from_split('val')
        test_dataset = TFRecordsDataset.from_split('test')

        return train_dataset, val_dataset, test_dataset
    
    def train_mlp(self, cfg):
        trainer = MLPTrainer(cfg)
        trainer.train()
        

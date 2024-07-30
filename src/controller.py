# from prepare_dataset.all_indices_dataset import create_serotype_mapping, create_pickled_dataset
from training.trainer import MLPTrainer, MLPTrainerBestFeatures, TransformerTrainer, MLPTrainerFilteredFeatures, TransformerTrainerVocab3
from training.trainer import TransformerTrainerWindowed
from postprocessing.explainer import DeepLiftExplainerPartialDataset, DeepLiftExplainerFiltered
from prepare_dataset.best_indices_dataset import BestFeaturesDataclassDataset
from prepare_dataset.windowed_dataset import WindowedDataset, Sample
from utils.logger import Logger
from preprocessing.dataloader import SingleClassDatasetDataclass
from utils.f1_utils import ConfusionMatrixGenerator
import pandas as pd
import time

class Controller:
    """Controller contains all individual services and 
    can be used to call them"""
    def __init__(self) -> None:
        pass

    # def get_sequence_map(self, config):
    #     return get_sequence_map(config)
    
    # def get_dataset(self, cfg):
    #     TFRecordsDataset.initialize_data(cfg)
    #     train_dataset = TFRecordsDataset.from_split('train')
    #     val_dataset = TFRecordsDataset.from_split('val')
    #     test_dataset = TFRecordsDataset.from_split('test')

    #     return train_dataset, val_dataset, test_dataset
    
    def train_mlp(self, cfg):
        logger = Logger(cfg)
        time_now = time.strftime("%Y%m%d-%H%M%S")
        logger.log(f"Initializing training at {time_now}")
        trainer = MLPTrainer(cfg)
        trainer.train()

    def explain_model(self, cfg):
        explainer = DeepLiftExplainerPartialDataset(cfg)
        explainer.explain_test()

    def train_with_best_features(self, cfg):
        trainer = MLPTrainerBestFeatures(cfg)
        trainer.train()

    def generate_confusion_matrix(self, cfg):
        cfg.preprocessing.dataset.input_size = 14692
        cmg = ConfusionMatrixGenerator(cfg)
        cm = cmg.generate_confusion_matrix()
        classes = cfg.preprocessing.dataset.classes
        df = pd.DataFrame(cm, columns=classes, index=classes)
        df.to_excel('results/best_indices_top_97/confusion_matrix.xlsx')

    def train_transformer_model(self, cfg):
        trainer = TransformerTrainer(cfg)
        trainer.train()

    def train_transformer_model_vocab3(self, cfg):
        trainer = TransformerTrainerVocab3(cfg)
        trainer.train()

    def create_windowed_dataset(self, cfg):
        dataset = WindowedDataset(cfg)
        dataset.create_windowed_dataset()
    
    def train_windowed_transformer_model(self, cfg):
        trainer = TransformerTrainerWindowed(cfg)
        trainer.train()

    def train_filtered_indices_mlp(self, cfg):
        trainer = MLPTrainerFilteredFeatures(cfg)
        trainer.train()

    def explain_filtered_model(self, cfg):
        explainer = DeepLiftExplainerFiltered(cfg)
        explainer.explain_test()

    def train_filtered_with_best_features(self, cfg):
        trainer = MLPTrainerBestFeatures(cfg)
        trainer.train()

    def generate_confusion_matrix_filtered(self, cfg):
        cfg.preprocessing.dataset.input_size = 3915
        cmg = ConfusionMatrixGenerator(cfg)
        cm = cmg.generate_confusion_matrix()
        classes = cfg.preprocessing.dataset.classes
        df = pd.DataFrame(cm, columns=classes, index=classes)
        df.to_excel('results/best_indices_filtered/confusion_matrix.xlsx')

    # def create_serotype_mapping(self, cfg):
    #     create_serotype_mapping(cfg)

    # def create_best_indices_dataset(self, cfg):
    #     dataset = BestFeaturesDataclassDataset(cfg, 97)
    #     dataset.generate_dataset()

    # def random_test(self, cfg):
    #     dataset = SingleClassDatasetDataclass(cfg, 'Typhimurium')
    #     print(dataset[0])
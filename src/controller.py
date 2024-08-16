# from prepare_dataset.all_indices_dataset import create_serotype_mapping, create_pickled_dataset
from training.trainer import MLPTrainer, MLPTrainerBestFeatures, TransformerTrainer, MLPTrainerFilteredFeatures
from training.trainer import MLPTrainerCorrFiltered, TransformerTrainerCorrFiltered
from postprocessing.explainer import DeepLiftExplainerPartialDataset, DeepLiftExplainerFiltered
from prepare_dataset.best_indices_dataset import BestFeaturesDataclassDataset, CorrelationFilteredDataset
from prepare_dataset.windowed_dataset import WindowedDataset, Sample
from utils.logger import Logger
from prepare_dataset.pd_utils import identify_best_features_cutoff, identify_best_features_ankle_point, create_best_features_dataset
from prepare_dataset.pd_utils import create_best_features_from_corr_dataset
from utils.f1_utils import ConfusionMatrixGenerator, CMTransformer
import pandas as pd
import time
import os

class Controller:
    """Controller contains all individual services and 
    can be used to call them"""
    def __init__(self) -> None:
        pass

    # def get_sequence_map(self, config):
    #     return get_sequence_map(config)
    
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

    def identify_best_features(self, cfg, method):
        # method = 'ankle_point', 'cutoff', combined
        if method == 'ankle_point':
            identify_best_features_ankle_point(cfg)
        elif method == 'cutoff':
            identify_best_features_cutoff(cfg)
        elif method == 'combined':
            identify_best_features_combined(cfg)
        else:
            raise ValueError('Method must be either "ankle_point" or "cutoff"')
        
    def create_best_indices_dataset(self, cfg):
        create_best_features_dataset(cfg)

    def create_best_indices_from_corr_dataset(self, cfg):
        create_best_features_from_corr_dataset(cfg)

    def train_filtered_with_best_features(self, cfg):
        trainer = MLPTrainerBestFeatures(cfg)
        trainer.train()

    def generate_confusion_matrix_filtered(self, cfg):
        
        cmg = ConfusionMatrixGenerator(cfg)
        cmg.generate_confusion_matrix()
    # def create_serotype_mapping(self, cfg):
    #     create_serotype_mapping(cfg)

    def train_correlation_filtered_mlp(self, cfg):
        trainer = MLPTrainerCorrFiltered(cfg)
        trainer.train()

    def train_correlation_filtered_transformer(self, cfg):
        trainer = TransformerTrainerCorrFiltered(cfg)
        trainer.train()

    def generate_confusion_matrix_filtered_transformer(self, cfg):
        
        cmg = CMTransformer(cfg)
        cmg.generate_confusion_matrix()
    
    # def create_serotype_mapping(self, cfg):
    #     create_serotype_mapping(cfg)

    def get_attention_matrix(self, cfg):
        pass
# from prepare_dataset.all_indices_dataset import create_serotype_mapping, create_pickled_dataset
from postprocessing.explainer import DeepLiftExplainerTop97, DeepLiftExplainerFiltered
from prepare_dataset.windowed_dataset import WindowedDataset
from utils import f1_utils, correlation_utils
from prepare_dataset import pd_utils
from training import trainer
import pandas as pd
import os

class Controller:
    """Controller contains all individual services and 
    can be used to call them"""
    def __init__(self) -> None:
        pass
    
    def train_mlp_top_97(self, cfg):
        trainer_handle = trainer.MLPTrainer(cfg)
        trainer_handle.train()

    def explain_model(self, cfg):
        explainer = DeepLiftExplainerTop97(cfg)
        explainer.explain_test()

    def create_best_indices_dataset(self, cfg, method):
        """ method = 'ankle_point', 'cutoff', combined"""

        # Identify the best features
        if method == 'ankle_point':
            pd_utils.identify_best_features_ankle_point(cfg)
        elif method == 'cutoff':
            pd_utils.identify_best_features_cutoff(cfg)
        elif method == 'combined':
            pd_utils.identify_best_features_combined(cfg)
        elif method == 'non_coding':
            pd_utils.identify_best_features_non_coding(cfg)
        elif method == 'coding':
            pd_utils.identify_best_features_coding(cfg)
        else:
            raise ValueError('Method must be either "ankle_point" or "cutoff"')
        
        # Create dataset with the identified best features
        pd_utils.create_best_features_dataset(cfg)

    def create_best_indices_dataset_from_corr_vals(self, cfg):
        correlation_utils.find_pearson_correlation(cfg)
        correlation_utils.select_from_correlated_indices(cfg)
        # pd_utils.create_best_features_dataset_from_corr_vals(cfg)

    def train_correlation_filtered_mlp(self, cfg):
        trainer_handle = trainer.MLPTrainerCorrFiltered(cfg)
        trainer_handle.train()    

    def generate_confusion_matrix_OHE(self, cfg):
        cfg.preprocessing.dataset.input_size = 50
        cmg = f1_utils.CMNormal(cfg)
        cmg.generate_confusion_matrix()

    def train_correlation_filtered_transformer(self, cfg):
        trainer_handle = trainer.TransformerTrainerCorrFiltered(cfg)
        trainer_handle.train()


    def train_with_best_features(self, cfg):
        trainer_handle = trainer.MLPTrainerBestFeatures(cfg)
        trainer_handle.train()



    def train_transformer_model_vocab3(self, cfg):
        trainer_handle = trainer.TransformerTrainerVocab3(cfg)
        trainer_handle.train()

    def create_windowed_dataset(self, cfg):
        dataset = WindowedDataset(cfg)
        dataset.create_windowed_dataset()
    
    def train_windowed_transformer_model(self, cfg):
        trainer_handle = trainer.TransformerTrainerWindowed(cfg)
        trainer_handle.train()

    def train_filtered_indices_mlp(self, cfg):
        trainer_handle = trainer.MLPTrainerFilteredFeatures(cfg)
        trainer_handle.train()

    def explain_filtered_model(self, cfg):
        explainer = DeepLiftExplainerFiltered(cfg)
        explainer.explain_test()

    def train_filtered_with_best_features(self, cfg):
        trainer_handle = trainer.MLPTrainerBestFeatures(cfg)
        trainer_handle.train()

    def generate_confusion_matrix_filtered(self, cfg):
        
        cmg = f1_utils.ConfusionMatrixGenerator(cfg)
        cmg.generate_confusion_matrix()
    # def create_serotype_mapping(self, cfg):
    #     create_serotype_mapping(cfg)

    def train_correlation_filtered_mlp(self, cfg):
        trainer_handle = trainer.MLPTrainerCorrFiltered(cfg)
        trainer_handle.train()

    def train_correlation_filtered_transformer(self, cfg):
        trainer_handle = trainer.TransformerTrainerCorrFiltered(cfg)
        trainer_handle.train()

    def generate_confusion_matrix_filtered_transformer(self, cfg):
        
        cmg = f1_utils.CMTransformer(cfg)
        cmg.generate_confusion_matrix()
    
    # def create_serotype_mapping(self, cfg):
    #     create_serotype_mapping(cfg)

    def get_attention_matrix(self, cfg):
        pass
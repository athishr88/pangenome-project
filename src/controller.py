from training.trainer_reduced_dimension import MLPTrainerReducedDimension
from primer_functions.primer_utils import PrimerSegmentSearch
from postprocessing.explainer import DeepLiftExplainerPartialDataset
from preprocessing.generate_dataset import BestFeaturesDataset
from utils.pangenome_graph_utils import get_sequence_map
from preprocessing.dataloader import TFRecordsDataset
from training.trainer import MLPTrainer
from utils.logger import Logger

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

    def explain_model(self, cfg):
        explainer = DeepLiftExplainerPartialDataset(cfg)
        explainer.explain_test()

    def train_with_top_features(self, cfg):
        logger = Logger(cfg)
        top_features_choices = cfg.training.explanation.num_top_features
        num_classes = cfg.preprocessing.dataset.num_classes
        for num_top_features in top_features_choices:
            cfg.model.model_params.input_dim = 2*num_top_features*num_classes
            print(cfg.model.model_params.input_dim)
            logger.log(f"Training with top {num_top_features} features")
            trainer = MLPTrainerReducedDimension(cfg, num_top_features)
            trainer.train()
    
    def get_primer_sites(self, cfg):
        pss = PrimerSegmentSearch(cfg)
        pss.get_primer_site_for_all_serotypes()

    def generate_new_dataset(self, cfg):
        best_n = cfg.preprocessing.dimensionality_reduction.best_n
        dataset = BestFeaturesDataset(cfg, best_n)
        dataset.generate_dataset()
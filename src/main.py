import hydra
import numpy as np
from pipeline import CorrelationFilteredPipeline
from dataclasses import dataclass
from utils.correlation_utils import select_from_correlated_indices
from controller import Controller

@dataclass
class BestSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

@dataclass
class Sample:
    name: str
    indices: np.ndarray
    sparse_vals: np.ndarray
    serotype: int

controller = Controller()

@hydra.main(config_path="../configs", config_name="top_97", version_base=None)
def main(cfg):
    controller.create_best_indices_dataset_from_corr_vals(cfg)
    

if __name__ == "__main__":
    main()
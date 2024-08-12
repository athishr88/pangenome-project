import hydra
import numpy as np
from controller import Controller
from grid_search import GridSearch
from pipeline import Pipeline
from prepare_dataset.windowed_dataset import Sample
from dataclasses import dataclass
import time
from utils.correlation_utils import select_from_correlated_indices, find_pearson_correlation

@dataclass
class BestSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

controller = Controller()
grid = GridSearch()

@hydra.main(config_path="../configs", config_name="excluded_indices", version_base=None)
def main(cfg):
    select_from_correlated_indices(cfg)
    

if __name__ == "__main__":
    main()
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

controller = Controller()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    controller.generate_confusion_matrix_filtered(cfg)
    

if __name__ == "__main__":
    main()
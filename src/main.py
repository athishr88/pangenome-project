import hydra
import numpy as np
from dataclasses import dataclass
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

@hydra.main(config_path="../configs", config_name="oh_excluded", version_base=None)
def main(cfg):
    controller.create_best_indices_dataset(cfg, 'oh_excluded')
    

if __name__ == "__main__":
    main()
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

@hydra.main(config_path="../configs", config_name="normal_400", version_base=None)
def main(cfg):
    controller.train_mlp(cfg)
    

if __name__ == "__main__":
    main()
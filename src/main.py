import hydra
import numpy as np
from controller import Controller
from prepare_dataset.windowed_dataset import Sample
from dataclasses import dataclass

@dataclass
class BestSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int


controller = Controller()

@hydra.main(config_path="../configs", config_name="excluded_indices", version_base=None)
def main(cfg):
    controller.generate_confusion_matrix_filtered(cfg)

if __name__ == "__main__":
    main()
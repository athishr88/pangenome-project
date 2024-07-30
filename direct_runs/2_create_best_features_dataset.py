import sys
import os
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from dataclasses import dataclass
import numpy as np

@dataclass
class Sample:
    name: str
    indices: np.ndarray
    sparse_vals: np.ndarray
    serotype: int

# os.chdir("..")
# sys.path.append("..")
sys.path.append("src")

from prepare_dataset.best_indices_dataset import BestFeaturesDataclassDataset


GlobalHydra.instance().clear()

initialize(config_path="../configs", version_base=None)

cfg = compose(config_name="excluded_indices")


dataset = BestFeaturesDataclassDataset(cfg, 97)
dataset.generate_dataset()
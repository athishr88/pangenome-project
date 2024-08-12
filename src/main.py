import hydra
import numpy as np
from pipeline import CorrelationFilteredPipeline
from dataclasses import dataclass

@dataclass
class BestSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int



@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    pipeline = CorrelationFilteredPipeline()
    pipeline.correlation_filtered_pipeline(cfg)
    

if __name__ == "__main__":
    main()
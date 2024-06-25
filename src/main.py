import hydra
import numpy as np
from controller import Controller

controller = Controller()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    controller.explain_model(cfg)

if __name__ == "__main__":
    main()
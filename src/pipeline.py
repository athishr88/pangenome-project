from controller import Controller
import os

class Pipeline:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cleanup_previous_files(cfg)
        self.controller = Controller()
        self.edit_config_files(cfg)

    def cleanup_previous_files(self, cfg):
        dataclass_out_folder = cfg.file_paths.new_dataset.out_folder
        # Delete the folder
        os.system(f"rm -r {dataclass_out_folder}")

    def edit_config_files(self, cfg,
                          cutoff=1):
        cfg.best_features_dataset.dataset.cutoff = cutoff

        logfile = cfg.logging.train.logfile
        logfile_stem = logfile.split('.')[0]
        cfg.logging.train.logfile = f"{logfile_stem}_cutoff_{cutoff}.log"

    def excluded_indices_pipeline_cutoff(self, cfg):

        # self.controller.identify_best_features(cfg, 'cutoff')
        self.controller.create_best_indices_dataset(cfg)
        self.controller.train_filtered_with_best_features(cfg)
        self.controller.generate_confusion_matrix_filtered(cfg)

    def correlation_filtered_pipeline(self, cfg):
        self.controller.create_best_indices_from_corr_dataset(cfg)
        self.controller.train_filtered_with_best_features(cfg)
        self.controller.generate_confusion_matrix_filtered(cfg)
        pass

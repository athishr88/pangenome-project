from pipeline import CorrelationFilteredPipeline, ExcludedIndicesPipeline
from utils.logger import Logger

class GridSearch:
    def __init__(self) -> None:
        
        pass

    def grid_search_excluded_indices(self, cfg):
        # Obsolete
        pipeline = ExcludedIndicesPipeline(cfg)
        logger = Logger(cfg)
        cutoffs = [1]
        for cutoff in cutoffs:
            logger.log(f"Starting pipeline with cutoff {cutoff}")
            pipeline.edit_config_files(cfg, cutoff)
            pipeline.excluded_indices_pipeline_cutoff(cfg)
        
    def grid_search_correlation_filtered(self, cfg):
        pass
import os
import zipfile
import numpy as np
import pandas as pd
from utils.logger import Logger
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DatasetHandler():
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = Logger(self.cfg)
        self.logger.log('DatasetHandler initialized')

class TFRecordDataset(Dataset):
    def __init__(self, cfg, dataset_handler):
        """
        @params 
        cfg: config file
        dataset_handler: DatasetHandler object 
            (contains properties common for train,
              test and val datasets)
        """
        self.cfg = cfg
        self.logger = Logger(self.cfg)
        self.dataset_handler = dataset_handler
        self.logger.log('TFRecordDataset initialized')
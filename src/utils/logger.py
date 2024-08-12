import time
import os

class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()
        self._create_folders()

    def _create_folders(self):
        logfile = self.cfg.file_paths.logging.logfile
        log_dir = os.path.dirname(logfile)
        os.makedirs(log_dir, exist_ok=True)

    def log(self, message: str):
        filename = self.cfg.file_paths.logging.logfile
        write_time = time.time()
        passed_time = (write_time - self.start_time) / 3600.00
        with open(filename, 'a') as f:
            f.write(message)
            f.write(f'\t | Time passed : {passed_time} hours')
            f.write('\n')
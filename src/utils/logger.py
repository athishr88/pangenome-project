import time

class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()

    def log(self, message: str):
        filename = self.cfg.logging.train.logfile
        write_time = time.time()
        passed_time = (write_time - self.start_time) / 3600.00
        with open(filename, 'a') as f:
            f.write(message)
            f.write(f'\t Time passed : {passed_time} hours')
            f.write('\n')
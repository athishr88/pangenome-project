class Logger:
    def __init__(self, cfg):
        self.cfg = cfg

    def log(self, message: str):
        filename = self.cfg.logging.train.logfile
        with open(filename, 'a') as f:
            f.write(message)
            f.write('\n')
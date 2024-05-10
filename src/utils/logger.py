def log(message: str, cfg):
    filename = cfg.logging.train.logfile
    with open(filename, 'a') as f:
        f.write(message)
        f.write('\n')
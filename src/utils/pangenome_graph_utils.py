# Define a wrapper to see if the sequence map exist in a pickle file
import os
import pickle

def pickle_exists(func):
    def wrapper(*args, **kwargs):
        config = args[0]
        logfile = config.utils.pgg.logfile
        pickle_file = config.utils.pgg.sequence_map_pickle_path
        if os.path.exists(pickle_file):
            with open(logfile, 'a') as log:
                log.write(f"Reading sequence map from {pickle_file}\n")
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        else:
            result = func(*args, **kwargs)
            with open(pickle_file, 'wb') as f:
                pickle.dump(result, f)
            return result
    return wrapper

@pickle_exists
def get_sequence_map(config):
    """
    Returns a dictionary with the sequence name as key and the sequence as value.
    @param pgg_file: The path to the pangenome graph file.
    """
    logfile = config.utils.pgg.logfile
    pgg_file = config.utils.pgg.pgg_file
    sequence_map = {}
    with open(logfile, 'a') as log:
        log.write(f"Reading sequences from {pgg_file}\n")
        with open(pgg_file, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith("S"):
                    sequence_map[line.split("\t")\
                        [1].strip()] = line.split("\t")[2].strip()
                if i % 100000 == 0:
                    log.write(f"Read {i} lines\n")

    return sequence_map
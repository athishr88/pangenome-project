import pickle

def cache(variable, path):
    with open(path, 'wb') as f:
        pickle.dump(variable, f)

def load_cache(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

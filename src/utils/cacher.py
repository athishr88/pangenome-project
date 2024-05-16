import pickle

def cache(variable, path):
    with open(path, 'wb') as f:
        pickle.dump(variable, f)

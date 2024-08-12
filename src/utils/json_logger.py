import json

def update_metrics(epoch, train_f1, val_f1, test_f1, train_loss, val_loss, test_loss, cfg):
    """ Update the training metrics history in a JSON file. """
    file_path = cfg.file_paths.logging.train_history
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    
    data[str(epoch)] = {
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def read_metrics(cfg):
    """ Read the training metrics history from a JSON file. """

    file_path = cfg.logging.train.train_history
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
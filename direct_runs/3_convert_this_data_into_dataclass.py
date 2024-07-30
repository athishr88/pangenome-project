from dataclasses import dataclass
import numpy as np
import pickle
import time
import os

# Wait 200 minutes before running this script
print("Waiting 200 minutes before running this script")
time.sleep(300 * 60)

@dataclass
class BestSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

os.makedirs('best_features_dataclass/best_indices_top_97_pickle/metadata', exist_ok=True)
os.makedirs('best_features_dataclass/best_indices_top_97_pickle/dataclasses', exist_ok=True)

file_path = 'best_features_dataclass/best_97_dataset.txt'
metadata_file_out = 'best_features_dataclass/best_indices_top_97_pickle/metadata/indices.txt'
dataclass_out_folder = 'best_features_dataclass/best_indices_top_97_pickle/dataclasses'

with open(file_path, 'r') as f:
    names = f.readline().strip().split(', ')

    indices = names[1:-1]
    with open(metadata_file_out, 'w') as meta_f:
        for idx in indices:
            meta_f.write(f"{idx}\n")
    i = 0
    while True:
        i += 1
        if i % 1000 == 0:
            print(f"Processing {i}/{500000} th sample")
        line = f.readline()
        if not line:
            break
        data = line.strip().split(', ')
        sample_name = data[0].split('.')[0]
        sparse_vals = np.array([int(float(x)) for x in data[1:-1]], dtype=np.uint8)
        serotype = int(data[-1])
        sample = BestSample(sample_name, sparse_vals, serotype)
        
        out_file = os.path.join(dataclass_out_folder, sample_name + '.pkl')
        with open(out_file, 'wb') as f_pickle:
            pickle.dump(sample, f_pickle)
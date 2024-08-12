import os
from dataclasses import dataclass
import numpy as np
import pickle
import pandas as pd

@dataclass
class FilteredSample:
    name: str
    sparse_vals: np.ndarray
    serotype: int

def calculate_all_correlations(averages):
    dataclass_folder = "/home/athishramdas/Desktop/Pangenome/pangenome-project/best_features_dataset/dataclass"
    dataclass_files = os.listdir(dataclass_folder)
    files = [f for f in dataclass_files if f.endswith('.pkl')]
    num_features = 66140
    corr_matrix = np.zeros((num_features, num_features))

    save_file_num = 0
    for file in files:
        file_path = os.path.join(dataclass_folder, file)
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
            sparse_vals = sample.sparse_vals
        rand_num = 0
        for col1 in range(num_features):
            if rand_num % 100 == 0:
                print(f"File {file}, col1 {col1}")
            rand_num += 1
            for col2 in range(col1, num_features):
                numerator = sparse_vals[col1] - averages[col1] * sparse_vals[col2] - averages[col2]
                denominator1 = sparse_vals[col1] - averages[col1]**2
                denominator2 = sparse_vals[col2] - averages[col2]**2
                corr_matrix[col1, col2] += numerator / (denominator1 * denominator2)**0.5
                corr_matrix[col2, col1] = corr_matrix[col1, col2]
            break
        
        # SAve file after each iteration
        df_corr = pd.DataFrame(corr_matrix)
        # df_corr.to_csv(f"correlation_matrix/correlation_matrix_{save_file_num}.csv")
        save_file_num += 1
        with open(f"df_corr_{save_file_num}.pkl", 'wb') as filepickle:
            pickle.dump(df_corr, filepickle)

    corr_matrix /= len(files)
    # Save to file
    df_corr = pd.DataFrame(corr_matrix)
    df_corr.to_csv("correlation_matrix/correlation_matrix.csv")
    return corr_matrix

# import os
# import numpy as np
# import pandas as pd
# import pickle
# from multiprocessing import Pool, cpu_count

# def process_file(args):
#     file_path, num_features, averages = args
#     with open(file_path, 'rb') as f:
#         sample = pickle.load(f)
#         sparse_vals = np.array(sample.sparse_vals)
        
#     # Calculate differences from averages
#     diff_vals = sparse_vals - averages
#     corr_matrix_partial = np.zeros((num_features, num_features))

#     for col1 in range(num_features):
#         for col2 in range(col1, num_features):
#             numerator = diff_vals[col1] * diff_vals[col2]
#             denominator1 = diff_vals[col1] ** 2
#             denominator2 = diff_vals[col2] ** 2
#             corr_matrix_partial[col1, col2] += numerator / (denominator1 * denominator2)**0.5
#             corr_matrix_partial[col2, col1] = corr_matrix_partial[col1, col2]

#     return corr_matrix_partial

# def calculate_all_correlations(averages):
#     dataclass_folder = "/home/athishramdas/Desktop/Pangenome/pangenome-project/best_features_dataset/dataclass"
#     dataclass_files = os.listdir(dataclass_folder)
#     files = [f for f in dataclass_files if f.endswith('.pkl')]
#     num_features = 66140

#     # Prepare arguments for parallel processing
#     args = [(os.path.join(dataclass_folder, file), num_features, averages) for file in files]

#     # Use multiprocessing to process files in parallel
#     with Pool(cpu_count()) as pool:
#         results = pool.map(process_file, args)

#     # Aggregate results
#     corr_matrix = np.sum(results, axis=0)
#     corr_matrix /= len(files)

#     # Save to file
#     df_corr = pd.DataFrame(corr_matrix)
#     df_corr.to_csv("correlation_matrix.csv")

#     return corr_matrix


    

def get_averages():
    dataclass_folder = "/home/athishramdas/Desktop/Pangenome/pangenome-project/best_features_dataset/dataclass"
    dataclass_files = os.listdir(dataclass_folder)
    dataclass_files = [f for f in dataclass_files if f.endswith('.pkl')]

    for i, file in enumerate(dataclass_files):
        with open(os.path.join(dataclass_folder, file), 'rb') as f:
            sample = pickle.load(f)

        if i == 0:
            sums = np.zeros(sample.sparse_vals.shape)
        else:
            sums += sample.sparse_vals

        if i % 10000 == 0:
            print(f"Processed {i} samples")
        
    averages = sums / i

    with open("out_file_averages.txt", 'w') as f:
        for i in range(len(averages)):
            f.write(f"{averages[i]}\n")


            
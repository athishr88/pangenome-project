import matplotlib.pyplot as plt
from kneed import KneeLocator
import pandas as pd
import os

fi_folder = 'explanations'
files = os.listdir(fi_folder)
files = [f for f in files if f.endswith('.csv')]

folds = 1
num_best_features = 500
all_indices = []
ankle_points = {}

for i, file in enumerate(files):
    print(file)
    df = pd.read_csv(os.path.join(fi_folder, file), names=["feature", "score"], skiprows=1, index_col=0)
    df = df.sort_values("score", ascending=False)
    top_500_scores = df.head(num_best_features)['score'].values.tolist()
    knee_locator = KneeLocator(range(1, num_best_features + 1), top_500_scores, curve="convex", direction="decreasing")
    ankle_point = knee_locator.knee
    num_top_choices = int(ankle_point * folds)
    top_features = df.head(num_top_choices).index.tolist()
    top_features = [int(x.split("_")[1]) for x in top_features]
    all_indices.extend(top_features)

    bottom_500_scores = df.tail(num_best_features)['score'].values.tolist()
    bottom_500_scores = bottom_500_scores[::-1]
    knee_locator = KneeLocator(range(1, num_best_features + 1), bottom_500_scores, curve="concave", direction="increasing")
    ankle_point = knee_locator.knee
    ankle_points[file+'_bottom'] = ankle_point
    num_bottom_choices = int(ankle_point * folds)
    bottom_features = df.tail(num_bottom_choices).index.tolist()
    bottom_features = [int(x.split("_")[1]) for x in bottom_features]
    all_indices.extend(bottom_features)

unique_indices = sorted(list(set(all_indices)))
os.makedirs('results/best_indices_filtered/', exist_ok=True)
with open('results/best_indices_filtered/RF_Important_Indices.txt', 'w') as f:
    for idx in unique_indices:
        f.write("%s\n" % idx)
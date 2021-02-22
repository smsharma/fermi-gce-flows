from pathlib import Path
import argparse
import joblib

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm


parser = argparse.ArgumentParser(description="High-level script that extract low-dim summaries")

parser.add_argument("--sample", type=str, default=None, help='Sample name, like "train" or "test".')
parser.add_argument("--n_components", type=int, default=96, help="Number of PCA dimensions",)
parser.add_argument("--n_files", type=int, default=500, help="Number of files to act on",)

args = parser.parse_args()

n_components = args.n_components
sample = args.sample
n_files = args.n_files

# Incremental PCA fit

incremental_pca = IncrementalPCA(n_components=n_components)

for i_file in tqdm(range(n_files)):

    filename = "data/samples/x_{}_{}.npy".format(sample, i_file)
    if not Path(filename).is_file():
        continue
    X = np.load(filename)
    
    incremental_pca.partial_fit(X[:,0,:])

# Save PCA model

model_filename = "data/models/pca_{}_{}.p".format(n_components, sample)

joblib.dump(incremental_pca, model_filename)
incremental_pca = joblib.load(model_filename)  # Load model to make sure that it works

# Do PCA decomposition

for i_file in tqdm(range(n_files)):

    filename = "data/samples/x_{}_{}.npy".format(sample, i_file)
    if not Path(filename).is_file():
        continue
    X = np.load(filename)
    X_pca = incremental_pca.transform(X[:,0,:])
    X_pca = np.expand_dims(X_pca, 1)  # Add channel dimension
    np.save("data/samples/x_pca_{}_{}_{}.npy".format(n_components, sample, i_file), X_pca)

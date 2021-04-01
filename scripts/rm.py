import os
from tqdm import tqdm

for i in tqdm(range(480, 1200)):
    os.remove("../data/samples/x_train_ModelO_gamma_fix_{}.npy".format(i))
import numpy as np
import config

from scipy.sparse import coo_matrix

import pickle

# fname = f'{config.OUTPUT_DIR}/../training_data_subs/training_mask.npy'

# arr = np.load(fname)
# print(arr.shape)

# d = [coo_matrix(arr)]
# print(d)

# with open(fname.replace('_mask.npy', '_mask.pkl'), 'wb') as new_fname:
#     pickle.dump(d, new_fname)


# with open(fname.replace('_mask.npy', '_mask.pkl'), 'rb') as new_fname:
#     d = pickle.load(new_fname)

# print(d)

from sys import getsizeof
a = np.ones((300, 3000,5000), bool)
# print(getsizeof(f))
l = round(getsizeof(a) / 1024 / 1024,2)
print(l)
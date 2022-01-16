from tifffile import imread
import pickle
import matplotlib.pyplot as plt
import numpy as np


train_file = '/home/loaloa/Documents/G001_red_compr.deflate.tif'
train_data = imread(train_file)
print(train_data)
train_data[train_data<43] = 0
print(train_data)
train_data_t0 = np.random.choice(train_data[0].flatten(), int(1e5))
print(train_data_t0)
train_data_tn1 = np.random.choice(train_data[-1].flatten(), int(1e5))
print(train_data_tn1)

inference_file = '/home/loaloa/Documents/timelapse13_processed/D12_G009_GFP_compr.deflate.tif'
inference_data = imread(inference_file)
inference_data -= inference_data.min()
inference_data[inference_data<43] = 0

inf_data_t0 = np.random.choice(inference_data[0].flatten(), int(1e5))
inf_data_tn1 = np.random.choice(inference_data[-1].flatten(), int(1e5))

print('hist....')
fig, axes = plt.subplots(4, figsize=(19,13), sharex=True, sharey=True)
axes[0].hist(train_data_t0, alpha=.7, label='train_t0', bins=2**12, range=(0,2**12))
axes[1].hist(train_data_tn1, alpha=.7, label='train_tn1', bins=2**12, range=(0,2**12))
axes[2].hist(inf_data_t0, alpha=.7, label='inf_data_t0', bins=2**12, range=(0,2**12))
axes[3].hist(inf_data_tn1, alpha=.7, label='inf_data_tn1', bins=2**12, range=(0,2**12))

axes[3].legend()
axes[2].legend()
axes[1].legend()
axes[0].legend()

plt.show()
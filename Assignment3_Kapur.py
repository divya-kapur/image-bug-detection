#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:20:29 2021

@author: divyakapur
"""

import numpy as np
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_extraction.image import extract_patches_2d

b = img.imread('bug.png')
im1 = img.imread('Chickens_with_bug.png')

plt.imshow(im1)
plt.imshow(b)
patches1 = extract_patches_2d(im1, patch_size=(544, 667), max_patches=150)
    
fig, axes = plt.subplots(10,10, figsize=(8,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(patches1[i])
    
patches1_flattened = patches1.flatten()
patches_series = pd.Series(patches1_flattened.tolist())

patches_series

b_flattened = b.flatten()
bug_series = pd.Series(b_flattened.tolist())
bug_series

rolling = patches_series.rolling(window=100)
rolling_mean = rolling.mean()

#plt.plot(patches_series.index, patches_series.values)
#plt.show()

plt.plot(rolling_mean.index, rolling_mean.values)
plt.show()

b_rolling = bug_series.rolling(window=100)
b_rolling_mean = b_rolling.mean()
plt.plot(b_rolling_mean.index, b_rolling_mean.values)
plt.show()

correlations = []
for i in range(len(patches1) - 1):
    patch_flattened = patches1[i].flatten()
    patch_series = pd.Series(patch_flattened.tolist())
    rolling_patch = patch_series.rolling(window=100)
    rolling_patch_mean = rolling_patch.mean()
    rolling_patch_mean = np.nan_to_num(rolling_patch_mean)
    b_rolling_mean = np.nan_to_num(b_rolling_mean)
    r, p = pearsonr(rolling_patch_mean, b_rolling_mean)
    correlations.append(r)

if (1 - abs(np.nanmax(correlations))) < (1 - abs(np.nanmin(correlations))):
    strongest_r = np.nanmax(correlations)
else:
    strongest_r = np.nanmin(correlations)
strongest_r
strongest_match_patch_index = correlations.index(strongest_r)

strongest_match_patch_flattened = patches1[strongest_match_patch_index].flatten()
strongest_match_patch_series = pd.Series(strongest_match_patch_flattened.tolist())

strongest_match_rolling = strongest_match_patch_series.rolling(window=100)
strongest_match_rolling_mean = strongest_match_rolling.mean()
plt.plot(strongest_match_rolling_mean.index, strongest_match_rolling_mean.values)
plt.show()

patches1[strongest_match_patch_index]
plt.imshow(patches1[strongest_match_patch_index])

x_coordinate = 667 * strongest_match_patch_index
y_coordinate = 544 * strongest_match_patch_index

print("Bug located at:", (x_coordinate, y_coordinate))

#!/usr/local/bin/python3
import os
import sys
import numpy as np
from scipy import misc
from sklearn.neighbors import NearestNeighbors

def read_data(data_dir):
    imgs = []
    for i in range(1, 482):
        img_path = os.path.join(data_dir, 'hand.seq{}.png'.format(i))
        img = misc.imread(img_path)
        imgs.append(img.flatten())
    
    return np.array(imgs)

def load_stds():
    data_path = 'stds.npy'
    return np.load(data_path)

def comp_std(imgs, stds):
    std = np.std(imgs)
    dists = np.empty(60)

    for idx, s in enumerate(stds):
        dists[idx] = np.mean(np.abs(s - std))

    min_dist_id = np.argmin(dists) + 1
    print('argmin_id = {}'.format(min_dist_id))

if __name__ == '__main__':
    data_dir = sys.argv[1]
    imgs = read_data(data_dir)
    stds = load_stds()
    comp_std(imgs, stds)


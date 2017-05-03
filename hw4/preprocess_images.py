#!/usr/local/bin/python3
import os
import argparse
import numpy as np
from PIL import Image

def preprocess(dir_path, save_path):
    print('Pre-process images in {} and save to {}'.format(dir_path, save_path))
    imgs = []
    for file in os.listdir(dir_path):
        if not file.endswith('.bmp'):
            continue
        file_path = os.path.join(dir_path, file)
        img = Image.open(file_path)
        imgs.append(np.array(img))

    np.save(save_path, np.array(imgs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dimensionality Reduction with PCA.')
    parser.add_argument('dir_path', help='the images directory path')
    parser.add_argument('--save_path',
        help='the path for saving numpy array',
        default='face_exp_db.npy')

    args = parser.parse_args()
    preprocess(args.dir_path, args.save_path)

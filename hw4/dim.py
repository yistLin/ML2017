#!/usr/local/bin/python3
import os
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Dim:
    def __init__(self, data_path):
        self.read_data(data_path)

    def read_data(self, data_path):
        self.raw_data = np.load(data_path)
        print('raw_data loaded from {}'.format(data_path))

    def load_stds(self):
        data_path = 'stds.npy'
        if os.path.exists(data_path):
            self.stds = np.load(data_path)
            print('stds loaded from {}'.format(data_path))
        else:
            from getdata import calc_std
            self.stds = calc_std(return_std=True)
            print('stds generated from getdata.py')

    def calc_std(self):
        self.predictions = np.empty(200)
        dists = np.empty(60)

        for i in range(200):
            std = np.std(self.raw_data[str(i)])
            print('dataset {}, std = {}'.format(i, std))

            for int_dim in range(60):
                dists[int_dim] = np.mean(np.abs(self.stds[int_dim] - std))
                # print('  ==> dists[{}] = {}'.format(int_dim, dists[int_dim]))

            idx_min_dist = np.argmin(dists)
            self.predictions[i] = idx_min_dist + 1

            print('  ==> idx_min_dist = {}\n'.format(idx_min_dist))

    def calc_nn(self):
        import getdata
        dists = getdata.calc_nn(return_data=True)
        print('distances generated from getdata.py')
        return

        self.predictions = np.empty(200)
        dists = np.empty(60)

        for i in range(200):
            std = np.std(self.raw_data[str(i)])
            print('dataset {}, std = {}'.format(i, std))

            for int_dim in range(60):
                dists[int_dim] = np.mean(np.abs(self.stds[int_dim] - std))
                # print('  ==> dists[{}] = {}'.format(int_dim, dists[int_dim]))

            idx_min_dist = np.argmin(dists)
            self.predictions[i] = idx_min_dist + 1

            print('  ==> idx_min_dist = {}\n'.format(idx_min_dist))

    def write_output(self, csv_path):
        print('write predictions to output file, {}'.format(csv_path))

        outputs = ['SetId,LogDim']
        for idx, dim in enumerate(self.predictions):
            outputs.append('{},{}'.format(idx, np.log(dim)))

        with open(csv_path, 'w') as output_file:
            output_file.write('\n'.join(outputs))
            output_file.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dimensionality Reduction with PCA.')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('output_path', help='output file path')
    args = parser.parse_args()

    dim = Dim(args.data_path)
    # dim.load_stds()
    # dim.calc_std()
    dim.calc_nn()
    # dim.write_output(args.output_path)
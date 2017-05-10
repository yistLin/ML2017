#!/usr/local/bin/python3
import argparse
import numpy as np

class Dim:
    def __init__(self, data_path):
        self.read_data(data_path)

    def read_data(self, data_path):
        self.raw_data = np.load(data_path)
        print('raw_data loaded from {}'.format(data_path))

    def load_stds(self, data_path):
        self.stds = np.load(data_path)
        print('stds loaded from {}'.format(data_path))

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

            print('  ==> idx_min_dist = {}'.format(idx_min_dist))
            print()

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
    dim.load_stds('stds.npy')
    dim.calc_std()
    dim.write_output(args.output_path)

import gen
import numpy as np

def calc_std(return_std=False):
    N = 10000
    stds = np.empty(shape=(60, 20))

    for dim in range(1, 61):
        print('dim =', dim)
        for hdim in range(60, 80):
            layer_dims = [hdim, 100]
            data = gen.gen_data(dim, layer_dims, N)
            std = np.std(data)
        
            stds[dim-1, hdim-60] = std
            # print('[dim = {}][hdim = {}] data: std = {}'.format(dim, hdim, std))

    if return_std:
        return stds
    else:
        np.save('stds.npy', stds)

if __name__ == '__main__':
    calc_std()

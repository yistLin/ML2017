import gen
import numpy as np

N = 10000

stds = np.empty(shape=(60, 20))

for dim in range(1, 61):
    print('dim =', dim)
    for hdim in range(60, 80):
        layer_dims = [hdim, 100]
        data = gen.gen_data(dim, layer_dims, N)
        std = np.std(data)
        mean = np.mean(data)
        
        stds[dim-1, hdim-60] = std
        print('[dim = {}][hdim = {}] data: mean = {}, std = {}'.format(dim, hdim, mean, std))
        break

# np.save('stds.npy', stds)
# np.save('mean_stds.npy', stds)

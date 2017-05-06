#!/usr/local/bin/python3
import argparse
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, images_path):
        print('initializing PCA...')
        self.images_path = images_path
        self.img_arr = self.get_images()

    def get_images(self):
        imgs = np.load(self.images_path)
        print('images array created from {}'.format(self.images_path))
        return imgs.reshape(-1, 4096)

    def select_data(self, first_sub=10, first_img=10):
        arr_a = self.img_arr.reshape(13, 75, 4096)
        self.img_arr = arr_a[:first_sub, :first_img, :].reshape(-1, 4096)

    def plot_average_face(self):
        ZOOM_SCALE = 3
        mean_img = np.mean(self.img_arr, axis=0)
        mean_img = mean_img.reshape(64, 64)
        zoomed_img = scipy.ndimage.zoom(mean_img, ZOOM_SCALE, order=3)
        plt.imsave(fname='average-face.png', arr=zoomed_img, cmap='gray')

    def plot_origin_faces(self):
        first_k = 100
        fig = plt.figure(figsize=(9, 9))
        for i in range(first_k):
            subplt = fig.add_subplot(first_k//10, 10, i+1)
            subplt.axes.get_xaxis().set_ticks([])
            subplt.axes.get_yaxis().set_ticks([])
            subplt.imshow(self.img_arr[i].reshape(64, 64), cmap='gray')
        fig.tight_layout()
        fig.savefig('origin-faces.png')

    def plot_eigen_faces(self, top_k=9):
        mean_img = np.mean(self.img_arr, axis=0)
        arr_a = self.img_arr - mean_img

        U, s, V = np.linalg.svd(arr_a.T, full_matrices=False)

        fig = plt.figure(figsize=(9, 9))
        for i in range(top_k):
            subplt = fig.add_subplot(top_k//3, 3, i+1)
            subplt.axes.get_xaxis().set_ticks([])
            subplt.axes.get_yaxis().set_ticks([])
            subplt.imshow(U[:, i].reshape(64, 64), cmap='gray')
        fig.tight_layout()
        fig.savefig('eigen-faces-top-{}.png'.format(top_k))

    def compute_RMSE(self, arr1, arr2):
        # arr1.shape = arr2.shape = (409600,)
        return np.sqrt(np.average((arr1 - arr2) ** 2))

    def reconstruct_faces(self, top_k=10):
        mean_img = np.mean(self.img_arr, axis=0)
        arr_a = self.img_arr - mean_img
        arr_b = np.empty(shape=arr_a.shape)

        U, s, V = np.linalg.svd(arr_a.T, full_matrices=False)
        weights = np.dot(arr_a, U)
        for i in range(arr_a.shape[0]):
            recon = mean_img + np.dot(weights[i, 0:j], U[:, 0:j].T)
            arr_b[i] = recon

        rmse_val = self.compute_RMSE(self.img_arr.flatten(), arr_b.flatten())
        print('rmse_val = {:4.2f}, error rate = {:6.4f}'.format(rmse_val, rmse_val / 256.))

        first_k = 100
        fig = plt.figure(figsize=(9, 9))
        for i in range(first_k):
            subplt = fig.add_subplot(first_k//10, 10, i+1)
            subplt.axes.get_xaxis().set_ticks([])
            subplt.axes.get_yaxis().set_ticks([])
            subplt.imshow(arr_b[i].reshape(64, 64), cmap='gray')
        fig.tight_layout()
        fig.savefig('reconstruct-with-{}-eigenfaces.png'.format(top_k))

def main(image_path, plot_ori, plot_avg, plot_eig, plot_rec):
    pca = PCA(image_path)
    pca.select_data(first_sub=10, first_img=10)

    if plot_ori:
        print('plot origin faces.')
        pca.plot_origin_faces()
    if plot_avg:
        print('plot average face.')
        pca.plot_average_face()
    if plot_eig:
        print('plot eigen faces.')
        pca.plot_eigen_faces(top_k=9)
    if plot_rec:
        print('plot reconstructed faces.')
        pca.reconstruct_faces(top_k=int(plot_rec))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dimensionality Reduction with PCA.')
    parser.add_argument('images_path', help='images path')
    parser.add_argument('--plot_ori', help='plot origin face', action='store_true')
    parser.add_argument('--plot_avg', help='plot average face', action='store_true')
    parser.add_argument('--plot_eig', help='plot eigen faces', action='store_true')
    parser.add_argument('--plot_rec', help='plot reconstructed faces')
    args = parser.parse_args()
    main(args.images_path, args.plot_ori, args.plot_avg, args.plot_eig, args.plot_rec)

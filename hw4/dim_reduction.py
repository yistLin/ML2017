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
        self.img_arr = arr_a[:first_sub, :first_img, :, :].reshape(-1, 4096)

    def plot_average_face(self):
        ZOOM_SCALE = 3
        mean_img = np.mean(self.img_arr, axis=0)
        mean_img = mean_img.reshape(64, 64)
        zoomed_img = scipy.ndimage.zoom(mean_img, ZOOM_SCALE, order=3)
        fig = plt.imsave(fname='average-face.png', arr=zoomed_img, cmap='gray')

    def plot_origin_faces(self, first_k=9):
        fig = plt.figure(figsize=(9, 9))
        for i in range(first_k):
            subplt = fig.add_subplot(first_k//5, 5, i+1)
            subplt.imshow(self.img_arr[i], cmap='gray')
        fig.tight_layout()
        fig.savefig('origin-faces-first{}.png'.format(first_k))

    def plot_eigen_faces(self, top_k=9):
        mean_img = np.mean(self.img_arr, axis=0)
        arr_a = self.img_arr - mean_img

        U, s, V = np.linalg.svd(arr_a.T, full_matrices=False)

        fig = plt.figure(figsize=(9, 9))
        for i in range(top_k):
            subplt = fig.add_subplot(top_k//3, 3, i+1)
            subplt.imshow(U[:, i].reshape(64, 64), cmap='gray')
        fig.tight_layout()
        fig.savefig('eigen-faces-top{}.png'.format(top_k))

    def reconstruct_faces(self, top_k=10):
        mean_img = np.mean(self.img_arr, axis=0)
        arr_a = self.img_arr - mean_img

        U, s, V = np.linalg.svd(arr_a.T, full_matrices=False)

        for i in range(arr_a.shape[0]):
            s[i][top_k:] = 0
            arr_b[i] = np.dot(U[i], np.dot(np.diag(s[i]), V[i]))

def main(image_path, plot_ori, plot_avg, plot_eig, plot_rec):
    pca = PCA(image_path)
    pca.select_data(first_sub=10, first_img=10)

    if plot_ori:
        print('plot origin faces.')
        pca.plot_origin_faces(first_k=25)
    if plot_avg:
        print('plot average face.')
        pca.plot_average_face()
    if plot_eig:
        print('plot eigen faces.')
        pca.plot_eigen_faces(top_k=9)
    if plot_rec:
        print('plot reconstructed faces.')
        pca.reconstruct_faces(top_k=20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dimensionality Reduction with PCA.')
    parser.add_argument('images_path', help='images path')
    parser.add_argument('--plot_ori', help='plot origin face', action='store_true')
    parser.add_argument('--plot_avg', help='plot average face', action='store_true')
    parser.add_argument('--plot_eig', help='plot eigen faces', action='store_true')
    parser.add_argument('--plot_rec', help='plot reconstructed faces', action='store_true')
    args = parser.parse_args()
    main(args.images_path, args.plot_ori, args.plot_avg, args.plot_eig, args.plot_rec)

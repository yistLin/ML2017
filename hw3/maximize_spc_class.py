#!/usr/local/bin/env python3
# -- coding: utf-8 --
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.utils import generic_utils
from scipy.ndimage.filters import gaussian_filter, median_filter

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def deprocess_image(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-20)
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_ascent(num_step, input_image_data, iter_func):
    lr = 0.05
    grad_lr = np.zeros(shape=(1, 48, 48, 1))
    for i in range(num_step):
        loss_val, grad_val = iter_func([input_image_data, 0])
        print('[{}] loss_val = {}'.format(i+1, loss_val))
        # if loss_val > 0.9999:
        #     break
        # grad_lr = grad_lr + grad_val ** 2
        # input_image_data += (lr / np.sqrt(grad_lr)) * grad_val
        input_image_data += grad_val * lr
        input_image_data *= 0.9999

        if i % 5 == 0:
            input_image_data = gaussian_filter(input_image_data, sigma=[0, 0, 0.3, 0.3])
        if i % 10 == 0:
            input_image_data = median_filter(input_image_data, size=(1, 1, 5, 5))
    return deprocess_image(input_image_data.squeeze())

def main():
    model_path = sys.argv[1]
    model = load_model(model_path)
    print('Model loaded')

    input_img = model.input
    NUM_STEPS = 200
    SPCIFIC_CLASS = 3 # class 'Happy'

    input_img_data = np.random.random_sample((1, 48, 48, 1))
    target = K.mean(model.output[:, SPCIFIC_CLASS])
    grads = normalize(K.gradients(target, input_img)[0])
    iterate = K.function([input_img, K.learning_phase()], [target, grads])
    img = grad_ascent(NUM_STEPS, input_img_data, iterate)

    print('img.shape =', img.shape)
    print('Draw the image')
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()

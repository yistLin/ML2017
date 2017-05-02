#!/usr/local/bin/env python3
# -- coding: utf-8 --
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.utils import generic_utils
from scipy.ndimage.filters import gaussian_filter, median_filter
import scipy.ndimage
from keras.utils import generic_utils

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def deprocess_image(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-20)
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def main():
    model_path = sys.argv[1]
    model = load_model(model_path)
    print('Model loaded')

    input_img = model.input
    SPEC_CLASS = 3
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    input_img_data = np.random.random((1, 48, 48, 1))
    target = K.mean(model.output[:, SPEC_CLASS])
    # grads = normalize(K.gradients(target, input_img)[0])
    grads = K.gradients(target, input_img)[0]
    iterate = K.function([input_img, K.learning_phase()], [target, grads])
    # img = grad_ascent(NUM_STEPS, input_img_data, iterate)

    lr = 1
    NUM_STEPS = 2500
    progbar = generic_utils.Progbar(NUM_STEPS)
    for i in range(NUM_STEPS):
        input_img_data *= 0.9999

        if (i+1) % 5 == 0:
            input_img_data = gaussian_filter(input_img_data, sigma=[0, 0, 0.3, 0.3])
        if (i+1) % 10 == 0:
            input_img_data = median_filter(input_img_data, size=(1, 1, 3, 3))

        loss_val, grad_val = iterate([input_img_data, 0])
        # print('[{:4d}] loss_val = {:10.8f}'.format(i+1, loss_val))
        input_img_data += grad_val * lr
        progbar.add(1, values=[('loss', loss_val)])

        if (i + 1) % 500 == 0:
            img = input_img_data.squeeze()
            draw_img = deprocess_image(img)
            draw_img = scipy.ndimage.zoom(draw_img, 4, order=3)
            plt.imsave(fname='max-{}-ep{}.png'.format(class_names[SPEC_CLASS], i+1), arr=draw_img, cmap='gray')
            # plt.show()
            prediction = model.predict(img.reshape(1, 48, 48, 1), verbose=1)
            print('prediction =', prediction)

if __name__ == "__main__":
    main()

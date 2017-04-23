#!/usr/local/bin/env python3
# -- coding: utf-8 --
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.utils import generic_utils

def read_features(filename):
    col0 = []
    col1 = []
    with open(filename, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            fields = line.strip().split(',')
            col0.append(int(fields[0]))
            col1.append(list(map(int, fields[1].split())))
            if idx == 300:
                break
    return np.array(col0), np.array(col1)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-10)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_ascent(num_step, record_freq, input_image_data, iter_func):
    filter_images = []
    for i in range(num_step):
        loss_val, grad_val = iter_func([input_image_data, 0])
        input_image_data += grad_val * 0.05
        if i % record_freq == 0:
            filter_images.append((deprocess_image(input_image_data.squeeze()), loss_val))
    return filter_images

def main():
    model_path = sys.argv[1]

    # Load model
    model = load_model(model_path)
    print('Model loaded')

    layer_dict = dict([layer.name, layer] for layer in model.layers[:])
    print('layer_dict =', layer_dict.keys)

    input_img = model.input
    name_ls = ['leaky_re_lu_1']

    NUM_STEPS = 90
    RECORD_FREQ = 30
    nb_filter = 16
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        print('Processing layer: {}'.format(name_ls[cnt]))

        filter_imgs = []
        print('Gradient ascent the filter')
        progbar = generic_utils.Progbar(nb_filter)
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1))
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])
            filter_imgs.append(grad_ascent(NUM_STEPS, RECORD_FREQ, input_img_data, iterate))
            # print progress
            progbar.add(1)

        print('Draw the image')
        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            print('Dealing filter state in epoch {}'.format(it*RECORD_FREQ))
            progbar = generic_utils.Progbar(nb_filter)
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/8, 8, i+1)
                ax.imshow(filter_imgs[i][it][0], cmap='Blues')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[i][it][1]))
                plt.tight_layout()
                # print progress
                progbar.add(1, values=[('subplot', i)])
            # fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            fig.savefig('{}_e{}'.format(name_ls[cnt], it*RECORD_FREQ))

if __name__ == "__main__":
    main()

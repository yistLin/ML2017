#!/usr/local/bin/python3
import argparse
import word2vec
import numpy as np

def training(data_path):
    phrase_path = data_path + '.phrase'
    model_path = data_path + '.bin'

    word2vec.word2phrase(data_path, phrase_path, verbose=True)
    word2vec.word2vec(phrase_path, model_path, size=100, verbose=True)

def plot_model(model_path):
    # plot most frequent k words
    TOP_FREQ = 500

    # load the model
    model = word2vec.load(model_path)

    # get vocabularies
    vocab = model.vocab
    print('vocab[:100] =', vocab[:100])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization of Word Vectors.')
    parser.add_argument('--train', help='training data path')
    parser.add_argument('--plot', help='plot the trained model')
    args = parser.parse_args()

    # Do something
    if args.train:
        training(args.train)
    elif args.plot:
        plot_model(args.plot)
    else:
        print('Nothing to do')

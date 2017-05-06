#!/usr/local/bin/python3
import argparse
import word2vec
import nltk
import numpy as np
from sklearn.manifold import TSNE

def training(data_path):
    phrase_path = data_path + '.phrase'
    model_path = data_path + '.bin'

    word2vec.word2phrase(data_path, phrase_path, verbose=True)
    word2vec.word2vec(phrase_path, model_path, size=100, verbose=True)

def pick_vocabs(vocabs, nb_keep=100):
    KEEP_TAGS = ['JJ', 'NN', 'NNP', 'NNS']
    IGNORE_PUNC = ['“', '”', ',', '.', ':', ';', '’', '!', '?']
    
    keep_vocabs = []
    word_cnt = 0
    for vocab in vocabs:
        if len(vocab) < 2:
            # print('{:<16} is too short (length < 2)'.format(vocab))
            continue
        elif any(punc in vocab for punc in IGNORE_PUNC):
            # print('{:<16} contains IGNORE_PUNC'.format(vocab))
            continue

        voc, tag = nltk.pos_tag([vocab])[0]
        if tag not in KEEP_TAGS:
            # print('{:<16} tagged as {}, dropped'.format(voc, tag))
            continue

        # print('{:<16} tagged as {}, keeped'.format(voc, tag))
        keep_vocabs.append(vocab)

        word_cnt += 1
        if word_cnt == nb_keep:
            break

    return keep_vocabs

def plot_model(model_path):
    # load the model
    model = word2vec.load(model_path)
    vocabs = model.vocab

    # keep frequently appeared vocabularies
    keep_vocabs = pick_vocabs(vocabs, nb_keep=500)

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

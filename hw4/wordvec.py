#!/usr/local/bin/python3
import argparse
import word2vec
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embedding_size = 100

def training(data_path):
    phrase_path = data_path + '.phrase'
    model_path = data_path + '.bin'

    word2vec.word2phrase(data_path, phrase_path, verbose=True)
    word2vec.word2vec(phrase_path, model_path, size=embedding_size, verbose=True)

def pick_vocabs(vocabs, nb_keep=100):
    KEEP_TAGS = ['JJ', 'NN', 'NNP', 'NNS']
    IGNORE_PUNC = ['“', '”', ',', '.', ':', ';', '’', '!', '?']
    
    keep_vocabs = []
    word_cnt = 0
    for vocab in vocabs:
        if len(vocab) < 2:
            continue
        elif any(punc in vocab for punc in IGNORE_PUNC):
            continue

        voc, tag = nltk.pos_tag([vocab])[0]
        if tag not in KEEP_TAGS:
            continue

        keep_vocabs.append(vocab)
        word_cnt += 1
        if word_cnt == nb_keep:
            break

    return keep_vocabs

def plot_model(model_path):
    # load the model
    model = word2vec.load(model_path)
    vocabs = model.vocab
    print('word2vec model loaded')

    # keep frequently appeared vocabularies
    nb_vocab = 500
    keep_vocabs = pick_vocabs(vocabs, nb_keep=nb_vocab)
    voc_vectors = np.array([model[voc] for voc in keep_vocabs])
    print('vocab vectors gotten')

    # reduce the dimension
    tsne = TSNE(n_components=2, random_state=0)
    voc_vectors_new = tsne.fit_transform(voc_vectors)
    print('vocab vectors reduced to 2-dim vectors')

    plt.plot(*zip(*voc_vectors_new), marker='o', ls='')
    plt.show()

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

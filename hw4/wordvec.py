#!/usr/local/bin/python3
import argparse
import word2vec
import nltk
import random
import numpy as np
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
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # load the model
    model = word2vec.load(model_path)
    vocabs = model.vocab
    print('word2vec model loaded')

    # keep frequently appeared vocabularies
    nb_vocab = 500
    keep_vocabs = np.array(pick_vocabs(vocabs, nb_keep=nb_vocab))
    voc_vectors = np.array([model[voc] for voc in keep_vocabs])
    print('vocab vectors gotten')

    # reduce the dimension
    tsne = TSNE(n_components=2, random_state=0)
    voc_vectors_new = tsne.fit_transform(voc_vectors)
    print('vocab vectors reduced to 2-dim vectors')

    xs = voc_vectors_new[:, 0] * 1000.
    ys = voc_vectors_new[:, 1] * 1000.

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        ax.plot(x, y, 'o', label=str(idx))

    texts = []
    for x, y, voc in zip(xs, ys, keep_vocabs):
        texts.append(ax.text(x, y, voc))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    fig.savefig('wordvec-{}.png'.format(nb_vocab))

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
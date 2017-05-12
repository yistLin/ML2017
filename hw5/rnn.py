#!/usr/local/bin/python3
import nltk
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Embedding
from keras.callbacks import ModelCheckpoint, CSVLogger


def read_data(data_path):
    data = []
    with open(data_path, 'r', encoding='latin1') as data_file:
        next(data_file)
        cnt = 0
        for line in data_file:
            sp_line = line.strip().split('"', 2)
            line_id = int(sp_line[0][:-1])
            tags = sp_line[1].split(',')
            text_str = sp_line[2][1:].lower()
            texts = nltk.word_tokenize(text_str)
            
            data.append((line_id, tags, texts))
    return data


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    """Compute the F1 score
    
    Parameters:
        y_true: pass by fit function
        y_pred: pass by fit function
    """

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = 1
    f_score = (1 + bb) * (p * r) / (bb * p + r + k.epsilon())
    return f_score


class ArtikelKlassfizier:
    def __init__(self, max_seq_len=500):
        self.max_seq_len = max_seq_len

    def fit(self, data):
        """Fit the training data

        Parameters:
            data <numpy.array>
                the data for model training
        """
        lineids, tags, texts = zip(*data)

        # calculate number of words
        word2id = {'_PAD': 0}
        word_cnt = len(word2id)
        for s in texts:
            for word in s:
                if word not in word2id:
                    word2id[word] = word_cnt
                    word_cnt += 1
        nb_words = len(word2id)
        print('nb_words =', nb_words)

        # transfer to ids
        id_arr = []
        for s in texts:
            ids = [word2id[word] for word in s]
            id_arr.append(ids)

        # pad to specific length
        X_train = []
        for ids in id_arr:
            pad_ids = [0] * (self.max_seq_len - len(ids)) + ids
            X_train.append(pad_ids)
        X_train = np.array(X_train)

        # calculate number of tags
        tag2id = {}
        id2tag = []
        tag_cnt = 0
        for t in tags:
            for tag in t:
                if tag not in tag2id:
                    tag2id[tag] = tag_cnt
                    id2tag.append(tag)
                    tag_cnt += 1
        nb_tags = len(tag2id)
        print('nb_tags =', nb_tags)

        # convert tags to tag_ids
        Y_train = np.zeros(shape=(len(X_train), nb_tags))
        for i, t in enumerate(tags):
            Y_train[i, [tag2id[tag] for tag in t]] = 1

        self.nb_words = nb_words
        self.nb_tags = nb_tags
        self.word2id = word2id
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.X_train = X_train
        self.Y_train = Y_train

    def build_model(self, embedding_size=32):
        """Build the model

        Parameters:
            embedding_size <int>
                the embedding size of word embedding
        """

        self.model = Sequential()
        self.model.add(Embedding(self.nb_words, embedding_size, input_length=self.max_seq_len))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dense(self.nb_tags, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    def train(self):
        print('train with training data.')
        self.model.fit(self.X_train, self.Y_train, )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RNN tag predictor.')
    parser.add_argument('data_path', help='images path')
    parser.add_argument('--plot', help='plot', action='store_true')
    args = parser.parse_args()

    print('Running rnn.py as main program.')

    data = read_data(args.data_path)

    # max sequence length is 365
    ak = ArtikelKlassfizier(max_seq_len=400)
    ak.fit(data)
    ak.build_model(embedding_size=64)
    ak.train()

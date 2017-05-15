#!/usr/local/bin/python3
import pickle
import numpy as np
import keras.backend as K
from utils import DataReader
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, SimpleRNN, Embedding, TimeDistributed, RepeatVector, Bidirectional
from keras.callbacks import ModelCheckpoint, CSVLogger


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
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
    f_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return f_score


class ArtikelKlassfizier:
    def __init__(self, max_seq_len=500):
        self.max_seq_len = max_seq_len

    def fit(self, tags, texts):
        """Fit the training data

        Parameters:
            data <numpy.array>
                the data for model training
        """

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
        # self.model.add(LSTM(64, return_sequences=True))
        # self.model.add(LSTM(64))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(RepeatVector(38))
        # self.model.add(LSTM(64, return_sequences=True))
        # self.model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        # self.model.compile(loss='mse', optimizer='adam', metrics=[f1_score])
        
        # self.model.add(LSTM(64, return_sequenes=True))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(0.4))
        self.model.add(SimpleRNN(64))
        self.model.add(Dense(self.nb_tags, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])

        print(self.model.summary())

    def train(self):
        print('train with training data.')
        self.model.fit(self.X_train, self.Y_train, epochs=30, batch_size=32, validation_split=0.1)

    def dump(self):
        with open('word2id.pkl', 'wb') as f:
            pickle.dump(self.word2id, f)

        with open('id2tag.pkl', 'wb') as f:
            pickle.dump(self.id2tag, f)

        self.model.save('model.hdf5')

    def predict(self, data, output_path):
        with open('word2id.pkl', 'rb') as f:
            word2id = pickle.load(f)
        with open('id2tag.pkl', 'rb') as f:
            id2tag = pickle.load(f)
        
        X_test = []
        for words in data:
            ids = [word2id.get(word, 0) for word in words]
            if len(ids) < self.max_seq_len:
                ids = [0] * (self.max_seq_len - len(ids)) + ids
            elif len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            X_test.append(ids)
    
        X_test = np.array(X_test)
        predictions = self.model.predict(X_test)
        print('predictions.shape =', predictions.shape)
    
        results = []
        for dist in predictions:
            tags = [id2tag[id[0]] for id in np.argwhere(dist > 0.5)]
            results.append(tags)
    
        with open(output_path, 'w') as out_f:
            outputs = ['id,tags\n']
            for idx, tags in enumerate(results):
                outputs.append('{},"{}"\n'.format(idx, ','.join(tags)))
            out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RNN tag predictor.')
    parser.add_argument('data_path', help='images path')
    parser.add_argument('--predict', help='predict from saved model')
    args = parser.parse_args()

    print('Running rnn.py as main program.')

    reader = DataReader()

    if not args.predict:
        _, tags, texts = reader.read_data(args.data_path)
        print(texts)
        exit()
        ak = ArtikelKlassfizier(max_seq_len=400)
        ak.fit(tags, texts)
        ak.build_model(embedding_size=128)
        ak.train()
        ak.dump()
    else:
        _, texts = reader.read_test_data(args.data_path)
        ak = ArtikelKlassfizier(max_seq_len=400)
        ak.model = load_model(args.predict, custom_objects={'f1_score': f1_score})
        ak.predict(texts, 'rnn-output.csv')


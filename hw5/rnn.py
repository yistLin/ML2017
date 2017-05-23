#!/usr/local/bin/python3
import pickle
import numpy as np
from utils import DataReader
from utils import f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, SimpleRNN, GRU, Embedding
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import ModelCheckpoint


class ArtikelKlassfizier:
    def __init__(self):
        self.tokenizer = None
        self.binarizer = None
        self.embedding_dim = 100
        self.max_seq_len = 300

    def get_embedding_dict(self, path):
        embedding_dict = {}
        with open(path, 'r') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_dict[word] = coefs
        return embedding_dict

    def fit(self, tags, sentences, wordvec_path):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(sentences)
        self.binarizer = MultiLabelBinarizer().fit(tags)

        self.word_index = self.tokenizer.word_index
        self.nb_words = len(self.tokenizer.word_index)
        self.nb_tags = len(self.binarizer.classes_)
        print('# of word_index =', self.nb_words)
        print('# of tags =', self.nb_tags)

        embedding_dict = self.get_embedding_dict(wordvec_path)
        self.embedding_matrix = np.zeros(shape=(self.nb_words+1, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embedding_dict.get(word, np.zeros(shape=(self.embedding_dim)))
            self.embedding_matrix[i] = embedding_vector

    def train(self, X, Y, model_path):
        X = self.tokenizer.texts_to_sequences(X)
        Y = self.binarizer.transform(Y)
        X = pad_sequences(X)
        self.max_seq_len = X.shape[1]

        print('# of training data =', X.shape[0])
        print('max_seq_len =', self.max_seq_len)

        # shuffle and split training data
        indices = np.arange(X.shape[0])
        np.random.seed(5)
        np.random.shuffle(indices)
        X_data = X[indices]
        Y_data = Y[indices]
        X_train, X_valid = X_data[490:], X_data[:490]
        Y_train, Y_valid = Y_data[490:], Y_data[:490]

        model = Sequential()
        model.add(Embedding(
                self.nb_words + 1,
                self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_seq_len,
                trainable=False))
        # model.add(GRU(128, activation='tanh', dropout=0.2, return_sequences=True))
        model.add(GRU(128, activation='tanh', dropout=0.3, return_sequences=True))
        model.add(GRU(256, dropout=0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.nb_tags, activation='sigmoid'))

        adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])

        ckpt = ModelCheckpoint('rnn-model.hdf5', monitor='val_f1_score', save_best_only=True, mode='max')

        model.summary()
        model.fit(X_train, Y_train, epochs=5000, batch_size=128,
                validation_data=(X_valid, Y_valid),
                callbacks=[ckpt])

        model.save(model_path)

    def predict(self, sentences, model, output_path):
        X_test = self.tokenizer.texts_to_sequences(sentences)
        X_test = pad_sequences(X_test, maxlen=306)
        predictions = model.predict(X_test)
        print('predictions.shape =', predictions.shape)

        results = self.binarizer.inverse_transform(np.round(predictions))

        with open(output_path, 'w') as out_f:
            outputs = ['"id","tags"\n']
            for idx, tags in enumerate(results):
                all_tags = ' '.join(tags) if len(tags) > 0 else 'FICTION NOVEL'
                outputs.append('"{}","{}"\n'.format(idx, all_tags))
            out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RNN tag predictor.')
    parser.add_argument('--train_data', help='data path')
    parser.add_argument('--test_data', help='test data path')
    parser.add_argument('--wordvec', help='glove word2vec path')
    parser.add_argument('--class_path', default='rnn-classifier.pkl')
    parser.add_argument('--model_path', default='rnn-model.hdf5')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    print('Running rnn.py as main program.')

    reader = DataReader()
    model_path = args.model_path
    class_path = args.class_path

    if args.fit:
        _, tags, train_sentences = reader.read_data(args.train_data)
        _, test_sentences = reader.read_test_data(args.test_data)
        ak = ArtikelKlassfizier()
        ak.fit(tags, list(train_sentences) + list(test_sentences), args.wordvec)

        with open(args.class_path, 'wb') as class_f:
            pickle.dump(ak, class_f)

    if args.train:
        _, tags, train_sentences = reader.read_data(args.train_data)
        with open(args.class_path, 'rb') as class_f:
            ak = pickle.load(class_f)
        ak.train(train_sentences, tags, args.model_path)

    if args.predict:
        _, texts = reader.read_test_data(args.test_data)
        with open(args.class_path, 'rb') as class_f:
            ak = pickle.load(class_f)
        model = load_model(args.model_path, custom_objects={'f1_score': f1_score})
        ak.predict(list(texts), model, 'rnn-output.csv')

#!/usr/local/bin/python3
import pickle
import numpy as np
from utils import DataReader
from utils import f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, SimpleRNN, GRU, Embedding, Bidirectional
from sklearn.preprocessing import MultiLabelBinarizer


class ArtikelKlassfizier:
    def __init__(self):
        self.tokenizer = None
        self.binarizer = None
        self.embedding_dim = 200

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
        self.tag_classes = self.binarizer.classes_
        self.nb_words = len(self.word_index)
        self.nb_tags = len(self.tag_classes)
        print('# of word_index =', self.nb_words)
        print('# of tags =', self.nb_tags)

        embedding_dict = self.get_embedding_dict(wordvec_path)
        self.embedding_matrix = np.zeros(shape=(self.nb_words+1, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embedding_dict.get(word, np.zeros(shape=(self.embedding_dim)))
            self.embedding_matrix[i] = embedding_vector

    def train(self, X, Y, model_path):
        X = self.tokenizer.texts_to_sequences(X)
        self.Y = self.binarizer.transform(Y)
        self.X = pad_sequences(X)
        self.max_seq_len = self.X.shape[1]

        print('# of training data =', self.X.shape[0])
        print('max_seq_len =', self.max_seq_len)

        model = Sequential()
        model.add(Embedding(
                self.nb_words + 1,
                self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_seq_len,
                trainable=False))
        model.add(LSTM(128, activation='tanh', dropout=0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.nb_tags, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])

        model.summary()
        model.fit(self.X, self.Y, epochs=50, batch_size=32, validation_split=0.1)

        model.save(model_path)

    def predict(self, tokens_list, output_path):
        with open('rnn-labeler.pkl', 'rb') as f:
            labeler = pickle.load(f)
        with open('rnn-binarizer.pkl', 'rb') as f:
            binarizer = pickle.load(f)

        for i, tokens in enumerate(tokens_list):
            if len(tokens) < self.max_seq_len:
                tokens_list[i] += ['_PAD'] * (self.max_seq_len - len(tokens))
            elif len(tokens) > self.max_seq_len:
                tokens_list[i] = tokens[:self.max_seq_len]

        X_test = labeler.transform(tokens_list)
        predictions = self.model.predict(X_test)
        print('predictions.shape =', predictions.shape)

        results = binarizer.inverse_transform(np.round(predictions))

        with open(output_path, 'w') as out_f:
            outputs = ['"id","tags"\n']
            for idx, tags in enumerate(results):
                outputs.append('"{}","{}"\n'.format(idx, ' '.join(tags)))
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
        ak.predict(list(texts), 'rnn-output.csv')


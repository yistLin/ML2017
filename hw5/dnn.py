#!/usr/local/bin/python3
import nltk
import pickle
import numpy as np
from utils import DataReader
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import ModelCheckpoint


def f1_score(y_true, y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred, thresh), dtype='float32')
    tp = K.sum(y_true * y_pred, axis=-1)
    precision = tp / (K.sum(y_pred, axis=-1) + K.epsilon())
    recall = tp / (K.sum(y_true, axis=-1) + K.epsilon())
    return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))


class Classifier:
    def __init__(self):
        pass

    def train_tfidf(self, sentences, tags):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=40000, sublinear_tf=True)
        self.vectorizer.fit(sentences)
        self.binarizer = MultiLabelBinarizer()
        self.binarizer.fit(tags)

    def train_svm(self, sentences, tags, model_path):
        X = self.vectorizer.transform(sentences)
        Y = self.binarizer.transform(tags)
        print('data_vec.shape =', X.shape)
        print('tag_vec.shape =', Y.shape)

        indices = np.arange(X.shape[0])
        np.random.seed(5)
        np.random.shuffle(indices)
        X_data = X[indices]
        Y_data = Y[indices]
        X_train, X_valid = X_data[490:], X_data[:490]
        Y_train, Y_valid = Y_data[490:], Y_data[:490]

        model = Sequential()
        model.add(Dense(512, input_shape=(40000,), activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(38, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])

        model.summary()

        ckpt = ModelCheckpoint('dnn-model.hdf5', monitor='val_f1_score', save_best_only=True, mode='max')
        model.summary()
        model.fit(X_train.toarray(), Y_train, epochs=5000, batch_size=128,
                validation_data=(X_valid.toarray(), Y_valid),
                callbacks=[ckpt]
                )

    def predict(self, sentences, output_path):
        model = load_model('dnn-model.hdf5', custom_objects={'f1_score': f1_score})
        test_vec = self.vectorizer.transform(sentences)
        preds = model.predict(test_vec.toarray())
        preds[preds < 0.4] = 0
        preds[preds >= 0.4] = 1
        results = self.binarizer.inverse_transform(preds)

        with open(output_path, 'w') as out_f:
            outputs = ['"id","tags"\n']
            for idx, tags in enumerate(results):
                alltags = 'FICTION NOVEL' if len(tags) == 0 else ' '.join(tags)
                outputs.append('"{}","{}"\n'.format(idx, alltags))
            out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('--train', help='predict')
    parser.add_argument('--predict', help='doc2vec')
    parser.add_argument('--model', default='dnn-classifier.pkl')
    parser.add_argument('--cv', type=int, default=0)
    args = parser.parse_args()

    reader = DataReader()
    model_path = args.model

    if args.train:
        _, tags, texts = reader.read_data(args.train)

        clf = Classifier()
        clf.train_tfidf(texts, tags)
        with open(model_path, 'wb') as model_f:
            pickle.dump(clf, model_f)
        clf.train_svm(texts, tags, model_path)

    if args.predict:
        _, texts = reader.read_test_data(args.predict)
        with open(model_path, 'rb') as model_f:
            clf = pickle.load(model_f)
        clf.predict(texts, 'dnn-output.csv')


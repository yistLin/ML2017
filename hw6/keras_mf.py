import numpy as np
from utils import read_data, write_output
from keras.models import Sequential, load_model
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


class MF_Model(Sequential):
    def __init__(self, nb_users, nb_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(nb_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))

        Q = Sequential()
        Q.add(Embedding(nb_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))

        super(MF_Model, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
        self.add(Dropout(0.3))
        self.add(Dense(k_factors, activation='relu'))
        self.add(Dropout(0.3))
        self.add(Dense(1, activation='linear'))


def train(train_data_path, model_path):
    train_data = read_data(train_data_path)
    max_userid = 6040
    max_movieid = 3952

    print('train_data.shape =', train_data.shape)
    print('max_userid = {}, max_movieid = {}'.format(max_userid, max_movieid))

    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm]
    valid_split = int(0.1 * train_data.shape[0])
    user_train, user_valid = train_data[valid_split:, 1], train_data[:valid_split, 1]
    item_train, item_valid = train_data[valid_split:, 2], train_data[:valid_split, 2]
    rate_train, rate_valid = train_data[valid_split:, 3], train_data[:valid_split, 3]
    print('# of validation data =', valid_split)

    mf_model = MF_Model(max_userid, max_movieid, 256)
    mf_model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping('val_loss', patience=3)
    ckpt = ModelCheckpoint(model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1)
    mf_model.fit(
        [user_train, item_train],
        rate_train,
        epochs=30,
        validation_data=([user_valid, item_valid], rate_valid),
        verbose=1,
        callbacks=[early_stop, ckpt])


def predict(test_data_path, model_path, output_path):
    test_data = read_data(test_data_path)
    mf_model = MF_Model(6040, 3952, 256)
    mf_model.load_weights(model_path)

    user_test = test_data[:, 1]
    item_test = test_data[:, 2]
    predictions = mf_model.predict([user_test, item_test]).squeeze()
    print(predictions.shape)
    print(predictions)

    write_output(predictions, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--model_path', default='mf_model.hdf5')
    parser.add_argument('--output_path', default='output.csv')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    if args.train:
        train(args.data_path, args.model_path)
    elif args.predict:
        predict(args.data_path, args.model_path, args.output_path)
    else:
        print('You must specify --train or --predict.')

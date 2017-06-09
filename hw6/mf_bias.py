import numpy as np
import keras
from utils import read_data, write_output
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Reshape, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

nb_factors = 16

def build_model(nb_users, nb_items, k_factors):
    user_input = Input(shape=(1,))
    P = Embedding(input_dim=nb_users, output_dim=k_factors, input_length=1)(user_input)
    user_emb_output = Reshape((k_factors,))(P)

    item_input = Input(shape=(1,))
    Q = Embedding(input_dim=nb_items, output_dim=k_factors, input_length=1)(item_input)
    item_emb_output = Reshape((k_factors,))(Q)

    R = Embedding(input_dim=nb_users, output_dim=1, input_length=1)(user_input)
    user_bias = Reshape((1,))(R)

    S = Embedding(input_dim=nb_items, output_dim=1, input_length=1)(item_input)
    item_bias = Reshape((1,))(S)

    mf_output = keras.layers.dot([user_emb_output, item_emb_output], axes=-1)
    mf_output_bias = keras.layers.add([mf_output, user_bias, item_bias])
    return Model(inputs=[user_input, item_input], outputs=[mf_output_bias])


def train(train_data_path, model_path):
    train_data = read_data(train_data_path)
    max_userid = 6040
    max_movieid = 3952

    print('train_data.shape =', train_data.shape)
    print('max_userid = {}, max_movieid = {}'.format(max_userid, max_movieid))

    np.random.seed(seed=7)
    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm]
    valid_split = int(0.1 * train_data.shape[0])
    user_train, user_valid = train_data[valid_split:, 1], train_data[:valid_split, 1]
    item_train, item_valid = train_data[valid_split:, 2], train_data[:valid_split, 2]
    rate_train, rate_valid = train_data[valid_split:, 3], train_data[:valid_split, 3]
    print('# of validation data =', valid_split)
    print('user_train.shape =', user_train.shape)
    print('item_train.shape =', item_train.shape)
    print('rate_train.shape =', rate_train.shape)

    rate_mean = 3.581712
    rate_std = 1.116897
    rate_train = (rate_train - rate_mean) / rate_std
    rate_valid = (rate_valid - rate_mean) / rate_std

    mf_model = build_model(max_userid, max_movieid, nb_factors)
    mf_model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping('val_loss', patience=3)
    ckpt = ModelCheckpoint(model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1)
    mf_model.fit(
        [user_train, item_train],
        rate_train,
        epochs=100,
        batch_size=256,
        validation_data=([user_valid, item_valid], rate_valid),
        verbose=1,
        callbacks=[ckpt, early_stop])


def predict(test_data_path, model_path, output_path):
    test_data = read_data(test_data_path)
    mf_model = load_model(model_path)

    user_test = test_data[:, 1]
    item_test = test_data[:, 2]
    predictions = mf_model.predict([user_test, item_test]).squeeze()
    predictions = predictions * 1.116897 + 3.581712
    print(predictions.shape)
    print(predictions)

    write_output(test_data[:, 0], np.clip(predictions, 1, 5), output_path)


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

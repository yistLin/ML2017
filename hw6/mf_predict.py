import argparse
import numpy as np
from utils import read_data, write_output
from keras.models import load_model


def predict(test_data_path, model_path, output_path):
    test_data = read_data(test_data_path)
    mf_model = load_model(model_path)

    user_test = test_data[:, 1]
    item_test = test_data[:, 2]
    predictions = mf_model.predict([user_test, item_test]).squeeze()

    write_output(test_data[:, 0], np.clip(predictions, 1, 5), output_path)


parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('--model_path', default='mf_model.hdf5')
parser.add_argument('--output_path', default='output.csv')
args = parser.parse_args()

predict(args.data_path, args.model_path, args.output_path)

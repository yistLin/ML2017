import pickle
import keras.backend as K

class DataReader():
    def __init__(self):
        pass

    def read_data(self, data_path):
        data = []
        with open(data_path, 'r') as data_file:
            next(data_file)
            for line in data_file:
                sp_line = line.strip().split('"', 2)
                line_id = int(sp_line[0][:-1])
                tags = sp_line[1].split()
                text_str = sp_line[2][1:].lower()
                data.append((line_id, tags, text_str))
        return zip(*data)
 
    def read_test_data(self, data_path):
        data = []
        with open(data_path, 'r') as data_file:
            next(data_file)
            for line in data_file:
                sp_line = line.strip().split(',', 1)
                line_id = int(sp_line[0])
                text_str = sp_line[1].lower()
                data.append((line_id, text_str))
        return zip(*data)


def precision(y_true, y_pred, thresh=0.5):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred, thresh=0.5):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def old_f1_score(y_true, y_pred, thresh=0.5):
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
    f_score = K.mean((1 + bb) * (p * r) / (bb * p + r + K.epsilon()))
    return f_score


def f1_score(y_true, y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred, thresh), dtype='float32')
    tp = K.sum(y_true * y_pred, axis=-1)
    precision = tp / (K.sum(y_pred, axis=-1) + K.epsilon())
    recall = tp / (K.sum(y_true, axis=-1) + K.epsilon())
    return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))

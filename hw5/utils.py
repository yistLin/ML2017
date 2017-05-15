import nltk
import string
import pickle

class DataReader():
    def __init__(self):
        self.translator = str.maketrans('', '', string.punctuation)
        with open('eng-stop-words.pkl', 'rb') as f:
            stop_words_list = pickle.load(f)
            self.stop_words = set(stop_words_list)

    def remove_punc(self, sent):
        return sent.translate(self.translator)

    def remove_stop_words(self, tokens):
        return [token if token in self.stop_words for token in tokens]

    def read_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='latin1') as data_file:
            next(data_file)
            for line in data_file:
                sp_line = line.strip().split('"', 2)
                line_id = int(sp_line[0][:-1])
                tags = sp_line[1].split(',')
                text_str = sp_line[2][1:].lower()
                text_str = self.remove_punc(text_str)
                tokens = nltk.word_tokenize(text_str)
                tokens = self.remove_stop_words(tokens)
                data.append((line_id, tags, tokens))
        return zip(*data)
    
    def read_test_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='latin1') as data_file:
            next(data_file)
            for line in data_file:
                sp_line = line.strip().split(',', 1)
                line_id = int(sp_line[0])
                text_str = sp_line[1].lower()
                text_str = self.remove_punc(text_str)
                tokens = nltk.word_tokenize(text_str)
                tokens = self.remove_stop_words(tokens)
                data.append((line_id, tokens))
        return zip(*data)


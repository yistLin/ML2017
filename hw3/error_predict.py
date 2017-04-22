import sys
import numpy as np
from keras.models import load_model
from PIL import Image

def read_features(filename):
    col0 = []
    col1 = []
    with open(filename, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            fields = line.strip().split(',')
            col0.append(int(fields[0]))
            col1.append(list(map(int, fields[1].split())))
            if idx == 300:
                break

    return np.array(col0), np.array(col1)

def predict(model, features):
    X = features / 255
    X.shape = (-1, 48, 48, 1)
    return model.predict_classes(X)

def main():
    labels, features = read_features(sys.argv[1])
    model = load_model(sys.argv[2])

    results = predict(model, features[:100])
    answers = labels[:100]
    results.flatten()
    answers.flatten()

    print('Sad  in results:', np.argwhere(results == 4).reshape(-1))
    print('Fear in answers:', np.argwhere(answers == 2).reshape(-1))
    print('Fear in results:', np.argwhere(results == 2).reshape(-1))
    print('Sad  in answers:', np.argwhere(answers == 4).reshape(-1))

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # labels[2] is Fear, but is predicted as Sad
    img = Image.new('L', (48, 48))
    img.putdata(features[2])
    img = img.resize((256, 256), resample=Image.ANTIALIAS)
    img.save('train-image-2.png', 'PNG')

    # labels[6] is Sad, but is predicted as Fear
    img = Image.new('L', (48, 48))
    img.putdata(features[6])
    img = img.resize((256, 256), resample=Image.ANTIALIAS)
    img.save('train-image-6.png', 'PNG')

if __name__ == '__main__':
    main()

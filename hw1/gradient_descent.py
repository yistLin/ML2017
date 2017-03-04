import sys
import numpy as np

# Read in training data
with open(sys.argv[1], 'r', encoding='big5') as train_file:
    train_data = []
    for line in train_file:
        line = line.replace('NR', '0.0')
        fields = line[:-1].split(',')
        train_data.append(fields[3:])

# Prepare training data
train_data = np.array(train_data[1:]) # shape = (4320, 24)
train_data = train_data.reshape(12, -1, 24) # shape = (12, 360, 24)
train_data = train_data.reshape(12, -1, 18, 24) # shape = (12, 20, 18, 24)
train_data = train_data.swapaxes(1, 2).reshape(12, 18, -1) # shape = (12, 18, 480)
needed_cols = [8, 9, 10, 15, 16]
target_col = 9 # PM2.5
num_col = len(needed_cols)
X = []
Y = []
for mon in range(12):
    for i in range(480 - 9):
        X.append(np.array([train_data[mon][col][i:i+9] for col in needed_cols]))
        Y.append(train_data[mon][target_col][i+9].astype(float))

# Declare basic settings
iteration = 300000
lr = 0.1
b_lr = 1e-2
w_lr = np.full((num_col * 9), 1e-2)
b = 0.0
w = np.ones(shape=(num_col * 9))

# Training
x_flat = np.array(X).reshape(len(X), 9 * num_col).astype(float)
# print(x_flat)
# print('x_flat shape =', x_flat.shape)
# print('w shape =', w.shape)

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros(shape=(num_col * 9))

    predictions = x_flat.dot(w) + b
    errors = Y - predictions
    # print('predictions =', predictions)
    # print('predictions shape =', predictions.shape)
    # print('errors =', errors)
    # print('errors shape =', errors.shape)
    
    b_grad = b_grad - 2.0 * np.sum(errors) * 1.0
    w_grad = w_grad - 2.0 * np.dot(x_flat.T, errors)

    # for n in range(len(X)):
    #     x_flat = X[n].flatten().astype(float)
    #     b_grad = b_grad - 2.0 * (Y[n] - b - x_flat.dot(w)) * 1.0
    #     w_grad = w_grad - 2.0 * (Y[n] - b - x_flat.dot(w)) * x_flat
    
    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2
    # Update parameters
    b = b - lr / np.sqrt(b_lr) * b_grad
    w = w - lr / np.sqrt(w_lr) * w_grad

    if (i+1) % 100 == 0:
        print('Epoch %d' % (i+1))

np.set_printoptions(threshold=np.nan)
print('b =', b)
print('w =', w)

# Read in testing data
with open(sys.argv[2], 'r', encoding='big5') as test_file:
    test_data = []
    for line in test_file:
        line = line.replace('NR', '0.0')
        fields = line[:-1].split(',')
        test_data.append(fields[2:])

# Prepare testing data
test_data = np.array(test_data) # shape = (4320, 9)
test_data = test_data.reshape(-1, 18, 9) # shape = (240, 18, 9)
test_X = []
test_Y = []
for i in range(test_data.shape[0]):
    test_X.append(np.array([test_data[i][col][:] for col in needed_cols]))

# Testing
for x in test_X:
    x_flat = x.flatten().astype(float)
    test_Y.append(x_flat.dot(w) + b)

# Write predictions to output file
with open(sys.argv[3], 'w') as output_file:
    outputs = ['id,value']
    for i in range(len(test_Y)):
        outputs.append('id_%d,%f' % (i, test_Y[i]))
    output_file.write('\n'.join(outputs) + '\n')

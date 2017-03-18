import sys
import numpy as np

# Read in training data
with open(sys.argv[1], 'r', encoding='big5') as train_file:
    train_data = []
    for line in train_file:
        fields = line[:-1].split(',')
        train_data.append([0.0 if x in ['NR', ''] else float(x) for x in fields[3:]])

# Prepare training data
train_data = np.array(train_data[1:]) # shape = (4320, 24)
train_data = train_data.reshape(12, -1, 24) # shape = (12, 360, 24)
train_data = train_data.reshape(12, -1, 18, 24) # shape = (12, 20, 18, 24)
train_data = train_data.swapaxes(1, 2).reshape(12, 18, -1) # shape = (12, 18, 480)

needed_cols = [2, 7, 8, 9, 10, 12, 14, 15, 16, 17]
# needed_cols = range(18)
target_col = 9 # PM2.5
num_col = len(needed_cols)
prev_col = 9
X = []
Y = []
for mon in range(12):
    for i in range(480 - prev_col):
        X.append(np.array([train_data[mon][col][i:i+prev_col] for col in needed_cols]))
        Y.append(train_data[mon][target_col][i+prev_col])

# Training
X = np.array(X)
col_pm2 = X[:, 2].reshape(len(X), prev_col)
col_pm10 = X[:, 3].reshape(len(X), prev_col)
col_mul = col_pm2 * col_pm10
X = X.reshape(len(X), prev_col * num_col)
Y = np.array(Y)
# X = np.concatenate((X, X ** 3, col_mul), axis=1)

# Normalization
feat_max = np.max(X, axis=0)
feat_min = np.min(X, axis=0)
X = (X - feat_min) / (feat_max - feat_min + 1e-20)

# Separate valid set and train set
# randomize = np.random.permutation(len(X))
# X, Y = X[randomize], Y[randomize]
# valid_X, train_X = X[:1000], X[1000:]
# valid_Y, train_Y = Y[:1000], Y[1000:]
train_X = X
train_Y = Y

# Declare basic settings
iteration = 100000
lr = 0.5
b_lr = 1e-20
w_lr = np.full((X.shape[1]), 1e-20)
b = 0.0
w = np.ones(shape=(X.shape[1]))
prev_rmse = 1e20
rising_cnt = 0

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros(shape=(X.shape[1]))

    predictions = train_X.dot(w) + b
    errors = train_Y - predictions

    b_grad = b_grad - 2.0 * np.sum(errors)
    w_grad = w_grad - 2.0 * np.dot(train_X.T, errors)

    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2
    # Update parameters
    b = b - lr / np.sqrt(b_lr) * b_grad
    w = w - lr / np.sqrt(w_lr) * w_grad

    if (i+1) % 1000 == 0:
        print('Epoch %d' % (i+1))
        print('  --> [Train] RMSE Loss = %f' % np.sqrt(np.mean(errors ** 2)))
        # predictions = valid_X.dot(w) + b
        # errors = valid_Y - predictions
        # rmse = np.sqrt(np.mean(errors ** 2))
        # if prev_rmse < rmse:
        #     rising_cnt += 1
        #     if rising_cnt > 3:
        #         print('  --> [Warning] valid loss rise more than 3 times')
        #         break
        # else:
        #     rising_cnt = 0
        # prev_rmse = rmse
        # print('  --> [Valid] RMSE Loss = %f' % rmse)

np.set_printoptions(threshold=np.nan)
print('b =', b)
print('w =', w)
# np.save('b-' + sys.argv[3], b)
# np.save('w-' + sys.argv[3], w)
# np.save('fmax-' + sys.argv[3], feat_max)
# np.save('fmin-' + sys.argv[3], feat_min)

# Read in testing data
with open(sys.argv[2], 'r', encoding='big5') as test_file:
    test_data = []
    for line in test_file:
        fields = line[:-1].split(',')
        test_data.append([0.0 if x in ['NR', ''] else float(x) for x in fields[2:]])

# Prepare testing data
test_data = np.array(test_data) # shape = (4320, 9)
test_data = test_data.reshape(-1, 18, 9) # shape = (240, 18, 9)
test_X = []
test_Y = []
for i in range(test_data.shape[0]):
    test_X.append(np.array([test_data[i][col][9-prev_col:] for col in needed_cols]))

test_X = np.array(test_X)

# Testing
for x in test_X:
    col_pm2 = x[2].flatten()
    col_pm10 = x[3].flatten()
    col_mul = col_pm2 * col_pm10
    x_flat = x.flatten()
    # x_flat = np.concatenate((x_flat, x_flat ** 3, col_mul))
    x_flat = (x_flat - feat_min) / (feat_max - feat_min + 1e-20)
    test_Y.append(x_flat.dot(w) + b)

# Write predictions to output file
with open(sys.argv[3], 'w') as output_file:
    outputs = ['id,value']
    for i in range(len(test_Y)):
        outputs.append('id_%d,%f' % (i, test_Y[i]))
    output_file.write('\n'.join(outputs) + '\n')

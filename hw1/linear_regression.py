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
col_o3 = X[:, 1].reshape(len(X), prev_col)
col_pm2 = X[:, 3].reshape(len(X), prev_col)
col_mul = col_o3 * col_pm2
X = X.reshape(len(X), prev_col * num_col)
Y = np.array(Y)
X = np.concatenate((X, X ** 2), axis=1)

# Normalization
feat_max = np.max(X, axis=0)
feat_min = np.min(X, axis=0)
X = (X - feat_min) / (feat_max - feat_min + 1e-20)

# Separate valid set and train set
randomize = np.random.permutation(len(X))
X, Y = X[randomize], Y[randomize]
valid_X, train_X = X[5000:], X[:5000]
valid_Y, train_Y = Y[5000:], Y[:5000]
# train_X = X[:2000]
# train_Y = Y[:2000]
print('# of valid set =', len(valid_X))

# Declare basic settings
w = np.ones(shape=(X.shape[1]))

# closed-form
X_bias = np.concatenate((train_X, np.ones(shape=(train_X.shape[0],1))), axis=1)
w = np.linalg.lstsq(X_bias, train_Y)[0]
predictions = X_bias.dot(w)
errors = train_Y - predictions
print('  --> [Train] RMSE Loss = %f' % np.sqrt(np.mean(errors ** 2)))

# closed-form for valid set
valid_X = np.concatenate((valid_X, np.ones(shape=(valid_X.shape[0],1))), axis=1)
predictions = valid_X.dot(w)
errors = valid_Y - predictions
print('  --> [Valid] RMSE Loss = %f' % np.sqrt(np.mean(errors ** 2)))

# np.set_printoptions(threshold=np.nan)
# print('w =', w)

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
    col_o3 = x[1].flatten()
    col_pm2 = x[3].flatten()
    col_mul = col_o3 * col_pm2
    x_flat = x.flatten()

    x_flat = np.concatenate((x_flat, x_flat ** 2))
    x_flat = (x_flat - feat_min) / (feat_max - feat_min + 1e-20)

    # closed-form
    x_bias = np.concatenate((x_flat, np.ones(1)), axis=0)
    test_Y.append(x_bias.dot(w))

# Write predictions to output file
with open(sys.argv[3], 'w') as output_file:
    outputs = ['id,value']
    for i in range(len(test_Y)):
        outputs.append('id_%d,%f' % (i, test_Y[i]))
    output_file.write('\n'.join(outputs) + '\n')

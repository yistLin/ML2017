import sys
import numpy as np

# Read in training data
with open(sys.argv[1], 'r', encoding='big5') as train_file:
    train_data = []
    for line in train_file:
        train_data.append((line.strip('\n').split(','))[3:])

# Prepare training data
train_data = np.array(train_data[1:]) # shape = (4320, 24)
train_data = train_data.reshape(-1, 18, 24) # shape = (240, 18, 24)
train_data = train_data.swapaxes(0, 1).reshape(18, -1) # shape = (18, 5760)
needed_cols = [8, 9] # PM10, PM2.5
target_col = 9
X = []
Y = []
for i in range(5760 - 9):
    X.append(train_data[col][i:i+9] for col in needed_cols)
    Y.append(train_data[target_col][i+9])

sys.exit(1)

# Declare basic settings
iteration = 100

# Iterations
for i in range(iteration):

    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad  - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad  - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2

    # Update parameters
    b = b - lr / np.sqrt(b_lr) * b_grad
    w = w - lr / np.sqrt(w_lr) * w_grad

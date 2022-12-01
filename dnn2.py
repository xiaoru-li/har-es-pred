import time
import h5py
import os
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import torch
import sys
sys.stdout = open('mlp-opp-500.txt', 'w')

# time the script
start = time.time()

print(torch.__version__)
# %matplotlib inline
plt.style.use('ggplot')


# select dataset
ds = 'opportunity'

# path to the dataset in the data folder
path = os.path.join('data', ds + '.h5')  # change this to another dataset if you want

# load the dataset
f = h5py.File(path, 'r')
x_train = f.get('train').get('inputs')[()]
y_train = f.get('train').get('targets')[()]
x_test = f.get('test').get('inputs')[()]
y_test = f.get('test').get('targets')[()]

print("x_train shape =", x_train.shape)
print("y_train shape =", y_train.shape)
print("x_test shape =", x_test.shape)
print("y_test shape =", y_test.shape)


def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += (size // 2)  # 50% overlap in the sliding windows.


def segment(x_train, y_train, window_size):
    # 9 is the number of features. we put window_size samples in the same line
    # with 9 * window_size per row. The format is, if we had 3 features, xyz
    # x1,y1,z1,x2,y2,z2,x3,y3,z3 etc, per row. 1->2->3...->window_size
    shapeX = x_train.shape
    segments = np.zeros(((len(x_train)//(window_size//2))-1, window_size*shapeX[1]))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0

    for (start, end) in windowz(x_train, window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            offset_st = 0
            offset_fin = window_size
            for i in range(shapeX[1]):
                segments[i_segment][offset_st:offset_fin] = (x_train[start:end])[:, [i]].T
                offset_st = (i+1)*window_size
                offset_fin = (i+2)*window_size
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


input_width = 23  # (creates approximately 650k samples)

print("segmenting signal...")
train_x, train_y = segment(x_train, y_train, input_width)
test_x, test_y = segment(x_test, y_test, input_width)
print("signal segmented.")

print("train_x shape =", train_x.shape)
print("train_y shape =", train_y.shape)
print("test_x shape =", test_x.shape)
print("test_y shape =", test_y.shape)

# http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/
train = pd.get_dummies(train_y)
test = pd.get_dummies(test_y)
train, test = train.align(test, join='inner', axis=1)

train_y = np.asarray(train)
test_y = np.asarray(test)

print("unique test_y", np.unique(test_y))
print("unique train_y", np.unique(train_y))
print("test_y[1]=", test_y[1])
print("train_y shape(1-hot) =", train_y.shape)
print("test_y shape(1-hot) =", test_y.shape)


# Parameters
input_height = 1
num_labels = 18
num_channels = 77

learning_rate = 0.0005
# training_epochs = 20  # early stopping
training_epochs = 500
batch_size = 64
display_step = 1

# Network Parameters
# MLP has 3 hidden layers with 256 neurons each, 1 fully connected layer with 512 neurons
# Each layer passes through ReLU activation function and a max pooling layer.
# The dropout rate is 0.1 for the first layer, 0.25 for the second, and 0.5 for the third one, and the fully connected layer.

n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_hidden_3 = 256  # 3rd layer number of features
n_hidden_4 = 512  # 4th layer number of features
dropout_rate_1 = 0.1
dropout_rate_2 = 0.25
dropout_rate_3 = 0.5

# Create MLP model with parameters above


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_width*num_channels, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.fc5 = nn.Linear(n_hidden_4, num_labels)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate_1)
        self.dropout2 = nn.Dropout(dropout_rate_2)
        self.dropout3 = nn.Dropout(dropout_rate_3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


# Create model
model = MLP()
print(model)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Stohastic Gradient Descent optimizer to minimize negative log likelihood
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(train_x.shape[0]/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(batch_y).long()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, torch.max(batch_y, 1)[1])
        loss.backward()
        optimizer.step()
        avg_cost += loss.item()/total_batch
    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("Optimization Finished!")

# Test the Model
correct = 0
total = 0
test_x = test_x.reshape(-1, input_width*num_channels)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(np.argmax(test_y, 1))
outputs = model(test_x.float())
_, predicted = torch.max(outputs.data, 1)
total += test_y.size(0)
correct += (predicted == test_y).sum()
print('Accuracy of the network: %d %%' % (100 * correct / total))

y_pred = predicted.numpy()
y_true = test_y.numpy()

# Print out F1 score
f1_score_mean = metrics.f1_score(y_true, y_pred, average='macro')
f1_score_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
f1_score_per_class = metrics.f1_score(y_true, y_pred, average=None)
print("F1 score (mean):", f1_score_mean)
print("F1 score (weighted):", f1_score_weighted)
print("F1 score (per class):", f1_score_per_class)

# Print out the confusion matrix
print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))


# Save the Model
torch.save(model.state_dict(), 'output/mlp_opp_model_{}_{}.pkl'.format(training_epochs, batch_size))

# time end
end = time.time()
print("time elapsed: ", end - start)

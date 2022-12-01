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
sys.stdout = open('cnn-dap-200.txt', 'w')

# time the script
start = time.time()

print(torch.__version__)
# %matplotlib inline
plt.style.use('ggplot')


# select dataset
ds = 'daphnet'

# path to the dataset in the data folder
path = os.path.join('data', ds + '.h5')  # change this to another dataset if you want

# load the dataset
f = h5py.File(path, 'r')
x_train = f.get('train').get('inputs')[()]
y_train = f.get('train').get('targets')[()]
x_test = f.get('test').get('inputs')[()]
y_test = f.get('test').get('targets')[()]

# downsample to 30 Hz
x_train = x_train[::2, :]
y_train = y_train[::2]
x_test = x_test[::2, :]
y_test = y_test[::2]

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


input_width = 25  # (creates approximately 470k samples)

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
num_labels = 2
num_channels = 9

learning_rate = 0.001
# training_epochs = 20  # early stopping
training_epochs = 200
batch_size = 64
display_step = 1


# Network Parameters
# CNN has 2 hidden layers of 128 neurons each, 1 fully connected layer with 512 neurons
# Each layer passes through ReLU activation function and a max pooling layer.
# The dropout rate is 0.1 for the first layer, 0.25 for the second, and 0.5 for the third one (fully connected).
# Kernel sizes (for the convolution) :
kernel_size_1 = 7
kernel_size_2 = 3
dropout_rate_1 = 0.1
dropout_rate_2 = 0.25
dropout_rate_3 = 0.5


class CNN(nn.Module):
    def __init__(self, input_height, input_width, num_channels, num_labels):
        super(CNN, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=128,
                               kernel_size=(kernel_size_1, self.num_channels), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(kernel_size_2, 1), stride=1, padding=0)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, self.num_labels)
        self.dropout1 = nn.Dropout(dropout_rate_1)
        self.dropout2 = nn.Dropout(dropout_rate_2)
        self.dropout3 = nn.Dropout(dropout_rate_3)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# Create model
model = CNN(input_height, input_width, num_channels, num_labels)
print(model)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Adam optimizer to minimize negative log likelihood
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Create a function to test the model
def test_model(model, test_x, test_y):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(test_x)):
        test = torch.from_numpy(test_x[i]).float()
        test = test.view(1, num_channels, input_width)
        outputs = model(test)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == torch.max(torch.from_numpy(test_y[i]).float(), 0)[1]).sum()
    print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))


# Train the Model
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(train_x.shape[0]/batch_size)

    for i in range(total_batch):
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        batch_x = torch.from_numpy(batch_x).float()
        batch_x = batch_x.view(batch_size, num_channels, input_width)
        batch_y = torch.from_numpy(batch_y).float()
        batch_y = torch.max(batch_y, 1)[1]

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        avg_cost += loss.item() / total_batch
    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))


print("Optimization Finished!")

# Test the Model
test_model(model, test_x, test_y)


# Create a function to print out F1 scores
def f1_score(model, test_x, test_y):
    model.eval()
    y_true = []
    y_pred = []
    for i in range(len(test_x)):
        test = torch.from_numpy(test_x[i]).float()
        test = test.view(1, num_channels, input_width)
        outputs = model(test)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(torch.max(torch.from_numpy(test_y[i]).float(), 0)[1].numpy())
        y_pred.append(predicted.numpy())
    f1_score_mean = metrics.f1_score(y_true, y_pred, average='macro')
    f1_score_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    f1_score_per_class = metrics.f1_score(y_true, y_pred, average=None)
    print("F1 score (mean):", f1_score_mean)
    print("F1 score (weighted):", f1_score_weighted)
    print("F1 score (per class):", f1_score_per_class)


# Print out F1 scores
f1_score(model, test_x, test_y)


# Create a function to print out confusion matrix
def confusion_matrix(model, test_x, test_y):
    model.eval()
    y_true = []
    y_pred = []
    for i in range(len(test_x)):
        test = torch.from_numpy(test_x[i]).float()
        test = test.view(1, num_channels, input_width)
        outputs = model(test)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(torch.max(torch.from_numpy(test_y[i]).float(), 0)[1].numpy())
        y_pred.append(predicted.numpy())
    cm = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)


# Print out confusion matrix
confusion_matrix(model, test_x, test_y)

# Save the Model
torch.save(model.state_dict(), 'output/cnn_dap_model_{}_{}.pkl'.format(training_epochs, batch_size))

# time end
end = time.time()
print("time elapsed: ", end - start)

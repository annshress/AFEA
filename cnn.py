'''implements cnn calssification on the faces'''

import numpy as np
import lasagne
import sys
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import detect
from stratify import shuffle
# from nolearn.lasagne import TrainSplit
from sklearn.externals import joblib

sys.setrecursionlimit(5000)

images, classes, filenames = detect.main()

d = {'DI': 0, 'NE': 1, 'SU': 2, 'AN': 3, 'FE': 4, 'SA': 5, 'HA': 6}

X = images
y = [d[i] for i in classes]

# Splitting dateset into train and test using stratified sampling
train_index, test_index = shuffle(classes)

X_train, X_test, y_train, y_test = [], [], [], []
for i in train_index:
    X_train.append(X[i])
    y_train.append(y[i])
for i in test_index:
    X_test.append(X[i])
    y_test.append(y[i])

# making data theano compatible
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train.reshape((-1, 1, 157, 157))
X_test = X_test.reshape((-1, 1, 157, 157))
# theano works with uint8
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

print 'training values : %d ' % (len(X_train))
print 'testing  values : %d ' % (len(X_test))

# Neural Net Architecture
net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('LRN1', layers.LocalResponseNormalization2DLayer),
            ('conv2a', layers.Conv2DLayer),
            ('conv2b', layers.Conv2DLayer),
            ('maxpool2a', layers.MaxPool2DLayer),
            ('conv2c', layers.Conv2DLayer),
            ('concat2', layers.ConcatLayer),
            ('maxpool2b', layers.MaxPool2DLayer),
            ('conv3a', layers.Conv2DLayer),
            ('conv3b', layers.Conv2DLayer),
            ('maxpool3a', layers.MaxPool2DLayer),
            ('conv3c', layers.Conv2DLayer),
            ('concat3', layers.ConcatLayer),
            ('maxpool3b', layers.MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 157, 157),
    # layer conv1
    conv1_num_filters=64,
    conv1_filter_size=(7, 7),
    conv1_stride=2,
    conv1_pad=3,
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
    conv1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(3, 3),
    maxpool1_stride=2,
    maxpool1_pad=0,
    # layer LRN1
    LRN1_incoming='maxpool1',
    # layer conv2a
    conv2a_num_filters=96,
    conv2a_filter_size=(1, 1),
    conv2a_stride=1,
    conv2a_pad=0,
    conv2a_nonlinearity=lasagne.nonlinearities.rectify,
    # layer conv2b
    conv2b_num_filters=208,
    conv2b_filter_size=(3, 3),
    conv2b_stride=1,
    conv2b_pad=1,
    conv2b_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2a
    maxpool2a_incoming='LRN1',
    maxpool2a_pool_size=(3, 3),
    maxpool2a_stride=1,
    maxpool2a_pad=1,
    # layer conv2c
    conv2c_incoming='maxpool2a',
    conv2c_num_filters=64,
    conv2c_filter_size=(1, 1),
    conv2c_stride=1,
    conv2c_pad=0,
    conv2c_nonlinearity=lasagne.nonlinearities.rectify,
    # layer concat2
    concat2_incomings=['conv2b', 'conv2c'],
    # layer maxpool2b
    maxpool2b_pool_size=(3, 3),
    maxpool2b_stride=2,
    maxpool2b_pad=0,
    # layer conv3a
    conv3a_num_filters=96,
    conv3a_filter_size=(1, 1),
    conv3a_stride=1,
    conv3a_pad=0,
    conv3a_nonlinearity=lasagne.nonlinearities.rectify,
    # layer conv3b
    conv3b_num_filters=208,
    conv3b_filter_size=(3, 3),
    conv3b_stride=1,
    conv3b_pad=1,
    conv3b_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool3a
    maxpool3a_incoming='maxpool2b',
    maxpool3a_pool_size=(3, 3),
    maxpool3a_stride=1,
    maxpool3a_pad=1,
    # layer conv3c
    conv3c_incoming='maxpool3a',
    conv3c_num_filters=64,
    conv3c_filter_size=(1, 1),
    conv3c_stride=1,
    conv3c_pad=0,
    conv3c_nonlinearity=lasagne.nonlinearities.rectify,
    # layer concat3
    concat3_incomings=['conv3b', 'conv3c'],
    # layer maxpool3b
    maxpool3b_incoming='concat3',
    maxpool3b_pool_size=(3, 3),
    maxpool3b_stride=2,
    maxpool3b_pad=0,
    # dropout1
    dropout1_p=0.5,
    # dense1
    dense1_num_units=256,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    # dense2
    dense2_num_units=512,
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=7,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.0005,
    update_momentum=0.9,
    max_epochs=100,
    # train_split=TrainSplit(eval_size=0.25),
    verbose=1,
    )

nn = net.fit(X_train, y_train)
# to save the model
# joblib.dump(nn, 'cnn.pkl')
preds = nn.predict(X_test)
print preds
print accuracy_score(y_test, preds)

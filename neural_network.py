#!/usr/local/bin/python3
import numpy as np
from copy import deepcopy
from random import randrange

"""
Activation function for hidden layers
- 1e-9
"""
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

"""
Derivative of sigmoid function
"""
def sigmoid_derivative(x):
    y = sigmoid(x)
    return y * (1 - y)

"""
Activation function for output layer
"""
def softmax(x):
    # print(x)
    x_e = np.exp(x - np.max(x))
    return x_e / np.sum(x_e, axis = 0, keepdims = True)

"""
Implementation of a neural network with 2 hidden layers to classify hand written digits. Tunable parameters are
1. n_nodes - # of nodes in hidden layers
2. batch_size - # of data to train before updating weights and biases
3. epochs - # of times to train entire data set
4. learning_rate - the learning rate
"""
class NeuralNetwork:
    def __init__(self, data, labels, n_labels = 10, n_nodes = 512, batch_size = 20, epochs = 50, learning_rate = 0.65):
        self.data = data
        self.labels = labels
        self.n_samples = self.data.shape[0]
        self.n_attributes = self.data.shape[1]
        self.n_labels = n_labels
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # First Hidden Layer
        self.weights1 = np.random.normal(size = (self.n_nodes, self.n_attributes))
        self.bias1 = np.zeros((self.n_nodes, 1))

        # Second Hidden Layer
        self.weights2 = np.random.normal(size = (self.n_nodes, self.n_nodes))
        self.bias2 = np.zeros((self.n_nodes, 1))

        # Output Layer
        self.weights3 = np.random.normal(size = (self.n_labels, self.n_nodes))
        self.bias3 = np.zeros((self.n_labels, 1))

    """
    Forward propagation step
    """
    def __feed_forward(self, data):
        # First Hidden Layer
        i1 = np.dot(self.weights1, data) + self.bias1
        o1 = sigmoid(i1)

        # Second Hidden Layer
        i2 = np.dot(self.weights2, o1) + self.bias2
        o2 = sigmoid(i2)

        # Output Layer
        i3 = np.dot(self.weights3, o2) + self.bias3
        o3 = softmax(i3)

        return i1, o1, i2, o2, i3, o3

    """
    Backward propagation for adjusting weights and biases
    """
    def __back_propagate(self, data, labels, i1, o1, i2, o2, i3, o3):
        # Output Layer
        di3 = o3 - labels
        dw3 = np.dot(di3, o2.T) / self.batch_size
        db3 = np.sum(di3, axis=1, keepdims=True) / self.batch_size
        # print(o3_delta)
        
        # Second Hidden Layer
        di2 = self.weights3.T.dot(di3) * sigmoid_derivative(i2)
        dw2 = np.dot(di2, o1.T) / self.batch_size
        db2 = np.sum(di2, axis=1, keepdims=True) / self.batch_size

        # First Hidden Layer
        di1 = self.weights2.T.dot(di2) * sigmoid_derivative(i1)
        dw1 = np.dot(di1, data.T) / self.batch_size
        db1 = np.sum(di1, axis=1, keepdims=True) / self.batch_size

        return dw1, db1, dw2, db2, dw3, db3

    """
    Update weights and biases
    """
    def __update_weights_and_biases(self, dw1, db1, dw2, db2, dw3, db3):
        # First Hidden Layer
        self.weights1 -= self.learning_rate * dw1
        self.bias1 -= self.learning_rate * db1

        # Second Hidden Layer
        self.weights2 -= self.learning_rate * dw2
        self.bias2 -= self.learning_rate * db2

        # Output Layer
        self.weights3 -= self.learning_rate * dw3
        self.bias3 -= self.learning_rate * db3
    
    """
    Train the neural network given the hyper-parameters
    """
    def train(self):
        train_size = self.labels.shape[0]
        for _ in range(self.epochs):
            for i in range(0, train_size, self.batch_size):
                batch_data = self.data[i : i + self.batch_size]
                batch_data = batch_data.T
                batch_labels = self.labels[i : i + self.batch_size]
                batch_labels = batch_labels.T
            
                i1, o1, i2, o2, i3, o3 = self.__feed_forward(batch_data)
                dw1, db1, dw2, db2, dw3, db3 = self.__back_propagate(batch_data, batch_labels, i1, o1, i2, o2, i3, o3)
                self.__update_weights_and_biases(dw1, db1, dw2, db2, dw3, db3)

    """
    Make predictions of new data based on trained model
    """
    def predict(self, data):
        data = data.T

        # First Hidden Layer
        i1 = np.dot(self.weights1, data) + self.bias1
        o1 = sigmoid(i1)

        # Second Hidden Layer
        i2 = np.dot(self.weights2, o1) + self.bias2
        o2 = sigmoid(i2)

        # Output Layer
        i3 = np.dot(self.weights3, o2) + self.bias3
        o3 = softmax(i3)

        return np.around(o3)

    def accuracy(self, labels_pred, labels):
        return np.average(labels_pred == labels)

#!/usr/local/bin/python3
import numpy as np
import pandas as pd
import csv
import sys
from neural_network import NeuralNetwork

"""
Converts labels for data using one-hot encoding
"""
def one_hot_encode(data, labels):
    one_hot_labels = np.zeros((data.shape[0], 10))
    for i in range(data.shape[0]):
        for j in range(10):
            if labels[i] == j:
                one_hot_labels[i][j] = 1
    return one_hot_labels
    

def main():
    # Get input files
    train_data_df = pd.read_csv('train_image.csv', sep=',',header=None)
    train_labels_df = pd.read_csv('train_label.csv', sep=',',header=None)
    test_data_df = pd.read_csv('test_image.csv', sep=',',header=None)
    test_labels_df = pd.read_csv('test_label.csv', sep=',',header=None)

    # Format data
    train_data = train_data_df.astype(float).values
    train_labels = train_labels_df.values
    test_data = test_data_df.astype(float).values
    test_labels = test_labels_df.values

    # Normalization
    train_data /= 255.0
    test_data /= 255.0
    
    # One hot encode train and test labels
    one_hot_train_labels = one_hot_encode(train_data, train_labels)
    one_hot_test_labels = one_hot_encode(test_data, test_labels)

    # Train Neural Network on training set
    neural_network = NeuralNetwork(train_data, one_hot_train_labels)
    neural_network.train()
    
    # Make predictions for test data
    labels_pred = neural_network.predict(test_data)
    labels_pred = labels_pred.T
    
    accuracy = neural_network.accuracy(labels_pred, one_hot_test_labels)
    print("Accuracy is {}".format(accuracy))

if __name__ == "__main__":
    main()

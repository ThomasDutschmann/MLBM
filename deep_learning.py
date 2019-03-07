
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, TimeDistributed, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import random


def read_window_file(path):
    windows = list()
    structures = list()
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            window = [float(x) for x in parts[0].split(',')]
            struct = [int(x) for x in parts[1].split(',')]
            windows.append(window)
            structures.append(struct)
    return windows, structures


file_based_windows, file_based_structures = read_window_file('windows_and_structures.csv')


def run_learning_on_window_sequences(points, labels, length, verbose=1, test_fraction=0.2, epochs=150, batch_size=100):
    # Function to automatically run an RNN training
    X = X = np.array([[point] for point in points])
    Y = np.array(labels)
    
    indices = range(len(X))
    test_indices = random.sample(indices, int(len(X) * test_fraction))
    
    X_train = np.array([X[i] for i in indices if i not in test_indices])
    Y_train = np.array([Y[i] for i in indices if i not in test_indices])
    
    X_test = np.array([X[i] for i in test_indices])
    Y_test = np.array([Y[i] for i in test_indices])
    
    if verbose > 0:
        print('Input shape:')
        get_shape(X)
        print('Output shape:')
        get_shape(Y)

    # create model
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(1, 5 * length)))
    model.add(Dense(3, activation='sigmoid'))

    if verbose > 0: 
        print(model.summary(90))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate the model
    scores = model.evaluate(X_test, Y_test)
    
    if verbose > 0:
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores[1]


run_learning_on_window_sequences(blomapped_windows, consensus_structures, 21, verbose=2)

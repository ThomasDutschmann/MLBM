
'''
Script to play around with neural networks.
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, TimeDistributed, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random
import sys


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


def get_shape(satz):
    counter = 1
    current = satz
    print('(', end='')
    while True:
        if hasattr(current, '__len__'):
            if counter != 1:
                print(', ', end='')
            print(str(len(current)), end='')
            current = current[0]
            counter += 1
        else:
            print(')')
            print('First entry in last dimension:', str(current))
            break


def run_learning_on_window_sequences(points, labels, length, verbose=1, test_fraction=0.2, epochs=150, batch_size=100, network='dutsch'):
    # Function to automatically run an RNN training
    print('Selecting train and test set.')

    X = X = np.array([[point] for point in points])
    Y = np.array(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=666)
    
    if verbose > 0:
        print('Input shape:')
        get_shape(X)
        print('Output shape:')
        get_shape(Y)

    # create model
    model = Sequential()

    if network == 'dutsch':
        model.add(LSTM(units=100, input_shape=(1, 5 * length)))
        model.add(Dense(3, activation='sigmoid'))
    if network == 'heumos':
        model.add(GRU(256, batch_input_shape=(None, None, 640), return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(128, batch_input_shape=(None, None, 640), return_sequences=False, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(384, activation="softmax"))

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


def main():
    epochs = 150
    batch_size = 1024
    network_type = 'dutsch'
    if len(sys.argv) > 0:
        epochs = int(sys.argv[1])
        if len(sys.argv) > 1:
            batch_size = int(sys.argv[2])
            if len(sys.argv) > 2:
                network_type = sys.argv[3]
    print('Reading input file.')
    file_based_windows, file_based_structures = read_window_file('windows_and_structures.csv')
    run_learning_on_window_sequences(file_based_windows, file_based_structures, 21, verbose=2, batch_size=1024, epochs=25, network=network_type)


if __name__ == '__main__':
    main()


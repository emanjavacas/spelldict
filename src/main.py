# encoding: utf-8

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from utils import get_data

if __name__ == '__main__':
    N_SENTS = 10000
    RANDOM_STATE = 1001
    BATCH_SIZE = 25
    N_TARGETS = 5000
    N_FILTERS = 2000
    FILTER_LENGTH = 3

    # load data
    X, y, word_indexer, char_indexer = get_data(N_SENTS, N_TARGETS)
    X = X.transpose(0, 2, 1)
    y = np_utils.to_categorical(y, word_indexer.max)
    print("finished loading data...")

    # train-test split
    max_train = 8 * len(X) / 10
    X_train, y_train = X[:max_train], y[:max_train]

    # model
    model = Sequential()

    # convolutions
    model.add(Convolution1D(
        input_dim=char_indexer.max,  # vector size
        nb_filter=N_FILTERS,
        filter_length=FILTER_LENGTH,
        activation="relu",
        border_mode="valid",    # no padding
        subsample_length=1
    ))

    model.add(MaxPooling1D(pool_length=2))

    # model.add(LSTM((N_FILTERS/2, )))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    model.fit(X_train, y_train, validation_split=0.2,
              batch_size=BATCH_SIZE, nb_epoch=25,
              show_accuracy=True, verbose=1)

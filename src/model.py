# encoding: utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def model(EMBEDDING_INPUT, EMBEDDING_DIM, N_FILTERS, FILTER_LENGTH, N_CLASSES):
    m = Sequential()

    # embeddings
    if EMBEDDING_DIM:
        conv_input = EMBEDDING_DIM
        model.add(Embedding(EMBEDDING_INPUT, conv_input))
    else:
        conv_input = EMBEDDING_INPUT

    # convolutions
    model.add(Convolution1D(
        input_dim=conv_input,
        nb_filter=N_FILTERS,
        filter_length=FILTER_LENGTH,
        activation="relu",
        border_mode="valid",    # no padding
        subsample_length=1
    ))

    m.add(MaxPooling1D(pool_length=2))

    # LSTM
    m.add(LSTM(512, input_shape=(N_FILTERS/2, 1)))

    m.add(Dropout(0.5))
    m.add(Activation('relu'))

    m.add(Dense(N_CLASSES, input_dim=512))

    m.add(Activation('softmax'))

    m.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    return m

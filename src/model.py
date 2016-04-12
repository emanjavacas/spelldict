# encoding: utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def get_model(embedding_input, embedding_dim, n_filters, filter_length,
          n_classes, **kwargs):
    m = Sequential()

    # embeddings
    if embedding_dim:
        conv_input = embedding_dim
        model.add(embedding(embedding_input, conv_input))
    else:
        conv_input = embedding_input

    # convolutions
    m.add(Convolution1D(
        input_dim=conv_input,
        nb_filter=n_filters,
        filter_length=filter_length,
        activation="relu",
        border_mode="valid",    # no padding
        subsample_length=1
    ))

    m.add(MaxPooling1D(pool_length=2))

    # LSTM
    m.add(LSTM(512, input_shape=(n_filters/2, 1)))

    m.add(Dropout(0.5))
    m.add(Activation('relu'))

    m.add(Dense(n_classes, input_dim=512))
    m.add(Dropout(0.5))
    m.add(Activation('softmax'))

    m.summary()
    m.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    return m

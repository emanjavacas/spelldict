# encoding: utf-8

from theano.misc import pkl_utils

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from utils import get_data, save_model

if __name__ == '__main__':
    N_SENTS = 2000
    RANDOM_STATE = 1001
    BATCH_SIZE = 25
    NB_EPOCH = 1
    N_TARGETS = 50
    N_FILTERS = 2000
    FILTER_LENGTH = 3
    INPUT_TYPE = "one_hot"           # one_hot or input embedding dimension

    # load data
    X, y, word_idxr, char_idxr = \
        get_data('../data/post/', N_SENTS, N_TARGETS,
                 one_hot_encoding=True if INPUT_TYPE == "one_hot" else INPUT_TYPE)

    y = np_utils.to_categorical(y, word_idxr.vocab_len())
    print("finished loading data...")
    print("number of instances: [%d]" % len(y))
    print("input instance shape: (%d, %d)" % (X[0].shape))

    # train-test split
    max_train = 8 * len(X) / 10
    X_train, y_train = X[:max_train], y[:max_train]

    # model
    model = Sequential()

    # embeddings
    if INPUT_TYPE != "one_hot":
        model.add(Embedding(char_idxr.vocab_len(), INPUT_TYPE))

    # convolutions
    model.add(Convolution1D(
        input_dim=INPUT_TYPE if INPUT_TYPE != "one_hot" else char_idxr.vocab_len(),
        nb_filter=N_FILTERS,
        filter_length=FILTER_LENGTH,
        activation="relu",
        border_mode="valid",    # no padding
        subsample_length=1
    ))

    model.add(MaxPooling1D(pool_length=2))

    # LSTM
    model.add(LSTM(512, input_shape=(N_FILTERS/2, 1)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(word_idxr.vocab_len(), input_dim=512))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    model.fit(X_train, y_train, validation_split=0.2,
              batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
              show_accuracy=True, verbose=1)

    print('saving model...')
    save_model(model, word_idxr, char_idxr, 'model')


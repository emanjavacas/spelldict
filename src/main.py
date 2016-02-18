# encoding: utf-8

from __future__ import division
from canister.callback import DBCallback

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from utils import get_data, save_model

if __name__ == '__main__':
    params = {
        "N_SENTS": 200000,
        "RANDOM_STATE": 1001,
        "BATCH_SIZE": 50,
        "NB_EPOCH": 10,
        "TRAIN_TEST_SPLIT": 0.8,
        "N_TARGETS": 1000,
        "N_FILTERS": 2000,
        "FILTER_LENGTH": 3,
        "EMBEDDING_DIM": None,
        "INPUT_TYPE": "one_hot" # one_hot or embedding (requires EMBEDDING_DIM)
    }

    callback = DBCallback("CharConvLM", "tokenized", params)

    # load data
    encoding = "one_hot" if params["INPUT_TYPE"] == "one_hot" else params["EMBEDDING_DIM"]
    X, y, word_idxr, char_idxr = \
        get_data('../data/tokenized/', 
                 params["N_SENTS"], 
                 params["N_TARGETS"],
                 encoding=encoding)

    y = np_utils.to_categorical(y, word_idxr.vocab_len())
    print("finished loading data...")
    print("number of instances: [%d]" % len(y))
    print("input instance shape: (%d, %d)" % (X[0].shape))
    print("starting first iteration")

    # train-test split
    max_train = len(X) * params["TRAIN_TEST_SPLIT"]
    X_train, y_train = X[:max_train], y[:max_train]

    # model
    model = Sequential()

    # embeddings
    if params["EMBEDDING_DIM"]:
        conv_input = embedding_dim = params["EMBEDDING_DIM"]
        model.add(Embedding(char_idxr.vocab_len(), embedding_dim))
    else:
        conv_input = char_idxr.vocab_len()

    # convolutions
    model.add(Convolution1D(
        input_dim=conv_input,
        nb_filter=params["N_FILTERS"],
        filter_length=params["FILTER_LENGTH"],
        activation="relu",
        border_mode="valid",    # no padding
        subsample_length=1
    ))

    model.add(MaxPooling1D(pool_length=2))

    # LSTM
    model.add(LSTM(512, input_shape=(params["N_FILTERS"]/2, 1)))
    
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(word_idxr.vocab_len(), input_dim=512))
    
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    model.fit(X_train, y_train,
              validation_split=0.2,
              batch_size=params["BATCH_SIZE"], 
              nb_epoch=params["NB_EPOCH"],
              show_accuracy=True,
              verbose=1,
              callbacks=[remote])

    fname = '/home/manjavacas/code/python/spelldict/models/model'
    save_model(model, word_idxr, char_idxr, fname)

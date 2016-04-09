# encoding: utf-8

from __future__ import division
from canister.callback import DBCallback

from keras.utils import np_utils

from model import model
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
        # one_hot or embedding (requires EMBEDDING_DIM)
        "INPUT_TYPE": "one_hot"
    }

    # load data
    if params["INPUT_TYPE"] == "one_hot":
        encoding = "one_hot"
    else:
        encoding = params["EMBEDDING_DIM"]

    X, y, word_idxr, char_idxr = \
        get_data('../data/tokenized/',
                 params["N_SENTS"],
                 params["N_TARGETS"],
                 encoding=encoding)

    # add post-processing experiment params
    embedding_input = char_idxr.vocab_len()
    n_classes = word_idxr.vocab_len()
    params.update({"EMBEDDING_INPUT": embedding_input,
                   "N_CLASSES": n_classes})

    # canister callback
    callback = DBCallback("CharConvLM", "tokenized", params)

    y = np_utils.to_categorical(y, n_classes)
    print("finished loading data...")
    print("number of instances: [%d]" % len(y))
    print("input instance shape: (%d, %d)" % (X[0].shape))
    print("starting first iteration")

    # train-test split
    # TODO: add train mode params (on_batches, batch_size)
    max_train = len(X) * params["TRAIN_TEST_SPLIT"]
    X_train, y_train = X[:max_train], y[:max_train]

    # model
    m = model(**params)
    model.fit(X_train, y_train,
              validation_split=0.2,
              batch_size=params["BATCH_SIZE"],
              nb_epoch=params["NB_EPOCH"],
              show_accuracy=True,
              verbose=1,
              callbacks=[callback])

    fname = '/home/manjavacas/code/python/spelldict/models/model'
    save_model(model, word_idxr, char_idxr, fname)

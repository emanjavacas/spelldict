# encoding: utf-8

from __future__ import division
import argparse

# from canister.callback import DBCallback

from keras.utils import np_utils

from model import get_model
from utils import save_model, MiniBatchGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--n_sents', type=int, default=200000)
    parser.add_argument('-mb', '--mini_batch_size', type=int, default=128)
    parser.add_argument('-b', '--batch_size', type=int, default=100000)  
    parser.add_argument('-e', '--n_epoch', type=int, default=10)
    parser.add_argument('-t', '--n_targets', type=int, default=10000)
    parser.add_argument('-f', '--n_filters', type=int, default=3000)
    parser.add_argument('-l', '--filter_length', type=int, default=3)
    parser.add_argument('-n', '--encoding', default='one_hot',
                        help='One of "one_hot" or "embedding"' +
                        'the latter requires arg --embedding_dim')
    parser.add_argument('-d', '--embedding_dim', default=None)
    parser.add_argument('-r', '--random_state', type=int, default=1001)
    parser.add_argument('-bt', '--batch_training', default=True)

    params = vars(parser.parse_args())
    if params["encoding"] == 'embedding' and not params["embedding_dim"]:
        raise ValueError('embedding requires arg "embedding_dim"')

    # load data
    print("loading data...")
    if params["encoding"] == "one_hot":
        encoding = "one_hot"
    else:
        encoding = params["embedding_dim"]

    batches = MiniBatchGenerator('../data/post/',
                                 params["n_sents"],
                                 params["n_targets"],
                                 params["batch_size"],
                                 encoding)

    # add post-processing experiment params
    embedding_input = batches.char_indexer.vocab_len()
    n_classes = batches.word_indexer.vocab_len()
    params.update({"embedding_input": embedding_input, "n_classes": n_classes})

    X_test, y_test = batches.train_test_split()
    y_test = np_utils.to_categorical(y_test, n_classes)

    # model
    print("compiling model...")
    m = get_model(**params)

    print("learning...")
    if params["batch_training"]:
        for e in range(params["n_epoch"]):
            print("epoch number: %d" % e)
            for X_batch, y_batch in batches:
                
                y_batch = np_utils.to_categorical(y_batch, n_classes)
                m.fit(X_batch, y_batch, batch_size=params["mini_batch_size"],
                      nb_epoch=1, verbose=1, validation_data=(X_test, y_test))
                # loss = m.train_on_batch(X_batch, y_batch)
                # print("loss: [%d]" % loss[0])

    else:
        # canister callback
        # callback = DBCallback("CharConvLM", "tokenized", params)
        
        X_train = batches.X
        y_train = np_utils.to_categorical(batches.y, n_classes)
        m.fit(X_train, y_train,
              validation_split=0.05,
              batch_size=params["batch_size"],
              nb_epoch=params["n_epoch"],
              show_accuracy=True,
              verbose=1# ,
              # callbacks=[callback]
        )

    print("saving model params...")
    fname = '/home/manjavacas/code/python/spelldict/models/model'
    save_model(m, batches.word_indexer, batches.char_indexer, fname)

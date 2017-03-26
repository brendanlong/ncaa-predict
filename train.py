#!/usr/bin/env python3
import argparse
import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten

from ncaa_predict.data_loader import load_data_multiyear, \
    N_PLAYERS, PLAYER_FEATURE_COLUMNS
from ncaa_predict.util import list_arg


DEFAULT_BATCH_SIZE = 1000
DEFAULT_N_THREADS = 16
DEFAULT_STEPS = sys.maxsize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", "-b", default=DEFAULT_BATCH_SIZE, type=int,
        help="The training batch size. Smaller numbers will train faster but "
        "may not converge. (default: %(default)s)")
    parser.add_argument(
        "--model-out", "-o", default=None,
        help="Folder to save the model to. This folder must not exist, as "
        "tensorflow won't let us save over an old model. (default: don't "
        "save)")
    parser.add_argument(
        "--n-threads", "-j", default=DEFAULT_N_THREADS, type=int,
        help="Number of threads to use for some Pandas data-loading "
        "processes. (default: %(default)s)")
    parser.add_argument(
        "--steps", "-s", default=DEFAULT_STEPS, type=int,
        help="The maximum number of training steps. Note that you can stop "
        "training at any time and save the output with ctrl+c. (default: "
        "%(default)s)")
    parser.add_argument(
        "--train-years", "-y", default=list(range(2002, 2017)),
        type=list_arg(type=int, container=frozenset),
        help="A comma-separated list of years to train on.")
    args = parser.parse_args()

    model = Sequential([
        Flatten(input_shape=(2, N_PLAYERS, len(PLAYER_FEATURE_COLUMNS))),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(16, activation="relu"),
        Dense(2, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy", optimizer="adagrad",
        metrics=["accuracy"])

    features, labels = load_data_multiyear(
        args.train_years, n_threads=args.n_threads)
    try:
        model.fit(
            x=features, y=labels,
            batch_size=args.batch_size, epochs=args.steps // args.batch_size,
            shuffle=True, validation_split=0.1)
    except KeyboardInterrupt:
        print("Stopped training due to keyboard interrupt")
    if args.model_out is not None:
        model.save(args.model_out)

    # Workaround for TensorFlow bug:
    # https://github.com/tensorflow/tensorflow/issues/3388
    import gc
    gc.collect()

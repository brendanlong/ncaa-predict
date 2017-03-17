#!/usr/bin/env python3
import argparse

from ncaa_predict.estimator import *
from ncaa_predict.model import ModelType
from ncaa_predict.util import list_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", "-b", default=DEFAULT_BATCH_SIZE, type=int,
        help="The training batch size. Smaller numbers will train faster but "
        "may not converge. (default: %(default)s)")
    parser.add_argument(
        "--hidden-units", "-u", default=DEFAULT_HIDDEN_UNITS,
        type=list_arg(type=int),
        help="A comma seperated list of hidden units in each DNN layer.")
    parser.add_argument(
        "--model-out", "-o", default=None,
        help="Folder to save the model to. This folder must not exist, as "
        "tensorflow won't let us save over an old model. (default: don't "
        "save)")
    parser.add_argument(
        "--model-type", "-t", default=ModelType.dnn_classifier,
        type=ModelType, choices=list(ModelType))
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
        "--test-year", "-yt", default=2016, type=int,
        help="The year to use for the validation set.")
    parser.add_argument(
        "--train-years", "-y", default=list(range(2002, 2016)),
        type=list_arg(type=int, container=frozenset),
        help="A comma-separated list of years to train on.")
    args = parser.parse_args()

    # With verbose logging, we get training feedback every 100 steps
    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = Estimator(
        args.model_type, hidden_units=args.hidden_units,
        n_threads=args.n_threads, feature_year=args.test_year)
    estimator.train(
        args.train_years, batch_size=args.batch_size, steps=args.steps,)
    if args.model_out is not None:
        estimator.save(args.model_out)
    estimator.evaluate(args.test_year)

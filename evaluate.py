#!/usr/bin/env python3
import argparse

from ncaa_predict.estimator import *
from ncaa_predict.model import ModelType
from ncaa_predict.util import list_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-units", "-u", default=DEFAULT_HIDDEN_UNITS,
        type=list_arg(type=int),
        help="A comma seperated list of hidden units in each DNN layer.")
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument(
        "--model-type", "-t", default=ModelType.dnn_classifier,
        type=ModelType, choices=list(ModelType))
    parser.add_argument(
        "--n-threads", "-j", default=DEFAULT_N_THREADS, type=int,
        help="Number of threads to use for some Pandas data-loading "
        "processes. (default: %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_const", const=True)
    parser.add_argument("--year", "-y", default=2016, type=int)
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    estimator = Estimator(
        args.model_type, hidden_units=args.hidden_units,
        model_in=args.model_in, n_threads=args.n_threads,
        feature_year=args.year)
    print(estimator.evaluate(args.year))

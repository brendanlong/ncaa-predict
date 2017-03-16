#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

from constants import DNN_HIDDEN_UNITS
from data_loader import load_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", default=1000, type=int)
    parser.add_argument("--steps", "-s", default=10000, type=int)
    parser.add_argument("--n-threads", "-j", default=16, type=int)
    parser.add_argument("--model-out", "-o", default=None)
    parser.add_argument("--model-in", "-i", default=None)
    parser.add_argument("--predict-year", "-p", default=2016, type=int)
    parser.add_argument(
        "--train-years", "-y", default=list(range(2002, 2017)),
        type=lambda v: list(map(int, v.split(","))))
    parser.add_argument("--verbose", "-v", action="store_const", const=True)
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    data = [load_data(year, args.n_threads) for year in args.train_years]
    features = np.vstack([features for features, _ in data])
    labels = np.vstack([labels for _, labels in data])
    assert len(features) == len(labels)

    feature_cols = \
        tf.contrib.learn.infer_real_valued_columns_from_input(features)

    estimator = tf.contrib.learn.DNNClassifier(
        hidden_units=DNN_HIDDEN_UNITS,
        model_dir=args.model_in, feature_columns=feature_cols)
    estimator.fit(
        x=features, y=labels, steps=args.steps, batch_size=args.batch_size)
    if args.model_out:
        estimator.export(export_dir=args.model_out)

    test_features, test_labels = load_data(args.predict_year)
    print(estimator.evaluate(x=test_features, y=test_labels))

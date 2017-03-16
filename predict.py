#!/usr/bin/env python3
import argparse
import tensorflow as tf

from data_loader import load_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", default="model")
    parser.add_argument("--predict-year", "-y", default=2016, type=int)
    parser.add_argument("--verbose", "-v", action="store_const", const=True)
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    features, labels = load_data(args.predict_year)
    feature_cols = \
        tf.contrib.learn.infer_real_valued_columns_from_input(features)
    estimator = tf.contrib.learn.LinearClassifier(
        model_dir=args.model_in, feature_columns=feature_cols)

    print(estimator.evaluate(x=features, y=labels))

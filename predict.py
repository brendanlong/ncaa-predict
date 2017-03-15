#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf


def read_data_csv(path):
    dtypes = {
        "year": np.int,
        "team_id": np.int,
        "opponent_id": np.int,
        "team_score": np.int,
        "opponent_score": np.int
    }
    this_dir = os.path.dirname(__file__)
    path = os.path.join(this_dir, "data", path)
    df = pd.read_csv(path, usecols=list(dtypes))
    df = df[~df.isnull().any(axis=1)]
    return df.apply(pd.to_numeric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_const", const=True)
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    dfs = [read_data_csv("ncaa/csv/ncaa_games_%s.csv" % year)
           for year in range(2002, 2017)]
    df = pd.concat(dfs)

    feature_cols = ["year", "team_id", "opponent_id"]

    features = tf.contrib.learn.extract_pandas_data(df[feature_cols])
    labels = tf.contrib.learn.extract_pandas_labels(
        df["team_score"] > df["opponent_score"])

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        features)
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns)
    train = df["year"] != 2016
    estimator.fit(
        x=features[train], y=labels[train], steps=10000, batch_size=100)
    print(estimator.evaluate(x=features[~train], y=labels[~train]))

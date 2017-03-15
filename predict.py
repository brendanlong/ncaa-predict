#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf


def load_csv(path, columns):
    this_dir = os.path.dirname(__file__)
    path = os.path.join(this_dir, path)
    df = pd.read_csv(path, usecols=list(columns))
    df = df[~df.isnull().any(axis=1)]
    return df.apply(pd.to_numeric)


def load_ncaa_games(year):
    columns = {
        "year": np.int,
        "school_id": np.int,
        "opponent_id": np.int,
        "score": np.int,
        "opponent_score": np.int
    }
    path = "csv/ncaa_games_%s.csv" % year
    return load_csv(path, columns)


def load_ncaa_players(year):
    columns = {
        "school_id": np.int,
        "player_id": np.int,
        "points_avg": np.int
    }
    path = "csv/ncaa_players_%s.csv" % year
    return load_csv(path, columns)


def load_data(year):
    games = load_ncaa_games(year)
    players = load_ncaa_players(year)
    return pd.merge(games, players, how="inner", on="school_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-year", "-p", default=2016, type=int)
    parser.add_argument(
        "--train-years", "-y", default=list(range(2002, 2017)),
        type=lambda v: list(map(int, v.split(","))))
    parser.add_argument("--verbose", "-v", action="store_const", const=True)
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    dfs = [load_data(year) for year in args.train_years + [args.predict_year]]
    df = pd.concat(dfs)

    feature_cols = ["year", "school_id", "opponent_id"]

    features = tf.contrib.learn.extract_pandas_data(df[feature_cols])
    labels = tf.contrib.learn.extract_pandas_labels(
        df["score"] > df["opponent_score"])

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        features)
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns)
    train = df["year"] != args.predict_year
    estimator.fit(
        x=features[train], y=labels[train], steps=10000, batch_size=100)
    print(estimator.evaluate(x=features[~train], y=labels[~train]))

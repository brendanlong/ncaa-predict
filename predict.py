#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


def load_csv(path, columns, header=None):
    this_dir = os.path.dirname(__file__)
    path = os.path.join(this_dir, "data", path)
    df = pd.read_csv(
        path, usecols=list(columns), names=header,
        header=None if header is not None else 'infer')
    df = df[~df.isnull().any(axis=1)]
    return df.apply(pd.to_numeric)


def load_ncaa_games(year):
    columns = {
        "year": np.int,
        "team_id": np.int,
        "opponent_id": np.int,
        "team_score": np.int,
        "opponent_score": np.int
    }
    path = "ncaa/csv/ncaa_games_%s.csv" % year
    return load_csv(path, columns)


def load_ncaa_players(year):
    columns = {
        "team_id": np.int,
        "player_id": np.int,
        "points_avg": np.int
    }
    header = [
        "team_name", "team_id", "year", "name", "player_id", "class", "season",
        "pos", "height", "g", "fg_made", "fg_atts", "fg_pct", "3pt_made",
        "3pt_atts", "3pt_pct", "ft_made", "ft_atts", "ft_pct", "rebound_num",
        "rebound_avg", "assist_num", "assist_avg", "blocks_num", "blocks_avg",
        "steals_num", "steals_avg", "points_num", "points_avg", "turnovers",
        "dd", "td"]
    path = "ncaa/csv/ncaa_games_%s.csv" % year
    return load_csv(path, columns, header)


def load_data(year):
    games = load_ncaa_games(year)
    players = load_ncaa_players(year)
    return pd.merge(games, players, how="inner", on="team_id")


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

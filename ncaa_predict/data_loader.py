import functools
import multiprocessing
import os

import keras
import numpy as np
import pandas as pd


# All teams need to be the same size, so we pad them to this size
# or reduce to this size
N_PLAYERS = 10

PLAYER_FEATURE_COLUMNS = [
    # g = games
    "g", "height", "fg_percent", "3pt_percent", "freethrows_percent",
    "points_avg", "rebounds_avg", "assists_avg", "blocks_avg",
    "steals_avg"]

THIS_DIR = os.path.dirname(__file__)


def load_csv(path, columns, to_numeric=True):
    path = os.path.join(THIS_DIR, "..", path)
    df = pd.read_csv(path, usecols=list(columns))
    if to_numeric:
        df = df.apply(pd.to_numeric)
    return df


def load_ncaa_games(year):
    columns = ["year", "school_id", "opponent_id", "score", "opponent_score"]
    path = "csv/ncaa_games_%s.csv" % year
    return load_csv(path, columns)


def load_ncaa_players(year):
    columns = PLAYER_FEATURE_COLUMNS + ["school_id"]
    path = "csv/ncaa_players_%s.csv" % year
    players = load_csv(path, columns)
    players = players.fillna(0)
    players = players.sort_values("g", ascending=False).groupby("school_id")
    return players


def load_ncaa_schools():
    path = "csv/ncaa_schools.csv"
    return load_csv(path, ["school_id", "school_name"], to_numeric=False)


def get_players_for_team(players, school_id):
    try:
        team = players.get_group(school_id)
    except KeyError:
        return None

    team = team.as_matrix(columns=PLAYER_FEATURE_COLUMNS)
    if len(team) > N_PLAYERS:
        team = team[:N_PLAYERS]

    missing_players = N_PLAYERS - len(team)
    if missing_players > 0:
        missing_player = [0] * team.shape[1]
        team = np.vstack([team] + [missing_player] * missing_players)
    return team


def load_game(players, p):
    i, game = p
    this_team = get_players_for_team(players, game["school_id"])
    other_team = get_players_for_team(players, game["opponent_id"])
    if i % 1000 == 0:
        print("Handled row %s" % i)

    if this_team is None or other_team is None:
        return None, None
    teams = [this_team, other_team]
    label = [game["score"] > game["opponent_score"]]
    return np.stack(teams), label


def load_data(year):
    features_path = os.path.join(
        THIS_DIR, "../data_cache/features_%s.npy" % year)
    labels_path = os.path.join(
        THIS_DIR, "../data_cache/labels_%s.npy" % year)
    if not os.path.exists(features_path) \
            or not os.path.exists(labels_path):
        games = load_ncaa_games(year)
        players = load_ncaa_players(year)
        len_rows = games.shape[0]
        print("Iterating through %s games" % len_rows)
        f = functools.partial(load_game, players)
        with multiprocessing.Pool(64) as pool:
            res = pool.map(f, games.iterrows())
        features = [feature for feature, _ in res if feature is not None]
        labels = [label for _, label in res if label is not None]
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int8)
        labels = keras.utils.to_categorical(labels, num_classes=2)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        np.save(features_path, features)
        np.save(labels_path, labels)
    features, labels = np.load(features_path), np.load(labels_path)
    return features, labels


def load_data_multiyear(years):
    data = [load_data(year)
            for year in years]
    features = np.vstack([features for features, _ in data])
    labels = np.vstack([labels for _, labels in data])
    assert len(features) == len(labels)
    return features, labels

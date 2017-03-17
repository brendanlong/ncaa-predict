import functools
import multiprocessing
import os

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
    players[players["height"].isnull()] = players["height"].mean()
    players.fillna(0)
    return players


def load_ncaa_schools():
    path = "csv/ncaa_schools.csv"
    return load_csv(path, ["school_id", "school_name"], to_numeric=False)


def get_players_for_team(players, school_id):
    team = players[players["school_id"] == school_id]
    team = team.sort_values("g", ascending=False)
    team = team.as_matrix(columns=PLAYER_FEATURE_COLUMNS)
    if len(team) == 0:
        return None
    elif len(team) > N_PLAYERS:
        team = team[:N_PLAYERS]
    elif len(team) < N_PLAYERS:
        n = N_PLAYERS - len(team)
        i = np.random.choice(team.shape[0], size=n)
        team = np.vstack([team, team[i]])
    return team


def load_game(players, predict_score, p):
    i, game = p
    this_team = get_players_for_team(players, game["school_id"])
    other_team = get_players_for_team(players, game["opponent_id"])

    if this_team is None or other_team is None:
        return None, None
    teams = [this_team, other_team]
    if i % 1000 == 0:
        print("Handled row %s" % i)
    if predict_score:
        label = [game["score"] + game["opponent_score"]]
    else:
        label = [game["score"] > game["opponent_score"]]
    return np.stack(teams), label


def load_data(year, n_threads=16, predict_score=False):
    suffix = "_score" if predict_score else ""
    features_path = os.path.join(
        THIS_DIR, "../data_cache%s/features_%s.npy" % (suffix, year))
    labels_path = os.path.join(
        THIS_DIR, "../data_cache%s/labels_%s.npy" % (suffix, year))
    if not os.path.exists(features_path) \
            or not os.path.exists(labels_path):
        games = load_ncaa_games(year)
        players = load_ncaa_players(year)
        len_rows = games.shape[0]
        print("Iterating through %s games" % len_rows)
        f = functools.partial(load_game, players, predict_score)
        with multiprocessing.Pool(n_threads) as pool:
            res = pool.map(f, games.iterrows())
        features = [feature for feature, _ in res if feature is not None]
        labels = [label for _, label in res if label is not None]
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int8)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        np.save(features_path, features)
        np.save(labels_path, labels)
    return np.load(features_path), np.load(labels_path)


def load_data_multiyear(years, n_threads):
    data = [load_data(year, n_threads)
            for year in years]
    features = np.vstack([features for features, _ in data])
    labels = np.vstack([labels for _, labels in data])
    assert len(features) == len(labels)
    return features, labels
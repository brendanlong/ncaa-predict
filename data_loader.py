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


def load_csv(path, columns):
    this_dir = os.path.dirname(__file__)
    path = os.path.join(this_dir, path)
    df = pd.read_csv(path, usecols=list(columns))
    return df.apply(pd.to_numeric)


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


def load_game(players, p):
    i, game = p
    this_team = get_players_for_team(players, game["school_id"])
    other_team = get_players_for_team(players, game["opponent_id"])

    if this_team is None or other_team is None:
        return None, None
    teams = [this_team, other_team]
    if i % 1000 == 0:
        print("Handled row %s" % i)
    win = game["score"] > game["opponent_score"]
    return np.stack(teams), [win]


def load_data(year, n_threads=16):
    features_path = "data/features_%s.npy" % year
    labels_path = "data/labels_%s.npy" % year
    if not os.path.exists(features_path) \
            or not os.path.exists(labels_path):
        games = load_ncaa_games(year)
        players = load_ncaa_players(year)
        len_rows = games.shape[0]
        print("Iterating through %s games" % len_rows)
        f = functools.partial(load_game, players)
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

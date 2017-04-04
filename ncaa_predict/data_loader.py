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
N_FEATURES = len(PLAYER_FEATURE_COLUMNS)

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
    return load_csv(path, columns) \
        .dropna()


def load_ncaa_players(year):
    columns = PLAYER_FEATURE_COLUMNS + ["school_id"]
    path = "csv/ncaa_players_%s.csv" % year
    players = load_csv(path, columns)
    players = players.fillna(0)  # N/A games presumably means 0
    players = players.sort_values("g", ascending=False).groupby("school_id")
    return players


def load_ncaa_schools():
    path = "csv/ncaa_schools.csv"
    return load_csv(path, ["school_id", "school_name"], to_numeric=False)


def _setup_players(team):
    team = team.as_matrix(columns=PLAYER_FEATURE_COLUMNS)
    if len(team) > N_PLAYERS:
        team = team[:N_PLAYERS]

    missing_players = N_PLAYERS - len(team)
    if missing_players > 0:
        missing_player = [0] * team.shape[1]
        team = np.vstack([team] + [missing_player] * missing_players)
    return team


def get_players_for_team(players, school_id):
    try:
        team = players.get_group(school_id)
    except KeyError:
        return None

    return _setup_players(team)


def load_data(year):
    print("Loading data for %s" % year)
    games = load_ncaa_games(year)
    players = load_ncaa_players(year)
    print("Setting up teams")
    teams = {school_id: _setup_players(team) for school_id, team in players}
    num_games = games.shape[0]
    print("Getting valid games from %s games" % num_games)
    games = [game for game in games.itertuples()
             if game.school_id in teams and game.opponent_id in teams]
    num_games = len(games)
    print("Setting up %s games" % num_games)
    features = np.empty(shape=[num_games, 2, N_PLAYERS, N_FEATURES],
                        dtype=np.float32)
    labels = np.empty(shape=[num_games, 2], dtype=np.int8)
    for i, game in enumerate(games):
        this_team = teams[game.school_id]
        other_team = teams[game.opponent_id]
        features[i] = [this_team, other_team]
        labels[i] = [1, 0] if game.score > game.opponent_score else [0, 1]
    assert i == num_games - 1
    return features, labels


def load_data_multiyear(years):
    data = [load_data(year)
            for year in years]
    features = np.vstack([features for features, _ in data])
    labels = np.vstack([labels for _, labels in data])
    assert len(features) == len(labels)
    return features, labels

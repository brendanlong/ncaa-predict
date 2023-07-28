from enum import Enum, unique
import multiprocessing
import os

import keras
import numpy as np
import pandas as pd


# All teams need to be the same size, so we pad them to this size
# or reduce to this size
N_PLAYERS = 10


@unique
class Class(Enum):
    FRESHMAN = (1, 0, 0, 0, 0)
    JUNIOR = (0, 1, 0, 0, 0)
    SOPHOMORE = (0, 0, 1, 0, 0)
    SENIOR = (0, 0, 0, 1, 0)
    UNKNOWN = (0, 0, 0, 0, 1)

    @staticmethod
    def from_col(col):
        if col == "Fr.":
            return Class.FRESHMAN
        elif col == "Jr.":
            return Class.JUNIOR
        elif col == "So.":
            return Class.SOPHOMORE
        elif col == "Sr.":
            return Class.SENIOR
        elif col == "---":
            return Class.UNKNOWN
        else:
            raise NotImplementedError("%s is not a known Class" % col)


@unique
class Position(Enum):
    NONE = (1, 0, 0, 0)
    GUARD = (0, 1, 0, 0)
    FORWARD = (0, 0, 1, 0)
    CENTER = (0, 0, 0, 1)

    @staticmethod
    def from_col(col):
        if col in ("G", "Guard"):
            return Position.GUARD
        elif col in ("F", "Forward"):
            return Position.FORWARD
        elif col == "C":
            return Position.CENTER
        elif col is np.nan:
            return Position.NONE
        else:
            raise NotImplementedError("%s is not a known Position" % col)


PLAYER_FLOAT_COLUMNS = [
    # g = games
    "g",
    "height",
    "fg_made",
    "fg_attempts",
    "fg_percent",
    "3pt_made",
    "3pt_attempts",
    "3pt_percent",
    "freethrows_made",
    "freethrows_attempts",
    "freethrows_percent",
    "rebounds_num",
    "rebounds_avg",
    "assists_num",
    "assists_avg",
    "blocks_num",
    "blocks_avg",
    "steals_num",
    "steals_avg",
    "points_num",
    "points_avg",
    "turnovers",
    "dd",
    "td",
]
PLAYER_CATEGORICAL_COLUMNS = ["position", "class"]
PLAYER_FEATURE_COLUMNS = PLAYER_FLOAT_COLUMNS + PLAYER_CATEGORICAL_COLUMNS
N_FEATURES = len(PLAYER_FLOAT_COLUMNS) + len(Position) + len(Class)


THIS_DIR = os.path.dirname(__file__)


def load_csv(path, columns):
    path = os.path.join(THIS_DIR, "..", path)
    df = pd.read_csv(path, usecols=list(columns))
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def load_ncaa_games(year):
    columns = ["year", "school_id", "opponent_id", "score", "opponent_score"]
    path = "csv/ncaa_games_%s.csv" % year
    return load_csv(path, columns).dropna()


def load_ncaa_players(year):
    columns = PLAYER_FEATURE_COLUMNS + ["school_id"]
    path = "csv/ncaa_players_%s.csv" % year
    players = load_csv(path, columns)
    players["position"] = players["position"].apply(Position.from_col)
    players["class"] = players["class"].apply(Class.from_col)
    players = players.fillna(0)  # N/A games presumably means 0
    players = players.sort_values("g", ascending=False).groupby("school_id")
    return players


def load_ncaa_schools():
    path = "csv/ncaa_schools.csv"
    return load_csv(path, ["school_id", "school_name"])


def _setup_players(team):
    team = np.hstack(
        [
            team[PLAYER_FLOAT_COLUMNS].values,
            [p.value for p in team["position"].values],
            [c.value for c in team["class"].values],
        ]
    )
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
    teams = {school_id: _setup_players(team) for school_id, team in players}
    print("Loaded %s teams" % len(teams))

    games = [
        game
        for game in games.itertuples()
        if game.school_id in teams and game.opponent_id in teams
    ]
    num_games = len(games)
    features = np.empty(shape=[num_games, 2, N_PLAYERS, N_FEATURES], dtype=np.float32)
    labels = np.empty(shape=[num_games, 2], dtype=np.int8)
    for i, game in enumerate(games):
        this_team = teams[game.school_id]
        other_team = teams[game.opponent_id]
        features[i] = [this_team, other_team]
        labels[i] = [1, 0] if game.score > game.opponent_score else [0, 1]
    print("Loaded %s games" % num_games)
    assert i == num_games - 1
    return features, labels


def load_data_multiyear(years):
    with multiprocessing.Pool() as p:
        data = p.map(load_data, years)
    features = np.vstack([features for features, _ in data])
    labels = np.vstack([labels for _, labels in data])
    assert len(features) == len(labels)
    return features, labels

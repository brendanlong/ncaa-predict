#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

from constants import DNN_HIDDEN_UNITS
from data_loader import load_ncaa_players, load_ncaa_schools, \
    get_players_for_team


BRACKET = (
    (
        (
            # East
            (
                (
                    ("Villanova", "Mt. St. Mary's"),
                    ("Wisconsin", "Virginia Tech"),
                ), (
                    ("Virginia", "UNCW"),
                    ("Florida", "ETSU"),
                ),
            ), (
                (
                    ("SMU", "Southern California"),
                    ("Baylor", "New Mexico St."),
                ), (
                    ("South Carolina", "Marquette"),
                    ("Duke", "Troy"),
                ),
            ),
        ), (
            # West
            (
                (
                    ("Gonzaga", "South Dakota St."),
                    ("Northwestern", "Vanderbilt"),
                ), (
                    ("Notre Dame", "Princeton"),
                    ("West Virginia", "Bucknell"),
                ),
            ), (
                (
                    ("Maryland", "Xavier"),
                    ("Florida St.", "FGCU"),
                ), (
                    ("Saint Mary's (CA)", "VCU"),
                    ("Arizona", "North Dakota"),
                ),
            ),
        ),
    ),
    (
        (
            # Midwest
            (
                (
                    ("Kansas", "UC Davis"),
                    ("Miami (FL)", "Michigan St."),
                ), (
                    ("Iowa St.", "Nevada"),
                    ("Purdue", "Vermont"),
                ),
            ), (
                (
                    ("Creighton", "Rhode Island"),
                    ("Oregon", "Iona"),
                ), (
                    ("Michigan", "Oklahoma St."),
                    ("Louisville", "Jacksonville St."),
                ),
            ),
        ), (
            # South
            (
                (
                    ("North Carolina", "Texas Southern"),
                    ("Arkansas", "Seton Hall"),
                ), (
                    ("Minnesota", "Middle Tenn."),
                    ("Butler", "Winthrop"),
                ),
            ), (
                (
                    ("Cincinnati", "Kansas St."),
                    ("UCLA", "Kent St."),
                ), (
                    ("Dayton", "Wichita St."),
                    ("Kentucky", "Northern Ky."),
                ),
            ),
        ),
    ),
)


def team_id_to_name(id, all_teams):
    return all_teams[all_teams["school_id"] == id]["school_name"].values[0]


def team_name_to_id(name, all_teams):
    try:
        return \
            all_teams[all_teams["school_name"] == name]["school_id"].values[0]
    except IndexError:
        raise Exception("Couldn't find ID for school [%s]" % name)


def predict(estimator, all_teams, all_players, bracket):
    team_a, team_b = bracket
    if isinstance(team_a, tuple):
        team_a = predict(estimator, all_teams, all_players, team_a)
    if isinstance(team_b, tuple):
        team_b = predict(estimator, all_teams, all_players, team_b)
    teams = [team_a, team_b]
    team_ids = [team_name_to_id(name, all_teams) for name in teams]
    players_a = get_players_for_team(all_players, team_ids[0])
    players_b = get_players_for_team(all_players, team_ids[1])
    x = np.array([np.stack([players_a, players_b])])
    # classifier tells us 1 if team_a wins, 2 if team_b wins
    c = next(estimator.predict(x=x))
    winner = teams[not c]
    print("%s vs %s: %s wins" % (team_a, team_b, winner))
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", default="model")
    parser.add_argument("--year", "-y", default=2017, type=int)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.ERROR)

    players = load_ncaa_players(args.year)
    all_teams = load_ncaa_schools()
    example_team = get_players_for_team(players, 697)
    features = np.array([np.stack([example_team, example_team])])
    feature_cols = \
        tf.contrib.learn.infer_real_valued_columns_from_input(features)
    estimator = tf.contrib.learn.DNNClassifier(
        hidden_units=DNN_HIDDEN_UNITS,
        model_dir=args.model_in, feature_columns=feature_cols)

    predict(estimator, all_teams, players, BRACKET)

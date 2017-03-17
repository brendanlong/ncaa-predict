#!/usr/bin/env python3
import argparse

import numpy as np

from ncaa_predict.data_loader import load_ncaa_players, load_ncaa_schools, \
    get_players_for_team
from ncaa_predict.estimator import *
from ncaa_predict.util import list_arg, team_name_to_id


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


def predict(estimator, all_teams, all_players, bracket, wait=False):
    team_a, team_b = bracket
    if isinstance(team_a, tuple):
        team_a = predict(estimator, all_teams, all_players, team_a, wait)
    if isinstance(team_b, tuple):
        team_b = predict(estimator, all_teams, all_players, team_b, wait)
    teams = [team_a, team_b]
    team_ids = [team_name_to_id(name, all_teams) for name in teams]
    players_a = get_players_for_team(all_players, team_ids[0])
    players_b = get_players_for_team(all_players, team_ids[1])
    x = np.array([np.stack([players_a, players_b])])
    # classifier tells us 1 if team_a wins, 2 if team_b wins
    c = estimator.predict(x=x)
    winner = teams[not c]
    print("%s vs %s: %s wins" % (team_a, team_b, winner))
    if wait:
        input()
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-units", "-u", default=DEFAULT_HIDDEN_UNITS,
        type=list_arg(type=int),
        help="A comma seperated list of hidden units in each DNN layer.")
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument(
        "--model-type", "-t", default=ModelType.dnn_classifier,
        type=ModelType, choices=list(ModelType))
    parser.add_argument(
        "--n-threads", "-j", default=DEFAULT_N_THREADS, type=int,
        help="Number of threads to use for some Pandas data-loading "
        "processes. (default: %(default)s)")
    parser.add_argument("--year", "-y", default=2017, type=int)
    parser.add_argument(
        "--wait", "-w", default=False, action="store_const", const=True)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.ERROR)

    players = load_ncaa_players(args.year)
    all_teams = load_ncaa_schools()

    estimator = Estimator(
        args.model_type, hidden_units=args.hidden_units,
        model_in=args.model_in, n_threads=args.n_threads,
        feature_year=args.year)

    predict(estimator, all_teams, players, BRACKET, args.wait)

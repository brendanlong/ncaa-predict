#!/usr/bin/env python3
import argparse

import keras
import numpy as np

from ncaa_predict.data_loader import load_ncaa_players, load_ncaa_schools, \
    get_players_for_team
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


def predict(model, all_teams, all_players, bracket, wait=False):
    team_a, team_b = bracket
    if isinstance(team_a, tuple):
        team_a = predict(model, all_teams, all_players, team_a, wait)
    if isinstance(team_b, tuple):
        team_b = predict(model, all_teams, all_players, team_b, wait)
    teams = [team_a, team_b]
    team_ids = [team_name_to_id(name, all_teams) for name in teams]
    players_a = get_players_for_team(all_players, team_ids[0])
    players_b = get_players_for_team(all_players, team_ids[1])
    x = np.array([np.stack([players_a, players_b])])
    a_wins, b_wins = model.predict(x=x)[0]
    if a_wins > b_wins:
        winner = team_a
    else:
        winner = team_b
    print("%s vs %s: %s wins (p=%.2f)" % (team_a, team_b, winner, max(a_wins, b_wins)))
    if wait:
        input()
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument("--year", "-y", default=2017, type=int)
    parser.add_argument(
        "--wait", "-w", default=False, action="store_const", const=True)
    args = parser.parse_args()

    players = load_ncaa_players(args.year)
    all_teams = load_ncaa_schools()

    model = keras.models.load_model(args.model_in)
    predict(model, all_teams, players, BRACKET, args.wait)

    # Workaround for TensorFlow bug:
    # https://github.com/tensorflow/tensorflow/issues/3388
    import gc
    gc.collect()

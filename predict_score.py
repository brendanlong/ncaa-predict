#!/usr/bin/env python3
import argparse

import numpy as np

from ncaa_predict.data_loader import (
    load_ncaa_players,
    load_ncaa_schools,
    load_ncaa_games,
    get_players_for_team,
)
from ncaa_predict.estimator import *
from ncaa_predict.util import list_arg, team_name_to_id


def get_historical_score(team_id, all_games):
    games = all_games[all_games["school_id"] == team_id]
    normal_score = games["score"].mean()
    diffs = []
    for school_id in all_games["school_id"].unique():
        if school_id == team_id:
            continue
        g = all_games[all_games["school_id"] == school_id]
        normal = g["score"].mean()
        against_team = g[g["opponent_id"] == team_id]
        against_team_score = against_team["score"].mean()
        if np.isnan(against_team_score):
            continue
        diffs.append(normal - against_team_score)
    return normal_score, np.mean(diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument(
        "--hidden-units",
        "-u",
        default=DEFAULT_HIDDEN_UNITS,
        type=list_arg(type=int),
        help="A comma seperated list of hidden units in each DNN layer.",
    )
    parser.add_argument("--model-in", "-m")
    parser.add_argument(
        "--model-type",
        "-t",
        default=ModelType.dnn_classifier,
        type=ModelType,
        choices=list(ModelType),
    )
    parser.add_argument(
        "--n-threads",
        "-j",
        default=DEFAULT_N_THREADS,
        type=int,
        help="Number of threads to use for some Pandas data-loading "
        "processes. (default: %(default)s)",
    )
    parser.add_argument("--year", "-y", default=2017, type=int)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.ERROR)

    players = load_ncaa_players(args.year)
    all_teams = load_ncaa_schools()
    team_a_id = team_name_to_id(args.team_a, all_teams)
    team_b_id = team_name_to_id(args.team_b, all_teams)
    players_a = get_players_for_team(players, team_a_id)
    players_b = get_players_for_team(players, team_b_id)

    if args.model_in:
        features = np.array([np.stack([players_a, players_b])])

        estimator = Estimator(
            args.model_type,
            hidden_units=args.hidden_units,
            model_in=args.model_in,
            n_threads=args.n_threads,
            feature_year=args.year,
        )

        score = next(estimator.predict(x=features))
        print(
            "NN Prediction: %s vs. %s final score: %s"
            % (args.team_a, args.team_b, score)
        )

    else:
        # Use each team's games against other teams to figure out how much
        # worse an average team does when playing against them (vs. against
        # other teams).
        # Use that to adjust each team's historical mean score to predict how
        # well they'll do against each other.
        games = load_ncaa_games(args.year - 1)
        a_score, a_diff = get_historical_score(team_a_id, games)
        b_score, b_diff = get_historical_score(team_b_id, games)
        print(
            "Historical prediction: %s %.1f to %s %.1f (total: %.1f)"
            % (
                args.team_a,
                a_score - b_diff,
                args.team_b,
                b_score - a_diff,
                a_score + b_score - a_diff - b_diff,
            )
        )

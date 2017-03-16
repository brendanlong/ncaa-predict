#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

from constants import DNN_HIDDEN_UNITS
from data_loader import load_ncaa_players, load_ncaa_schools, \
    load_ncaa_games, get_players_for_team


def team_name_to_id(name, all_teams):
    try:
        return \
            all_teams[all_teams["school_name"] == name]["school_id"].values[0]
    except IndexError:
        raise Exception("Couldn't find ID for school [%s]" % name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument("--year", "-y", default=2017, type=int)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.ERROR)

    players = load_ncaa_players(args.year)
    all_teams = load_ncaa_schools()
    team_a_id = team_name_to_id(args.team_a, all_teams)
    team_b_id = team_name_to_id(args.team_b, all_teams)
    players_a = get_players_for_team(players, team_a_id)
    players_b = get_players_for_team(players, team_b_id)
    features = np.array([np.stack([players_a, players_b])])
    feature_cols = \
        tf.contrib.learn.infer_real_valued_columns_from_input(features)

    estimator = tf.contrib.learn.DNNRegressor(
        hidden_units=DNN_HIDDEN_UNITS,
        model_dir=args.model_in, feature_columns=feature_cols)
    score = next(estimator.predict(x=features))
    print("%s vs. %s final score: %s" % (args.team_a, args.team_b, score))

    # Since we want to know the combined score, average the average score and
    # opponent score for each team and multiply by two.
    # We care about the opponent score becaue it gives us some information
    # about how good the team is at preventing the other team from scoring.
    games = load_ncaa_games(args.year - 1)
    a_scores = []
    b_scores = []
    for team_id in (team_a_id, team_b_id):
        g = games[games["school_id"] == team_id]
        us_score = g["score"].mean()
        opponent_score = g["opponent_score"].mean()
        if team_id == team_a_id:
            a_scores.append(us_score)
            b_scores.append(opponent_score)
        else:
            b_scores.append(opponent_score)
            a_scores.append(us_score)
    a = np.mean(a_scores)
    b = np.mean(b_scores)
    print(
        "Or historical prediction: %s %.1f to %s %.1f (total: %.1f)"
        % (args.team_a, a, args.team_b, b, a + b))

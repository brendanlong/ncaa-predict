This is a project to choose NCAA brackets for me, using historical data
fed through a TensorFlow classifier.

All of the Python scripts in this repo have help info accessible with `--help`.

See also [the TODO list](TODO.md).

## Setup

To setup, install Python 3 and then run:

```
./setup.sh
```

This will setup a virtualenv, install [the requirements](requirement.txt) from
`pip`, and activate the [`direnv`](https://direnv.net/) [`.envrc`](.envrc) (if
applicable).

If you don't have `direnv`, you will need to `source bin/activate`.

If you have a CUDA GPU, you might want to run `pip install tensorflow-gpu` at
this point to get the GPU-enabled version instead.

You might also want to consider building TensorFlow from source, since the
default package doesn't have SSE enabled.

## Getting data

This repo comes with data for 2002-2017 in the [`csv`](csv) folder. If you need
more data:

```
./fetch_csvs.py get_schools
./fetch_csvs.py get_games -y $YEAR
./fetch_csvs.py get_players -y $YEAR
```

The loaders try to cache as much as possible since the NCAA site is really
slow, so if you want to update a CSV, you need to delete it, *then* run
`./fetch_csvs.py` with the relevant command and year.

See [the README in the `csv` folder](csv/README.md) for details.

## Training a model

```
./train.py -y $TRAINING_YEARS -p $VALIDATION_YEAR -o $MODEL_OUT
```

Example:

```
./train.py -y 2002,2003,2004 -p 2005 -o model_2002-2004
```

See other training options with `./train.py --help`. You may also want to
change the hidden layers, but this currently isn't a command-line option and
you will need to edit the code.

This trains a [Keras](https://keras.io/) classifier
using player stats to predict which team will win in a matchup.

## Evaluating a model

`./evaluate.py` can be used to evaluate a trained model against a given year's
data. You should also see the training set accuracy during training at the end
of each epoch.

## Predicting games

Once you have a trained model, open [`predict.py`](predict.py) and edit the
bracket at the top. TODO: Make this a config file or something.

The format is a tree of tuples where `(a, b)` means that `a` will play against
`b`. If `a` or `b` are tuples, then we will first determine the winner of that
match up, and then use the winner for the next one.

Example:

```
(
  ("Team A", "Team B"),
  ("Team C", "Team D"),
)
```

This means that the first round is Team A vs Team B and Team C vs Team D, and
the second round will be the winners of the first round.

Each string needs to be the name of a team from the "Schools" dropdown in
[the NCAA stats page](http://web1.ncaa.org/stats/StatsSrv/careersearch). An
easy way to find these is to look through the
[`ncaa_schools.csv`](csv/ncaa_schools.csv) file.

Once you've added your bracket:

```
./predict.py -m $YOUR_MODEL -y $YEAR
```

And you should get output like:

```
Villanova vs Mt. St. Mary's: Villanova wins
Wisconsin vs Virginia Tech: Wisconsin wins
Villanova vs Wisconsin: Villanova wins
Virginia vs UNCW: Virginia wins
Florida vs ETSU: Florida wins
Virginia vs Florida: Virginia wins
Villanova vs Virginia: Villanova wins
```

For filling in a bracket, add `--wait` to the command for it to stop after
each line and wait for you to hit enter.

## Predicting scores

The neural network version of this doesn't really work right now, so
`./predict_score.py` uses a simpler algorithm based on the model's historical
scores.

To run:

```
./predict_score.py -y $YEAR "Team a" "Team b"
```

Where `$YEAR` is the year to use for the prediction (most likely this is the
year before the year you want to predict, since the year you want to predict
presumably doesn't have win/loss data).

Each team parameter needs to be the name from the "Schools" dropdown in
[the NCAA stats page](http://web1.ncaa.org/stats/StatsSrv/careersearch). An
easy way to find these is to look through the
[`ncaa_schools.csv`](csv/ncaa_schools.csv) file.

The algorithm uses:
  - The average score for every team in the given year. This gives us a base
    prediction for how many points that team will score in a game.
  - For the two teams we're considering, the mean of the difference between
    their opponent's mean score that season and their opponent's mean score
    against them.

    This tells us how the opponent team will effect the score (i.e. if the
    opponent is not very good, our team will score more points than usual,
    vice-versa).

To predict each team's score, take its mean score and add the other team's
mean effect on the score.

For example, in 2016, Villanova's average score was 76.5 points, and the
average team scored 8 points fewer against them. Kansas's average was 71.1
points and the average team scored 3.9 points fewer against them.

Because of this we predict that in Villanova vs Kansas, Villanova will score
72.6 points (`76.5 - 3.9`) and Kansas will score 63.1 points (`71.1 - 8`).

```
$ ./predict_score.py -y 2016 Villanova Kansas
Historical prediction: Villanova 72.6 to Kansas 63.2 (total: 135.8)
```

This algorithm could use work but it comes up with plausible numbers.

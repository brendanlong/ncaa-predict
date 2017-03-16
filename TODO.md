## Model

  - Do some research on what kinds of models other people use for
    this kind of data.
  - Figure out a way to visualize weights so I can figure out which
    input features are most useful.
  - Figure out a way to make the structure of the feature data more
    obvious (the current version is a fully-connected network, so
    I think it collapses the data to 1D, even though we have a nice
    3D structure (teams, players, stats). Maybe convolution layers,
    although that seems weird since the stats columns are all
    different.
  - Get a score predictor working.

## Usability / General

  - Factor the model out into it's own Python code and save it with
    the model training data, so we can train different models and
    still load them in a reasonable way.
  - Load the data into an SQLite database.
  - Automatically save all models along with all of the information
    we can look up about them (training steps, batch size, evaluator
    type, hidden layers, etc.).
  - Move all of the Python code into a module directory and create
    one CLI input with multiple subcommands (i.e.
    `./ncaa-predict.py train`, `./ncaa-predict.py predict-game`,
    etc.).
  - Make the Pandas code less terrible (my first use of this
    library).

## Data

  - Use all of the numerical player stats (excluding ID's).
  - Figure out how to use categorical data like class and home
    game (although this second one would need to be optional so we
    can still predict games if we don't know if they're home or
    away).
  - Normalize integer and floating point data to be floating point
    between 0 and 1 (or -1 and 1) where possible. For example,
    height is currently an integer number of inches, but it would
    be easier for the model to understand if we made 0 = shortest
    player in the league, 1 = tallest player in the league, and
    everyone else is between those two. This would also apply to
    pretty much every integer or float input.
  - Figure out a way to include information about coaches.
  - Brainstorm other kinds of data we could include.

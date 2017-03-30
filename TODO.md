## Model

  - Do some research on what kinds of models other people use for
    this kind of data.
  - Figure out a way to visualize weights so I can figure out which
    input features are most useful.
  - Get a score predictor working.

## Usability / General

  - Load the data into an SQLite database.
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
  - Figure out a way to include information about coaches.
  - Brainstorm other kinds of data we could include.

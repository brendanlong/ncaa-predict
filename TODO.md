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

## Data

  - Include team division (seems to be confusing the model since it's
    predicting very low scores).
  - Figure out a way to include information about coaches.
  - Brainstorm other kinds of data we could include.

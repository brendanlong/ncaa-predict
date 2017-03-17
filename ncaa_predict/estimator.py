import sys

import numpy as np
import tensorflow as tf

from .data_loader import load_data, load_data_multiyear
from .model import ModelType


DEFAULT_BATCH_SIZE = 1000
DEFAULT_N_THREADS = 16
DEFAULT_STEPS = sys.maxsize
DEFAULT_HIDDEN_UNITS = [128, 64, 16]


class Estimator:
    def __init__(
            self, model_type, feature_year, model_in=None,
            n_threads=DEFAULT_N_THREADS, hidden_units=DEFAULT_HIDDEN_UNITS):
        self.model_type = ModelType(model_type)
        self.hidden_units = hidden_units
        self.n_threads = int(n_threads)

        # FIXME: Don't load arbitrary years here just to get the feature
        # description.
        features, _ = load_data(feature_year)
        self.estimator = self.model_type.get_estimator(
            features, hidden_units=self.hidden_units, model_in=model_in)

    def train(self, years, batch_size=DEFAULT_BATCH_SIZE, steps=DEFAULT_STEPS):
        features, labels = load_data_multiyear(years, self.n_threads)
        try:
            self.estimator.fit(
                x=features, y=labels, steps=steps, batch_size=batch_size)
        except KeyboardInterrupt:
            pass
        return self.estimator

    def evaluate(self, year):
        features, labels = load_data(year, self.n_threads)
        return self.estimator.evaluate(x=features, y=labels)

    def predict(self, x):
        return next(self.estimator.predict(x=x))

    def save(self, export_dir):
        """Export a trained model to a directory.

        If the directory exists, prompt for a different one until given one
        that doesn't exist.
        """
        while True:
            try:
                self.estimator.export(export_dir=export_dir)
                break
            except RuntimeError as e:
                if "Duplicate export dir" not in str(e):
                    raise
                print(
                    "%s already exists. Pick a different model out folder."
                    % export_dir)
                export_dir = input("Save model to? ")

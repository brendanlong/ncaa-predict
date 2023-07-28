#!/usr/bin/env python3
import argparse

import keras

from ncaa_predict.data_loader import load_data


def evaluate(model, year):
    features, labels = load_data(year)
    print("\nEvaluating accuracy")
    model.evaluate(x=features, y=labels, verbose=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument("--year", "-y", default=2016, type=int)
    args = parser.parse_args()

    model = keras.models.load_model(args.model_in)
    evaluate(model, args.year)

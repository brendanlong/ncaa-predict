from enum import Enum, unique

import tensorflow as tf


@unique
class ModelType(Enum):
    classifier = "classifier"
    regressor = "regressor"
    dnn_classifier = "dnn_classifier"
    dnn_regressor = "dnn_regressor"

    @property
    def estimator_factory(self):
        if self == ModelType.classifier:
            return tf.contrib.learn.LinearClassifier
        elif self == ModelType.regressor:
            return tf.contrib.learn.LinearRegressor
        elif self == ModelType.dnn_classifier:
            return tf.contrib.learn.DNNClassifier
        elif self == ModelType.dnn_regressor:
            return tf.contrib.learn.DNNRegressor

    @property
    def has_hidden_units(self):
        return self in (ModelType.classifier, ModelType.dnn_classifier)

    def get_estimator(self, features, hidden_units=None, model_in=None):
        feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(
            features)
        return self.estimator_factory(
            feature_columns=feature_cols, model_dir=model_in,
            hidden_units=hidden_units)

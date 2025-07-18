# Note: This code makes use of code snippets provided under the following links
# https://towardsdatascience.com/how-to-train-a-classification-model-with-tensorflow-in-10-minutes-fd2b7cfba86
# https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3

import numpy as np
import os
import json
import argparse
import errno
import datetime
import shutil
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import random
from ast import literal_eval
import pandas as pd
from pathlib import Path

class RiskDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 data_dir,
                 batch_size,
                 risky_state_rebalancing_fraction=None,
                 shuffle=False):

        self._data_dir = data_dir
        self._batch_size = batch_size
        self._risky_state_rebalancing_fraction = risky_state_rebalancing_fraction
        self._shuffle = shuffle
        self._state_action_risk = False
        self._state_size = None
        self._action_size = None
        self._x = []
        self._y = []

        if risky_state_rebalancing_fraction is not None and not shuffle:
            raise ValueError("risky_state_rebalancing_fraction requires shuffle to be True")

        # list all csv files in data_dir
        for csv_file in sorted(Path(self._data_dir).glob('*.csv')):
            df = pd.read_csv(csv_file, converters={'state': literal_eval, 'action': literal_eval})
            df_y = list(df["risk"])
            if "action" in df:
                self._state_action_risk = True
                state = list(df["state"])
                action = list(df["action"])
                df_x = [state[i] + action[i] for i in range(len(state))]
                self._state_size = len(state[0])
                self._action_size = len(action[0])
            else:
                df_x = list(df["state"])
                self._state_size = len(df_x[0])
            self._x.extend(df_x)
            self._y.extend(df_y)

        if not self._x:
            raise FileNotFoundError("Could not find valid data files in {}.".format(self._data_dir))

        self._len = int(len(self._x) / self._batch_size)
        if self._len == 0:
            # batch_size bigger than dataset
            self._batch_size = len(self._x)
            self._len = 1

        if not self._shuffle:
            # remove last entries that do not fit into a full batch
            self._x = self._x[:self._len * self._batch_size]
            self._y = self._y[:self._len * self._batch_size]

        # convert lists to numpy array
        self._x = np.array(self._x)
        self._y = np.array(self._y)

        if self._risky_state_rebalancing_fraction is not None:
            self._batch_size_risky = int(self._risky_state_rebalancing_fraction * self._batch_size)
            self._batch_size_not_risky = self._batch_size - self._batch_size_risky
            risky_indices = self._y == 1.0

            self._x_risky = self._x[risky_indices]
            self._y_risky = self._y[risky_indices]

            if len(self._x_risky) < self._batch_size_risky:
                raise ValueError("Not enough risky datapoints for the selected batch_size and rebalancing_fraction "
                                 "(required {}, available {}.)".format(self._batch_size_risky, len(self._x_risky)))

            self._x_not_risky = self._x[np.logical_not(risky_indices)]
            self._y_not_risky = self._y[np.logical_not(risky_indices)]

            if len(self._x_not_risky) < self._batch_size_not_risky:
                raise ValueError("Not enough unrisky datapoints for the selected batch_size and rebalancing_fraction "
                                 "(required {}, available {}).".format(self._batch_size_not_risky,
                                                                      len(self._x_not_risky)))

            self._random_indices_risky = np.array([1] * self._batch_size_risky
                                                  + [0] * (len(self._x_risky) - self._batch_size_risky), dtype=bool)
            self._random_indices_not_risky = np.array([1] * self._batch_size_not_risky
                                                      + [0] * (len(self._x_not_risky) - self._batch_size_not_risky),
                                                      dtype=bool)
            np.random.shuffle(self._random_indices_risky)
            np.random.shuffle(self._random_indices_not_risky)

        else:
            if self._shuffle:
                self._random_indices = np.array([1] * self._batch_size + [0] * (len(self._x) - self._batch_size),
                                                dtype=bool)
                np.random.shuffle(self._random_indices)
            else:
                self._random_indices = None

    def on_epoch_end(self):
        if self._shuffle:
            if self._risky_state_rebalancing_fraction is not None:
                np.random.shuffle(self._random_indices_risky)
                np.random.shuffle(self._random_indices_not_risky)
            else:
                np.random.shuffle(self._random_indices)

    def __getitem__(self, index):
        if self._shuffle:
            if self._risky_state_rebalancing_fraction is not None:
                x_batch = np.concatenate((self._x_not_risky[self._random_indices_not_risky],
                                          self._x_risky[self._random_indices_risky]))
                y_batch = np.concatenate((self._y_not_risky[self._random_indices_not_risky],
                                          self._y_risky[self._random_indices_risky]))
            else:
                x_batch = self._x[self._random_indices]
                y_batch = self._y[self._random_indices]
        else:
            x_batch = self._x[index * self._batch_size:(index + 1) * self._batch_size]
            y_batch = self._y[index * self._batch_size:(index + 1) * self._batch_size]

        return x_batch, y_batch

    def __len__(self):
        return self._len

    @property
    def state_action_risk(self):
        return self._state_action_risk

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk_data_dir', type=str, required=True, default=None)
    parser.add_argument('--experiment_name', type=str, required=True, default=None)
    parser.add_argument('--hidden_layer_activation', default='relu', choices=['relu', 'selu', 'tanh', 'sigmoid', 'elu',
                                                                              'gelu', 'swish', 'leaky_relu'])
    parser.add_argument('--last_layer_activation', default='sigmoid', choices=['linear', 'sigmoid'])
    parser.add_argument('--fcnet_hiddens', type=json.loads, default=[128, 256, 256])
    parser.add_argument('--precision_threshold', type=float, default=0.5)
    parser.add_argument('--recall_threshold', type=float, default=0.5)
    parser.add_argument('--risky_state_class_weight', type=float, default=None)
    parser.add_argument('--risky_state_rebalancing_fraction', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_per_checkpoint', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--kernel_constraint', type=float, default=None)
    parser.add_argument('--kernel_regularizer', type=float, default=None)
    parser.add_argument('--bias_regularizer', type=float, default=None)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    training_data_generator = RiskDataGenerator(data_dir=os.path.join(args.risk_data_dir, "train"),
                                                batch_size=args.batch_size,
                                                risky_state_rebalancing_fraction=args.risky_state_rebalancing_fraction,
                                                shuffle=args.shuffle)

    test_data_generator = RiskDataGenerator(data_dir=os.path.join(args.risk_data_dir, "test"),
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle)

    log_dir = os.path.join(Path.home(), "risk_results") if args.logdir is None else args.logdir
    # add state_risk / state_action_risk folder
    if training_data_generator.state_action_risk:
        log_dir = os.path.join(log_dir, "state_action_risk")
    else:
        log_dir = os.path.join(log_dir, "state_risk")
    # add args.experiment_name folder and current time stamp
    log_dir = os.path.join(log_dir, args.experiment_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))

    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise exc

    # copy config file from data dir to log_dir if available

    config_file_path = os.path.join(os.path.dirname(args.risk_data_dir), "risk_config.json")
    if os.path.isfile(config_file_path):
        destination_path = os.path.join(log_dir, "risk_config.json")
        shutil.copy(config_file_path, destination_path)

        with open(destination_path, 'r') as f:
            config = json.load(f)

        if "observation_size" in config:
            if training_data_generator.state_size != config["observation_size"]:
                raise ValueError("The observation size of the risk data ({}) does not match with the "
                                 "observations size specified in risk_config.json ({}).".format(
                                    training_data_generator.state_size, config["observation_size"]))
        else:
            config["observation_size"] = training_data_generator.state_size

        # action_size to config
        config["action_size"] = training_data_generator.action_size

        # store updated config
        with open(destination_path, 'w') as f:
            f.write(json.dumps(config, sort_keys=True))
            f.flush()

    # store arguments
    with open(os.path.join(log_dir, "arguments.json"), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True))
        f.flush()

    callbacks = []

    # tensorboard callback
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=True, write_images=False)
    callbacks.append(tensorboard)

    if args.epochs_per_checkpoint is not None:
        # checkpoint callback
        checkpoint_callback = ModelCheckpoint(filepath=log_dir,
                                              save_freq=args.epochs_per_checkpoint,
                                              save_weights_only=False)
        callbacks.append(checkpoint_callback)

    if args.kernel_regularizer is not None:
        kernel_regularizer = l2(args.kernel_regularizer)
    else:
        kernel_regularizer = None

    if args.bias_regularizer is not None:
        bias_regularizer = l2(args.bias_regularizer)
    else:
        bias_regularizer = None

    if args.kernel_constraint is not None:
        kernel_constraint = tf.keras.constraints.MaxNorm(args.kernel_constraint)
    else:
        kernel_constraint = None

    if args.risky_state_class_weight is not None:
        class_weight = {0: 1.0, 1: args.risky_state_class_weight}
    else:
        class_weight = None

    model = tf.keras.Sequential()
    for i in range(len(args.fcnet_hiddens)):
        model.add(tf.keras.layers.Dense(args.fcnet_hiddens[i], activation=args.hidden_layer_activation,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                        kernel_constraint=kernel_constraint))
        if args.dropout is not None:
            model.add(tf.keras.layers.Dropout(args.dropout))
    # last layer
    model.add(tf.keras.layers.Dense(1, activation=args.last_layer_activation,
                                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision_0.01', thresholds=0.01),
            tf.keras.metrics.Precision(name='precision_0.02', thresholds=0.02),
            tf.keras.metrics.Precision(name='precision_0.03', thresholds=0.03),
            tf.keras.metrics.Precision(name='precision_0.04', thresholds=0.04),
            tf.keras.metrics.Precision(name='precision_0.05', thresholds=0.05),
            tf.keras.metrics.Precision(name='precision_0.1', thresholds=0.1),
            tf.keras.metrics.Precision(name='precision_0.2', thresholds=0.2),
            tf.keras.metrics.Precision(name='precision_0.3', thresholds=0.3),
            tf.keras.metrics.Precision(name='precision_0.4', thresholds=0.4),
            tf.keras.metrics.Precision(name='precision_0.5', thresholds=0.5),
            tf.keras.metrics.Precision(name='precision_0.6', thresholds=0.6),
            tf.keras.metrics.Precision(name='precision_0.7', thresholds=0.7),
            tf.keras.metrics.Precision(name='precision_0.8', thresholds=0.8),
            tf.keras.metrics.Precision(name='precision_0.9', thresholds=0.9),
            tf.keras.metrics.Recall(name='recall_0.01', thresholds=0.01),
            tf.keras.metrics.Recall(name='recall_0.02', thresholds=0.02),
            tf.keras.metrics.Recall(name='recall_0.03', thresholds=0.03),
            tf.keras.metrics.Recall(name='recall_0.04', thresholds=0.04),
            tf.keras.metrics.Recall(name='recall_0.05', thresholds=0.05),
            tf.keras.metrics.Recall(name='recall_0.1', thresholds=0.1),
            tf.keras.metrics.Recall(name='recall_0.2', thresholds=0.2),
            tf.keras.metrics.Recall(name='recall_0.3', thresholds=0.3),
            tf.keras.metrics.Recall(name='recall_0.4', thresholds=0.4),
            tf.keras.metrics.Recall(name='recall_0.5', thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_0.6', thresholds=0.6),
            tf.keras.metrics.Recall(name='recall_0.7', thresholds=0.7),
            tf.keras.metrics.Recall(name='recall_0.8', thresholds=0.8),
            tf.keras.metrics.Recall(name='recall_0.9', thresholds=0.9),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR')
        ]
    )

    history = model.fit(x=training_data_generator,
                        batch_size=None,
                        epochs=args.epochs,
                        validation_data=test_data_generator,
                        validation_freq=1,
                        class_weight=class_weight,
                        shuffle=args.shuffle,
                        callbacks=callbacks)

    model.save(filepath=log_dir)





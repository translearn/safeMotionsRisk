import numpy as np
import os
import argparse
import logging
from ast import literal_eval
import pandas as pd
from pathlib import Path


class RiskDataReader:

    def __init__(self,
                 data_dir,
                 print_stats_for_each_file=False):

        self._data_dir = data_dir
        self._state_action_risk = False
        self._state_size = None
        self._action_size = None
        self._x = []
        self._y_ground_truth = []
        self._y_prediction = []
        self._state = []
        self._action = []

        self._pid = os.getpid()

        self._risk_network = None
        self._risk_network_graph = None

        csv_files = sorted(Path(self._data_dir).glob('*.csv'))

        if len(csv_files) == 0:
            raise FileNotFoundError("Could not find risk data (*.csv) in the specified directory.")

        self._training_data_dir = os.path.join(args.risk_data_dir, "train")
        self._test_data_dir = os.path.join(args.risk_data_dir, "test")

        for dir in [self._training_data_dir, self._test_data_dir]:
            os.makedirs(dir, exist_ok=True)
            if os.listdir(dir):
                raise ValueError("Directory {} is not empty. Delete the directory and try again.".format(dir))

        # list all csv files in data_dir
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, converters={'state': literal_eval, 'action': literal_eval})
            df_y_ground_truth = list(df["risk"])

            if "action" in df:
                self._state_action_risk = True
                state = list(df["state"])
                action = list(df["action"])
                df_x = [state[i] + action[i] for i in range(len(state))]
                self._state_size = len(state[0])
                self._action_size = len(action[0])
                self._state.extend(state)
                self._action.extend(action)
            else:
                df_x = list(df["state"])
                self._state_size = len(df_x[0])
                self._state.extend(df_x)

            if print_stats_for_each_file:
                logging.info("Processing file {}.csv".format(str(csv_file.stem)))
                self.print_risk_stats(x=df_x, y_ground_truth=df_y_ground_truth)

            self._x.extend(df_x)
            self._y_ground_truth.extend(df_y_ground_truth)

        if not self._x:
            raise FileNotFoundError("Could not find valid data files in {}.".format(self._data_dir))

        logging.info("----------------------------------------- Full dataset "
                     "-----------------------------------------")

        self.print_risk_stats()

    def print_risk_stats(self, x=None, y_ground_truth=None):
        if x is None or y_ground_truth is None:
            x = self._x
            y_ground_truth = self._y_ground_truth

        x = np.array(x)
        y_ground_truth = np.array(y_ground_truth)
        logging.info("Number of datapoints: {}".format(len(x)))
        logging.info("Risky action rate: {:.3f} %".format(np.mean(y_ground_truth) * 100))

    def store_risk_data(self, test_data_fraction, datapoints_per_file):
        logging.info("------------------------------------------------"
                     "------------------------------------------------")

        state = self._state

        if self._state_action_risk:
            action = self._action
        else:
            action = None

        risk = self._y_ground_truth

        num_files = int(len(state) / datapoints_per_file)
        ignored_datapoints = len(state) - num_files * datapoints_per_file
        logging.info("Exporting a total of {} datapoints to {} files with {} datapoints each.".format(
            len(state), num_files, datapoints_per_file))
        logging.info("Ignoring {} datapoints.".format(ignored_datapoints))
        logging.info("Export directory: {}".format(self._data_dir))
        num_training_files = int(num_files * (1.0 - test_data_fraction))
        logging.info("Splitting the data into {} training files and {} test files.".format(
            num_training_files, num_files - num_training_files))

        for i in range(num_files):
            risk_dict = {}
            datapoint_start = i * datapoints_per_file
            datapoint_end = i * datapoints_per_file + datapoints_per_file - 1
            risk_dict["state"] = state[datapoint_start:datapoint_end + 1]
            if self._state_action_risk:
                risk_dict["action"] = action[datapoint_start:datapoint_end + 1]
            risk_dict["risk"] = risk[datapoint_start:datapoint_end + 1]

            mean_datapoint_risk = np.mean(risk_dict["risk"])

            if i < num_training_files:
                export_dir = self._training_data_dir
            else:
                export_dir = self._test_data_dir

            data_frame = pd.DataFrame(risk_dict)
            export_file_name = "datapoints_{}_to_{}_risk_{:.2f}_pid_{}.csv".format(
                datapoint_start,
                datapoint_end,
                mean_datapoint_risk,
                self._pid)

            with open(os.path.join(export_dir, export_file_name), 'w') as file:
                data_frame.to_csv(path_or_buf=file)

            logging.info("File {} / {} exported to {}".format(
                i + 1, num_files, os.path.join(os.path.basename(export_dir), export_file_name)))

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
    parser.add_argument('--test_data_fraction', type=float, default=0.1)
    parser.add_argument('--datapoints_per_file', type=int, default=1000)
    parser.add_argument("--logging_level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s'
    logging_formatter = logging.Formatter(logging_format)
    logging.basicConfig(format=logging_format)
    logging.getLogger().setLevel(args.logging_level)

    if not (0 <= args.test_data_fraction <= 1.0):
        raise ValueError ("--test_data_fraction must be within 0.0 and 1.0")

    risk_data_reader = RiskDataReader(data_dir=args.risk_data_dir,
                                      print_stats_for_each_file=True)

    risk_data_reader.store_risk_data(test_data_fraction=args.test_data_fraction,
                                     datapoints_per_file=args.datapoints_per_file)



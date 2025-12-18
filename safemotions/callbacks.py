from collections import defaultdict
from ray.rllib.callbacks.callbacks import RLlibCallback
import numpy as np

METRICS_EPISODE_WINDOW = None  # compute metrics over a window of the last N episodes, e.g. N = 50
METRIC_OPS = ['sum', 'mean', 'max', 'min']

class CustomTrainCallbacks(RLlibCallback):

    def on_episode_start(self, *, episode, **kwargs):
        episode.custom_data['op'] = {}
        for op in METRIC_OPS:
            episode.custom_data['op'][op] = defaultdict(list)

    def on_episode_step(self, *, episode, **kwargs):
        episode_info = episode.infos[-1]
        if episode_info:
            for op in list(episode_info.keys() & METRIC_OPS):
                for k, v in episode_info[op].items():
                    episode.custom_data['op'][op][k].append(v)

    def on_episode_end(self, *, episode, env, metrics_logger, **kwargs):
        def __apply_op_on_list(operator, data_list):
            if operator == 'sum':
                return sum(data_list)
            elif operator == 'mean':
                return sum(data_list) / len(data_list)
            elif operator == 'max':
                return max(data_list)
            elif operator == 'min':
                return min(data_list)

        episode_info = episode.infos[-1]
        env = env.envs[-1].unwrapped

        for op in METRIC_OPS:
            for k, v in episode.custom_data['op'][op].items():
                data_after_op = __apply_op_on_list(op, episode.custom_data['op'][op][k])
                reduce_method = op if op != "sum" else "mean"
                metrics_logger.log_value(k + '_' + op, data_after_op, reduce=reduce_method,
                                         window=METRICS_EPISODE_WINDOW, clear_on_reduce=True)

        episode.custom_data['op'] = {}

        for k, v in episode_info.items():
            log_metric = False
            if k.startswith('obstacles'):
                log_metric = True
            if "moving_object" in k and not np.isnan(v):
                log_metric = True
            if "ball_machine" in k and not np.isnan(v):
                log_metric = True
            if "collision_avoidance" in k and not np.isnan(v):
                log_metric = True
            if "risk_network" in k and not np.isnan(v):
                log_metric = True

            if log_metric:
                metrics_logger.log_value(k, v, reduce="mean",
                                         window=METRICS_EPISODE_WINDOW, clear_on_reduce=True)

        min_max_mean_log_values = {'episode_length': float(episode_info['episode_length']),
                                   'trajectory_length': episode_info['trajectory_length']}

        if 'trajectory_fraction' in episode_info:
            min_max_mean_log_values['trajectory_fraction'] = episode_info['trajectory_fraction']

        if 'trajectory_successful' in episode_info:
            min_max_mean_log_values['success_rate'] = episode_info['trajectory_successful']
        else:
            min_max_mean_log_values['success_rate'] = 0.0

        for k, v in min_max_mean_log_values.items():
            for op in ['mean', 'max', 'min']:
                metrics_logger.log_value(k + '_' + op, v, reduce=op, window=METRICS_EPISODE_WINDOW,
                                         clear_on_reduce=True)

        termination_reasons_dict = \
            {env.TERMINATION_TRAJECTORY_LENGTH: 'termination_rate_trajectory_length',
             env.TERMINATION_JOINT_LIMITS: 'termination_rate_joint_limit_violation',
             env.TERMINATION_SELF_COLLISION: 'termination_rate_collision_self',
             env.TERMINATION_COLLISION_WITH_STATIC_OBSTACLE: 'termination_rate_collision_static_obstacles',
             env.TERMINATION_COLLISION_WITH_MOVING_OBSTACLE: 'termination_rate_collision_moving_obstacles'}

        for k, v in termination_reasons_dict.items():
            termination_log_value = 1.0 if episode_info['termination_reason'] == k else 0.0
            metrics_logger.log_value(v, termination_log_value, reduce="mean", window=METRICS_EPISODE_WINDOW,
                                     clear_on_reduce=True)
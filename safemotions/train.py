#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
import numpy as np
import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback  # tensorboard fix ray 1.5 https://github.com/ray-project/ray/issues/17366
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from typing import Dict
import multiprocessing
from collections import defaultdict
import logging
import klimits


METRIC_OPS = ['sum', 'average', 'max', 'min']

RISK_STATE_CONFIG = {'RISK_CHECK_CURRENT_STATE': 0,
                     'RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING': 1,
                     'RISK_CHECK_NEXT_STATE_FULL_FORECASTING': 2,
                     'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP': 3,
                     'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY': 4}


class CustomTrainCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        episode.user_data['op'] = {}
        for op in METRIC_OPS:
            episode.user_data['op'][op] = defaultdict(list)

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        episode_info = episode.last_info_for()
        if episode_info:
            for op in list(episode_info.keys() & METRIC_OPS):
                for k, v in episode_info[op].items():
                    episode.user_data['op'][op][k].append(v)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        def __apply_op_on_list(operator, data_list):
            if operator == 'sum':
                return sum(data_list)
            elif operator == 'average':
                return sum(data_list) / len(data_list)
            elif operator == 'max':
                return max(data_list)
            elif operator == 'min':
                return min(data_list)

        episode_info = episode.last_info_for()
        episode_length = episode_info['episode_length']
        trajectory_length = episode_info['trajectory_length']
        env = base_env.get_unwrapped()[-1]

        for op in METRIC_OPS:
            for k, v in episode.user_data['op'][op].items():
                episode.custom_metrics[k + '_' + op] = __apply_op_on_list(op, episode.user_data['op'][op][k])

        for k, v in episode_info.items():
            if k.startswith('obstacles'):
                episode.custom_metrics[k] = v
            if "moving_object" in k and not np.isnan(v):
                episode.custom_metrics[k] = v
            if "ball_machine" in k and not np.isnan(v):
                episode.custom_metrics[k] = v
            if "collision_avoidance" in k and not np.isnan(v):
                episode.custom_metrics[k] = v
            if "risk_network" in k and not np.isnan(v):
                episode.custom_metrics[k] = v

        episode.custom_metrics['episode_length'] = float(episode_length)
        episode.custom_metrics['trajectory_length'] = trajectory_length

        if 'trajectory_fraction' in episode_info:
            episode.custom_metrics['trajectory_fraction'] = episode_info['trajectory_fraction']

        if 'trajectory_successful' in episode_info:
            episode.custom_metrics['success_rate'] = episode_info['trajectory_successful']
        else:
            episode.custom_metrics['success_rate'] = 0.0

        termination_reasons_general_dict = \
            {env.TERMINATION_TRAJECTORY_LENGTH: 'trajectory_length_termination_rate',
             env.TERMINATION_JOINT_LIMITS: 'joint_limit_violation_termination_rate'}

        for k, v in termination_reasons_general_dict.items():
            episode.custom_metrics[v] = 1.0 if episode_info['termination_reason'] == k else 0.0

        termination_reasons_collision_dict = \
            {env.TERMINATION_SELF_COLLISION: 'collision_self_termination_rate',
             env.TERMINATION_COLLISION_WITH_STATIC_OBSTACLE: 'collision_static_obstacles_termination_rate',
             env.TERMINATION_COLLISION_WITH_MOVING_OBSTACLE: 'collision_moving_obstacles_termination_rate'}

        for k, v in termination_reasons_collision_dict.items():
            episode.custom_metrics[v] = 1.0 if episode_info['termination_reason'] == k else 0.0

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result['callback_ok'] = True


def resample_wrapper(method, env, policy):

    def resample_wrapping_method(*args, **kwargs):
        initial_time_step = policy.global_timestep
        for i in range(env.max_resampling_attempts + 1):
            if i > 0:
                policy.global_timestep = initial_time_step

            return_value = method(*args, **kwargs)

            action = np.core.umath.clip(return_value[0][0], env.action_space.low, env.action_space.high)
            if i < env.max_resampling_attempts:
                if hasattr(env, "is_action_safe"):
                    if env.is_action_safe(action):
                        break
                else:
                    raise NotImplementedError("Action resampling requires a function called "
                                              "is_action_safe in env")

        return return_value

    return resample_wrapping_method


class ResampleUnsafeActionsCallbacks(CustomTrainCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        if not hasattr(policies["default_policy"], "resample_wrapping_active"):
            setattr(policies["default_policy"], "resample_wrapping_active", True)
            env = base_env.get_unwrapped()[-1]

            policies["default_policy"].compute_actions_from_input_dict = \
                resample_wrapper(policies["default_policy"].compute_actions_from_input_dict, env,
                                 policies["default_policy"])

            policies["default_policy"].compute_actions = \
                resample_wrapper(policies["default_policy"].compute_actions, env, policies["default_policy"])

        super().on_episode_start(worker=worker, base_env=base_env,
                                 policies=policies,
                                 episode=episode, env_index=env_index, **kwargs)


def _make_env_config():

    env_config = {
        'experiment_name': args.name,
        'pos_limit_factor': args.pos_limit_factor,
        'vel_limit_factor': args.vel_limit_factor,
        'acc_limit_factor': args.acc_limit_factor,
        'jerk_limit_factor': args.jerk_limit_factor,
        'torque_limit_factor': args.torque_limit_factor,
        'action_mapping_factor': args.action_mapping_factor,
        'action_preprocessing_function': args.action_preprocessing_function,
        'normalize_reward_to_frequency': args.normalize_reward_to_frequency,
        'trajectory_duration': args.trajectory_duration,
        'trajectory_time_step': args.trajectory_time_step,
        'obs_add_target_point_pos': args.obs_add_target_point_pos,
        'obs_add_target_point_relative_pos': args.obs_add_target_point_relative_pos,
        'punish_action': args.punish_action,
        'action_punishment_min_threshold': args.action_punishment_min_threshold,
        'action_max_punishment': args.action_max_punishment,
        'punish_adaptation': args.punish_adaptation,
        'adaptation_max_punishment': args.adaptation_max_punishment,
        'punish_end_min_distance': args.punish_end_min_distance,
        'end_min_distance_max_threshold': args.end_min_distance_max_threshold,
        'end_min_distance_max_punishment': args.end_min_distance_max_punishment,
        'punish_end_max_torque': args.punish_end_max_torque,
        'end_max_torque_min_threshold': args.end_max_torque_min_threshold,
        'end_max_torque_max_punishment': args.end_max_torque_max_punishment,
        'obstacle_scene': args.obstacle_scene,
        'activate_obstacle_collisions': args.activate_obstacle_collisions,
        'log_obstacle_data': False,
        'check_braking_trajectory_collisions': args.check_braking_trajectory_collisions,
        'check_braking_trajectory_torque_limits': args.check_braking_trajectory_torque_limits,
        'collision_check_time': args.collision_check_time,
        'closest_point_safety_distance': args.closest_point_safety_distance,
        'use_target_points': args.use_target_points,
        'acc_limit_factor_braking': args.acc_limit_factor_braking,
        'jerk_limit_factor_braking': args.jerk_limit_factor_braking,
        'punish_braking_trajectory_min_distance': args.punish_braking_trajectory_min_distance,
        'braking_trajectory_min_distance_max_threshold': args.braking_trajectory_min_distance_max_threshold,
        'braking_trajectory_max_punishment': args.braking_trajectory_max_punishment,
        'punish_braking_trajectory_max_torque': args.punish_braking_trajectory_max_torque,
        'braking_trajectory_max_torque_min_threshold': args.braking_trajectory_max_torque_min_threshold,
        'robot_scene': args.robot_scene,
        'no_self_collision': args.no_self_collision,
        'terminate_on_robot_stop': args.terminate_on_robot_stop,
        'use_controller_target_velocities': args.use_controller_target_velocities,
        'starting_point_cartesian_range_scene': args.starting_point_cartesian_range_scene,
        'target_point_cartesian_range_scene': args.target_point_cartesian_range_scene,
        'target_point_relative_pos_scene': args.target_point_relative_pos_scene,
        'target_point_radius': args.target_point_radius,
        'target_point_sequence': args.target_point_sequence,
        'target_point_reached_reward_bonus': args.target_point_reached_reward_bonus,
        'target_point_reward_factor': args.target_point_reward_factor,
        'target_point_use_actual_position': args.target_point_use_actual_position,
        'normalize_reward_to_initial_target_point_distance': args.normalize_reward_to_initial_target_point_distance,
        'ball_machine_mode': args.ball_machine_mode,
        'use_moving_objects': args.use_moving_objects,
        'moving_object_sequence': args.moving_object_sequence,
        'moving_object_area_center': args.moving_object_area_center,
        'moving_object_area_width_height': args.moving_object_area_width_height,
        'moving_object_sphere_center': args.moving_object_sphere_center,
        'moving_object_sphere_radius': args.moving_object_sphere_radius,
        'moving_object_sphere_height_min_max': args.moving_object_sphere_height_min_max,
        'moving_object_sphere_angle_min_max': args.moving_object_sphere_angle_min_max,
        'moving_object_speed_meter_per_second': args.moving_object_speed_meter_per_second,
        'moving_object_active_number_single': args.moving_object_active_number_single,
        'moving_object_aim_at_current_robot_position': args.moving_object_aim_at_current_robot_position,
        'moving_object_check_invalid_target_link_point_positions':
            args.moving_object_check_invalid_target_link_point_positions,
        'moving_object_random_initial_position': args.moving_object_random_initial_position,
        'collision_avoidance_mode': args.collision_avoidance_mode,
        'collision_avoidance_kinematic_state_sampling_mode': args.collision_avoidance_kinematic_state_sampling_mode,
        'collision_avoidance_kinematic_state_sampling_probability':
            args.collision_avoidance_kinematic_state_sampling_probability,
        'collision_avoidance_stay_in_state_probability': args.collision_avoidance_stay_in_state_probability,
        'collision_avoidance_new_state_sample_time_range': args.collision_avoidance_new_state_sample_time_range,
        'collision_avoidance_self_collision_max_reward': args.collision_avoidance_self_collision_max_reward,
        'collision_avoidance_self_collision_max_reward_distance':
            args.collision_avoidance_self_collision_max_reward_distance,
        'collision_avoidance_static_obstacles_max_reward': args.collision_avoidance_static_obstacles_max_reward,
        'collision_avoidance_static_obstacles_max_reward_distance':
            args.collision_avoidance_static_obstacles_max_reward_distance,
        'collision_avoidance_moving_obstacles_max_reward': args.collision_avoidance_moving_obstacles_max_reward,
        'collision_avoidance_moving_obstacles_max_reward_distance':
            args.collision_avoidance_moving_obstacles_max_reward_distance,
        'collision_avoidance_low_acceleration_max_reward': args.collision_avoidance_low_acceleration_max_reward,
        'collision_avoidance_low_acceleration_threshold': args.collision_avoidance_low_acceleration_threshold,
        'collision_avoidance_low_velocity_max_reward': args.collision_avoidance_low_velocity_max_reward,
        'collision_avoidance_low_velocity_threshold': args.collision_avoidance_low_velocity_threshold,
        'collision_avoidance_episode_termination_bonus': args.collision_avoidance_episode_termination_bonus,
        'collision_avoidance_episode_early_termination_punishment':
            args.collision_avoidance_episode_early_termination_punishment,
        'terminate_on_self_collision': args.terminate_on_self_collision,
        'terminate_on_collision_with_static_obstacle': args.terminate_on_collision_with_static_obstacle,
        'terminate_on_collision_with_moving_obstacle': args.terminate_on_collision_with_moving_obstacle,
        'planet_mode': args.planet_mode,
        'planet_one_center': args.planet_one_center,
        'planet_one_radius_xy': args.planet_one_radius_xy,
        'planet_one_euler_angles': args.planet_one_euler_angles,
        'planet_one_period': args.planet_one_period,
        'planet_two_center': args.planet_two_center,
        'planet_two_radius_xy': args.planet_two_radius_xy,
        'planet_two_euler_angles': args.planet_two_euler_angles,
        'planet_two_period': args.planet_two_period,
        'planet_two_time_shift': args.planet_two_time_shift,
        'obs_planet_size_per_planet': args.obs_planet_size_per_planet,
        'risk_config_dir': args.risk_config_dir,
        'risk_threshold': args.risk_threshold,
        'risk_state_config': RISK_STATE_CONFIG[args.risk_state_config],
        'risk_state_backup_trajectory_steps': args.risk_state_backup_trajectory_steps,
        'risk_state_deterministic_backup_trajectory': args.risk_state_deterministic_backup_trajectory,
        'risk_check_initial_backup_trajectory': args.risk_check_initial_backup_trajectory,
        'risk_state_initial_backup_trajectory_steps': args.risk_state_initial_backup_trajectory_steps,
        'risk_use_backup_agent_for_initial_backup_trajectory_only':
            args.risk_use_backup_agent_for_initial_backup_trajectory_only,
        'risk_store_ground_truth': args.risk_store_ground_truth,
        'risk_ground_truth_episodes_per_file': args.risk_ground_truth_episodes_per_file,
        'risk_ignore_estimation_probability': args.risk_ignore_estimation_probability,
        'human_network_checkpoint': args.human_network_checkpoint,
        'human_network_use_full_observation': args.human_network_use_full_observation,
        'human_network_use_collision_avoidance_starting_point_sampling':
            args.human_network_use_collision_avoidance_starting_point_sampling,
        'human_network_collision_avoidance_kinematic_state_sampling_probability':
            args.human_network_collision_avoidance_kinematic_state_sampling_probability,
        'human_network_collision_avoidance_stay_in_state_probability':
            args.human_network_collision_avoidance_stay_in_state_probability,
        'target_link_offset': args.target_link_offset,
        'obstacle_use_computed_actual_values': args.obstacle_use_computed_actual_values,
        'solver_iterations': args.solver_iterations,
        'max_resampling_attempts': args.max_resampling_attempts,
        'logging_level': args.logging_level,
        'episodes_per_simulation_reset': args.episodes_per_simulation_reset,
        'use_gui': args.use_gui
    }

    if hasattr(klimits, '__version__'):
        env_config['klimits_version'] = klimits.__version__

    if hasattr(ray, '__version__'):
        env_config['ray_version'] = ray.__version__

    return env_config


if __name__ == '__main__':
    config = {
        'model': {
            'conv_filters': None,
            'fcnet_hiddens': [256, 128],
            'fcnet_activation': None,  # set at a later point
            'use_lstm': False,
        },
        'gamma': None,
        'use_gae': True,  # can be set to false with --no_use_gae
        'lambda': 1.0,
        'kl_coeff': 0.2,
        'rollout_fragment_length': None,  # set at a later point
        'train_batch_size': 49920,
        'sgd_minibatch_size': 1024,
        'num_sgd_iter': None,
        'lr': None,
        'lr_schedule': None,
        'vf_loss_coeff': 1.0,
        'entropy_coeff': None,
        'clip_param': 0.3,
        'vf_clip_param': None,
        'kl_target': None,
        'batch_mode': 'complete_episodes',
        'normalize_actions': False,
        'evaluation_interval': None,
        'evaluation_num_episodes': 624,
        'evaluation_parallel_to_training': False,
        'evaluation_config': {
            "explore": False,
            "rollout_fragment_length": 1},
        'evaluation_num_workers': 0,
    }

    algorithm = 'PPO'
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default_name')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint if training should be continued.')
    parser.add_argument('--time', type=int, required=True,
                        help='Total time of the training in hours.')
    parser.add_argument('--iterations_per_checkpoint', type=int, default=500,
                        help='The number of training iterations per checkpoint')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--pos_limit_factor', type=float, default=1.0)
    parser.add_argument('--vel_limit_factor', type=float, default=1.0)
    parser.add_argument('--acc_limit_factor', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0)
    parser.add_argument('--torque_limit_factor', type=float, default=1.0)
    parser.add_argument('--action_mapping_factor', type=float, default=1.0)
    parser.add_argument('--action_preprocessing_function', default=None, choices=['tanh'])
    parser.add_argument('--normalize_reward_to_frequency', dest='normalize_reward_to_frequency', action='store_true',
                        default=False)
    parser.add_argument('--batch_size_factor', type=float, default=1.0)
    parser.add_argument('--trajectory_duration', type=float, default=8.0)
    parser.add_argument('--trajectory_time_step', type=float, default=0.1)
    parser.add_argument('--obs_add_target_point_pos', action='store_true', default=False)
    parser.add_argument('--obs_add_target_point_relative_pos', action='store_true', default=False)
    parser.add_argument('--punish_action', action='store_true', default=False)
    parser.add_argument('--action_punishment_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_max_punishment', type=float, default=0.5)
    parser.add_argument('--punish_adaptation', action='store_true', default=False)
    parser.add_argument('--adaptation_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_end_min_distance', action='store_true', default=False)
    parser.add_argument('--end_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--end_min_distance_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_end_max_torque', action='store_true', default=False)
    parser.add_argument('--end_max_torque_min_threshold', type=float, default=0.9)
    parser.add_argument('--end_max_torque_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_min_distance', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--braking_trajectory_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_max_torque', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_max_torque_min_threshold', type=float, default=0.8)
    parser.add_argument('--obstacle_scene', type=int, default=0)
    parser.add_argument('--activate_obstacle_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--closest_point_safety_distance', type=float, default=0.1)
    parser.add_argument('--use_target_points', action='store_true', default=False)
    parser.add_argument('--acc_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--terminate_on_robot_stop', action='store_true', default=False)
    parser.add_argument('--use_controller_target_velocities', action='store_true', default=False)
    parser.add_argument('--starting_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--target_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--target_point_relative_pos_scene', type=int, default=0)
    parser.add_argument('--target_point_radius', type=float, default=0.065)
    parser.add_argument('--target_point_sequence', type=int, default=0)
    parser.add_argument('--target_point_reached_reward_bonus', type=float, default=0.00)
    parser.add_argument('--target_point_use_actual_position', action='store_true', default=False)
    parser.add_argument('--target_link_offset', type=json.loads, default=None)
    parser.add_argument('--target_point_reward_factor', type=float, default=1.0)
    parser.add_argument('--normalize_reward_to_initial_target_point_distance', action='store_true', default=False)
    parser.add_argument('--ball_machine_mode', action='store_true', default=False)
    # moving object settings
    parser.add_argument('--use_moving_objects', action='store_true', default=False)
    parser.add_argument('--moving_object_sequence', type=int, default=0)
    parser.add_argument('--moving_object_area_center', type=json.loads, default=None)
    parser.add_argument('--moving_object_area_width_height', type=json.loads, default=None)
    parser.add_argument('--moving_object_sphere_center', type=json.loads, default=None)
    parser.add_argument('--moving_object_sphere_radius', type=float, default=None)
    parser.add_argument('--moving_object_sphere_height_min_max', type=json.loads, default=None)
    parser.add_argument('--moving_object_sphere_angle_min_max', type=json.loads, default=None)
    parser.add_argument('--moving_object_speed_meter_per_second', type=float, default=1.0)
    parser.add_argument('--moving_object_active_number_single', type=int, default=1)
    parser.add_argument('--moving_object_aim_at_current_robot_position', action='store_true', default=False)
    parser.add_argument('--moving_object_check_invalid_target_link_point_positions', action='store_true', default=False)
    parser.add_argument('--moving_object_random_initial_position', action='store_true', default=False)
    # collision avoidance setting
    parser.add_argument('--collision_avoidance_mode', action='store_true', default=False)
    parser.add_argument('--collision_avoidance_kinematic_state_sampling_mode', action='store_true', default=False)
    parser.add_argument('--collision_avoidance_kinematic_state_sampling_probability', type=float, default=1.0)
    parser.add_argument('--collision_avoidance_stay_in_state_probability', type=float, default=0.3)
    parser.add_argument('--collision_avoidance_new_state_sample_time_range', type=json.loads, default=None)
    parser.add_argument('--collision_avoidance_self_collision_max_reward', type=float, default=0.0)
    parser.add_argument('--collision_avoidance_self_collision_max_reward_distance', type=float, default=0.05)
    parser.add_argument('--collision_avoidance_static_obstacles_max_reward', type=float, default=0.0)
    parser.add_argument('--collision_avoidance_static_obstacles_max_reward_distance', type=float, default=0.1)
    parser.add_argument('--collision_avoidance_moving_obstacles_max_reward', type=float, default=0.0)
    parser.add_argument('--collision_avoidance_moving_obstacles_max_reward_distance', type=float, default=0.3)
    parser.add_argument('--collision_avoidance_low_acceleration_max_reward', type=float, default=1.0)
    parser.add_argument('--collision_avoidance_low_acceleration_threshold', type=float, default=0.1)
    parser.add_argument('--collision_avoidance_low_velocity_max_reward', type=float, default=1.0)
    parser.add_argument('--collision_avoidance_low_velocity_threshold', type=float, default=0.1)
    parser.add_argument('--collision_avoidance_episode_termination_bonus', type=float, default=0.0)
    parser.add_argument('--collision_avoidance_episode_early_termination_punishment', type=float, default=0.0)
    parser.add_argument('--terminate_on_self_collision', action='store_true', default=False)
    parser.add_argument('--terminate_on_collision_with_static_obstacle', action='store_true', default=False)
    parser.add_argument('--terminate_on_collision_with_moving_obstacle', action='store_true', default=False)
    # end of collision avoidance setting
    # planet mode
    parser.add_argument('--planet_mode', action='store_true', default=False)
    parser.add_argument('--planet_one_center', type=json.loads, default=None)
    parser.add_argument('--planet_one_radius_xy', type=json.loads, default=None)
    parser.add_argument('--planet_one_euler_angles', type=json.loads, default=None)
    parser.add_argument('--planet_one_period', type=float, default=None)
    parser.add_argument('--planet_two_center', type=json.loads, default=None)
    parser.add_argument('--planet_two_radius_xy', type=json.loads, default=None)
    parser.add_argument('--planet_two_euler_angles', type=json.loads, default=None)
    parser.add_argument('--planet_two_period', type=float, default=None)
    parser.add_argument('--planet_two_time_shift', type=float, default=None)
    parser.add_argument('--obs_planet_size_per_planet', type=int, default=1)
    # end of planet mode
    # risk network settings
    parser.add_argument('--risk_config_dir', type=str, default=None)
    parser.add_argument('--risk_threshold', type=float, default=None)
    parser.add_argument("--risk_state_config", default='RISK_CHECK_CURRENT_STATE',
                        choices=['RISK_CHECK_CURRENT_STATE', 'RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING',
                                 'RISK_CHECK_NEXT_STATE_FULL_FORECASTING', 'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP',
                                 'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY'])
    parser.add_argument('--risk_state_backup_trajectory_steps', type=int, default=None)
    parser.add_argument('--risk_state_deterministic_backup_trajectory', action='store_true', default=False)
    parser.add_argument('--risk_check_initial_backup_trajectory', action='store_true', default=False)
    parser.add_argument('--risk_state_initial_backup_trajectory_steps', type=int, default=None)
    parser.add_argument('--risk_use_backup_agent_for_initial_backup_trajectory_only', action='store_true',
                        default=False)
    parser.add_argument('--risk_store_ground_truth', action='store_true', default=False)
    parser.add_argument('--risk_ground_truth_episodes_per_file', type=int, default=None)
    parser.add_argument('--risk_ignore_estimation_probability', type=float, default=0.0)
    # end of risk network settings
    # human network settings
    parser.add_argument('--human_network_checkpoint', type=str, default=None)
    parser.add_argument('--human_network_use_full_observation', action='store_true', default=False)
    parser.add_argument('--human_network_use_collision_avoidance_starting_point_sampling',
                        action='store_true', default=False)
    parser.add_argument('--human_network_collision_avoidance_kinematic_state_sampling_probability',
                        type=float, default=0.3)
    parser.add_argument('--human_network_collision_avoidance_stay_in_state_probability', type=float, default=0.3)
    # end of human network settings
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--logging_level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--hidden_layer_activation', default='selu', choices=['relu', 'selu', 'tanh', 'sigmoid', 'elu',
                                                                              'gelu', 'swish', 'leaky_relu'])
    parser.add_argument('--last_layer_activation', default=None, choices=['linear', 'tanh'])
    parser.add_argument('--no_log_std_activation', action='store_true', default=False)
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--max_resampling_attempts', type=int, default=0)
    parser.add_argument('--use_dashboard', action='store_true', default=False)
    parser.add_argument('--evaluation_interval', type=int, default=None)
    parser.add_argument('--vf_clip_param', type=float, default=10.0)
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--kl_target', type=float, default=0.01)
    parser.add_argument('--fcnet_hiddens', type=json.loads, default=None)
    parser.add_argument('--vf_hiddens', type=json.loads, default=None)
    parser.add_argument('--log_std_range', type=json.loads, default=None)
    parser.add_argument('--no_use_gae', action='store_true', default=False)
    parser.add_argument('--num_sgd_iter', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--clip_param', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--episodes_per_simulation_reset', type=int, default=None)
    parser.add_argument('--use_gui', action='store_true', default=False)

    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)

    config['model']['fcnet_activation'] = args.hidden_layer_activation
    if args.fcnet_hiddens is not None:
        config['model']['fcnet_hiddens'] = args.fcnet_hiddens
    config['evaluation_interval'] = args.evaluation_interval
    config['vf_clip_param'] = args.vf_clip_param
    config['entropy_coeff'] = args.entropy_coeff
    config['kl_target'] = args.kl_target
    config['num_sgd_iter'] = args.num_sgd_iter
    config['lr'] = args.lr
    config['gamma'] = args.gamma

    if args.no_use_gae:
        config['use_gae'] = False

    if args.last_layer_activation is not None and args.last_layer_activation != 'linear':
        use_keras_model = True
        if use_keras_model:
            from safemotions.model.keras_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('keras_fcnet_last_layer_activation',
                                               FullyConnectedNetworkLastLayerActivation)
            config['model']['custom_model'] = 'keras_fcnet_last_layer_activation'
        else:
            from safemotions.model.fcnet_v2_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
            config['model']['custom_model'] = 'fcnet_last_layer_activation'

        config['model']['custom_model_config'] = {'last_layer_activation': args.last_layer_activation,
                                                  'no_log_std_activation': args.no_log_std_activation,
                                                  'log_std_range': args.log_std_range,
                                                  'vf_hiddens': args.vf_hiddens}
        if use_keras_model:
            for key in ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation',
                        'no_final_layer', 'vf_share_layers', 'free_log_std']:
                if key in config['model']:
                    config['model']['custom_model_config'][key] = config['model'][key]

    if args.action_preprocessing_function == "tanh":
        config['normalize_actions'] = False
        config['clip_actions'] = False

    if args.checkpoint is not None:
        if not os.path.isdir(args.checkpoint) and not os.path.isfile(args.checkpoint):
            checkpoint_path = os.path.join(current_dir, 'trained_networks', args.checkpoint)
        else:
            checkpoint_path = args.checkpoint

        if os.path.isdir(checkpoint_path):
            if os.path.basename(checkpoint_path) == 'checkpoint':
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint')
            else:
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint', 'checkpoint')

        if not os.path.isfile(checkpoint_path):
            raise ValueError('Could not find checkpoint {}'.format(checkpoint_path))

        params_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        params_path = os.path.join(params_dir, 'params.json')

        with open(params_path) as params_file:
            checkpoint_config = json.load(params_file)
        config['env_config'] = checkpoint_config['env_config']
        config['train_batch_size'] = checkpoint_config['train_batch_size']
        config['sgd_minibatch_size'] = checkpoint_config['sgd_minibatch_size']

    else:
        checkpoint_path = None
        config['env_config'] = _make_env_config()
        config['train_batch_size'] = int(config['train_batch_size'] * args.batch_size_factor)
        config['sgd_minibatch_size'] = int(config['sgd_minibatch_size'] * args.batch_size_factor)

    if args.logdir is None:
        experiment_path = config['env_config']['experiment_name']
    else:
        experiment_path = os.path.join(args.logdir, config['env_config']['experiment_name'])

    if args.seed is not None:
        config['seed'] = args.seed
        config['env_config']['seed'] = args.seed
        np.random.seed(args.seed)

    if args.num_workers is None:
        config['num_workers'] = int(multiprocessing.cpu_count() * 0.75)
    else:
        config['num_workers'] = args.num_workers

    local_mode = False

    if config['num_workers'] == 0:
        local_mode = True

    config['rollout_fragment_length'] = int(config['train_batch_size'] / max(config['num_workers'], 1))

    # define number of threads per worker for parallel execution based on OpenMP
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    if config['env_config']['collision_avoidance_mode']:
        from safemotions.envs.safe_motions_env import SafeMotionsEnvCollisionAvoidance as Env
    else:
        from safemotions.envs.safe_motions_env import SafeMotionsEnv as Env

    env_name = Env.__name__
    config.update(env=env_name)
    tune.register_env(env_name, lambda config_args: Env(**config_args))

    ray.init(dashboard_host='0.0.0.0', include_dashboard=args.use_dashboard, ignore_reinit_error=True,
             logging_level=args.logging_level, local_mode=local_mode)
    config['callbacks'] = CustomTrainCallbacks

    if args.max_resampling_attempts > 0:
        config['callbacks'] = ResampleUnsafeActionsCallbacks

    if args.num_gpus is not None:
        config['num_gpus'] = args.num_gpus

    stop = {'time_total_s': args.time * 3600}

    experiment = {
        experiment_path: {
            'run': algorithm,
            'env': env_name,
            'stop': stop,
            'config': config,
            'checkpoint_freq': args.iterations_per_checkpoint,
            'checkpoint_at_end': True,
            'keep_checkpoints_num': 10,
            'max_failures': 0,
            'restore': checkpoint_path
        }
    }

    tune.run_experiments(experiment, callbacks=[TBXLoggerCallback()])

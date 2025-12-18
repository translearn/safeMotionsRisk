#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os

# limit the number of threads per worker for parallel execution to one
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)

os.environ['RAY_DEDUP_LOGS'] = str(0)  # no log deduplication
os.environ['RAY_AIR_NEW_OUTPUT'] = str(0)  # old (more detailed) cli logging


import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import TBXLoggerCallback  # tensorboard fix ray 1.5 https://github.com/ray-project/ray/issues/17366
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from torch import Tensor
import multiprocessing
import logging
import pathlib
import klimits

from safemotions.common_functions import (register_envs, get_checkpoint_path_and_config,
                                          get_model_config_and_rl_module_spec_from_checkpoint_config,
                                          betas_tensor_to_float)
from safemotions.callbacks import CustomTrainCallbacks


RISK_STATE_CONFIG = {'RISK_CHECK_CURRENT_STATE': 0,
                     'RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING': 1,
                     'RISK_CHECK_NEXT_STATE_FULL_FORECASTING': 2,
                     'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP': 3,
                     'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY': 4}


'''
def resample_wrapper(method, env, policy):

    def resample_wrapping_method(*args, **kwargs):
        initial_time_step = policy.global_timestep
        for i in range(env.max_resampling_attempts + 1):
            if i > 0:
                policy.global_timestep = initial_time_step

            return_value = method(*args, **kwargs)

            action = np.clip(return_value[0][0], env.action_space.low, env.action_space.high)
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
'''

def _determine_env():
    if args.collision_avoidance_mode:
        env = 'SafeMotionsEnvCollisionAvoidance'
    else:
        env = 'SafeMotionsEnv'

    return env

def store_args(path, args):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "arguments.json"), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True))
        f.flush()

def _get_num_gpus_per_learner(num_gpus, num_gpus_per_env_runner, num_env_runners):

    if num_gpus_per_env_runner * num_env_runners > num_gpus:
        raise ValueError("Cannot assign {} * {} GPUs to the workers with args.num_gpus == {}".format(
            num_env_runners, num_gpus_per_env_runner, num_gpus))

    num_gpus_per_learner = num_gpus - num_gpus_per_env_runner * num_env_runners

    if num_gpus_per_env_runner != int(num_gpus_per_env_runner) or num_gpus_per_learner != int(num_gpus_per_learner):
        raise ValueError("Fractional GPUs currently not supported.")

    return num_gpus_per_learner


def _make_ppo_config(config_dict=None):

    if config_dict is None:  # initial training run or checkpoint_config=="from_args"
        if args.num_workers is None:
            num_env_runners = int(multiprocessing.cpu_count() * 0.75)
        else:
            num_env_runners = args.num_workers

        num_gpus = args.num_gpus if args.num_gpus is not None else 0
        num_gpus_per_env_runner = args.num_gpus_per_worker if args.num_gpus_per_worker is not None else 0.0

        num_gpus_per_learner = _get_num_gpus_per_learner(num_gpus, num_gpus_per_env_runner, num_env_runners)

        config_dict = {'env': _determine_env(),
                       'env_config': _make_env_config(),
                       'num_gpus_per_learner': num_gpus_per_learner,
                       'num_gpus_per_env_runner': num_gpus_per_env_runner,
                       '_train_batch_size_per_learner': int(49920 * args.batch_size_factor),
                       'minibatch_size': int(1024 * args.batch_size_factor),
                       'num_env_runners': num_env_runners,
                       'clip_actions': True if args.action_preprocessing_function != "tanh" else False,
                       'gamma': args.gamma,
                       'lr': args.lr,
                       'use_critic': not args.no_critic,
                       'use_gae': not args.no_use_gae,
                       'lambda': args.lambda_,
                       # setting for gae, 0.0: high bias - low variance, 1.0: low bias, high variance
                       'use_kl_loss': not args.no_kl_loss,
                       'kl_coeff': args.kl_coeff,
                       'kl_target': args.kl_target,
                       'vf_loss_coeff': args.vf_loss_coeff,
                       'entropy_coeff': args.entropy_coeff,
                       'clip_param': args.clip_param,
                       'vf_clip_param': args.vf_clip_param}

        config_dict['rollout_fragment_length']=int(config_dict['_train_batch_size_per_learner']
                                                   / max(num_env_runners, 1))

        # specify rl module
        use_custom_model = False
        fcnet_hiddens = args.fcnet_hiddens if args.fcnet_hiddens is not None else [256, 128]

        pi_head_fcnet_hiddens = args.pi_head_fcnet_hiddens
        vf_head_fcnet_hiddens = args.vf_head_fcnet_hiddens

        if pi_head_fcnet_hiddens != vf_head_fcnet_hiddens or args.last_layer_activation != 'linear':
            use_custom_model = True

        if use_custom_model:
            # load a custom model to gain more control about the activation functions
            from safemotions.model.custom_ppo_torch_rl_module import CustomPPOTorchRLModule

            config_dict['rl_module_spec'] = RLModuleSpec(
                module_class=CustomPPOTorchRLModule,
                model_config={'fcnet_hiddens': fcnet_hiddens,
                              'fcnet_activation': args.hidden_layer_activation,
                              'head_fcnet_activation': args.hidden_layer_activation,
                              'log_std_clip_param':args.log_std_clip_param,
                              'log_std_range': args.log_std_range,
                              'pi_head_last_layer_activation': args.last_layer_activation,
                              'pi_head_no_log_std_activation': args.no_log_std_activation,
                              'pi_head_fcnet_hiddens': pi_head_fcnet_hiddens,
                              'vf_head_fcnet_hiddens': vf_head_fcnet_hiddens,
                              'vf_share_layers': args.vf_share_layers
                              },
            )
            config_dict['model_config'] = NotProvided

        else:
            # use default model config

            config_dict['rl_module_spec'] = NotProvided
            config_dict['model_config'] = DefaultModelConfig(fcnet_hiddens=fcnet_hiddens,
                                                             fcnet_activation=args.hidden_layer_activation,
                                                             head_fcnet_hiddens=pi_head_fcnet_hiddens,
                                                             log_std_clip_param=args.log_std_clip_param,
                                                             vf_share_layers=args.vf_share_layers)

    else:
        # load model config from checkpoint
        config_dict['model_config'], config_dict['rl_module_spec'] = (
            get_model_config_and_rl_module_spec_from_checkpoint_config(
                model_config_str=config_dict['_model_config'],
                rl_module_spec_str=config_dict['_rl_module_spec']))

        # some parameters are taken from args if provided:
        num_gpus = args.num_gpus if args.num_gpus is not None else (
                config_dict['num_gpus_per_learner'] +
                config_dict['num_gpus_per_env_runner'] * config_dict['num_env_runners'])

        if args.num_gpus_per_env_runner is not None:
            config_dict['num_gpus_per_env_runner'] = args.num_gpus_per_env_runner

        if args.num_workers is not None:
            num_workers = args.num_workers
            if config_dict['num_env_runners'] != num_workers:
                config_dict['rollout_fragment_length'] = (
                    int(config_dict['_train_batch_size_per_learner'] / max(args.num_workers, 1)))
                config_dict['num_env_runners'] = args.num_workers

        config_dict['num_gpus_per_learner'] = _get_num_gpus_per_learner(num_gpus,
                                                                        config_dict['num_gpus_per_env_runner'],
                                                                        config_dict['num_env_runners'])

        if args.logging_level is not None:
            config_dict['env_config']['logging_level'] = args.logging_level

        if args.name is not None:
            config_dict['env_config']['experiment_name'] = (
                    config_dict['env_config']['experiment_name'] + "_" + args.name)


    ppo_config = (
        PPOConfig()
        .environment(
            env=config_dict['env'],
            env_config=config_dict['env_config'],
            clip_rewards=False,
            normalize_actions=False,
            clip_actions=config_dict['clip_actions'],
            disable_env_checking=True,
            is_atari=False,
        )
        .env_runners(
            num_env_runners=config_dict['num_env_runners'],
            create_local_env_runner=True,
            create_env_on_local_worker=False,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=config_dict['num_gpus_per_env_runner'],
            rollout_fragment_length=config_dict['rollout_fragment_length'],
            batch_mode='complete_episodes',
            explore=True,
        )
        .learners(
            num_learners=0,
            num_cpus_per_learner=NotProvided,
            num_gpus_per_learner=config_dict['num_gpus_per_learner'],
            num_aggregator_actors_per_learner=NotProvided,
            max_requests_in_flight_per_aggregator_actor=NotProvided,
            local_gpu_idx=NotProvided,
            max_requests_in_flight_per_learner=NotProvided,
        )
        .training(
            gamma=config_dict['gamma'],
            lr=config_dict['lr'],
            grad_clip=NotProvided,
            grad_clip_by=NotProvided,
            train_batch_size_per_learner=config_dict['_train_batch_size_per_learner'],
            num_epochs=NotProvided,
            minibatch_size=config_dict['minibatch_size'],
            shuffle_batch_per_epoch=NotProvided,
            learner_class=NotProvided,
            learner_connector=NotProvided,
            add_default_connectors_to_learner_pipeline=NotProvided,
            learner_config_dict=NotProvided,
            # PPO specific settings
            use_critic=config_dict['use_critic'],
            use_gae=config_dict['use_gae'],
            lambda_=config_dict['lambda'],
            # setting for gae, 0.0: high bias - low variance, 1.0: low bias, high variance
            use_kl_loss=config_dict['use_kl_loss'],
            kl_coeff=config_dict['kl_coeff'],
            kl_target=config_dict['kl_target'],
            vf_loss_coeff=config_dict['vf_loss_coeff'],
            entropy_coeff=config_dict['entropy_coeff'],
            clip_param=config_dict['clip_param'],
            vf_clip_param=config_dict['vf_clip_param'],
            )
        .rl_module(
            model_config=config_dict['model_config'],
            rl_module_spec=config_dict['rl_module_spec']
        )
    )

    return ppo_config

def _make_env_config():

    env_config = {
        'experiment_name': args.name if args.name is not None else "no_experiment_name_specified",
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
        'risk_network_use_gpu': args.risk_network_use_gpu,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--logging_level', default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    # training: general settings
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint if training should be continued.')
    parser.add_argument('--checkpoint_config_from_args', action='store_true', default=False)
    parser.add_argument('--time', type=int, default=1000,
                        help='Total time of the training in hours.')
    parser.add_argument('--iterations_per_checkpoint', type=int, default=50,
                        help='The number of training iterations per checkpoint')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--num_gpus_per_worker', type=float, default=None)
    parser.add_argument('--use_dashboard', action='store_true', default=False)
    parser.add_argument('--evaluation_interval', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--overwrite_tuner_file', action='store_true', default=False)
    parser.add_argument('--num_checkpoints_to_keep', type=int, default=5)
    # training: hyperparameters
    parser.add_argument('--batch_size_factor', type=float, default=1.0)
    parser.add_argument('--lambda_', type=float, default=1.0)
    parser.add_argument('--no_critic', action='store_true', default=False)
    parser.add_argument('--no_kl_loss', action='store_true', default=False)
    parser.add_argument('--kl_coeff', type=float, default=0.2)
    parser.add_argument('--vf_clip_param', type=float, default=10.0)
    parser.add_argument('--vf_loss_coeff', type=float, default=1.0)
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--kl_target', type=float, default=0.01)
    parser.add_argument('--no_use_gae', action='store_true', default=False)
    parser.add_argument('--num_sgd_iter', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--clip_param', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.99)
    # training: network settings
    parser.add_argument('--hidden_layer_activation', default='selu', choices=['relu', 'selu', 'tanh', 'sigmoid', 'elu',
                                                                              'gelu', 'swish', 'leaky_relu'])
    parser.add_argument('--last_layer_activation', default='linear', choices=['linear', 'tanh'])
    parser.add_argument('--no_log_std_activation', action='store_true', default=False)
    parser.add_argument('--fcnet_hiddens', type=json.loads, default=None)
    parser.add_argument('--pi_head_fcnet_hiddens', type=json.loads, default=[])
    parser.add_argument('--vf_head_fcnet_hiddens', type=json.loads, default=[])
    parser.add_argument('--vf_share_layers', action='store_true', default=False)
    parser.add_argument('--log_std_clip_param', type=float, default=20) # for linear activation
    parser.add_argument('--log_std_range', type=json.loads, default=None) # for tanh activation
    # env config: pybullet
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--episodes_per_simulation_reset', type=int, default=None)
    parser.add_argument('--use_gui', action='store_true', default=False)
    # env config: robot joint limits
    parser.add_argument('--pos_limit_factor', type=float, default=1.0)
    parser.add_argument('--vel_limit_factor', type=float, default=1.0)
    parser.add_argument('--acc_limit_factor', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0)
    parser.add_argument('--torque_limit_factor', type=float, default=1.0)
    # env config: trajectory generation and control
    parser.add_argument('--trajectory_duration', type=float, default=8.0)
    parser.add_argument('--trajectory_time_step', type=float, default=0.1)
    parser.add_argument('--use_controller_target_velocities', action='store_true', default=False)
    # env config: robot, obstacle and collision settings
    parser.add_argument('--obstacle_scene', type=int, default=0)
    parser.add_argument('--activate_obstacle_collisions', action='store_true', default=False)
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--ball_machine_mode', action='store_true', default=False)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--target_link_offset', type=json.loads, default=None)
    parser.add_argument('--starting_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--terminate_on_robot_stop', action='store_true', default=False)
    # env config: action settings (general)
    parser.add_argument('--action_mapping_factor', type=float, default=1.0)
    parser.add_argument('--action_preprocessing_function', default=None, choices=['tanh'])
    parser.add_argument('--max_resampling_attempts', type=int, default=0)
    # env config: reward settings (general)
    parser.add_argument('--normalize_reward_to_frequency', dest='normalize_reward_to_frequency', action='store_true',
                        default=False)
    parser.add_argument('--punish_action', action='store_true', default=False)
    parser.add_argument('--action_punishment_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_max_punishment', type=float, default=0.5)
    # env config: reaching task general settings
    parser.add_argument('--use_target_points', action='store_true', default=False)
    parser.add_argument('--target_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--target_point_relative_pos_scene', type=int, default=0)
    parser.add_argument('--target_point_radius', type=float, default=0.065)
    parser.add_argument('--target_point_sequence', type=int, default=0)
    parser.add_argument('--target_point_reached_reward_bonus', type=float, default=0.00)
    parser.add_argument('--target_point_use_actual_position', action='store_true', default=False)
    # env config: reaching task observation settings
    parser.add_argument('--obs_add_target_point_pos', action='store_true', default=False)
    parser.add_argument('--obs_add_target_point_relative_pos', action='store_true', default=False)
    # env config: reaching task reward settings
    parser.add_argument('--target_point_reward_factor', type=float, default=1.0)
    parser.add_argument('--normalize_reward_to_initial_target_point_distance', action='store_true', default=False)
    # env config: braking trajectories as backup trajectories general settings
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--acc_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--closest_point_safety_distance', type=float, default=0.1)
    # env config: braking trajectories as backup trajectories reward settings
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
    # planet mode settings
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
    parser.add_argument('--risk_network_use_gpu', action='store_true', default=False)
    # human network settings
    parser.add_argument('--human_network_checkpoint', type=str, default=None)
    parser.add_argument('--human_network_use_full_observation', action='store_true', default=False)
    parser.add_argument('--human_network_use_collision_avoidance_starting_point_sampling',
                        action='store_true', default=False)
    parser.add_argument('--human_network_collision_avoidance_kinematic_state_sampling_probability',
                        type=float, default=0.3)
    parser.add_argument('--human_network_collision_avoidance_stay_in_state_probability', type=float, default=0.3)

    args = parser.parse_args()

    register_envs()

    storage_path = args.logdir

    if args.checkpoint is not None:
        checkpoint_path, checkpoint_config = get_checkpoint_path_and_config(args.checkpoint)
        default_storage_path = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))

        # default config
        if args.checkpoint_config_from_args: # new config from args
            ppo_config = _make_ppo_config(config_dict=None)

        else:
            # load checkpoint config from params.json file
            ppo_config = _make_ppo_config(config_dict=checkpoint_config)

        def load_checkpoint_on_algorithm_init(algorithm, **kwargs):
            algorithm.restore_from_path(checkpoint_path)
            algorithm.learner_group.foreach_learner(betas_tensor_to_float)

        ppo_config.callbacks(on_algorithm_init=load_checkpoint_on_algorithm_init)
        # add checkpoint name to experiment name
        checkpoint_name = os.path.basename(checkpoint_path)
        ppo_config.env_config['experiment_name'] = ppo_config.env_config['experiment_name'] + "_" + checkpoint_name

    else:
        default_storage_path = os.path.join(pathlib.Path.home(), "ray_results")
        ppo_config = _make_ppo_config()

    if storage_path is None:
        storage_path = default_storage_path
    elif os.path.abspath(storage_path) != storage_path:
        storage_path = os.path.join(default_storage_path, storage_path)

    # Check if an experiment with the same name already exists at the given storage path.
    # While Tune does not throw an error if two experiments have the same name, Tune will override the
    # tuner.pkl file required to resume the experiment if the name is not unique.
    experiment_path = os.path.join(storage_path, ppo_config.env_config['experiment_name'])

    if os.path.isdir(experiment_path) and os.listdir(experiment_path) and not args.overwrite_tuner_file:
        raise FileExistsError("There is already an experiment with the same name at the specified "
                              "storage path. "
                              "To continue, select a different name, add the argument --overwrite_tuner_file or "
                              "delete the content of the directory {}.".format(experiment_path))

    # store args
    store_args(experiment_path, args)

    num_workers = ppo_config.num_env_runners
    num_gpus = ppo_config.num_gpus_per_learner

    if ppo_config.env_config['logging_level'] is None:
        ppo_config.env_config['logging_level'] = 'WARNING'

    logging.basicConfig()
    logging.getLogger().setLevel(ppo_config.env_config['logging_level'])
    ppo_config.debugging(log_level=ppo_config.env_config['logging_level'])

    if args.seed is None:
        ppo_config.debugging(seed=None)
        ppo_config['env_config']['seed'] = None
    else:
        ppo_config.debugging(seed=args.seed)
        np.random.seed(args.seed)
        ppo_config['env_config']['seed'] = args.seed

    ppo_config.callbacks(CustomTrainCallbacks)

    # fix to avoid "beta1 as a Tensor is not supported for capturable=False and foreach=True" error when
    # restoring checkpoints using tuner.restore (in resume_training.py) with num_gpu != 0
    # see https://github.com/ray-project/ray/issues/51560

    def on_checkpoint_loaded(algorithm, **kwargs) -> None:
        def betas_tensor_to_float(learner):
            param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
            if not param_grp['capturable'] and isinstance(param_grp["betas"][0], Tensor):
                param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)

    ppo_config.callbacks(on_checkpoint_loaded=on_checkpoint_loaded)

    if num_workers == 0:
        local_mode = True
        if num_gpus != 0:
            raise ValueError("GPUs are not supported with --num_workers=0. Set --num_gpus=0 or --num_workers>=1 "
                             "to continue!")
    else:
        local_mode = False

    ray.init(dashboard_host='0.0.0.0', include_dashboard=args.use_dashboard, ignore_reinit_error=True,
             num_gpus=num_gpus, logging_level=logging.INFO, local_mode=local_mode)

    # if args.max_resampling_attempts > 0:
    #    config['callbacks'] = ResampleUnsafeActionsCallbacks

    stop = {'time_total_s': args.time * 3600}

    results = tune.Tuner(
        PPO,
        param_space=ppo_config,
        run_config=tune.RunConfig(
            name=ppo_config.env_config['experiment_name'],
            storage_path=storage_path,
            stop=stop,
            verbose=3,
            callbacks=[TBXLoggerCallback()],
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=args.num_checkpoints_to_keep,
                checkpoint_frequency=args.iterations_per_checkpoint,
                checkpoint_at_end=True),
            log_to_file=True),
        tune_config=None).fit()

    ray.shutdown()

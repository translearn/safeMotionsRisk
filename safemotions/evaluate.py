# This file makes use of code from
# https://github.com/ray-project/ray/blob/master/rllib/examples/inference/policy_inference_after_training_w_connector.py
# Copyright 2023 Ray Authors
# Licensed under the Apache License, Version 2.0;

import argparse
import json
import os
import time

os.environ['RAY_DEDUP_LOGS'] = str(0)  # no log deduplication
os.environ['RAY_AIR_NEW_OUTPUT'] = str(0)  # old (more detailed) cli logging

import sys
import inspect
import logging
import datetime
import ray
import numpy as np
from pathlib import Path
from collections import defaultdict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.core.columns import Columns

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

from safemotions.common_functions import (register_envs, get_checkpoint_path_and_config,
                                          get_model_config_and_rl_module_spec_from_checkpoint_config,
                                          betas_tensor_to_float, store_metrics, make_dir,
                                          make_metrics_dir_and_store_args,
                                          store_args, store_env_config)

from safemotions.callbacks import CustomTrainCallbacks
import klimits

RENDERER = {'opengl': 0,
            'egl': 1,
            'cpu': 2,
            'imagegrab': 3}

RISK_STATE_CONFIG = {'RISK_CHECK_CURRENT_STATE': 0,
                     'RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING': 1,
                     'RISK_CHECK_NEXT_STATE_FULL_FORECASTING': 2,
                     'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP': 3,
                     'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY': 4}

METRIC_OPS = ['sum', 'mean', 'max', 'min']


class CustomEvaluationCallbacks(CustomTrainCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        episode.custom_data['start_time'] = time.time()
        if args.store_metrics:
            super().on_episode_start(episode=episode, **kwargs)

    def on_episode_step(self, *, episode, **kwargs):
        if args.store_metrics:
            super().on_episode_step(episode=episode, **kwargs)

    def on_episode_end(self, *, episode, env, metrics_logger, **kwargs):
        episode_computation_time = time.time() - episode.custom_data['start_time']
        env = env.envs[-1].unwrapped
        print("Computing episode {} took {} seconds".format(env.episode_counter, episode_computation_time))
        last_info = episode.infos[-1]
        episode_length = last_info['episode_length']
        print("Trajectory duration: {} seconds".format(episode_length * env.trajectory_time_step))
        episode_return = np.sum(episode.get_rewards())
        print("Episode return: {}".format(episode_return))
        if args.store_metrics:
            store_metrics(env.evaluation_dir, env.use_real_robot, env.pid,
                          env.episode_counter, episode_return, last_info, episode.custom_data['op'])

def update_env_config():  # update env_config based on the script parameters
    if args.name is not None:
        env_config['experiment_name'] = args.name

    env_config['logging_level'] = args.logging_level

    if args.render:
        if args.num_workers == 0:
            env_config.update(render_video=True)
        env_config['camera_angle'] = args.camera_angle
        env_config['renderer'] = RENDERER[args.renderer]
        env_config['render_no_shadows'] = args.render_no_shadows
        env_config['video_frame_rate'] = args.video_frame_rate
        env_config['video_height'] = args.video_height
        env_config['video_dir'] = args.video_dir
        env_config['fixed_video_filename'] = args.fixed_video_filename
        env_config['video_add_text'] = args.video_add_text
        if args.fixed_video_filename and args.num_workers is not None and args.num_workers >= 2:
            raise ValueError("fixed_video_filename requires num_workers < 2")

    else:
        env_config.update(render_video=False)

    if args.num_workers == 0:
        env_config.update(use_gui=args.use_gui)
    else:
        env_config.update(use_gui=False)  # overwritten in evaluation config
        # no gui for local env created to infer the observation space and action space
    env_config.update(use_real_robot=args.use_real_robot)
    env_config['time_stamp'] = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    env_config['evaluation_dir'] = evaluation_dir

    if args.store_actions:
        env_config['store_actions'] = True

    if args.store_trajectory:
        env_config['store_trajectory'] = True

    if args.log_obstacle_data:
        env_config['log_obstacle_data'] = True

    if args.save_trajectory_plot:
        env_config['save_trajectory_plot'] = True

    if args.switch_gui_to_obstacle_client:
        env_config['switch_gui_to_obstacle_client'] = True

    if args.switch_gui_to_backup_client:
        env_config['switch_gui_to_backup_client'] = True

    if args.plot_actual_torques:
        env_config['plot_actual_torques'] = True

    if args.use_thread_for_movement:
        env_config['use_thread_for_movement'] = True

    if args.use_process_for_movement:
        env_config['use_process_for_movement'] = True

    if args.no_use_control_rate_sleep:
        env_config['use_control_rate_sleep'] = False

    if args.obstacle_use_computed_actual_values:
        env_config['obstacle_use_computed_actual_values'] = True

    if args.obstacle_scene is not None:
        env_config['obstacle_scene'] = args.obstacle_scene

    if args.trajectory_duration is not None:
        env_config['trajectory_duration'] = args.trajectory_duration

    if args.plot_trajectory:

        env_config['plot_trajectory'] = True

        if args.plot_acc_limits:
            env_config['plot_acc_limits'] = True

        if args.plot_actual_values:
            env_config['plot_actual_values'] = True

        if args.plot_computed_actual_values:
            env_config['plot_computed_actual_values'] = True

        if args.plot_joint is not None:
            env_config['plot_joint'] = args.plot_joint

        if args.plot_value:
            env_config['plot_value'] = args.plot_value

    if args.random_agent:
        env_config['random_agent'] = True

    if args.collision_check_time is not None:
        env_config['collision_check_time'] = args.collision_check_time

    if args.torque_limit_factor is not None:
        env_config['torque_limit_factor'] = args.torque_limit_factor

    if args.solver_iterations:
        env_config['solver_iterations'] = args.solver_iterations

    if args.visualize_debug_lines:
        env_config['visualize_debug_lines'] = args.visualize_debug_lines

    if args.real_robot_debug_mode:
        env_config['real_robot_debug_mode'] = True

    if args.no_self_collision:
        env_config['no_self_collision'] = True

    if args.time_step_fraction_sleep_observation is not None:
        env_config['time_step_fraction_sleep_observation'] = args.time_step_fraction_sleep_observation

    if args.control_time_step is not None:
        env_config['control_time_step'] = args.control_time_step

    if args.moving_object_no_random_initial_position:
        env_config['moving_object_random_initial_position'] = False

    if args.collision_avoidance_kinematic_state_sampling_probability is not None:
        env_config['collision_avoidance_kinematic_state_sampling_probability'] = \
            args.collision_avoidance_kinematic_state_sampling_probability

    if args.collision_avoidance_stay_in_state_probability is not None:
        env_config['collision_avoidance_stay_in_state_probability'] = \
            args.collision_avoidance_stay_in_state_probability

    if args.collision_avoidance_new_state_sample_time_range is not None:
        env_config['collision_avoidance_new_state_sample_time_range'] = \
            args.collision_avoidance_new_state_sample_time_range

    if args.no_link_coloring:
        env_config['no_link_coloring'] = True

    if args.risk_config_dir is not None:
        env_config['risk_config_dir'] = args.risk_config_dir

    if args.risk_threshold is not None:
        env_config['risk_threshold'] = args.risk_threshold

    if args.risk_state_config is not None:
        env_config['risk_state_config'] = RISK_STATE_CONFIG[args.risk_state_config]

    if args.risk_state_backup_trajectory_steps is not None:
        env_config['risk_state_backup_trajectory_steps'] = args.risk_state_backup_trajectory_steps

    if args.risk_state_deterministic_backup_trajectory:
        env_config['risk_state_deterministic_backup_trajectory'] = True

    if args.risk_store_ground_truth:
        env_config['risk_store_ground_truth'] = True

    if args.risk_ground_truth_episodes_per_file is not None:
        env_config['risk_ground_truth_episodes_per_file'] = args.risk_ground_truth_episodes_per_file

    if args.risk_ignore_estimation_probability is not None:
        env_config['risk_ignore_estimation_probability'] = args.risk_ignore_estimation_probability


    if ('risk_store_ground_truth' in env_config and env_config['risk_store_ground_truth']) and \
            ('risk_config_dir' not in env_config or env_config['risk_config_dir'] is None) \
            and args.risk_state_config == 'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY':
        # use network at checkpoint_path to generate risk data and create risk_config for that purpose
        env_config["risk_config"] = {"action_size": None,
                                     "checkpoint": checkpoint_path,
                                     "observation_size": None}


    if args.visualize_risk:
        env_config['visualize_risk'] = True

    if args.episodes_per_simulation_reset is not None:
        env_config['episodes_per_simulation_reset'] = args.episodes_per_simulation_reset

    if args.add_value_debug_text:
        env_config['add_value_debug_text'] = True

    if args.static_robot:
        env_config['static_robot'] = True

    if args.max_resampling_attempts is not None:
        env_config['max_resampling_attempts'] = args.max_resampling_attempts

    env_config['seed'] = args.seed

    if args.check_braking_trajectory_collisions:
        env_config['check_braking_trajectory_collisions'] = True

    if args.check_braking_trajectory_torque_limits:
        env_config['check_braking_trajectory_torque_limits'] = True


def update_checkpoint_config():
    if args.num_gpus is not None:
        checkpoint_config['num_gpus_per_learner'] = args.num_gpus

    checkpoint_config['rollout_fragment_length'] = 1

def get_ppo_config():
    def load_checkpoint_on_algorithm_init(algorithm, **kwargs):
        algorithm.restore_from_path(checkpoint_path)
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)

    # load model config from checkpoint
    checkpoint_config['model_config'], checkpoint_config['rl_module_spec'] = (
        get_model_config_and_rl_module_spec_from_checkpoint_config(
            model_config_str=checkpoint_config['_model_config'],
            rl_module_spec_str=checkpoint_config['_rl_module_spec'], inference_only=True))

    ppo_config = (
        PPOConfig()
        .environment(
            env=checkpoint_config['env'],
            env_config=checkpoint_config['env_config'],
            clip_rewards=False,
            normalize_actions=False,
            clip_actions=checkpoint_config['clip_actions'],
            disable_env_checking=True,
            is_atari=False,
        )
        .env_runners(
            num_env_runners=0,
            create_local_env_runner=True,
            create_env_on_local_worker=True if args.num_workers == 0 else False,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=checkpoint_config['rollout_fragment_length'],
            batch_mode='complete_episodes',
            explore=not args.no_exploration,
        )
        .evaluation(
            evaluation_num_env_runners=args.num_workers,
            evaluation_duration=max(args.num_workers, 1) * args.episodes,
            evaluation_duration_unit="episodes",
            evaluation_config=AlgorithmConfig.overrides(
                env_config={"use_gui": args.use_gui,
                            "render_video": args.render},
            offline_eval_rl_module_inference_only=True,
            ),

        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=checkpoint_config['num_gpus_per_learner'],
        )
        .rl_module(
            model_config=checkpoint_config['model_config'],
            rl_module_spec=checkpoint_config['rl_module_spec']
        )
        .callbacks(
            on_algorithm_init=load_checkpoint_on_algorithm_init
        )
        .debugging(
            seed=args.seed
        )
    )

    if args.num_workers >= 1:
        ppo_config.callbacks(CustomEvaluationCallbacks)

    return ppo_config


def check_package_versions():
    if 'ray_version' in env_config and hasattr(ray, '__version__'):
        if env_config['ray_version'] != ray.__version__:
            logging.warning('This network was trained with ray=={} but you are using ray=={}'.format(
                env_config['ray_version'], ray.__version__))

    if 'klimits_version' in env_config and hasattr(klimits, '__version__'):
        if env_config['klimits_version'] != klimits.__version__:
            logging.warning('This network was trained with klimits=={} but you are using klimits=={}'.format(
                env_config['klimits_version'], klimits.__version__))

def make_dirs_and_store_args(env_evaluation_dir):
    if args.store_logs:
        make_dir(env_evaluation_dir)
        file_handler = logging.FileHandler(os.path.join(env_evaluation_dir, "logs"))
        file_handler.setFormatter(logging_formatter)
        file_handler.setLevel(args.logging_level)
        logging.getLogger().addHandler(file_handler)
    if args.store_metrics:
        make_metrics_dir_and_store_args(env_evaluation_dir, args.use_real_robot, args)
    if args.store_trajectory or args.render or \
            ('risk_store_ground_truth' in env_config and env_config['risk_store_ground_truth']) or args.store_logs:
        store_env_config(env_evaluation_dir, checkpoint_config["env_config"])
        store_args(env_evaluation_dir, args)

def rollout_single_worker_manually():

    env = algo.env_runner_group.local_env_runner.env.unwrapped.envs[0].unwrapped

    # use the connectors from env_runner
    env_to_module = algo.env_runner._env_to_module
    module_to_env = algo.env_runner._module_to_env
    # alternatively, the connectors can be build from ppo_config
    # env_to_module = ppo_config.build_env_to_module_connector(env=env)
    # module_to_env = ppo_config.build_module_to_env_connector(env=env)
    # or loaded from the checkpoint directory (see policy_inference_after_training_w_connector.py)

    rl_module = algo.get_module()
    make_dirs_and_store_args(env_evaluation_dir=env.evaluation_dir)

    episodes_sampled = 0
    episode_return_list = []
    episode_computation_time_list = []
    episode_control_phase_list = []
    episode_trajectory_duration_list = []
    start = time.time()

    while True:
        if args.episodes:
            if episodes_sampled >= args.episodes:
                break

        obs, _ = env.reset()
        reward_total = 0.0
        episode_info = {}
        info = {}
        step = -1

        episode = SingleAgentEpisode(
            observations=[obs],
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        start_episode_timer = time.time()
        while not episode.is_done:
            step = step + 1
            shared_data = {}
            input_dict = env_to_module(
                episodes=[episode],
                rl_module=rl_module,
                explore=not args.no_exploration,
                shared_data=shared_data,
            )

            if args.no_exploration:
                rl_module_out = rl_module.forward_inference(input_dict)
            else:
                rl_module_out = rl_module.forward_exploration(input_dict)

            output_dict = module_to_env(
                batch=rl_module_out,
                episodes=[episode],
                rl_module=rl_module,
                explore=not args.no_exploration,
                shared_data=shared_data,
            )

            action = output_dict.pop(Columns.ACTIONS_FOR_ENV)[0]

            if args.store_metrics:
                if not episode_info:
                    for op in METRIC_OPS:
                        episode_info[op] = defaultdict(list)

                obs, reward, terminated, truncated, info = env.step(action)

                for op in list(info.keys() & METRIC_OPS):
                    for k, v in info[op].items():
                        episode_info[op][k].append(v)

            else:
                obs, reward, terminated, truncated, _ = env.step(action)

            episode.add_env_step(
                obs,
                action,
                reward,
                terminated=terminated,
                truncated=truncated,
                extra_model_outputs={k: v[0] for k, v in output_dict.items()},
            )

            reward_total += reward

        end_episode_timer = time.time()
        episode_computation_time = end_episode_timer - start_episode_timer
        logging.info("Computing episode %s took %s seconds", episodes_sampled + 1, episode_computation_time)
        episode_return_list.append(reward_total)
        episode_computation_time_list.append(episode_computation_time)
        trajectory_duration = (step + 1) * env.trajectory_time_step
        episode_trajectory_duration_list.append(trajectory_duration)
        if env.precomputation_timer is not None:
            control_phase_duration = end_episode_timer - env.precomputation_timer
            logging.info("Trajectory duration: %s seconds. Control phase: %s seconds.",
                         trajectory_duration, control_phase_duration)
            episode_control_phase_list.append(control_phase_duration)
        else:
            logging.info("Trajectory duration: %s seconds", (step + 1) * env.trajectory_time_step)
        logging.info("Episode return: %s", reward_total)
        episodes_sampled += 1

        if args.store_metrics:
            store_metrics(env.evaluation_dir, env.use_real_robot, env.pid, episodes_sampled, reward_total,
                          info, episode_info)

    end = time.time()
    logging.info("Computed %s episode(s) in %s seconds.", len(episode_computation_time_list), end - start)
    logging.info("Mean episode return: {:.2f}".format(np.mean(episode_return_list)))
    logging.info("Mean computation time: %s seconds, Max computation time: %s seconds.",
                 np.mean(episode_computation_time_list),
                 np.max(episode_computation_time_list))
    computation_time_fraction = np.array(episode_computation_time_list) * 100 / \
                                np.array(episode_trajectory_duration_list)
    logging.info("Mean computation time fraction: %s %%, Max computation time fraction: %s %%.",
                 np.mean(computation_time_fraction),
                 np.max(computation_time_fraction))
    if episode_control_phase_list:
        control_rate_fraction = np.array(episode_control_phase_list) / np.array(episode_trajectory_duration_list)
        max_relative_delay = (np.max(control_rate_fraction) - 1) * 100
        mean_relative_fraction = np.mean(control_rate_fraction) * 100
        logging.info("Mean control phase: %s seconds, Max control phase: %s seconds. Mean fraction: %s percent, "
                     "Max delay: %s percent.",
                     np.mean(episode_control_phase_list),
                     np.max(episode_control_phase_list),
                     mean_relative_fraction,
                     max_relative_delay)

def rollout_multiple_workers():
    if args.use_real_robot:
        raise ValueError("--use_real_robot requires --num_workers=0")

    env_evaluation_dir = algo.env_runner_group.local_env_runner.env.unwrapped.envs[0].unwrapped.evaluation_dir
    make_dirs_and_store_args(env_evaluation_dir=env_evaluation_dir)
    algo.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help="The name of the evaluation.")
    parser.add_argument('--evaluation_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the checkpoint for evaluation.")
    parser.add_argument('--episodes', type=int, default=20,
                        help="The number of episodes for evaluation per worker.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--use_real_robot', action='store_true', default=None)
    parser.add_argument('--real_robot_debug_mode', dest='real_robot_debug_mode', action='store_true', default=False)
    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--switch_gui_to_obstacle_client', action='store_true', default=False)
    parser.add_argument('--switch_gui_to_backup_client', action='store_true', default=False)
    parser.add_argument('--trajectory_duration', type=float, default=None)
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--store_metrics', action='store_true', default=False)
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--save_trajectory_plot', action='store_true', default=False)
    parser.add_argument('--plot_acc_limits', action='store_true', default=False)
    parser.add_argument('--plot_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_actual_torques', action='store_true', default=False)
    parser.add_argument('--plot_value', action='store_true', default=False)
    parser.add_argument('--plot_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_joint', type=json.loads, default=None)
    parser.add_argument('--visualize_debug_lines', action='store_true', default=False)
    # moving object settings
    parser.add_argument('--moving_object_no_random_initial_position', action='store_true', default=False)
    # end of moving object settings
    # collision avoidance settings
    parser.add_argument('--collision_avoidance_kinematic_state_sampling_probability', type=float, default=None)
    parser.add_argument('--collision_avoidance_stay_in_state_probability', type=float, default=None)
    parser.add_argument('--collision_avoidance_new_state_sample_time_range', type=json.loads, default=None)
    # end of collision avoidance settings
    # risk settings
    parser.add_argument('--risk_config_dir', type=str, default=None)
    parser.add_argument('--risk_threshold', type=float, default=None)
    parser.add_argument("--risk_state_config", default=None,
                        choices=['RISK_CHECK_CURRENT_STATE', 'RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING',
                                 'RISK_CHECK_NEXT_STATE_FULL_FORECASTING', 'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP',
                                 'RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY'])
    parser.add_argument('--risk_state_backup_trajectory_steps', type=int, default=None)
    parser.add_argument('--risk_state_deterministic_backup_trajectory', action='store_true', default=False)
    parser.add_argument('--risk_store_ground_truth', action='store_true', default=False)
    parser.add_argument('--risk_ground_truth_episodes_per_file', type=int, default=None)
    parser.add_argument('--risk_ignore_estimation_probability', type=float, default=0.0)
    parser.add_argument('--visualize_risk', action='store_true', default=False)
    # end of risk settings
    parser.add_argument('--torque_limit_factor', type=float, default=None)
    parser.add_argument('--store_actions', action='store_true', default=False)
    parser.add_argument('--store_trajectory', action='store_true', default=False)
    parser.add_argument('--store_logs', action='store_true', default=False)
    parser.add_argument('--no_exploration', action='store_true', default=False)
    parser.add_argument('--log_obstacle_data', action='store_true', default=False)
    parser.add_argument('--obstacle_scene', type=int, default=None)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--random_agent', action='store_true', default=False)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--use_thread_for_movement', action='store_true', default=False)
    parser.add_argument('--use_process_for_movement', action='store_true', default=False)
    parser.add_argument('--no_use_control_rate_sleep', action='store_true', default=False)
    parser.add_argument('--control_time_step', type=float, default=None)
    parser.add_argument('--time_step_fraction_sleep_observation', type=float, default=None)
    parser.add_argument("--logging_level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--no_link_coloring', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False,
                        help="If set, videos of the generated episodes are recorded.")
    parser.add_argument("--renderer", default='opengl', choices=['opengl', 'egl', 'cpu', 'imagegrab'])
    parser.add_argument('--render_no_shadows', action='store_true', default=False)
    parser.add_argument('--camera_angle', type=int, default=0)
    parser.add_argument('--video_frame_rate', type=float, default=None)
    parser.add_argument('--video_height', type=int, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--video_add_text', action='store_true', default=False)
    parser.add_argument('--fixed_video_filename', action='store_true', default=False)
    parser.add_argument('--add_value_debug_text', action='store_true', default=False)
    parser.add_argument('--static_robot', action='store_true', default=False)
    parser.add_argument('--max_resampling_attempts', type=int, default=None)
    parser.add_argument('--episodes_per_simulation_reset', type=int, default=None)
    parser.add_argument('--use_dashboard', action='store_true', default=False)

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    if args.render and args.renderer == 'egl':
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s'
    logging_formatter = logging.Formatter(logging_format)
    logging.basicConfig(format=logging_format)
    logging.getLogger().setLevel(args.logging_level)

    if args.evaluation_dir is None:
        evaluation_dir = os.path.join(Path.home(), "safe_motions_risk_evaluation")
    else:
        evaluation_dir = os.path.join(args.evaluation_dir, "safe_motions_risk_evaluation")

    checkpoint_path, checkpoint_config = get_checkpoint_path_and_config(args.checkpoint)

    env_config = checkpoint_config['env_config']

    check_package_versions()
    update_env_config()
    update_checkpoint_config()

    ppo_config = get_ppo_config()
    register_envs()

    if args.seed is not None:
        np.random.seed(args.seed)

    if checkpoint_config['num_gpus_per_learner'] == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    ray.init(dashboard_host='0.0.0.0', include_dashboard=args.use_dashboard, ignore_reinit_error=True,
             num_gpus=checkpoint_config['num_gpus_per_learner'], logging_level=logging.INFO,
             local_mode=args.num_workers == 0)

    algo = ppo_config.build_algo()

    if args.num_workers > 0:
        rollout_multiple_workers()
    else:
        rollout_single_worker_manually()

    algo.stop()

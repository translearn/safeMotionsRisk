# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import logging
import os
import time
from abc import abstractmethod
from functools import partial
from multiprocessing import Pool
import json
import errno
import tempfile
import random
import copy
import inspect
from collections import defaultdict
from itertools import chain
from pathlib import Path
from threading import Thread

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pybullet as p

from safemotions.robot_scene.real_robot_scene import RealRobotScene
from safemotions.robot_scene.simulated_robot_scene import SimRobotScene
from safemotions.utils.control_rate import ControlRate
from safemotions.utils.trajectory_manager import TrajectoryManager

torch = None  # import later if needed

SIM_TIME_STEP = 1. / 240.
CONTROLLER_TIME_STEP = 1. / 200.
EPISODES_PER_SIMULATION_RESET = 12500  # to avoid out of memory error

RISK_GROUND_TRUTH_EPISODES_PER_FILE = 100

# Renderer
OPENGL_GUI_RENDERER = 0
OPENGL_EGL_RENDERER = 1
CPU_TINY_RENDERER = 2

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

def resampling_decorator(func):
    def resampling_wrapper(self, action):
        action_double = np.array(action, dtype=np.float64)
        action_is_safe = func(self, action_double)

        if action_is_safe:
            self._safe_action = action_double
        else:
            self._resampling_attempts = self._resampling_attempts + 1
            
        return action_is_safe

    return resampling_wrapper


class SafeMotionsBase(gym.Env):
    # Termination reason
    TERMINATION_UNSET = -1
    TERMINATION_SUCCESS = 0  # unused
    TERMINATION_JOINT_LIMITS = 1
    TERMINATION_TRAJECTORY_LENGTH = 2
    TERMINATION_SELF_COLLISION = 3
    TERMINATION_COLLISION_WITH_STATIC_OBSTACLE = 4
    TERMINATION_COLLISION_WITH_MOVING_OBSTACLE = 5

    # State-based risk option
    RISK_CHECK_CURRENT_STATE = 0  # check the risk of the current state (without considering the selected action)
    RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING = 1  # check the risk of the next state considering the kinematic state
    # resulting from the selected action
    RISK_CHECK_NEXT_STATE_FULL_FORECASTING = 2  # additionally forecast the non-kinematic components of the next risk
    # observation e.g., position of moving obstacles (requires extended knowledge about the environment)
    RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP = 3  # run a background simulation to simulate the next time step
    # considering the selected action. If the simulation does not lead to a safety violation, use a risk network to
    # estimate the risk of the subsequent state
    RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY = 4  # run a background simulation to simulate the
    # next time step considering and a subsequent backup trajectory using a backup agent. This option does not require
    # a risk network. However, knowledge about the future development of the environment is required.

    def __init__(self,
                 experiment_name,
                 time_stamp=None,
                 evaluation_dir=None,
                 use_real_robot=False,
                 real_robot_debug_mode=False,
                 use_gui=False,
                 switch_gui_to_obstacle_client=False,
                 switch_gui_to_backup_client=False,
                 control_time_step=None,
                 use_control_rate_sleep=True,
                 use_thread_for_movement=False,
                 use_process_for_movement=False,
                 pos_limit_factor=1,
                 vel_limit_factor=1,
                 acc_limit_factor=1,
                 jerk_limit_factor=1,
                 torque_limit_factor=1,
                 acceleration_after_max_vel_limit_factor=0.01,
                 eval_new_condition_counter=1,
                 log_obstacle_data=False,
                 save_obstacle_data=False,
                 store_actions=False,
                 store_trajectory=False,
                 trajectory_duration=8.0,
                 trajectory_time_step=0.1,
                 position_controller_time_constants=None,
                 plot_computed_actual_values=False,
                 plot_actual_torques=False,
                 plot_value=False,
                 robot_scene=0,
                 obstacle_scene=0,
                 activate_obstacle_collisions=False,
                 observed_link_point_scene=0,
                 obstacle_use_computed_actual_values=False,
                 visualize_bounding_spheres=False,
                 visualize_debug_lines=False,
                 check_braking_trajectory_collisions=False,
                 collision_check_time=None,
                 distance_calculation_check_observed_points=False,
                 check_braking_trajectory_torque_limits=False,
                 closest_point_safety_distance=0.1,
                 observed_point_safety_distance=0.1,
                 starting_point_cartesian_range_scene=0,
                 use_target_points=False,
                 target_point_cartesian_range_scene=0,
                 target_point_relative_pos_scene=0,
                 target_point_radius=0.05,
                 target_point_sequence=0,
                 target_point_reached_reward_bonus=0.0,
                 target_point_use_actual_position=False,
                 obs_add_target_point_pos=False,
                 obs_add_target_point_relative_pos=False,
                 use_moving_objects=False,
                 moving_object_sequence=0,
                 moving_object_area_center=None,
                 moving_object_area_width_height=None,
                 moving_object_sphere_center=None,  # e.g [0, 0, 0]
                 moving_object_sphere_radius=None,
                 moving_object_sphere_height_min_max=None,  # e.g. [-0.5, 2]
                 moving_object_sphere_angle_min_max=None,  # e.g. [0. 2 * np.pi]
                 moving_object_speed_meter_per_second=1.0,
                 moving_object_aim_at_current_robot_position=False,
                 moving_object_check_invalid_target_link_point_positions=False,
                 moving_object_active_number_single=1,
                 moving_object_random_initial_position=False,
                 planet_mode=False,
                 planet_one_center=None,
                 planet_one_radius_xy=None,
                 planet_one_euler_angles=None,
                 planet_one_period=None,
                 planet_two_center=None,
                 planet_two_radius_xy=None,
                 planet_two_euler_angles=None,
                 planet_two_period=None,
                 planet_two_time_shift=None,
                 ball_machine_mode=False,
                 terminate_on_self_collision=False,
                 terminate_on_collision_with_static_obstacle=False,
                 terminate_on_collision_with_moving_obstacle=False,
                 collision_avoidance_mode=False,
                 collision_avoidance_kinematic_state_sampling_mode=False,
                 collision_avoidance_kinematic_state_sampling_probability=1.0,
                 collision_avoidance_stay_in_state_probability=0.3,
                 collision_avoidance_new_state_sample_time_range=None,
                 always_use_collision_avoidance_starting_point_sampling=False,   # even without collision avoidance mode
                 risk_config=None,
                 risk_config_dir=None,
                 risk_threshold=None,
                 risk_state_config=RISK_CHECK_CURRENT_STATE,
                 risk_state_backup_trajectory_steps=None,
                 risk_check_initial_backup_trajectory=False,
                 risk_state_initial_backup_trajectory_steps=None,
                 risk_use_backup_agent_for_initial_backup_trajectory_only=False,
                 risk_state_deterministic_backup_trajectory=False,
                 risk_store_ground_truth=False,
                 risk_ground_truth_episodes_per_file=None,
                 risk_ignore_estimation_probability=0.0,
                 risk_network_use_gpu=False,
                 visualize_risk=False,
                 human_network_checkpoint=None,
                 human_network_use_full_observation=False,
                 human_network_use_collision_avoidance_starting_point_sampling=False,
                 human_network_collision_avoidance_kinematic_state_sampling_probability=0.3,
                 human_network_collision_avoidance_stay_in_state_probability=0.3,
                 target_link_name=None,
                 target_link_offset=None,
                 no_self_collision=False,
                 no_link_coloring=False,
                 no_target_link_coloring=False,
                 terminate_on_robot_stop=False,
                 use_controller_target_velocities=False,
                 time_step_fraction_sleep_observation=0.0,
                 seed=None,
                 solver_iterations=None,
                 logging_level="WARNING",
                 add_value_debug_text=False,
                 random_agent=False,
                 do_not_execute_robot_movement=False,
                 max_resampling_attempts=0,
                 static_robot=False,
                 episodes_per_simulation_reset=None,
                 physic_clients_dict=None,
                 **kwargs):

        self._risk_config_dir = risk_config_dir
        if evaluation_dir is None:
            evaluation_dir = os.path.join(Path.home(), "safe_motions_risk_evaluation")
        self._time_stamp = time_stamp
        if logging_level is not None:
            logging.getLogger().setLevel(logging_level)

        self._initial_seed = seed
        self._seed = None
        self.set_seed(seed)

        if self._time_stamp is None:
            self._time_stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        if risk_threshold is None:
            risk_threshold = 0.05

        self._experiment_name = experiment_name
        self._evaluation_dir = os.path.join(evaluation_dir, self.__class__.__name__,
                                            self._experiment_name, self._time_stamp)
        self._pid = os.getpid()

        if solver_iterations is None:
            self._solver_iterations = 150
        else:
            self._solver_iterations = solver_iterations

        self._target_link_name = target_link_name
        self._use_real_robot = use_real_robot
        self._use_gui = use_gui
        self._switch_gui_to_obstacle_client = switch_gui_to_obstacle_client
        self._switch_gui_to_backup_client = switch_gui_to_backup_client
        self._use_control_rate_sleep = use_control_rate_sleep
        self._num_physic_clients = 0
        self._gui_client_id = None

        if control_time_step is None:
            self._control_time_step = CONTROLLER_TIME_STEP if self._use_real_robot else SIM_TIME_STEP
        else:
            self._control_time_step = control_time_step

        self._simulation_time_step = SIM_TIME_STEP
        self._control_step_counter = 0
        self._episode_counter = 0

        self._obstacle_scene = obstacle_scene
        self._activate_obstacle_collisions = activate_obstacle_collisions
        self._observed_link_point_scene = observed_link_point_scene
        self._visualize_bounding_spheres = visualize_bounding_spheres
        self._visualize_debug_lines = visualize_debug_lines
        self._log_obstacle_data = log_obstacle_data
        self._save_obstacle_data = save_obstacle_data
        self._robot_scene_index = robot_scene
        self._check_braking_trajectory_collisions = check_braking_trajectory_collisions
        self._collision_check_time = collision_check_time
        self._distance_calculation_check_observed_points = distance_calculation_check_observed_points
        self._check_braking_trajectory_torque_limits = check_braking_trajectory_torque_limits
        self._closest_point_safety_distance = closest_point_safety_distance
        self._observed_point_safety_distance = observed_point_safety_distance
        self._starting_point_cartesian_range_scene = starting_point_cartesian_range_scene
        self._use_target_points = use_target_points
        self._target_point_cartesian_range_scene = target_point_cartesian_range_scene
        self._target_point_relative_pos_scene = target_point_relative_pos_scene
        self._target_point_radius = target_point_radius
        self._target_point_sequence = target_point_sequence
        self._target_point_reached_reward_bonus = target_point_reached_reward_bonus
        self._target_point_use_actual_position = target_point_use_actual_position
        self._obs_add_target_point_pos = obs_add_target_point_pos
        self._obs_add_target_point_relative_pos = obs_add_target_point_relative_pos
        self._use_moving_objects = use_moving_objects
        self._moving_object_sequence = moving_object_sequence
        self._moving_object_area_center = moving_object_area_center
        self._moving_object_area_width_height = moving_object_area_width_height
        self._moving_object_sphere_center = moving_object_sphere_center
        self._moving_object_sphere_radius = moving_object_sphere_radius
        self._moving_object_sphere_height_min_max = moving_object_sphere_height_min_max
        self._moving_object_sphere_angle_min_max = moving_object_sphere_angle_min_max
        self._moving_object_speed_meter_per_second = moving_object_speed_meter_per_second
        self._moving_object_aim_at_current_robot_position = moving_object_aim_at_current_robot_position
        self._moving_object_check_invalid_target_link_point_positions = \
            moving_object_check_invalid_target_link_point_positions
        self._moving_object_active_number_single = moving_object_active_number_single
        self._moving_object_random_initial_position = moving_object_random_initial_position
        self._planet_mode = planet_mode
        self._planet_one_center = planet_one_center
        self._planet_one_radius_xy = planet_one_radius_xy
        self._planet_one_euler_angles = planet_one_euler_angles
        self._planet_one_period = planet_one_period
        self._planet_two_center = planet_two_center
        self._planet_two_radius_xy = planet_two_radius_xy
        self._planet_two_euler_angles = planet_two_euler_angles
        self._planet_two_period = planet_two_period
        self._planet_two_time_shift = planet_two_time_shift
        self._ball_machine_mode = ball_machine_mode
        self._terminate_on_self_collision = terminate_on_self_collision
        self._terminate_on_collision_with_static_obstacle = terminate_on_collision_with_static_obstacle
        self._terminate_on_collision_with_moving_obstacle = terminate_on_collision_with_moving_obstacle

        self._collision_avoidance_mode = collision_avoidance_mode
        self._collision_avoidance_kinematic_state_sampling_mode = collision_avoidance_kinematic_state_sampling_mode
        self._collision_avoidance_kinematic_state_sampling_probability = \
            collision_avoidance_kinematic_state_sampling_probability
        self._collision_avoidance_stay_in_state_probability = collision_avoidance_stay_in_state_probability
        self._always_use_collision_avoidance_starting_point_sampling = \
            always_use_collision_avoidance_starting_point_sampling

        self._no_self_collision = no_self_collision
        self._no_link_coloring = no_link_coloring
        self._no_target_link_coloring = no_target_link_coloring
        self._terminate_on_robot_stop = terminate_on_robot_stop
        self._use_controller_target_velocities = use_controller_target_velocities
        self._trajectory_time_step = trajectory_time_step
        self._position_controller_time_constants = position_controller_time_constants
        self._plot_computed_actual_values = plot_computed_actual_values
        self._plot_actual_torques = plot_actual_torques
        self._plot_value = plot_value
        self._pos_limit_factor = pos_limit_factor
        self._vel_limit_factor = vel_limit_factor
        self._acc_limit_factor = acc_limit_factor
        self._jerk_limit_factor = jerk_limit_factor
        self._torque_limit_factor = torque_limit_factor
        self._acceleration_after_max_vel_limit_factor = acceleration_after_max_vel_limit_factor
        self._trajectory_duration = trajectory_duration
        self._eval_new_condition_counter = eval_new_condition_counter
        self._store_actions = store_actions
        self._store_trajectory = store_trajectory
        self._target_link_offset = target_link_offset
        self._real_robot_debug_mode = real_robot_debug_mode
        self._random_agent = random_agent
        self._do_not_execute_robot_movement = do_not_execute_robot_movement  # for the nested human env
        self._max_resampling_attempts = max_resampling_attempts
        self._static_robot = static_robot
        self._add_value_debug_text = add_value_debug_text
        self._value_debug_text = None

        self._risk_threshold = risk_threshold
        self._risk_state_config = risk_state_config
        self._risk_state_backup_trajectory_steps = risk_state_backup_trajectory_steps
        # the number of decision steps the backup trajectory should be checked for. If None, the number of steps is
        # derived from the risk config
        self._risk_check_initial_backup_trajectory = risk_check_initial_backup_trajectory
        # if true, the reset of the environment is repeated until a safe backup trajectory with a length of
        # self._risk_state_backup_trajectory_steps steps is found.
        self._risk_state_initial_backup_trajectory_steps = risk_state_initial_backup_trajectory_steps
        # the number of decision steps the initial backup trajectory should be checked for.
        # If None, risk_state_backup_trajectory_steps is used
        self._risk_use_backup_agent_for_initial_backup_trajectory_only = \
            risk_use_backup_agent_for_initial_backup_trajectory_only
        # use the backup agent to get a safe initial robot state but not to replace unsafe actions.
        # This mode is required for benchmarking to make sure that the initial state of the robot is selected according
        # to the same criteria - no matter whether risk estimation is used or not
        self._risk_state_deterministic_backup_trajectory = risk_state_deterministic_backup_trajectory
        self._risk_store_ground_truth = risk_store_ground_truth
        if risk_ground_truth_episodes_per_file is None:
            self._risk_ground_truth_episodes_per_file = RISK_GROUND_TRUTH_EPISODES_PER_FILE
        else:
            self._risk_ground_truth_episodes_per_file = risk_ground_truth_episodes_per_file

        self._risk_ignore_estimation_probability = risk_ignore_estimation_probability
        self._risk_network_use_gpu = risk_network_use_gpu

        self._visualize_risk = visualize_risk

        self._human_network_checkpoint = human_network_checkpoint
        self._human_network_use_full_observation = human_network_use_full_observation
        self._human_network_use_collision_avoidance_starting_point_sampling = \
            human_network_use_collision_avoidance_starting_point_sampling
        self._human_network_collision_avoidance_kinematic_state_sampling_probability = \
            human_network_collision_avoidance_kinematic_state_sampling_probability
        self._human_network_collision_avoidance_stay_in_state_probability = \
            human_network_collision_avoidance_stay_in_state_probability

        self._risk_config = risk_config
        self._risk_observation = None
        self._backup_agent = None
        if self._risk_config_dir is not None or self._risk_config is not None:
            self._load_risk_config()

        self._physic_clients_dict = physic_clients_dict
        self._init_physic_clients()

        self._episodes_per_simulation_reset = episodes_per_simulation_reset \
            if episodes_per_simulation_reset is not None else EPISODES_PER_SIMULATION_RESET

        self._collision_avoidance_new_state_sample_step_range = None
        self._collision_avoidance_new_state_sample_step = None

        if self._collision_avoidance_mode and collision_avoidance_new_state_sample_time_range is not None:
            self._collision_avoidance_new_state_sample_step_range = \
                [int(np.floor(collision_avoidance_new_state_sample_time_range[0] / self._trajectory_time_step)),
                 int(np.ceil(collision_avoidance_new_state_sample_time_range[1] / self._trajectory_time_step))]

        self._network_prediction_part_done = None
        self._use_thread_for_movement = use_thread_for_movement
        self._use_process_for_movement = use_process_for_movement
        if self._use_thread_for_movement and self._use_process_for_movement:
            raise ValueError("use_thread_for_movement and use_process_for_movement are not "
                             "allowed to be True simultaneously")
        self._use_movement_thread_or_process = self._use_thread_for_movement or self._use_process_for_movement
        if self._use_movement_thread_or_process and not self._use_control_rate_sleep:
            logging.warning("use_movement_thread_or_process without use_control_rate_sleep == True")
        if self._use_real_robot and not self._use_movement_thread_or_process:
            raise ValueError("use_real_robot requires either use_thread_for_movement or use_process_for_movement")
        if self._real_robot_debug_mode and not self._use_real_robot:
            raise ValueError("real_robot_debug_mode requires use_real_robot")

        self._time_step_fraction_sleep_observation = time_step_fraction_sleep_observation
        # 0..1; fraction of the time step,  the main thread sleeps before getting the next observation;
        # only relevant if self._use_real_robot == True
        if time_step_fraction_sleep_observation != 0:
            logging.info("time_step_fraction_sleep_observation %s", self._time_step_fraction_sleep_observation)
        self._obstacle_use_computed_actual_values = obstacle_use_computed_actual_values
        # use computed actual values to determine the distance between the robot and obstacles and as initial point
        # for torque simulations -> advantage: can be computed in advance, no measurements -> real-time capable
        # disadvantage: controller model might introduce inaccuracies
        if self._use_movement_thread_or_process and not self._obstacle_use_computed_actual_values:
            raise ValueError("Real-time execution requires obstacle_use_computed_actual_values to be True")

        if self._use_movement_thread_or_process:
            if self._use_thread_for_movement:
                logging.info("Using movement thread")
            else:
                logging.info("Using movement process")

        if self._use_process_for_movement:
            self._movement_process_pool = Pool(processes=1)
        else:
            self._movement_process_pool = None

        self._model_actual_values = self._use_movement_thread_or_process or self._obstacle_use_computed_actual_values \
            or self._plot_computed_actual_values or (self._use_real_robot and self._use_gui)

        if not self._use_movement_thread_or_process and self._control_time_step != self._simulation_time_step:
            raise ValueError("If no movement thread or process is used, the control time step must equal the control "
                             "time step of the obstacle client")

        self._start_position = None
        self._start_velocity = None
        self._start_acceleration = None
        self._end_acceleration = None
        self._position_deviation = None
        self._acceleration_deviation = None
        self._current_trajectory_point_index = None
        self._trajectory_successful = None
        self._total_reward = None
        self._episode_length = None
        self._action_list = []
        self._last_action = None
        self._termination_reason = self.TERMINATION_UNSET
        self._movement_thread = None
        self._movement_process = None
        self._brake = False

        self._adaptation_punishment = None
        self._end_min_distance = None
        self._end_max_torque = None  # for (optional) reward calculations
        self._punish_end_max_torque = False  # set in rewards.py

        self._moving_object_hit_robot_or_obstacle = None
        self._moving_object_missed_robot = None

        self._self_collision_detected = None
        self._collision_with_static_obstacle_detected = None
        self._collision_with_moving_obstacle_detected = None

        self._resampling_attempts = None
        self._safe_action = None
        self._last_risk_prediction = None
        self._precomputation_timer = None

        self._init_simulation()

        if self._gui_client_id is not None:
            # deactivate rendering temporarily to reduce the computational effort for the additional process that
            # ray spawns to detect the observation space and the action space
            # rendering is activated the first time that reset is called
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._gui_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._gui_client_id)

        self._risk_network = None
        self._risk_network_device = None
        self._risk_network_use_state_and_action = None
        self._risk_network_first_risky_action_step = None
        self._risk_ground_truth_dict = None
        if self._risk_config is not None:
            if self._risk_store_ground_truth:
                self._risk_ground_truth_dict = defaultdict(list)

        self._do_not_copy_keys = ["_risk_config", "_backup_agent", "_risk_network", "_risk_network_device",
                                  "_trajectory_plotter", "observation_space", "action_space",
                                  "_acc_limitation", "_braking_trajectory_generator", "_risk_ground_truth_dict",
                                  "_acc_limitation_joint", "_control_rate", "_video_recorder", '_np_random', 'spec']
        self._recursive_copy_keys = ["_robot_scene", "_trajectory_manager"]

    def _load_risk_config(self):
        if self._risk_config is None:
            if not os.path.isdir(self._risk_config_dir):
                self._risk_config_dir = os.path.join(current_dir, "trained_networks", self._risk_config_dir)
            risk_config_file_path = os.path.join(self._risk_config_dir, "risk_config.json")
            if os.path.isfile(risk_config_file_path):
                with open(risk_config_file_path, 'r') as f:
                    self._risk_config = json.load(f)
            else:
                raise FileNotFoundError("Could not find config file {}.".format(risk_config_file_path))

        # adjust checkpoint path in risk config if needed
        backup_checkpoint_path = self._risk_config["checkpoint"] if os.path.isdir(self._risk_config["checkpoint"]) \
            else os.path.join(current_dir, "trained_networks", self._risk_config["checkpoint"])

        # load env config from params.json
        backup_checkpoint_config_file_path = os.path.join(os.path.dirname(backup_checkpoint_path),
                                                          "params.json")

        if not os.path.isfile(backup_checkpoint_config_file_path):
            raise FileNotFoundError("Could not find checkpoint params file {}.".format(
                backup_checkpoint_config_file_path))

        with open(backup_checkpoint_config_file_path, 'r') as f:
            self._risk_config["config"] = json.load(f)

        # overwrite relevant env parameters based on the env_config provided by the risk network
        overwrite_parameters = ["planet_mode", "planet_one_center", "planet_one_radius_xy",
                                "planet_one_euler_angles", "planet_one_euler_angles",
                                "planet_one_period", "planet_two_center", "planet_two_radius_xy",
                                "planet_two_euler_angles", "planet_two_period", "planet_two_time_shift",
                                "use_moving_objects", "moving_object_sequence", "moving_object_area_center",
                                "moving_object_area_width_height", "moving_object_sphere_center",
                                "moving_object_sphere_radius", "moving_object_sphere_height_min_max",
                                "moving_object_sphere_angle_min_max", "moving_object_speed_meter_per_second",
                                "moving_object_aim_at_current_robot_position",
                                "moving_object_check_invalid_target_link_point_positions",
                                "moving_object_active_number_single",
                                "ball_machine_mode",
                                "human_network_checkpoint", "human_network_use_full_observation"]

        error_parameters = ["robot_scene", "trajectory_time_step",
                            "acc_limit_factor", "jerk_limit_factor", "pos_limit_factor", "vel_limit_factor"]

        warning_parameters = ["obstacle_scene", "closest_point_safety_distance", "starting_point_cartesian_range_scene",
                              "use_controller_target_velocities"]

        for parameter in overwrite_parameters:
            if parameter in self._risk_config["config"]["env_config"]:
                setattr(self, "_{}".format(parameter), self._risk_config["config"]["env_config"][parameter])

        for parameter in error_parameters:
            if parameter in self._risk_config["config"]["env_config"]:
                env_parameter_name = "_{}".format(parameter) if parameter != "robot_scene" else "_robot_scene_index"
                env_parameter = getattr(self, env_parameter_name)
                if env_parameter != self._risk_config["config"]["env_config"][parameter]:
                    raise ValueError("The {} parameter of the risk network does not match with the environment "
                                     "(risk: {} / env: {}).".format(
                                                    parameter,
                                                    self._risk_config["config"]["env_config"][parameter],
                                                    env_parameter))

        for parameter in warning_parameters:
            if parameter in self._risk_config["config"]["env_config"]:
                env_parameter = getattr(self, "_{}".format(parameter))
                if env_parameter != self._risk_config["config"]["env_config"][parameter]:
                    logging.warning("The {} parameter of the risk network does not match with the environment "
                                    "(risk: {} / env: {}).".format(
                                                    parameter,
                                                    self._risk_config["config"]["env_config"][parameter],
                                                    env_parameter))

        if self._risk_state_backup_trajectory_steps is None:
            # trajectory steps as during the training of the backup agent
            self._risk_state_backup_trajectory_steps = round(
                self._risk_config["config"]["env_config"]["trajectory_duration"] / self._trajectory_time_step)

        if self._risk_state_initial_backup_trajectory_steps is None:
            self._risk_state_initial_backup_trajectory_steps = self._risk_state_backup_trajectory_steps

    def _init_risk_network_and_backup_agent(self):
        from safemotions.utils.rl_agent import RLAgent

        ## import tensorflow as tf  # has to be imported after ray

        if not self._risk_use_backup_agent_for_initial_backup_trajectory_only:
            if self._risk_config["action_size"] is not None:
                self._risk_network_use_state_and_action = True
                if self._risk_config["action_size"] != self._num_manip_joints:
                    raise ValueError("The action size of the risk network does not match with num_manip_joints: "
                                     "action_size: {}, num_manip_joints: {}".format(self._risk_config["action_size"],
                                                                                    self._num_manip_joints))
            else:
                self._risk_network_use_state_and_action = False   # use state-based risk

            if self._risk_network_use_state_and_action or \
                    self._risk_state_config != self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY:

                global torch
                import torch as torch
                from safemotions.model.risk_network import RiskNetwork

                risk_network_device_name = 'cpu'

                if self._risk_network_use_gpu:
                    # make sure the GPU can be seen even if ray sets CUDA_VISIBLE_DEVICES to ''
                    if os.environ['CUDA_VISIBLE_DEVICES'] == '':
                        del os.environ['CUDA_VISIBLE_DEVICES']

                    cuda_available = torch.cuda.is_available()

                    if cuda_available:
                        risk_network_device_name = 'cuda'
                    else:
                        logging.warning("Risk network configured to use a GPU but no CUDA device available. "
                                        "Using the CPU instead.")

                self._risk_network_device = torch.device(risk_network_device_name)

                self._risk_network = RiskNetwork(**self._risk_config["risk_network_config"])

                self._risk_network.load_state_dict(
                    torch.load(os.path.join(self._risk_config_dir, "model.pt"),
                               map_location=self._risk_network_device, weights_only=True))

                self._risk_network = self._risk_network.to(self._risk_network_device)
                self._risk_network.eval()  # set to eval mode

        backup_checkpoint_path = self._risk_config["checkpoint"] if os.path.isdir(self._risk_config["checkpoint"]) \
                else os.path.join(current_dir, "trained_networks", self._risk_config["checkpoint"])

        self._backup_agent = RLAgent(checkpoint_path=backup_checkpoint_path,
                                     observation_size=self._risk_config["observation_size"],
                                     action_size=self._num_manip_joints,
                                     explore=False,
                                     clip_actions=self._risk_config["config"]["clip_actions"],
                                     policy_name="backup")

    def _init_physic_clients(self):
        self._num_physic_clients = 0

        if self._physic_clients_dict is None:
            if self._render_video:
                pybullet_options = "--width={} --height={}".format(self._video_width, self._video_height)
            else:
                pybullet_options = ""

            if self._use_gui and not (self._switch_gui_to_obstacle_client or self._switch_gui_to_backup_client):
                self._simulation_client_id = p.connect(p.GUI, options=pybullet_options)
                self._gui_client_id = self._simulation_client_id
                self._num_physic_clients += 1
            else:
                if not self._use_real_robot:
                    self._simulation_client_id = p.connect(p.DIRECT, options=pybullet_options)
                    self._num_physic_clients += 1
                else:
                    self._simulation_client_id = None

            self._main_client_id = self._simulation_client_id

            self._egl_plugin = None

            if self._simulation_client_id is not None:
                if self._renderer == OPENGL_GUI_RENDERER and self._render_video and not self._use_gui:
                    raise ValueError("renderer==OPENGL_GUI_RENDERER requires use_gui==True")
                if self._renderer == OPENGL_GUI_RENDERER or self._renderer == OPENGL_EGL_RENDERER:
                    self._pybullet_renderer = p.ER_BULLET_HARDWARE_OPENGL
                    if self._renderer == OPENGL_EGL_RENDERER and self._render_video:
                        import pkgutil
                        egl_renderer = pkgutil.get_loader('eglRenderer')
                        logging.warning(
                            "The usage of the egl renderer might lead to a segmentation fault on systems without "
                            "a GPU.")
                        if egl_renderer:
                            self._egl_plugin = p.loadPlugin(egl_renderer.get_filename(), "_eglRendererPlugin")
                        else:
                            self._egl_plugin = p.loadPlugin("eglRendererPlugin")
                else:
                    self._pybullet_renderer = p.ER_TINY_RENDERER
            else:
                self._pybullet_renderer = None

            if self._use_gui and self._switch_gui_to_obstacle_client:
                self._obstacle_client_id = p.connect(p.GUI, options=pybullet_options)
                self._gui_client_id = self._obstacle_client_id
            else:
                self._obstacle_client_id = p.connect(p.DIRECT)
            self._num_physic_clients += 1

            self._backup_client_id = None

            if self._risk_config is not None and \
                    (self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP
                     or self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY or
                     self._risk_check_initial_backup_trajectory or self._risk_store_ground_truth):

                if self._use_gui and self._switch_gui_to_backup_client:
                    if self._switch_gui_to_obstacle_client:
                        raise ValueError("switch_gui_to_obstacle_client and switch_gui_to_backup_client are not allowed"
                                         "to be True simultaneously")

                    self._backup_client_id = p.connect(p.GUI, options=pybullet_options)
                    self._gui_client_id = self._backup_client_id
                else:
                    self._backup_client_id = p.connect(p.DIRECT)

                self._num_physic_clients += 1
            else:
                if self._switch_gui_to_backup_client:
                    raise ValueError("switch_gui_to_backup_client requires a backup client to be initialized.")
                if self._risk_check_initial_backup_trajectory:
                    raise ValueError("risk_check_initial_backup_trajectory requires a valid risk config")
        else:
            if "main_client_id" in self._physic_clients_dict and \
                    self._physic_clients_dict["main_client_id"] is not None:
                self._main_client_id = self._physic_clients_dict["main_client_id"]
                self._simulation_client_id = self._main_client_id
                self._num_physic_clients += 1
            else:
                self._main_client_id = None
                self._simulation_client_id = None

            if "obstacle_client_id" in self._physic_clients_dict and \
                    self._physic_clients_dict["obstacle_client_id"] is not None:
                self._obstacle_client_id = self._physic_clients_dict["obstacle_client_id"]
                self._num_physic_clients += 1
            else:
                self._obstacle_client_id = None

            if "backup_client_id" in self._physic_clients_dict and \
                    self._physic_clients_dict["backup_client_id"] is not None:
                self._backup_client_id = self._physic_clients_dict["backup_client_id"]
                self._num_physic_clients += 1
            else:
                self._backup_client_id = None

    def _init_simulation(self):
        # reset the physics engine
        if self._physic_clients_dict is None:
            for i in range(self._num_physic_clients):
                p.resetSimulation(physicsClientId=i)  # to free memory
                if self._backup_client_id is not None:
                    p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=i)
                    # to ensure a deterministic behavior

        if self._render_video and self._use_gui and self._switch_gui_to_obstacle_client:
            capture_frame_function = partial(self._capture_frame_with_video_recorder, frames=2)
        else:
            capture_frame_function = None

        self._control_steps_per_action = int(round(self._trajectory_time_step / self._control_time_step))
        self._obstacle_client_update_steps_per_action = int(round(self._trajectory_time_step /
                                                                  self._simulation_time_step))

        logging.info("Trajectory time step: " + str(self._trajectory_time_step))

        # robot scene settings
        robot_scene_parameters = {'simulation_client_id': self._simulation_client_id,
                                  'simulation_time_step': self._simulation_time_step,
                                  'obstacle_client_id': self._obstacle_client_id,
                                  'backup_client_id': self._backup_client_id,
                                  'gui_client_id': self._gui_client_id,
                                  'trajectory_time_step': self._trajectory_time_step,
                                  'trajectory_duration': self._trajectory_duration,
                                  'use_real_robot': self._use_real_robot,
                                  'robot_scene': self._robot_scene_index,
                                  'obstacle_scene': self._obstacle_scene,
                                  'visual_mode': (self._use_gui or self._render_video),
                                  'capture_frame_function': capture_frame_function,
                                  'activate_obstacle_collisions': self._activate_obstacle_collisions,
                                  'observed_link_point_scene': self._observed_link_point_scene,
                                  'log_obstacle_data': self._log_obstacle_data,
                                  'visualize_bounding_spheres': self._visualize_bounding_spheres,
                                  'visualize_debug_lines': self._visualize_debug_lines,
                                  'acc_range_function': self.compute_next_acc_min_and_next_acc_max,
                                  'acc_braking_function': self.acc_braking_function,
                                  'violation_code_function': self.compute_violation_code_per_joint,
                                  'check_braking_trajectory_collisions': self._check_braking_trajectory_collisions,
                                  'collision_check_time': self._collision_check_time,
                                  'check_braking_trajectory_observed_points':
                                      self._distance_calculation_check_observed_points,
                                  'check_braking_trajectory_torque_limits':
                                      self._check_braking_trajectory_torque_limits,
                                  'closest_point_safety_distance': self._closest_point_safety_distance,
                                  'observed_point_safety_distance': self._observed_point_safety_distance,
                                  'starting_point_cartesian_range_scene':
                                      self._starting_point_cartesian_range_scene,
                                  'use_target_points': self._use_target_points,
                                  'target_point_cartesian_range_scene': self._target_point_cartesian_range_scene,
                                  'target_point_relative_pos_scene': self._target_point_relative_pos_scene,
                                  'target_point_radius': self._target_point_radius,
                                  'target_point_sequence': self._target_point_sequence,
                                  'target_point_reached_reward_bonus': self._target_point_reached_reward_bonus,
                                  'target_point_use_actual_position': self._target_point_use_actual_position,
                                  'use_moving_objects': self._use_moving_objects,
                                  'moving_object_sequence': self._moving_object_sequence,
                                  'moving_object_area_center': self._moving_object_area_center,
                                  'moving_object_area_width_height': self._moving_object_area_width_height,
                                  'moving_object_sphere_center': self._moving_object_sphere_center,
                                  'moving_object_sphere_radius': self._moving_object_sphere_radius,
                                  'moving_object_sphere_height_min_max': self._moving_object_sphere_height_min_max,
                                  'moving_object_sphere_angle_min_max': self._moving_object_sphere_angle_min_max,
                                  'moving_object_speed_meter_per_second': self._moving_object_speed_meter_per_second,
                                  'moving_object_aim_at_current_robot_position':
                                      self._moving_object_aim_at_current_robot_position,
                                  'moving_object_check_invalid_target_link_point_positions':
                                      self._moving_object_check_invalid_target_link_point_positions,
                                  'moving_object_active_number_single': self._moving_object_active_number_single,
                                  'moving_object_random_initial_position': self._moving_object_random_initial_position,
                                  'planet_mode': self._planet_mode,
                                  'planet_one_center': self._planet_one_center,
                                  'planet_one_radius_xy': self._planet_one_radius_xy,
                                  'planet_one_euler_angles': self._planet_one_euler_angles,
                                  'planet_one_period': self._planet_one_period,
                                  'planet_two_center': self._planet_two_center,
                                  'planet_two_radius_xy': self._planet_two_radius_xy,
                                  'planet_two_euler_angles': self._planet_two_euler_angles,
                                  'planet_two_period': self._planet_two_period,
                                  'planet_two_time_shift': self._planet_two_time_shift,
                                  'terminate_on_collision_with_moving_obstacle':
                                      self._terminate_on_collision_with_moving_obstacle,
                                  'collision_avoidance_mode': self._collision_avoidance_mode,
                                  'collision_avoidance_kinematic_state_sampling_mode':
                                      self._collision_avoidance_kinematic_state_sampling_mode,
                                  'collision_avoidance_kinematic_state_sampling_probability':
                                      self._collision_avoidance_kinematic_state_sampling_probability,
                                  'collision_avoidance_stay_in_state_probability':
                                      self._collision_avoidance_stay_in_state_probability,
                                  'always_use_collision_avoidance_starting_point_sampling':
                                      self._always_use_collision_avoidance_starting_point_sampling,
                                  'ball_machine_mode': self._ball_machine_mode,
                                  'risk_state_deterministic_backup_trajectory':
                                      self._risk_state_deterministic_backup_trajectory,
                                  'human_network_checkpoint': self._human_network_checkpoint,
                                  'human_network_use_collision_avoidance_starting_point_sampling':
                                      self._human_network_use_collision_avoidance_starting_point_sampling,
                                  'human_network_collision_avoidance_kinematic_state_sampling_probability':
                                      self._human_network_collision_avoidance_kinematic_state_sampling_probability,
                                  'human_network_collision_avoidance_stay_in_state_probability':
                                      self._human_network_collision_avoidance_stay_in_state_probability,
                                  'no_self_collision': self._no_self_collision,
                                  'no_link_coloring': self._no_link_coloring,
                                  'no_target_link_coloring': self._no_target_link_coloring,
                                  'target_link_name': self._target_link_name,
                                  'target_link_offset': self._target_link_offset,
                                  'pos_limit_factor': self._pos_limit_factor,
                                  'vel_limit_factor': self._vel_limit_factor,
                                  'acc_limit_factor': self._acc_limit_factor,
                                  'jerk_limit_factor': self._jerk_limit_factor,
                                  'torque_limit_factor': self._torque_limit_factor,
                                  'use_controller_target_velocities': self._use_controller_target_velocities,
                                  'reward_maximum_relevant_distance': self.reward_maximum_relevant_distance,
                                  'reward_consider_moving_obstacles': self.reward_consider_moving_obstacles,
                                  'static_robot': self._static_robot,
                                  'obstacle_client_update_steps_per_action':
                                      self._obstacle_client_update_steps_per_action,
                                  'do_not_execute_robot_movement': self._do_not_execute_robot_movement,
                                  'use_fixed_seed': True if self._seed is not None else False,
                                  }

        if self._use_real_robot:
            self._robot_scene = RealRobotScene(real_robot_debug_mode=self._real_robot_debug_mode,
                                               **robot_scene_parameters)
        else:
            self._robot_scene = SimRobotScene(**robot_scene_parameters)

        self._num_manip_joints = self._robot_scene.num_manip_joints
        if self._position_controller_time_constants is None:
            if self._use_controller_target_velocities:
                self._position_controller_time_constants = [0.0005] * self._num_manip_joints
            else:
                self._position_controller_time_constants = [0.0372] * self._num_manip_joints

        # trajectory manager settings
        self._trajectory_manager = TrajectoryManager(trajectory_time_step=self._trajectory_time_step,
                                                     trajectory_duration=self._trajectory_duration,
                                                     obstacle_wrapper=self._robot_scene.obstacle_wrapper,
                                                     env=self)

        self._robot_scene.compute_actual_joint_limits()

        # calculate model coefficients to estimate actual values if required
        if self._model_actual_values:
            self._trajectory_manager.compute_controller_model_coefficients(self._position_controller_time_constants,
                                                                           self._simulation_time_step)

        self._zero_joint_vector_list = [0.0] * self._num_manip_joints
        self._zero_joint_vector_array = np.array(self._zero_joint_vector_list)

        if (self._use_movement_thread_or_process or self._use_gui) and self._use_control_rate_sleep:
            self._control_rate = ControlRate(1. / self._control_time_step, skip_periods=True, debug_mode=False)
        else:
            self._control_rate = None

        for i in range(self._num_physic_clients):
            p.setGravity(0, 0, -9.81, physicsClientId=i)
            p.setPhysicsEngineParameter(numSolverIterations=self._solver_iterations,
                                        enableFileCaching=1,
                                        physicsClientId=i)
            p.setTimeStep(self._simulation_time_step, physicsClientId=i)

        if self._gui_client_id is not None:
            gpu_support = 'CUDA' in os.environ['PATH'].upper() if 'PATH' in os.environ else False
            use_shadows = 1 if not self._render_no_shadows else 0
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, use_shadows, lightPosition=[-15, 0, 28],
                                       shadowMapResolution=16384 if gpu_support else 4096,
                                       physicsClientId=self._gui_client_id)
            # set shadowMapResolution to 4096 when running the code without a dedicated GPU and to 16384 otherwise

    def reset(self, seed=None, repeated_reset=False, options=None):
        if seed is not None and seed != self._seed:
            self.set_seed(seed)

        self._episode_counter += 1
        if self._episode_counter == 1 and self._gui_client_id is not None:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._gui_client_id)
            if self._render_video and not self._renderer == self.IMAGEGRAB_RENDERER:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self._gui_client_id)

        if self._episode_counter % self._episodes_per_simulation_reset == 0 and not repeated_reset:
            self._disconnect_physic_clients()
            self._init_physic_clients()
            self._init_simulation()

        self._control_step_counter = 0
        self._precomputation_timer = None

        self._total_reward = 0
        self._episode_length = 0
        self._trajectory_successful = True
        self._current_trajectory_point_index = 0
        self._action_list = []

        self._network_prediction_part_done = False

        get_new_setup = (((self._episode_counter-1) % self._eval_new_condition_counter) == 0)

        self._robot_scene.obstacle_wrapper.reset_obstacles()

        duration_multiplier = None

        self._trajectory_manager.reset(get_new_trajectory=get_new_setup,
                                       duration_multiplier=duration_multiplier)


        self._start_position = np.array(self._get_trajectory_start_position())
        self._start_velocity = np.array(self._get_trajectory_start_velocity())
        self._start_acceleration = np.array(self._get_trajectory_start_acceleration())

        if self._collision_avoidance_mode and self._collision_avoidance_new_state_sample_step_range is not None:
            self._collision_avoidance_new_state_sample_step = np.random.randint(
                self._collision_avoidance_new_state_sample_step_range[0],
                self._collision_avoidance_new_state_sample_step_range[1] + 1)

        if self._use_real_robot:
            logging.info("Starting position: %s", self._start_position)
        else:
            logging.debug("Starting position: %s", self._start_position)
        self._robot_scene.pose_manipulator(self._start_position, self._start_velocity)

        compute_initial_braking_trajectory = \
            True if (self._collision_avoidance_mode or
                     self._always_use_collision_avoidance_starting_point_sampling) else False
        self._robot_scene.obstacle_wrapper.reset(start_position=self._start_position,
                                                 start_velocity=self._start_velocity,
                                                 start_acceleration=self._start_acceleration,
                                                 compute_initial_braking_trajectory=compute_initial_braking_trajectory)

        if not self._use_real_robot:
            # the initial torques are not zero due to gravity
            # the call to step simulation also leads to a call of p.performCollisionDetection meaning that
            # p.getContactPoints yields correct results
            self._robot_scene.set_motor_control(target_positions=self._start_position,
                                                target_velocities=self._start_velocity,
                                                target_accelerations=self._start_acceleration,
                                                initial_kinematic_state=True,
                                                physics_client_id=self._simulation_client_id)
            p.stepSimulation(physicsClientId=self._simulation_client_id)

        self._robot_scene.obstacle_wrapper.update(target_position=self._start_position,
                                                  target_velocity=self._start_velocity,
                                                  target_acceleration=self._start_acceleration,
                                                  actual_position=self._start_position,
                                                  actual_velocity=self._start_velocity,
                                                  update_after_reset=True)

        self._reset_plotter(self._start_position, self._start_velocity, self._start_acceleration)
        self._add_computed_actual_position_to_plot(self._start_position, self._start_velocity,
                                                   self._start_acceleration)

        if not self._use_real_robot:
            self._add_actual_position_to_plot(self._start_position)

        if self._plot_actual_torques and not self._use_real_robot:
            actual_joint_torques = self._robot_scene.get_actual_joint_torques()
            self._add_actual_torques_to_plot(actual_joint_torques)
        else:
            self._add_actual_torques_to_plot(self._zero_joint_vector_list)

        self._calculate_safe_acc_range(self._start_position, self._start_velocity, self._start_acceleration,
                                       self._current_trajectory_point_index)

        self._termination_reason = self.TERMINATION_UNSET
        self._last_action = None
        self._network_prediction_part_done = False
        self._movement_thread = None
        self._movement_process = None
        self._brake = False
        self._end_min_distance = None
        self._end_max_torque = None
        self._adaptation_punishment = None
        self._moving_object_hit_robot_or_obstacle = False
        self._moving_object_missed_robot = False
        self._end_acceleration = None

        self._resampling_attempts = 0
        self._safe_action = None
        self._last_risk_prediction = None
        self._risk_observation = None
        self._risk_network_first_risky_action_step = np.nan

        return None

    def step_add_value(self, value):
        # can be called before step to add the current value for plotting / visualization
        logging.debug("Value %s: %s", self._episode_length, value)
        if self._add_value_debug_text:
            if self._value_debug_text is None:
                self._value_debug_text = p.addUserDebugText(text="Value {}: {:.2f}".format(self._episode_length, value),
                                                            textPosition=[0.0, -0.8, 2],
                                                            textColorRGB=[0, 0, 0],
                                                            textSize=4,
                                                            physicsClientId=self._gui_client_id)
            else:
                self._value_debug_text = p.addUserDebugText(text="Value {}: {:.2f}".format(self._episode_length, value),
                                                            textPosition=[0.0, -0.8, 2],
                                                            textColorRGB=[0, 0, 0],
                                                            textSize=4,
                                                            replaceItemUniqueId=self._value_debug_text,
                                                            physicsClientId=self._gui_client_id)
        self._add_value_to_plot(value)

    def step(self, action):
        self._episode_length += 1
        self._robot_scene.clear_last_action()

        if self._random_agent:
            action = self.get_random_action()
            # overwrite the desired action with a random action
        else:
            action = np.asarray(action, dtype=np.float64)

        if self._store_actions or self._store_trajectory:
            self._action_list.append(action)

        logging.debug("Action %s: %s", self._episode_length - 1, action)

        self._self_collision_detected = False
        self._collision_with_static_obstacle_detected = False
        self._collision_with_moving_obstacle_detected = False

        (end_acceleration, controller_setpoints, obstacle_client_update_setpoints,
            initial_motor_action, action_info, robot_stopped) = \
            self._compute_controller_setpoints_from_action(action)

        self._last_action = initial_motor_action  # store the last action for reward calculation
        self._robot_scene.obstacle_wrapper.step()

        if self._store_trajectory:
            for i in range(len(controller_setpoints['positions'])):
                self._add_generated_trajectory_control_point(controller_setpoints['positions'][i],
                                                             controller_setpoints['velocities'][i],
                                                             controller_setpoints['accelerations'][i])
        
        for i in range(len(obstacle_client_update_setpoints['positions'])):

            if self._model_actual_values:

                last_position_setpoint = self._start_position if i == 0 else obstacle_client_update_setpoints[
                    'positions'][i - 1]
                computed_position_is = self._trajectory_manager.model_position_controller_to_compute_actual_values(
                    current_setpoint=obstacle_client_update_setpoints['positions'][i],
                    last_setpoint=last_position_setpoint)

                last_velocity_setpoint = self._start_velocity if i == 0 else obstacle_client_update_setpoints[
                    'velocities'][i - 1]
                computed_velocity_is = self._trajectory_manager.model_position_controller_to_compute_actual_values(
                    current_setpoint=obstacle_client_update_setpoints['velocities'][i],
                    last_setpoint=last_velocity_setpoint, key='velocities')

                last_acceleration_setpoint = self._start_acceleration if i == 0 else obstacle_client_update_setpoints[
                    'accelerations'][i - 1]
                computed_acceleration_is = self._trajectory_manager.model_position_controller_to_compute_actual_values(
                    current_setpoint=obstacle_client_update_setpoints['accelerations'][i],
                    last_setpoint=last_acceleration_setpoint, key='accelerations')


                self._add_computed_actual_trajectory_control_point(computed_position_is,
                                                                   computed_velocity_is,
                                                                   computed_acceleration_is)
                self._add_computed_actual_position_to_plot(computed_position_is, computed_velocity_is,
                                                           computed_acceleration_is)

                if self._use_movement_thread_or_process or self._obstacle_use_computed_actual_values:

                    self._robot_scene.obstacle_wrapper.update(
                        target_position=obstacle_client_update_setpoints['positions'][i],
                        target_velocity=obstacle_client_update_setpoints['velocities'][i],
                        target_acceleration=obstacle_client_update_setpoints['accelerations'][i],
                        actual_position=computed_position_is,
                        actual_velocity=computed_velocity_is)

        if self._episode_length == 1:
            if self._control_rate is not None:
                # start the control phase and compute the precomputation time
                if hasattr(self._control_rate, 'start_control_phase'):
                    self._control_rate.start_control_phase()
                else:
                    self._control_rate.sleep()
            self._precomputation_timer = time.time()

        movement_info = None

        if not self._do_not_execute_robot_movement:
            if self._use_movement_thread_or_process:
                if self._use_thread_for_movement:
                    movement_thread = Thread(target=self._execute_robot_movement,
                                             kwargs=dict(controller_setpoints=controller_setpoints))
                    if self._movement_thread is not None:
                        self._movement_thread.join()
                    movement_thread.start()
                    self._movement_thread = movement_thread
                if self._use_process_for_movement:
                    control_rate = None if self._control_rate is None else self._control_rate.control_rate
                    control_function = self._robot_scene.send_command_to_trajectory_controller \
                        if not self._real_robot_debug_mode else None
                    fifo_path = self._robot_scene.FIFO_PATH if not self._real_robot_debug_mode else None
                    if self._movement_process is not None:
                        last_time = self._movement_process.get()
                    else:
                        last_time = None

                    self._movement_process = \
                        self._movement_process_pool.apply_async(func=self._execute_robot_movement_as_process,
                                                                kwds=dict(control_function=control_function,
                                                                          fifo_path=fifo_path,
                                                                          controller_position_setpoints=
                                                                          controller_setpoints['positions'],
                                                                          control_rate=control_rate,
                                                                          last_time=last_time))

                    time.sleep(0.002)
                    # the movement process will start faster if the main process sleeps during the start-up phase
            else:
                self._movement_thread = None
                movement_info = self._execute_robot_movement(controller_setpoints=controller_setpoints)

            return self.process_step_outcome(end_acceleration, obstacle_client_update_setpoints,
                                             robot_stopped, movement_info, action_info)

        return end_acceleration, controller_setpoints, obstacle_client_update_setpoints, \
            movement_info, action_info, robot_stopped

    def process_step_outcome(self, end_acceleration, obstacle_client_update_setpoints,
                             robot_stopped, movement_info, action_info):

        self._robot_scene.obstacle_wrapper.process_step_outcome()

        if movement_info is None:
            movement_info = {'mean': {}, 'min': {}, 'max': {}}

        if action_info is None:
            action_info = {'mean': {}, 'min': {}, 'max': {}}

        if self._max_resampling_attempts != 0:
            for key in ["mean", "min", "max"]:
                movement_info[key]["resampling_attempts"] = self._resampling_attempts

        self._start_position = obstacle_client_update_setpoints['positions'][-1]
        self._start_velocity = obstacle_client_update_setpoints['velocities'][-1]
        self._start_acceleration = end_acceleration

        self._add_generated_trajectory_point(self._start_position, self._start_velocity, self._start_acceleration)

        self._current_trajectory_point_index += 1

        # sleep for a specified part of the time_step before getting the observation
        if self._time_step_fraction_sleep_observation != 0:
            time.sleep(self._trajectory_time_step * self._time_step_fraction_sleep_observation)

        observation, reward, done, info = self._process_action_outcome(movement_info, action_info, robot_stopped)

        termination = False
        truncation = False

        if not self._network_prediction_part_done:
            self._total_reward += reward
        else:
            done = True

        if done:
            self._network_prediction_part_done = True

        if not self._network_prediction_part_done:
            self._prepare_for_next_action()
        else:
            if not self._use_real_robot or robot_stopped:

                if self._movement_thread is not None:
                    self._movement_thread.join()
                if self._movement_process is not None:
                    self._movement_process.get()

                self._robot_scene.prepare_for_end_of_episode()
                self._prepare_for_end_of_episode()
                observation, reward, termination, truncation, info = self._process_end_of_episode(observation, reward,
                                                                                                  info)

                if self._store_actions:
                    self._store_action_list()
                if self._store_trajectory:
                    self._store_trajectory_data()
            else:
                self._brake = True  # slow down the robot prior to stopping the episode

        self._resampling_attempts = 0
        self._safe_action = None
        self._last_risk_prediction = None

        # distinguish between termination and truncation if needed
        # see https://farama.org/Gymnasium-Terminated-Truncated-Step-API

        return observation, reward, termination, truncation, dict(info)

    def _execute_robot_movement(self, controller_setpoints):
        # executed in real-time if required
        actual_joint_torques_rel_abs_list = []

        for i in range(len(controller_setpoints['positions'])):

            if self._control_rate is not None:
                self._control_rate.sleep()

            self._robot_scene.set_motor_control(controller_setpoints['positions'][i],
                                                target_velocities=controller_setpoints['velocities'][i],
                                                target_accelerations=controller_setpoints['accelerations'][i],
                                                computed_position_is=controller_setpoints['positions'][i],
                                                computed_velocity_is=controller_setpoints['velocities'][i])

            if not self._use_real_robot:
                self._sim_step()
                actual_joint_torques = self._robot_scene.get_actual_joint_torques()
                actual_joint_torques_rel_abs = np.abs(normalize_joint_values(actual_joint_torques,
                                                                             self._robot_scene.max_torques))
                actual_joint_torques_rel_abs_list.append(actual_joint_torques_rel_abs)

                if self._plot_actual_torques:
                    self._add_actual_torques_to_plot(actual_joint_torques)

            actual_position = None

            if (not self._use_movement_thread_or_process and not self._obstacle_use_computed_actual_values):
                actual_position, actual_velocity = self._robot_scene.get_actual_joint_position_and_velocity(
                    physics_client_id=self._simulation_client_id)

                if self._store_trajectory:
                    actual_acceleration = (actual_velocity - np.array(
                        self._get_measured_actual_trajectory_control_point(-1, key='velocities'))) / \
                                          self._simulation_time_step
                    self._add_measured_actual_trajectory_control_point(actual_position, actual_velocity,
                                                                       actual_acceleration)

            if not self._use_movement_thread_or_process and not self._obstacle_use_computed_actual_values:

                self._robot_scene.obstacle_wrapper.update(target_position=controller_setpoints['positions'][i],
                                                          target_velocity=controller_setpoints['velocities'][i],
                                                          target_acceleration=controller_setpoints['accelerations'][
                                                              i],
                                                          actual_position=actual_position,
                                                          actual_velocity=actual_velocity)

            if not self._use_real_robot:
                self._add_actual_position_to_plot(actual_position)

        movement_info = {'mean': {}, 'min': {}, 'max': {}}

        if not self._use_real_robot:
            # add torque info to movement_info
            torque_violation = 0.0
            actual_joint_torques_rel_abs = np.array(actual_joint_torques_rel_abs_list)
            if self._punish_end_max_torque and self._end_max_torque is None:
                self._end_max_torque = np.max(actual_joint_torques_rel_abs[-1])
            actual_joint_torques_rel_abs_swap = actual_joint_torques_rel_abs.T
            for j in range(self._num_manip_joints):
                movement_info['mean']['joint_{}_torque_abs'.format(j)] = np.mean(
                    actual_joint_torques_rel_abs_swap[j])
                actual_joint_torques_rel_abs_max = np.max(actual_joint_torques_rel_abs_swap[j])
                movement_info['max']['joint_{}_torque_abs'.format(j)] = actual_joint_torques_rel_abs_max
                if actual_joint_torques_rel_abs_max > 1.001 and self._simulation_client_id == self._main_client_id:
                    torque_violation = 1.0

            movement_info['max']['joint_torque_violation'] = torque_violation
            movement_info['mean']['joint_torque_violation'] = torque_violation

        return movement_info

    @staticmethod
    def _execute_robot_movement_as_process(control_function, fifo_path, controller_position_setpoints,
                                           control_rate=None, last_time=None):
        if control_rate is not None:
            control_rate = ControlRate(control_rate=control_rate, skip_periods=False, debug_mode=False,
                                       last_time=last_time, busy_wait=True)
        
        fifo_in = None
        if fifo_path is not None:
            fifo_in = os.open(fifo_path, os.O_WRONLY) 

        for i in range(len(controller_position_setpoints)):
            if control_rate is not None:
                control_rate.sleep()

            if control_function is not None and fifo_in is not None:
                control_function(controller_position_setpoints[i], fifo_in)
                
        if fifo_in is not None:
            os.close(fifo_in)

        if control_rate is not None:
            return control_rate.last_time
        else:
            return None

    def get_random_action(self):
        return np.random.uniform(-1, 1, self.action_space.shape)

    def _process_action_outcome(self, base_info, action_info, robot_stopped=False):

        reward, reward_info = self._get_reward()

        if self._collision_avoidance_mode and self._collision_avoidance_new_state_sample_step is not None:
            self._collision_avoidance_new_state_sample_step -= 1

        if self._collision_avoidance_mode and self._collision_avoidance_new_state_sample_step is not None \
                and self._collision_avoidance_new_state_sample_step == 0:
            # adjust start position after the reward calculation but before the safe acc range is computed
            self._start_position, self._start_velocity, self._start_acceleration \
                = self._trajectory_manager.get_new_trajectory_start_position_velocity_acceleration()
            self._collision_avoidance_new_state_sample_step = np.random.randint(
                self._collision_avoidance_new_state_sample_step_range[0],
                self._collision_avoidance_new_state_sample_step_range[1] + 1)
            self._robot_scene.pose_manipulator(self._start_position, self._start_velocity)

        self._calculate_safe_acc_range(self._start_position, self._start_velocity, self._start_acceleration,
                                       self._current_trajectory_point_index)

        for key in ['mean', 'max']:
            base_info[key]['collision_rate_self'] = 1.0 if self._self_collision_detected else 0.0
            base_info[key]['collision_rate_static_obstacles'] = \
                1.0 if self._collision_with_static_obstacle_detected else 0.0
            base_info[key]['collision_rate_moving_obstacles'] = \
                1.0 if self._collision_with_moving_obstacle_detected else 0.0

        observation, observation_info = self._get_observation()
        done = self._check_termination(robot_stopped)

        info = defaultdict(dict)

        for k, v in chain(base_info.items(), action_info.items(), observation_info.items(), reward_info.items()):
            info[k] = {**info[k], **v}

        return observation, reward, done, info

    def _process_end_of_episode(self, observation, reward, info):
        if self._trajectory_successful:
            info['trajectory_successful'] = 1.0
        else:
            info['trajectory_successful'] = 0.0

        info.update(trajectory_length=self._trajectory_manager.trajectory_length)
        info.update(episode_length=self._episode_length)
        info['termination_reason'] = self._termination_reason

        if self._risk_network is not None:
            info['risk_network_first_risky_action_step'] = self._risk_network_first_risky_action_step

        if self._simulation_client_id == self._main_client_id:
            logging.info("Termination reason: %s", self._termination_reason)

        self._robot_scene.obstacle_wrapper.process_end_of_episode()
        # get info from obstacle wrapper
        obstacle_info = self._robot_scene.obstacle_wrapper.get_info_and_print_stats()
        info = dict(info, **obstacle_info)  # concatenate dicts
        if not self._plot_value:  # if plot_value is True, display plot is called in evaluate to include the last value
            self.display_plot()
        self._save_plot(self.__class__.__name__, self._experiment_name)

        if self._risk_store_ground_truth and \
                self._episode_counter % self._risk_ground_truth_episodes_per_file == 0 and \
                self._simulation_client_id != self._backup_client_id:
            self._store_risk_ground_truth_data()

        termination = True
        truncation = False

        # note the difference between termination and truncation
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # overload this method to distinguish between the two cases

        return observation, reward, termination, truncation, info

    def _make_evaluation_dir(self, sub_dir=None):
        if sub_dir is not None:
            eval_dir = os.path.join(self._evaluation_dir, sub_dir)
        else:
            eval_dir = self._evaluation_dir

        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        return eval_dir

    def _store_risk_ground_truth_data(self):
        import pandas as pd

        if self._episode_counter % self._risk_ground_truth_episodes_per_file == 0:
            start_episode = self._episode_counter - self._risk_ground_truth_episodes_per_file
        else:
            start_episode = int(self._episode_counter / self._risk_ground_truth_episodes_per_file) \
                            * self._risk_ground_truth_episodes_per_file

        if "state_action_risk" in self._risk_ground_truth_dict:
            # export state_action_risk and state_risk in separate directories
            risk_dicts = [self._risk_ground_truth_dict["state_action_risk"],
                          self._risk_ground_truth_dict["state_risk"]]
            risk_dirs = ["state_action_risk", "state_risk"]
        else:
            risk_dicts = [self._risk_ground_truth_dict]
            risk_dirs = ["ground_truth_risk"]

        for i in range(len(risk_dicts)):

            eval_dir = self._make_evaluation_dir(sub_dir=risk_dirs[i])
            data_frame = pd.DataFrame(risk_dicts[i])

            if len(risk_dicts[i]["risk"]) > 0:
                mean_ground_truth_risk = np.mean(risk_dicts[i]["risk"])
                if "risk_prediction" in risk_dicts[i]:
                    mean_ground_truth_risk_prediction = np.mean(risk_dicts[i]["risk_prediction"])
                else:
                    mean_ground_truth_risk_prediction = 0.0

                with open(os.path.join(eval_dir, "episodes_{}_to_{}_risk_{:.2f}_prediction_{:.2f}_pid_{}.csv".format(
                        start_episode,
                        self._episode_counter - 1,
                        mean_ground_truth_risk,
                        mean_ground_truth_risk_prediction,
                        self._pid)), 'w') as file:

                    data_frame.to_csv(path_or_buf=file)

        # store risk config if it does not exist yet
        risk_config_file_path = os.path.join(self._evaluation_dir, "risk_config.json")
        if not os.path.isfile(risk_config_file_path):
            risk_config_export = self._risk_config.copy()
            risk_config_export.pop('config', None)  # do not export backup checkpoint config
            with open(risk_config_file_path, 'w') as f:
                f.write(json.dumps(risk_config_export, sort_keys=True))
                f.flush()

        self._risk_ground_truth_dict = defaultdict(list)

    def _store_action_list(self):
        action_dict = {'actions': np.asarray(self._action_list).tolist()}
        eval_dir = self._make_evaluation_dir(sub_dir="action_logs")

        with open(os.path.join(eval_dir, "episode_{}_{}.json".format(self._episode_counter, self.pid)), 'w') as f:
            f.write(json.dumps(action_dict))
            f.flush()

    def _store_trajectory_data(self):
        trajectory_dict = {'actions': np.asarray(self._action_list).tolist(),
                           'trajectory_setpoints': self._to_list(
                               self._trajectory_manager.generated_trajectory_control_points),
                           'trajectory_measured_actual_values': self._to_list(
                               self._trajectory_manager.measured_actual_trajectory_control_points),
                           'trajectory_computed_actual_values': self._to_list(
                               self._trajectory_manager.computed_actual_trajectory_control_points),
                           }

        eval_dir = self._make_evaluation_dir(sub_dir="trajectory_data")

        with open(os.path.join(eval_dir, "episode_{}_{}.json".format(self._episode_counter, self.pid)), 'w') as f:
            f.write(json.dumps(trajectory_dict))
            f.flush()

    def close(self):
        if self._risk_store_ground_truth and \
                self._episode_counter % self._risk_ground_truth_episodes_per_file != 0 and \
                self._simulation_client_id != self._backup_client_id:
            self._store_risk_ground_truth_data()
        self._robot_scene.disconnect()
        self._disconnect_physic_clients()
        if self._movement_process_pool is not None:
            self._movement_process_pool.close()
            self._movement_process_pool.join()

    def _is_action_risky(self, action, motor_action, end_acceleration=None):
        action_considered_as_risky = False
        risky_action_termination_reason = None
        risk_input = None
        if self._risk_network_use_state_and_action:
            risk_input = np.array(list(self._risk_observation) + list(motor_action))
        else:
            if self._risk_state_config == self.RISK_CHECK_CURRENT_STATE:
                # check the risk of the current state only
                risk_input = np.array(self._risk_observation)
            elif self._risk_state_config == self.RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING or \
                    self._risk_state_config == self.RISK_CHECK_NEXT_STATE_FULL_FORECASTING:
                forecast_non_kinematic_components = True if \
                    self._risk_state_config == self.RISK_CHECK_NEXT_STATE_FULL_FORECASTING else False

                if end_acceleration is None:
                    raise ValueError("end_acceleration not specified")
                next_risk_observation = \
                    self._get_next_risk_observation(
                        end_acceleration,
                        forecast_non_kinematic_components=
                        forecast_non_kinematic_components)
                risk_input = np.array(next_risk_observation)
            elif self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP or \
                    self._risk_state_config == \
                    self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY:

                state_action_ground_truth_state = self._risk_observation.copy()
                state_ground_truth_state = None

                stored_variables = self.switch_to_backup_client()
                self._safe_action = None
                if self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP:
                    simulation_steps = 1
                else:
                    simulation_steps = 1 + self._risk_state_backup_trajectory_steps

                for i in range(simulation_steps):
                    if i == 0:
                        simulated_action = action
                    else:
                        backup_action = np.array(
                            self._backup_agent.compute_action(self._risk_observation, full_fetch=False),
                            dtype=np.float64)
                        simulated_action = self._get_action_from_backup_action(backup_action)

                    _, _, termination, truncation, _ = self.step(simulated_action)

                    simulation_done = termination or truncation

                    if not simulation_done and state_ground_truth_state is None:
                        state_ground_truth_state = self._risk_observation.copy()

                    if simulation_done:
                        if self.termination_reason != self.TERMINATION_TRAJECTORY_LENGTH:
                            action_considered_as_risky = True
                            risky_action_termination_reason = self.termination_reason
                        break

                    if not action_considered_as_risky and \
                            self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP:
                        risk_input = np.copy(self._risk_observation)

                if self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY \
                        and self._risk_state_deterministic_backup_trajectory:
                    stored_variables["_robot_scene._obstacle_wrapper._moving_object_deterministic_list"] = \
                        self._robot_scene.obstacle_wrapper.moving_object_deterministic_list

                    if self.robot_scene.obstacle_wrapper.human is not None:
                        self.robot_scene.obstacle_wrapper.human.copy_deterministic_lists_to_main_client()

                self.switch_back_to_main_client(stored_variables)

                if action_considered_as_risky:
                    self._last_risk_prediction = 1.0
                elif self._risk_state_config == \
                        self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY:
                    self._last_risk_prediction = 0.0

                if self._risk_store_ground_truth and self._risk_state_config == \
                        self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY:
                    if action_considered_as_risky:
                        ground_truth_risk = 1.0
                    else:
                        ground_truth_risk = 0.0

                    self_collision = 1.0 if risky_action_termination_reason == self.TERMINATION_SELF_COLLISION else 0.0
                    static_obstacle = 1.0 if (risky_action_termination_reason ==
                                              self.TERMINATION_COLLISION_WITH_STATIC_OBSTACLE) else 0.0
                    moving_obstacle = 1.0 if (risky_action_termination_reason ==
                                              self.TERMINATION_COLLISION_WITH_MOVING_OBSTACLE) else 0.0

                    if "state_action_risk" not in self._risk_ground_truth_dict:
                        self._risk_ground_truth_dict["state_action_risk"] = defaultdict(list)
                        self._risk_ground_truth_dict["state_risk"] = defaultdict(list)

                    self._risk_ground_truth_dict["state_action_risk"]["state"].append(
                        state_action_ground_truth_state.tolist())
                    self._risk_ground_truth_dict["state_action_risk"]["action"].append(motor_action.tolist())
                    self._risk_ground_truth_dict["state_action_risk"]["risk"].append(ground_truth_risk)
                    self._risk_ground_truth_dict["state_action_risk"]["self_collision"].append(self_collision)
                    self._risk_ground_truth_dict["state_action_risk"]["static_obstacle"].append(static_obstacle)
                    self._risk_ground_truth_dict["state_action_risk"]["moving_obstacle"].append(moving_obstacle)

                    if state_ground_truth_state is not None:
                        self._risk_ground_truth_dict["state_risk"]["state"].append(state_ground_truth_state.tolist())
                        self._risk_ground_truth_dict["state_risk"]["risk"].append(ground_truth_risk)
                        self._risk_ground_truth_dict["state_risk"]["self_collision"].append(self_collision)
                        self._risk_ground_truth_dict["state_risk"]["static_obstacle"].append(static_obstacle)
                        self._risk_ground_truth_dict["state_risk"]["moving_obstacle"].append(moving_obstacle)

            else:
                raise NotImplementedError()

        if risk_input is not None:
            risk_input = np.float32(risk_input.reshape((1, -1)))

            risk_input_tensor = torch.from_numpy(risk_input).to(self._risk_network_device)
            with torch.no_grad():
                risk_prediction = self._risk_network(risk_input_tensor)
            self._last_risk_prediction = risk_prediction[0].item()

            if self._last_risk_prediction >= self._risk_threshold:
                action_considered_as_risky = True

            if self._risk_store_ground_truth:
                ground_truth_risk = 0.0

                stored_variables = self.switch_to_backup_client()
                self._safe_action = None
                if self._risk_network_use_state_and_action or \
                        self._risk_state_config == self.RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING or \
                        self._risk_state_config == self.RISK_CHECK_NEXT_STATE_FULL_FORECASTING or \
                        self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP:
                    simulation_steps = 1 + self._risk_state_backup_trajectory_steps
                    simulate_current_action = True
                else:
                    simulation_steps = self._risk_state_backup_trajectory_steps
                    simulate_current_action = False

                if self._risk_network_use_state_and_action or self._risk_state_config == self.RISK_CHECK_CURRENT_STATE:
                    ground_truth_state = self._risk_observation.copy()
                else:
                    ground_truth_state = None

                for i in range(simulation_steps):
                    if i == 0 and simulate_current_action:
                        simulated_action = action
                    else:
                        backup_action = np.array(
                            self._backup_agent.compute_action(self._risk_observation, full_fetch=False),
                            dtype=np.float64)
                        simulated_action = self._get_action_from_backup_action(backup_action)

                    _, _, termination, truncation, _ = self.step(simulated_action)

                    simulation_done = termination or truncation

                    if not simulation_done and ground_truth_state is None:
                        ground_truth_state = self._risk_observation.copy()

                    if simulation_done and self.termination_reason != self.TERMINATION_TRAJECTORY_LENGTH:
                        ground_truth_risk = 1.0
                        break

                self.switch_back_to_main_client(stored_variables)

                if ground_truth_state is not None:
                    self._risk_ground_truth_dict["state"].append(list(ground_truth_state))

                    if self._risk_network_use_state_and_action:
                        self._risk_ground_truth_dict["action"].append(list(motor_action))

                    self._risk_ground_truth_dict["risk"].append(ground_truth_risk)
                    self._risk_ground_truth_dict["risk_prediction"].append(self._last_risk_prediction)

        if action_considered_as_risky and self._risk_ignore_estimation_probability != 0.0:
            # option to overwrite the risk estimation with a certain probability when generating risk training data
            if np.random.uniform(low=0.0, high=1.0) < self._risk_ignore_estimation_probability:
                action_considered_as_risky = False

        return action_considered_as_risky

    @resampling_decorator
    def is_action_safe(self, action):
        motor_action, _, _, _ = self._split_action(action)
        action_is_safe = True
        valid_resampling_attempt = False
        end_acceleration = None

        if self._risk_state_config == self.RISK_CHECK_NEXT_STATE_KINEMATIC_FORECASTING or \
                self._risk_state_config == self.RISK_CHECK_NEXT_STATE_FULL_FORECASTING:
            end_acceleration = self._compute_end_acceleration_from_motor_action(motor_action)

        if self._risk_config is not None and self._risk_state_config != self.RISK_CHECK_CURRENT_STATE and not \
                self._risk_use_backup_agent_for_initial_backup_trajectory_only:
            valid_resampling_attempt = True
            action_is_safe = not self._is_action_risky(action, motor_action, end_acceleration)

        if action_is_safe and self._robot_scene.obstacle_wrapper.use_braking_trajectory_method:
            valid_resampling_attempt = True
            if not self._brake:
                if end_acceleration is None:
                    end_acceleration = self._compute_end_acceleration_from_motor_action(motor_action)

                execute_braking_trajectory = self._robot_scene.obstacle_wrapper.check_braking_trajectory_method(
                    current_acc=self._start_acceleration,
                    current_vel=self._start_velocity,
                    current_pos=self._start_position,
                    target_acc=end_acceleration,
                    time_step_counter=self._current_trajectory_point_index)

                action_is_safe = not execute_braking_trajectory

        if not valid_resampling_attempt:
            raise ValueError("Action resampling is activated but the current config does not make use of action "
                             "resampling.")

        return action_is_safe

    def _disconnect_physic_clients(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        for i in range(self._num_physic_clients):
            p.disconnect(physicsClientId=i)

    def set_seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            logging.info("Setting seed to %s", self._seed)
            np.random.seed(seed)
            random.seed(seed)
            super().reset(seed=seed)

        return [seed]

    @staticmethod
    def _to_list(dictionary):
        for key, value in dictionary.items():
            dictionary[key] = np.asarray(value).tolist()
        return dictionary

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError()

    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def display_plot(self):
        raise NotImplementedError()

    @abstractmethod
    def _save_plot(self, class_name, experiment_name):
        raise NotImplementedError()

    @abstractmethod
    def acc_braking_function(self):
        raise NotImplementedError()

    def compute_next_acc_min_and_next_acc_max(self, start_position, start_velocity, start_acceleration):
        raise NotImplementedError()

    def compute_violation_code_per_joint(self, joint_index, start_position, start_velocity, start_acceleration):
        raise NotImplementedError()

    @abstractmethod
    def _compute_controller_setpoints_from_action(self, action):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_position(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_velocity(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_acceleration(self, step):
        raise NotImplementedError()

    def _sim_step(self):
        self._robot_scene.obstacle_wrapper.prepare_sim_step()
        p.stepSimulation(physicsClientId=self._simulation_client_id)
        self._control_step_counter += 1

    def _prepare_for_next_action(self):
        return

    def _prepare_for_end_of_episode(self):
        return

    def _check_termination(self, robot_stopped):
        done = False

        if not done and self._terminate_on_self_collision:
            if self._self_collision_detected:
                done = True
                self._termination_reason = self.TERMINATION_SELF_COLLISION

        if not done and self._terminate_on_collision_with_static_obstacle:
            if self._collision_with_static_obstacle_detected:
                done = True
                self._termination_reason = self.TERMINATION_COLLISION_WITH_STATIC_OBSTACLE

        if not done and self._terminate_on_collision_with_moving_obstacle:
            if self._collision_with_moving_obstacle_detected:
                done = True
                self._termination_reason = self.TERMINATION_COLLISION_WITH_MOVING_OBSTACLE

        if not done:
            done, termination_reason = \
                self._trajectory_manager.is_trajectory_finished(self._episode_length)
            if done:
                self._termination_reason = termination_reason

        return done

    def switch_to_backup_client(self, nested_env=False):
        if not nested_env:
            # switch the simulation client to the backup client
            file_handle, temp_path = tempfile.mkstemp()
            try:
                p.saveBullet(temp_path, physicsClientId=self._main_client_id)
                p.restoreState(fileName=temp_path, physicsClientId=self._backup_client_id)
            finally:
                os.remove(temp_path)
                os.close(file_handle)

        # store variables
        stored_variables = self.copy_variables_to_dict()

        self._simulation_client_id = self._backup_client_id

        self._robot_scene.prepare_switch_to_backup_client(nested_env=nested_env)
        if not nested_env:
            self._trajectory_manager.trajectory_length = self._risk_state_backup_trajectory_steps + 2
        else:
            self._trajectory_manager.trajectory_length = 100000  # a big number to avoid termination

        self._episode_length = 0
        self._random_agent = False

        return stored_variables

    def switch_back_to_main_client(self, stored_variables, object=None):
        # switch the simulation client to the backup client by restoring stored_variables
        # this will also restore self._simulation_cliend_id = self._main_client_id
        def recursive_setattr(object, key, value):
            key_split = key.split(".", 1)
            if len(key_split) == 1:
                setattr(object, key_split[0], value)
            else:
                recursive_setattr(getattr(object, key_split[0]), key_split[1], value)

        if object is None:
            object = self

        for key, value in stored_variables.items():
            recursive_setattr(object, key, value)

        self._robot_scene.switch_back_to_main_client()

    def copy_variables_to_dict(self, object=None, dictionary=None, prefix="", debug=True, memo=None):
        if object is None:
            object = self
        if dictionary is None:
            dictionary = {}
        if memo is None:
            memo = {}

        if hasattr(object, "_do_not_copy_keys"):
            object._do_not_copy_keys.append("_do_not_copy_keys")
            ignored_keys = object._do_not_copy_keys
        else:
            ignored_keys = []
            
        if hasattr(object, "_recursive_copy_keys"):
            recursive_copy_keys = object._recursive_copy_keys
            ignored_keys.append("_recursive_copy_keys")
        else:
            recursive_copy_keys = []
            
        simple_types = [int, float, np.float64, np.float32, bool, str, type(None)]
        deep_copy_types = [list, tuple, dict]
        unduplicated_keys = []

        for key, value in object.__dict__.items():
            if key not in ignored_keys:
                if key in recursive_copy_keys:
                    dictionary = self.copy_variables_to_dict(object=value, dictionary=dictionary,
                                                             prefix="{}{}.".format(prefix, key),
                                                             memo=memo)
                elif type(value) in simple_types:
                    dictionary[prefix + key] = value
                elif type(value) in deep_copy_types:
                    dictionary[prefix + key] = copy.deepcopy(value, memo)
                elif type(value) == np.ndarray:
                    dictionary[prefix + key] = np.copy(value)
                else:
                    unduplicated_keys.append(key)

        if debug and unduplicated_keys:
            if prefix != "":
                print("Did not copy the following variables in {}: {}".format(prefix[1:-1], unduplicated_keys))
            else:
                print("Did not copy the following variables: {}".format(unduplicated_keys))

        return dictionary

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @property
    def pid(self):
        return self._pid

    @property
    def evaluation_dir(self):
        return self._evaluation_dir

    @property
    def use_real_robot(self):
        return self._use_real_robot

    @property
    def episode_counter(self):
        return self._episode_counter

    @property
    def precomputation_time(self):
        if self._control_rate is not None and hasattr(self._control_rate, 'precomputation_time'):
            return self._control_rate.precomputation_time
        else:
            return None

    @property
    def precomputation_timer(self):
        return self._precomputation_timer

    @property
    def use_moving_objects(self):
        return self._use_moving_objects

    @property
    def termination_reason(self):
        return self._termination_reason

    @property
    def max_resampling_attempts(self):
        return self._max_resampling_attempts

    @property
    def risk_network_use_state_and_action(self):
        return self._risk_network_use_state_and_action

    @property
    def risk_check_initial_backup_trajectory(self):
        return self._risk_check_initial_backup_trajectory

    @property
    def risk_use_backup_agent_for_initial_backup_trajectory_only(self):
        return self._risk_use_backup_agent_for_initial_backup_trajectory_only

    @property
    def robot_scene(self):
        return self._robot_scene

    @property
    @abstractmethod
    def pos_limits_min_max(self):
        pass

    @property
    @abstractmethod
    def vel_limits_min_max(self):
        pass

    @property
    @abstractmethod
    def acc_limits_min_max(self):
        pass

    @abstractmethod
    def _get_safe_acc_range(self):
        pass

    @property
    @abstractmethod
    def reward_maximum_relevant_distance(self):
        pass

    @property
    @abstractmethod
    def reward_consider_moving_obstacles(self):
        pass

    @abstractmethod
    def _reset_plotter(self, initial_joint_position, initial_joint_velocity, initial_joint_acceleration):
        pass

    @abstractmethod
    def _add_actual_position_to_plot(self, actual_position):
        pass

    @abstractmethod
    def _add_computed_actual_position_to_plot(self, computed_position_is, computed_velocity_is,
                                              computed_acceleration_is):
        pass

    @abstractmethod
    def _add_baseline_position_to_plot(self, baseline_position_is, baseline_velocity_is, baseline_acceleration_is):
        pass

    @abstractmethod
    def _add_actual_torques_to_plot(self, actual_torques):
        pass

    @abstractmethod
    def _add_value_to_plot(self, value):
        pass

    @abstractmethod
    def _calculate_safe_acc_range(self, start_position, start_velocity, start_acceleration, trajectory_point_index):
        pass

    def _get_trajectory_start_position(self):
        return self._trajectory_manager.get_trajectory_start_position()

    def _get_trajectory_start_velocity(self):
        return self._trajectory_manager.get_trajectory_start_velocity()

    def _get_trajectory_start_acceleration(self):
        return self._trajectory_manager.get_trajectory_start_acceleration()

    def _get_generated_trajectory_point(self, index, key='positions'):
        return self._trajectory_manager.get_generated_trajectory_point(index, key)

    def _get_measured_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        return self._trajectory_manager.get_measured_actual_trajectory_control_point(index, start_at_index, key)

    def _get_computed_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        return self._trajectory_manager.get_computed_actual_trajectory_control_point(index, start_at_index, key)

    def _get_generated_trajectory_control_point(self, index, key='positions'):
        return self._trajectory_manager.get_generated_trajectory_control_point(index, key)

    def _add_generated_trajectory_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_generated_trajectory_point(position, velocity, acceleration)

    def _add_measured_actual_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_measured_actual_trajectory_control_point(position, velocity, acceleration)

    def _add_computed_actual_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_computed_actual_trajectory_control_point(position, velocity, acceleration)

    def _add_generated_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_generated_trajectory_control_point(position, velocity, acceleration)


def normalize_joint_values(values, joint_limits):
    return list(np.asarray(values) / np.asarray(joint_limits))


class RiskDummyEnv(gym.Env):
    def __init__(self,
                 observation_size,
                 action_size):

        self.observation_space = Box(low=np.float32(-1), high=np.float32(1), shape=(observation_size,),
                                     dtype=np.float32)

        self.action_space = Box(low=np.float32(-1), high=np.float32(1),
                                shape=(action_size,), dtype=np.float32)




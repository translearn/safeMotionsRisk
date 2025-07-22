# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import inspect
import itertools
import logging
import os
import re
import warnings
import json
import types

import numpy as np
import pybullet as p
import copy
from klimits import compute_distance_c
from klimits import interpolate_position_batch as interpolate_position_batch
from klimits import interpolate_velocity_batch as interpolate_velocity_batch
from klimits import interpolate_position as interpolate_position
from klimits import interpolate_velocity as interpolate_velocity
from klimits import denormalize
from klimits import normalize

USE_CACHED_GRAPHICS_SHAPES = True

OBSERVED_POINT_NO_INFLUENCE_COLOR = (0, 1, 0, 0.5)  # green
OBSERVED_POINT_INFLUENCE_COLOR = (255 / 255, 84 / 255, 0 / 255, 0.5)  # orange
OBSERVED_POINT_VIOLATION_COLOR = (1, 0, 0, 0.5)  # red
LINK_OBJECT_COLLISION_INFLUENCE_COLOR = (0 / 255, 0 / 255, 170 / 255, 1.0)
LINK_SELF_COLLISION_INFLUENCE_COLOR = (117 / 255, 5 / 255, 45 / 255, 1.0)
# braking trajectory due to potential self collision
LINK_TORQUE_INFLUENCE_COLOR = (1, 0.33, 0.0, 1.0)  # braking trajectory due to potential torque violation

TARGET_POINT_SIMULTANEOUS = 0
TARGET_POINT_ALTERNATING = 1
TARGET_POINT_SINGLE = 2

MOVING_OBJECT_SIMULTANEOUS = 0
MOVING_OBJECT_ALTERNATING = 1
MOVING_OBJECT_SINGLE = 2

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class ObstacleWrapperBase:
    def __init__(self,
                 robot_scene=None,
                 obstacle_scene=None,
                 visual_mode=False,
                 obstacle_client_update_steps_per_action=24,
                 observed_link_point_scene=0,
                 log_obstacle_data=False,
                 check_braking_trajectory_collisions=False,
                 check_braking_trajectory_torque_limits=False,
                 collision_check_time=None,
                 trajectory_time_step=None,
                 simulation_time_step=1 / 240,
                 acc_range_function=None,
                 acc_braking_function=None,
                 violation_code_function=None,
                 distance_calculation_check_observed_points=False,
                 closest_point_safety_distance=0.1,
                 observed_point_safety_distance=0.1,
                 print_stats=False,
                 starting_point_cartesian_range_scene=0,
                 use_target_points=True,
                 target_point_cartesian_range_scene=0,
                 target_point_relative_pos_scene=0,
                 target_point_radius=0.05,
                 target_point_sequence=TARGET_POINT_SIMULTANEOUS,
                 target_point_reached_reward_bonus=0,
                 target_point_use_actual_position=False,
                 moving_object_sequence=MOVING_OBJECT_SINGLE,
                 moving_object_active_number_single=1,
                 # True: Check if a target point is reached based on the actual position, False: Use setpoints
                 reward_maximum_relevant_distance=None,
                 reward_consider_moving_obstacles=False,
                 moving_object_behind_the_robot_x_value=-1,
                 *vargs,
                 **kwargs):

        self._robot_scene = robot_scene
        self._obstacle_scene = obstacle_scene
        self._visual_mode = visual_mode
        self._observed_link_point_scene = observed_link_point_scene
        self._obstacle_list = []
        self._links_in_use = []
        self._links = []
        self._log_obstacle_data = log_obstacle_data
        self._trajectory_time_step = trajectory_time_step
        self._simulation_time_step = simulation_time_step
        self._obstacle_client_update_steps_per_action = obstacle_client_update_steps_per_action

        self._simulation_steps_per_action = int(round(self._trajectory_time_step / self._simulation_time_step))
        self._update_time_step = self._trajectory_time_step / self._obstacle_client_update_steps_per_action

        if collision_check_time is None:
            self._collision_check_time = 0.05
        else:
            self._collision_check_time = collision_check_time
        self._collision_checks_per_time_step = round(max(1, self._trajectory_time_step / self._collision_check_time))

        self._acc_range_function = acc_range_function
        self._acc_braking_function = acc_braking_function
        self._violation_code_function = violation_code_function
        self._check_braking_trajectory_collisions = check_braking_trajectory_collisions
        self._check_braking_trajectory_torque_limits = check_braking_trajectory_torque_limits
        self._braking_trajectory = None
        self._affected_observed_point = None
        self._affected_link_index_list = None
        self._braking_trajectory_collision_free = None
        self._braking_trajectory_torque_limited = None
        self._valid_braking_trajectories = None
        self._use_braking_trajectory_method = self._check_braking_trajectory_collisions \
            or self._check_braking_trajectory_torque_limits

        self._distance_calculation_check_observed_points = distance_calculation_check_observed_points
        self._distance_calculation_check_closest_points = True
        self._closest_point_safety_distance = closest_point_safety_distance
        self._observed_point_safety_distance = observed_point_safety_distance

        if self._check_braking_trajectory_collisions and not self._distance_calculation_check_closest_points \
                and not self._distance_calculation_check_observed_points:
            logging.warning("Warning: check_braking_trajectory_collisions is True but neither closest points nor "
                            "observed points are checked for collisions")

        self._print_stats = print_stats

        self._episode_counter = 0
        self._simulation_steps_per_action = None
        self._mean_num_points_in_safety_zone = 0
        self._mean_num_points_in_collision_zone = 0
        self._num_points_in_safety_zone_list = []
        self._num_points_in_collision_zone_list = []
        self._braking_duration_list = []  # duration of all computed braking trajectories
        self._active_braking_duration_list = []  # duration of all braking trajectories that led to action adaption
        self._active_braking_influence_time_list = []
        # for each introduced braking trajectory, the time that the action is influenced
        self._active_braking_influence_time = None

        self._mean_time_in_collision_zone = 0
        self._mean_time_in_safety_zone = 0
        self._mean_time_influenced_by_braking_trajectory_collision = 0
        self._mean_time_influenced_by_braking_trajectory_torque = 0

        self._time_in_object_observed_point_collision_zone_list = []
        self._time_in_object_closest_point_collision_zone_list = []
        self._time_in_self_collision_zone_list = []
        self._time_in_any_collision_zone_list = []
        self._time_in_safety_zone_list = []
        self._time_influenced_by_braking_trajectory_collision_list = []
        self._time_influenced_by_braking_trajectory_torque_list = []

        self._use_target_points = use_target_points
        self._target_point_radius = target_point_radius
        self._target_point_joint_pos_list = None
        self._sample_new_target_point_list = None
        self._sample_new_moving_object_list = None

        self._target_point_list = [[] for _ in range(self._robot_scene.num_robots)]

        self._target_position = None
        self._target_velocity = None
        self._actual_position = None
        self._actual_velocity = None
        self._last_actual_position = None
        self._last_actual_velocity = None

        if self._robot_scene.robot_name == "iiwa7":
            if starting_point_cartesian_range_scene == 0:
                self._starting_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                        [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            elif starting_point_cartesian_range_scene == 1:
                self._starting_point_cartesian_range = [[-1.1, 1.1], [-1.1, 1.1],
                                                        [-0.1, 1.5]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            elif starting_point_cartesian_range_scene == 2:
                self._starting_point_cartesian_range = [[-0.1, 0.6], [-0.8, 0.8],
                                                        [-0.1, 1.2]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            else:
                self._starting_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                        [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            self._target_point_relative_pos_min_max = np.array([[-1.6, -2, -1.5], [1.6, 2, 1.5]])

        elif self._robot_scene.robot_name.startswith("human"):
            self._starting_point_cartesian_range = [[0.0, 0.6], [-0.8, 0.8],
                                                    [0.075, 0.75]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]
            self._target_point_relative_pos_min_max = np.array([[-1.4, -1.6, -1.5], [1.4, 1.6, 1.5]])

        if target_point_cartesian_range_scene == 0:
            self._target_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        elif target_point_cartesian_range_scene == 1:
            self._target_point_cartesian_range = [[-0.6, 0.6], [-0.3, 0.3],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        elif target_point_cartesian_range_scene == 2:
            self._target_point_cartesian_range = [[-0.4, 0.4], [-0.4, 0.4],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        elif target_point_cartesian_range_scene == 3:
            self._target_point_cartesian_range = [[-0.2, 0.8], [-0.9, 0.9],
                                                  [0.3, 1.45]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 4:
            self._target_point_cartesian_range = [[-0.8, 0.8], [-0.9, 0.9],
                                                  [0.3, 1.45]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 5:
            self._target_point_cartesian_range = [[-0.1, 0.8], [-0.9, 0.9],
                                                  [0.3, 1.45]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 6:
            self._target_point_cartesian_range = [[-0.1, 0.8], [-1.0, 1.0],
                                                  [0.15, 1.65]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 7:
            self._target_point_cartesian_range = [[-0.1, 0.75], [-0.8, 0.8],
                                                  [1.0, 1.65]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 8:
            self._target_point_cartesian_range = [[-1.0, 1.0], [-1.0, 1.0],
                                                  [-0.7, 1.3]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 9:  # human
            self._target_point_cartesian_range = [[0.0, 0.6], [-0.8, 0.8],
                                                  [0.075, 0.75]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 10:  # iiwa, only in front of the robot
            self._target_point_cartesian_range = [[-0.1, 0.6], [-0.8, 0.8],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        else:
            self._target_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        self._log_target_point_relative_pos_min_max = True
        # whether to add the maximum target_point_relative_pos to info
        self._target_point_relative_pos_min_max_log = [[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]]

        self._target_point_cartesian_range_min_max = np.array(self._target_point_cartesian_range).T
        self._target_link_pos_list = [None] * self._robot_scene.num_robots
        self._target_point_pos_list = [None] * self._robot_scene.num_robots
        self._target_point_deterministic_list = None
        self._target_point_deterministic_list_index = None
        self._last_target_point_distance_list = [None] * self._robot_scene.num_robots
        self._target_point_pos_norm_list = [None] * self._robot_scene.num_robots
        self._target_point_joint_pos_norm_list = [None] * self._robot_scene.num_robots
        self._target_point_reached_list = [None] * self._robot_scene.num_robots
        self._target_point_id_reuse_list = None
        self._initial_target_point_distance_list = [[np.nan] for _ in range(self._robot_scene.num_robots)]
        self._num_target_points_reached_list = [0] * self._robot_scene.num_robots
        self._starting_point_sampling_attempts = 0
        self._target_point_sampling_attempts_list = [[] for _ in range(self._robot_scene.num_robots)]

        self._braking_trajectory_minimum_distance = np.inf
        self._braking_trajectory_maximum_rel_torque = 0

        self._torque_limits = None
        self._braking_timeout = False

        self._target_point_sequence = target_point_sequence
        # 0: target points for all robots, 1: alternating target points
        self._target_point_active_list = [False] * self._robot_scene.num_robots
        self._target_point_reached_reward_bonus = target_point_reached_reward_bonus
        self._target_link_point_list = []
        self._target_link_index_list = []

        self._moving_object_sequence = moving_object_sequence
        self._moving_object_active_number_single = moving_object_active_number_single

        if self._moving_object_sequence == MOVING_OBJECT_SINGLE and \
                self._moving_object_active_number_single > self._robot_scene.num_robots:
            raise ValueError("moving_object_active_number_single greater than num_robots")
        if self._moving_object_sequence == MOVING_OBJECT_SIMULTANEOUS:
            self._moving_object_active_number = self._robot_scene.num_robots
        elif self._moving_object_sequence == MOVING_OBJECT_SINGLE:
            self._moving_object_active_number = self._moving_object_active_number_single
        else:
            self._moving_object_active_number = 1  # alternating
        self._moving_object_active_list = [False] * self._robot_scene.num_robots
        self._moving_object_initial_active_list = [False] * self._robot_scene.num_robots
        self._moving_object_list = [[] for _ in range(self._robot_scene.num_robots)]
        self._moving_object_id_reuse_list = None
        self._moving_object_deterministic_list = None
        self._moving_object_deterministic_list_index = None
        self._moving_object_fade_out_list = None
        self._moving_object_distance_list = None
        self._moving_object_observation_robot_indices = None
        self._moving_object_behind_the_robot_x_value = moving_object_behind_the_robot_x_value

        self._human = None

        if self._robot_scene.moving_object_sphere_center is None:
            self._moving_object_position_min_max = np.array([
                [self._moving_object_behind_the_robot_x_value,
                 self._robot_scene.moving_object_area_center[0]],  # [x_min, x_max]
                [-2.0, 2.0],  # [y_min, y_max]
                [-0.3, 2.5]]).T  # [z_min, z_max]
            self._moving_object_direction_min_max = np.array([
                [-1.0, -0.94],  # [x_min, x_max]
                [-0.35, 0.35],  # [y_min, y_max]
                [-0.35, 0.35]]).T  # [z_min, z_max]
            self._moving_object_final_object_position_min_max = np.array([
                [self._moving_object_behind_the_robot_x_value, self._moving_object_behind_the_robot_x_value],
                # [x_min, x_max]
                [-1.3, 1.3],  # [y_min, y_max]
                [-0.3, 1.5]]).T  # [z_min, z_max]
        else:
            self._moving_object_final_object_position_min_max = np.array([
                [-1.0, 1.0],
                # [x_min, x_max]
                [-1.3, 1.3],  # [y_min, y_max]
                [-0.3, 1.5]]).T  # [z_min, z_max]
            self._moving_object_position_min_max = np.array([
                [-(self._robot_scene.moving_object_sphere_center[0] + self._robot_scene.moving_object_sphere_radius),
                 (self._robot_scene.moving_object_sphere_center[0] + self._robot_scene.moving_object_sphere_radius)],
                # [x_min, x_max]
                [-(self._robot_scene.moving_object_sphere_center[1] + self._robot_scene.moving_object_sphere_radius),
                 (self._robot_scene.moving_object_sphere_center[1] + self._robot_scene.moving_object_sphere_radius)],
                # [y_min, y_max]
                [self._moving_object_final_object_position_min_max[0][2],
                 self._moving_object_final_object_position_min_max[1][2]]]).T  # [z_min, z_max]
            self._moving_object_direction_min_max = np.array([
                [-1.0, 1.0],  # [x_min, x_max]
                [-1.0, 1.0],  # [y_min, y_max]
                [-1.0, 1.0]]).T  # [z_min, z_max]

        if self._robot_scene.moving_object_sphere_center is None:
            max_initial_height = self._robot_scene.moving_object_area_center[2] \
                + 0.5 * self._robot_scene.moving_object_area_width_height[1]
        else:
            max_initial_height = self._robot_scene.moving_object_sphere_center[2] + \
                                 self._robot_scene.moving_object_sphere_height_min_max[1]

        max_height_time = self._robot_scene.moving_object_speed_meter_per_second / 9.81
        max_height = max_initial_height + self._robot_scene.moving_object_speed_meter_per_second * max_height_time \
            - 0.5 * 9.81 * max_height_time ** 2
        self._moving_object_position_min_max[1][2] = max_height

        zero_height_time, _ = Ball.get_target_height_time(
            initial_height=max_initial_height,
            initial_z_speed=-self._robot_scene.moving_object_speed_meter_per_second,
            target_height=self._robot_scene.ball_radius,
            no_update_steps=True)

        min_speed = -self._robot_scene.moving_object_speed_meter_per_second - 9.81 * zero_height_time

        self._moving_object_ball_velocity_min_max = np.array([
            [-self._robot_scene.moving_object_speed_meter_per_second,
             self._robot_scene.moving_object_speed_meter_per_second],  # [x_min, x_max]
            [-self._robot_scene.moving_object_speed_meter_per_second,
             self._robot_scene.moving_object_speed_meter_per_second],  # [y_min, y_max]
            [min_speed, -min_speed]]).T  # [z_min, z_max]

        self._target_point_use_actual_position = target_point_use_actual_position

        self._reward_consider_moving_obstacles = reward_consider_moving_obstacles

        if reward_maximum_relevant_distance is None:
            self._closest_point_maximum_relevant_distance = self._closest_point_safety_distance + 0.002
            # for performance purposes the exact distance between links / obstacles is only computed if the
            # distance is smaller than self._closest_point_maximum_relevant_distance
        else:
            # set the maximum relevant distance for computing closest points based on the reward settings
            # -> higher values require more computational effort
            if reward_maximum_relevant_distance <= self._closest_point_safety_distance:
                raise ValueError("reward_maximum_relevant_distance {} needs to be greater than "
                                 "closest_point_safety_distance {}".format(reward_maximum_relevant_distance,
                                                                           self._closest_point_safety_distance))
            else:
                self._closest_point_maximum_relevant_distance = reward_maximum_relevant_distance + 0.002

    @property
    def obstacle_scene(self):
        return self._obstacle_scene

    @property
    def use_target_points(self):
        return self._use_target_points

    @property
    def log_obstacle_data(self):
        return self._log_obstacle_data

    @property
    def num_obstacles(self):
        return len(self._obstacle_list)

    @property
    def target_point_sequence(self):
        return self._target_point_sequence

    @property
    def moving_object_sequence(self):
        return self._moving_object_sequence

    @property
    def moving_object_active_number(self):
        return self._moving_object_active_number

    @property
    def closest_point_safety_distance(self):
        return self._closest_point_safety_distance

    @property
    def target_point_cartesian_range_min_max(self):
        return self._target_point_cartesian_range_min_max

    @property
    def use_braking_trajectory_method(self):
        return self._use_braking_trajectory_method

    @property
    def num_target_points(self):
        num_target_points = 0
        for i in range(len(self._target_point_list)):
            num_target_points += len(self._target_point_list[i])

        return num_target_points

    def get_num_target_points_reached(self, robot=None):
        # if robot is None -> consider all robots, otherwise only the specified robot_index
        num_target_points_reached = 0
        for i in range(self._robot_scene.num_robots):
            if robot is None or robot == i:
                num_target_points_reached += self._num_target_points_reached_list[i]

        return num_target_points_reached

    def get_moving_object_info(self):

        hit_robot_list = []
        hit_obstacle_list = []
        hit_robot_or_obstacle_list = []
        missed_robot_list = []
        no_final_status_list = []

        for i in range(self._robot_scene.num_robots):
            hit_robot_count = 0
            hit_obstacle_count = 0
            hit_robot_or_obstacle_count = 0
            missed_robot_count = 0
            no_final_status_count = 0

            if self._moving_object_list[i]:
                for j in range(len(self._moving_object_list[i])):
                    no_final_status = False
                    if self._moving_object_list[i][j].check_final_state(missed_robot=True):
                        missed_robot_count += 1
                    elif self._moving_object_list[i][j].check_final_state(hit_robot=True):
                        hit_robot_count += 1
                        hit_robot_or_obstacle_count += 1
                    elif self._moving_object_list[i][j].check_final_state(hit_obstacle=True):
                        hit_obstacle_count += 1
                        hit_robot_or_obstacle_count += 1
                    else:
                        no_final_status_count += 1
                        no_final_status = True

            hit_robot_list.append(hit_robot_count)
            hit_obstacle_list.append(hit_obstacle_count)
            hit_robot_or_obstacle_list.append(hit_robot_or_obstacle_count)
            missed_robot_list.append(missed_robot_count)
            no_final_status_list.append(no_final_status_count)

        return hit_robot_list, hit_obstacle_list, hit_robot_or_obstacle_list, missed_robot_list, \
            no_final_status_list

    def _get_initial_target_point_distance(self, robot=None):
        # if robot is None -> consider all robots, otherwise only the specified robot_index
        if robot is None:
            return np.nanmean(list(itertools.chain.from_iterable(self._initial_target_point_distance_list)))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(self._initial_target_point_distance_list[robot])

    @property
    def obstacle(self):
        return self._obstacle_list + list(itertools.chain.from_iterable(self._target_point_list))

    @property
    def target_point(self):
        return list(itertools.chain.from_iterable(self._target_point_list))

    @property
    def links_in_use(self):
        return self._links_in_use

    @property
    def links(self):
        return self._links

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @property
    def torque_limits(self):
        return self._torque_limits

    @torque_limits.setter
    def torque_limits(self, val):
        self._torque_limits = val

    def get_indices_of_observed_links_in_use(self, obstacle_index):
        indices_of_observed_links_in_use = []
        if obstacle_index >= self.num_obstacles:
            obstacle_index = obstacle_index - self.num_obstacles
            obstacle_list = list(itertools.chain.from_iterable(self._target_point_list))
        else:
            obstacle_list = self._obstacle_list
        for i in range(len(self._links_in_use)):
            if self._links_in_use[i] in obstacle_list[obstacle_index].observed_links:
                indices_of_observed_links_in_use.append(i)
        return indices_of_observed_links_in_use

    def get_index_of_link_in_use(self, link_number):
        for i in range(len(self._links_in_use)):
            if self._links_in_use[i] == link_number:
                return i
        raise ValueError("Desired link is not in use")


class ObstacleWrapperSim(ObstacleWrapperBase):
    OBSTACLE_CLIENT_AT_OTHER_POSITION = 0
    OBSTACLE_CLIENT_AT_TARGET_POSITION = 1
    OBSTACLE_CLIENT_AT_ACTUAL_POSITION = 2
    OBSTACLE_CLIENT_AT_DEFAULT_POSITION = 2

    def __init__(self,
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 backup_client_id=None,
                 gui_client_id=None,
                 use_real_robot=False,
                 link_name_list=None,
                 manip_joint_indices=None,
                 visualize_bounding_spheres=False,
                 visualize_bounding_sphere_actual=True,
                 visualize_debug_lines=False,
                 target_link_name="iiwa_link_7",
                 target_link_offset=None,
                 activate_obstacle_collisions=False,
                 *vargs,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        if target_link_offset is None:
            target_link_offset = [0, 0, 0.0]
        if manip_joint_indices is None:
            manip_joint_indices = []

        self._simulation_client_id = simulation_client_id
        if self._log_obstacle_data and self._simulation_client_id is None:
            raise ValueError("log_obstacle_data requires an active physics client (physicsClientId is None)")
        self._obstacle_client_id = obstacle_client_id
        self._backup_client_id = backup_client_id
        self._gui_client_id = gui_client_id
        self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION
        self._use_real_robot = use_real_robot

        self._manip_joint_indices = manip_joint_indices

        self._visualize_bounding_spheres = visualize_bounding_spheres
        if self._visualize_bounding_spheres and not self._log_obstacle_data:
            raise ValueError("visualize_bounding_spheres requires log_obstacle_data to be True")
        self._visualize_bounding_sphere_actual = visualize_bounding_sphere_actual
        self._activate_obstacle_collisions = activate_obstacle_collisions

        self._visualize_debug_lines = visualize_debug_lines

        if target_link_name is not None:
            target_link_name_list = self._robot_scene.get_link_names_for_multiple_robots(target_link_name)
            if len(target_link_name_list) < self._robot_scene.num_robots:
                raise ValueError("Could not find a target link for each robot. Found " + str(target_link_name_list))
        else:
            target_link_name = None

        self._target_link_name = target_link_name

        closest_point_active_link_name_list = []

        if self._robot_scene.robot_name == "iiwa7":
            closest_point_active_link_name_list = ["iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                                   "iiwa_link_6", "iiwa_link_7"]
        elif self._robot_scene.robot_name == "human":
            closest_point_active_link_name_list = ["forearm", "hand"]

        if self._robot_scene.ball_machine_mode:
            closest_point_active_link_name_list += [self._target_link_name]

        closest_point_active_link_name_multiple_robots_list = \
            self._robot_scene.get_link_names_for_multiple_robots(closest_point_active_link_name_list)

        visual_shape_data = p.getVisualShapeData(self._robot_scene.robot_id,
                                                 flags=p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS)
        visualize_target_link_point = False  # bounding sphere around the target link point

        self._moving_object_target_link_point_list = []

        for i in range(len(link_name_list)):
            observed_points = self._specify_observed_points(link_name=link_name_list[i], link_index=i)
            self_collision_links = self._specify_self_collision_links(link_name=link_name_list[i],
                                                                      link_name_list=link_name_list)
            if link_name_list[i] in closest_point_active_link_name_multiple_robots_list:
                closest_point_active = True
            else:
                closest_point_active = False

            # link default color
            default_color = visual_shape_data[i][7]

            if self._use_target_points:
                for j in range(self._robot_scene.num_robots):
                    if link_name_list[i] in self._robot_scene.get_link_names_for_multiple_robots(self._target_link_name,
                                                                                                 robot_indices=[j]) \
                            and not self._robot_scene.robot_name.startswith("human"):

                        robot_index = j if self._target_point_sequence != TARGET_POINT_SINGLE else 0
                        transparency = 1.0

                        if not self._robot_scene.ball_machine_mode and not self._robot_scene.no_target_link_coloring:
                            default_color = self.get_target_point_color(robot=robot_index, transparency=transparency)

            self._links.append(
                LinkBase(name=link_name_list[i], observe_closest_point=True, closest_point_active=closest_point_active,
                         observed_points=observed_points, index=i,
                         closest_point_safety_distance=self._closest_point_safety_distance,
                         robot_id=self._robot_scene.robot_id,
                         robot_index=self._robot_scene.get_robot_index_from_link_name(link_name_list[i]),
                         self_collision_links=self_collision_links,
                         default_color=default_color,
                         simulation_client_id=self._simulation_client_id,
                         obstacle_client_id=self._obstacle_client_id,
                         use_real_robot=self._use_real_robot,
                         set_robot_position_in_obstacle_client_function=self.set_robot_position_in_obstacle_client,
                         ))

            if target_link_name_list is not None:
                if link_name_list[i] in target_link_name_list:
                    self._target_link_point_list.append(LinkPointBase(name="Target", offset=target_link_offset,
                                                                      bounding_sphere_radius=0.01
                                                                      if visualize_target_link_point else 0.0,
                                                                      safety_distance=0.0,
                                                                      active=False,
                                                                      visualize_bounding_sphere=
                                                                      visualize_target_link_point,
                                                                      num_clients=self._robot_scene.num_clients))
                    self._target_link_point_list[-1].link_object = self._links[-1]
                    self._target_link_index_list.append(i)

            if self._robot_scene.moving_object_aim_at_current_robot_position:
                if link_name_list[i] in self._robot_scene.moving_object_current_robot_position_target_link_list:
                    self._moving_object_target_link_point_list.append(LinkPointBase(name="Target_" + link_name_list[i],
                                                                                   offset=[0, 0, 0],
                                                                                   bounding_sphere_radius=0.0,
                                                                                   safety_distance=0.0,
                                                                                   active=False,
                                                                                   visualize_bounding_sphere=
                                                                                   False,
                                                                                   num_clients=
                                                                                   self._robot_scene.num_clients))
                    self._moving_object_target_link_point_list[-1].link_object = self._links[-1]


        if self._use_target_points:
            if len(self._target_link_point_list) < self._robot_scene.num_robots:
                raise ValueError("Could not find a target link named " + str(self._target_link_name) +
                                 " for each robot. Found " + str(self._target_link_point_list))

        # Visualize the distance between an obstacle and a selected point by a debug line
        self._debug_line = None
        self._debug_line_obstacle = 0
        self._debug_line_link = 0
        self._debug_line_point = 0  # 0: closest point if observed, else: first observed point

        visualize_starting_point_cartesian_range = False or self._visualize_debug_lines
        if visualize_starting_point_cartesian_range:
            self._visualize_cartesian_range(c_range=self._starting_point_cartesian_range, line_color=[1, 0, 0])
        visualize_target_point_cartesian_range = False or self._visualize_debug_lines
        if visualize_target_point_cartesian_range:
            self._visualize_cartesian_range(c_range=self._target_point_cartesian_range, line_color=[0, 0, 1])

        if self._robot_scene.use_moving_objects:
            self._moving_object_check_counter = None
            visualize_moving_object_release_area = False or self._visualize_debug_lines
            if visualize_moving_object_release_area:
                if self._robot_scene.moving_object_sphere_center is None:
                    self._visualize_cartesian_area(center=self._robot_scene.moving_object_area_center,
                                                   width=self._robot_scene.moving_object_area_width_height[0],
                                                   height=self._robot_scene.moving_object_area_width_height[1],
                                                   line_color=[1, 0, 0],
                                                   line_width=2)
                else:
                    self._visualize_spherical_segment(center=self._robot_scene.moving_object_sphere_center,
                                                      radius=self._robot_scene.moving_object_sphere_radius,
                                                      height_list=
                                                      [self._robot_scene.moving_object_sphere_height_min_max[0], 0,
                                                       self._robot_scene.moving_object_sphere_height_min_max[1]],
                                                      angle_min_max=self._robot_scene.moving_object_sphere_angle_min_max,
                                                      line_color=[1, 0, 0],
                                                      line_width=2,
                                                      adjust_radius=True)

            visualize_moving_object_position_range = False or self._visualize_debug_lines
            if visualize_moving_object_position_range:
                self._visualize_cartesian_range(c_range=self._moving_object_position_min_max.T,
                                                line_color=[1, 1, 1])

            self._moving_object_invalid_target_link_point_position_min_max = np.array([
                [-0.4, 0.0],
                # [x_min, x_max]
                [-0.2, 0.2],  # [y_min, y_max]
                [0.0, 0.5]]).T  # [z_min, z_max]

            visualize_moving_object_invalid_target_link_point_position_range = False or self._visualize_debug_lines
            if visualize_moving_object_invalid_target_link_point_position_range:
                self._visualize_cartesian_range(c_range=
                                                self._moving_object_invalid_target_link_point_position_min_max.T,
                                                line_color=[1, 0, 0])

        self._collision_avoidance_starting_point_selected_step = None
        self._collision_avoidance_starting_point_computation_steps = None

        self._planet_list = []
        self._planet_check_counter = None
        if self._robot_scene.planet_mode:
            observed_links, num_observed_points_per_link = \
                self.get_link_indices_in_link_name_list(self._robot_scene.planet_observed_link_names)
            if self._robot_scene.planet_one_center is not None:
                self._planet_list.append(Planet(center=self._robot_scene.planet_one_center,
                                                radius_xy=self._robot_scene.planet_one_radius_xy,
                                                euler_angles=self._robot_scene.planet_one_euler_angles,
                                                period=self._robot_scene.planet_one_period,
                                                update_time_step=self._update_time_step,
                                                rotations_per_period=1,
                                                default_orn=p.getQuaternionFromEuler([0, 0, -np.pi/2]),
                                                initial_time=-1,
                                                orbit_color=[90/255, 91/255, 118/255],
                                                urdf_file_name="ISS",
                                                urdf_scaling_factor=0.6,
                                                simulation_client_id=self._simulation_client_id,
                                                obstacle_client_id=self._obstacle_client_id,
                                                backup_client_id=self._backup_client_id,
                                                gui_client_id=self._gui_client_id,
                                                use_real_robot=self._use_real_robot,
                                                observed_links=observed_links,
                                                num_observed_points_per_link=num_observed_points_per_link,
                                                visual_mode=self._visual_mode,
                                                create_collision_shape_in_simulation_client=True,
                                                create_collision_shape_in_obstacle_client=True,
                                                update_position_in_obstacle_client=False,
                                                num_clients=self._robot_scene.num_clients
                                                ))
            if self._robot_scene.planet_two_center is not None:
                self._planet_list.append(Planet(center=self._robot_scene.planet_two_center,
                                                radius_xy=self._robot_scene.planet_two_radius_xy,
                                                euler_angles=self._robot_scene.planet_two_euler_angles,
                                                period=self._robot_scene.planet_two_period,
                                                time_shift=self._robot_scene.planet_two_time_shift,
                                                update_time_step=self._update_time_step,
                                                rotations_per_period=4,
                                                default_orn=p.getQuaternionFromEuler([0, 0, 0]),
                                                initial_time=-1,
                                                orbit_color=[0.5, 0.5, 0.5],
                                                urdf_file_name="asteroid",
                                                urdf_scaling_factor=1.0,
                                                simulation_client_id=self._simulation_client_id,
                                                obstacle_client_id=self._obstacle_client_id,
                                                backup_client_id=self._backup_client_id,
                                                gui_client_id=self._gui_client_id,
                                                use_real_robot=self._use_real_robot,
                                                observed_links=observed_links,
                                                num_observed_points_per_link=num_observed_points_per_link,
                                                visual_mode=self._visual_mode,
                                                create_collision_shape_in_simulation_client=True,
                                                create_collision_shape_in_obstacle_client=True,
                                                update_position_in_obstacle_client=False,
                                                num_clients=self._robot_scene.num_clients
                                                ))

                if self._robot_scene.planet_two_time_shift:
                    self._planet_list[0].coupled_planet = self._planet_list[1]

            visualize_planet_global_pos = False or self._visualize_debug_lines
            if visualize_planet_global_pos:
                self._visualize_cartesian_range(c_range=self._robot_scene.planet_obs_global_pos_min_max.T,
                                                line_color=[0/255, 91/255, 118/255])

        if self._robot_scene.human_network_checkpoint is not None:
            # set robot to default position in obstacle client
            self.set_robot_position_in_obstacle_client(set_to_default=True)
            observed_link_names = ["iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                   "iiwa_link_6"]
            if self._robot_scene.ball_machine_mode:
                observed_link_names += [self._target_link_name]
            else:
                observed_link_names.append("iiwa_link_7")

            observed_links, num_observed_points_per_link = self.get_link_indices_in_link_name_list(observed_link_names)

            self._human = \
                Human(human_network_checkpoint=self._robot_scene.human_network_checkpoint,
                      robot_id=self._robot_scene.robot_id,
                      observed_links=observed_links,
                      num_observed_points_per_link=num_observed_points_per_link,
                      risk_state_deterministic_backup_trajectory=
                      self._robot_scene.risk_state_deterministic_backup_trajectory,
                      use_collision_avoidance_starting_point_sampling=
                      self._robot_scene.human_network_use_collision_avoidance_starting_point_sampling,
                      collision_avoidance_kinematic_state_sampling_probability=
                      self._robot_scene.human_network_collision_avoidance_kinematic_state_sampling_probability,
                      collision_avoidance_stay_in_state_probability=
                      self._robot_scene.human_network_collision_avoidance_stay_in_state_probability,
                      trajectory_duration=self._robot_scene.trajectory_duration,
                      simulation_client_id=self._simulation_client_id,
                      obstacle_client_id=self._obstacle_client_id,
                      backup_client_id=self._backup_client_id,
                      use_gui=True if self._gui_client_id is not None else False,
                      no_link_coloring=self._robot_scene.no_link_coloring,
                      use_fixed_seed=self._robot_scene.use_fixed_seed,
                      num_clients=self._robot_scene.num_clients)

        self._do_not_copy_keys = ['_robot_scene', '_acc_range_function', '_acc_braking_function',
                                  '_violation_code_function', '_human']

    def _visualize_cartesian_range(self, c_range, line_color, line_width=2):
        # c_range: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        indices = [[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1, 1],
                   [0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 0, 1],
                   [0, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 1],
                   [1, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 1]]

        for client in [self._simulation_client_id, self._obstacle_client_id]:
            if client is not None:
                for index in indices:
                    p.addUserDebugLine([c_range[0][index[0]], c_range[1][index[1]], c_range[2][index[2]]],
                                       [c_range[0][index[3]], c_range[1][index[4]], c_range[2][index[5]]],
                                       lineColorRGB=line_color, lineWidth=line_width,
                                       physicsClientId=client)

    def _visualize_cartesian_area(self, center, width, height, line_color, line_width=2):
        # cartesian area with fixed x-value, width -> y-direction, height -> z-direction
        # centered around a center point -> center[x, y, z]
        indices = [[-1, -1, -1, 1],
                   [-1, -1, 1, -1],
                   [-1, 1, 1, 1],
                   [1, -1, 1, 1]]

        for client in [self._simulation_client_id, self._obstacle_client_id]:
            if client is not None:
                for index in indices:
                    p.addUserDebugLine([center[0], center[1] + index[0] * width / 2,
                                        center[2] + index[1] * height / 2],
                                       [center[0], center[1] + index[2] * width / 2,
                                        center[2] + index[3] * height / 2],
                                       lineColorRGB=line_color, lineWidth=line_width,
                                       physicsClientId=client)

    def _visualize_spherical_segment(self, center, radius, height_list, angle_min_max, line_color, line_width=2,
                                     adjust_radius=False):
        if angle_min_max[1] - angle_min_max[0] < 2 * np.pi - 0.01:
            draw_lines_to_center = True
        else:
            draw_lines_to_center = False
        for height in height_list:
            circular_segment_radius = np.sqrt(radius ** 2 - height ** 2) if adjust_radius else radius
            self._visualize_circular_segment(segment_lines=None, center=center, radius=circular_segment_radius,
                                             height=height, min_angle=angle_min_max[0], max_angle=angle_min_max[1],
                                             line_color=line_color, line_width=line_width,
                                             draw_lines_to_center=draw_lines_to_center)

    def _visualize_circular_segment(self, segment_lines, center, radius, height, min_angle, max_angle, line_color,
                                    line_width=2, draw_lines_to_center=False, lines_per_circle=40):
        # approximate circle with straight lines

        if segment_lines is None:
            segment_lines = []

        start = center + np.array([radius * np.cos(min_angle), radius * np.sin(min_angle), height])

        for i in range(1, lines_per_circle + 1):
            angle = min_angle + (max_angle - min_angle) * i / lines_per_circle
            end = center + np.array([radius * np.cos(angle), radius * np.sin(angle), height])

            if len(segment_lines) > i - 1:
                segment_lines[i - 1] = p.addUserDebugLine(start, end, lineColorRGB=line_color, lineWidth=line_width,
                                                          replaceItemUniqueId=segment_lines[i - 1])
            else:
                segment_lines.append(p.addUserDebugLine(start, end, lineColorRGB=line_color))

            start = end

        if draw_lines_to_center:
            start = center
            for i in range(2):
                angle = min_angle if i == 0 else max_angle
                end = center + np.array([radius * np.cos(angle), radius * np.sin(angle), height])
                if len(segment_lines) > lines_per_circle + i:
                    segment_lines[lines_per_circle + i] = \
                        p.addUserDebugLine(start, end, lineColorRGB=line_color,
                                           replaceItemUniqueId=segment_lines[lines_per_circle + i])
                else:
                    segment_lines.append(p.addUserDebugLine(start, end, lineColorRGB=line_color))

        return segment_lines

    def is_obstacle_client_at_other_position(self):
        if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_OTHER_POSITION:
            return True
        else:
            return False

    def prepare_switch_to_backup_client(self, nested_env=False):
        self._simulation_client_id = self._backup_client_id
        if self._use_target_points:
            # move target points in backup client to avoid the sampling of new target points
            self._target_point_deterministic_list_index = 0
            for i in range(self._robot_scene.num_robots):
                if self._target_point_active_list[i]:
                    self._target_point_list[i][-1].set_simulation_client_to_backup_client()
                    if not nested_env:
                        self._target_point_pos_list[i] = \
                            np.array(self._target_point_list[i][-1].get_position(actual=False))
                        self._target_point_list[i][-1].set_to_unreachable_position_in_backup_client()

        if self._robot_scene.planet_mode:
            for i in range(len(self._planet_list)):
                self._planet_list[i].set_simulation_client_to_backup_client()

        if self._robot_scene.use_moving_objects:
            for moving_object in list(itertools.chain.from_iterable(self._moving_object_list)):
                moving_object.set_simulation_client_to_backup_client()
            self._moving_object_deterministic_list_index = 0

        if self._human is not None:
            self._human.switch_to_backup_client()

    def switch_back_to_main_client(self):
        if self._human is not None:
            self._human.switch_back_to_main_client()

    def reset_obstacles(self):
        self._delete_all_target_points()
        self._delete_all_moving_objects()

        if not self._obstacle_list:
            self._add_obstacles()
        else:
            for obstacle in self._obstacle_list:
                obstacle.reset()

        if self._human is not None:
            # set robot to default position in obstacle client
            self.set_robot_position_in_obstacle_client(set_to_default=True)
            self._human.reset()

    def step(self):
        if self._human is not None:
            # set robot to default position in obstacle client
            self.set_robot_position_in_obstacle_client(set_to_default=True)
            self._human.step()

    def process_step_outcome(self):
        if self._human is not None:
            # set robot to default position in obstacle client
            self.set_robot_position_in_obstacle_client(set_to_default=True)
            self._human.process_step_outcome()

    def prepare_sim_step(self):
        if self._human is not None:
            self._human.prepare_sim_step()

    def process_end_of_episode(self):
        if self._human is not None:
            self._human.process_end_of_episode()

    def reset(self, start_position, start_velocity, start_acceleration, compute_initial_braking_trajectory=False):

        self._episode_counter += 1
        self._time_in_safety_zone_list = []
        self._time_in_object_observed_point_collision_zone_list = []
        self._time_in_object_closest_point_collision_zone_list = []
        self._time_in_self_collision_zone_list = []
        self._time_in_any_collision_zone_list = []
        self._num_points_in_safety_zone_list = []
        self._num_points_in_collision_zone_list = []
        self._time_influenced_by_braking_trajectory_collision_list = []
        self._time_influenced_by_braking_trajectory_torque_list = []
        self._braking_duration_list = []
        self._active_braking_duration_list = []
        self._active_braking_influence_time_list = []
        self._active_braking_influence_time = 0

        self._valid_braking_trajectories = {'current': None, 'last': None}
        self._braking_trajectory_minimum_distance = np.inf
        self._braking_trajectory_maximum_rel_torque = 0

        self._braking_timeout = False

        self._target_position = start_position
        self._target_velocity = start_velocity
        self._actual_position = start_position
        self._actual_velocity = start_velocity
        self._last_actual_position = None
        self._last_actual_velocity = None

        self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION

        self._affected_observed_point = None
        self._affected_link_index_list = None
        self._braking_trajectory_collision_free = None
        self._braking_trajectory_torque_limited = None

        for i in range(len(self._links)):
            self._links[i].reset()

        if self._use_target_points:
            self._target_point_sampling_attempts_list = [[] for _ in range(self._robot_scene.num_robots)]
            self._last_target_point_distance_list = [None] * self._robot_scene.num_robots
            self._sample_new_target_point_list = [False] * self._robot_scene.num_robots
            self._target_point_pos_norm_list = [None] * self._robot_scene.num_robots
            self._target_point_joint_pos_list = [None] * self._robot_scene.num_robots
            self._target_point_joint_pos_norm_list = [None] * self._robot_scene.num_robots
            self._target_point_active_list = [False] * self._robot_scene.num_robots
            self._target_point_id_reuse_list = []
            self._target_point_deterministic_list = []
            self._target_point_reached_list = [False] * self._robot_scene.num_robots
            self._initial_target_point_distance_list = [[np.nan] for _ in range(self._robot_scene.num_robots)]
            self._num_target_points_reached_list = [0] * self._robot_scene.num_robots

            self._target_point_relative_pos_min_max_log = [[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]]

            if self._target_point_sequence == TARGET_POINT_SIMULTANEOUS:
                self._target_point_active_list = [True] * self._robot_scene.num_robots
            else:
                target_point_active_robot_index = np.random.randint(0, self._robot_scene.num_robots)
                self._target_point_active_list[target_point_active_robot_index] = True

            active_robots_list = []
            for i in range(self._robot_scene.num_robots):
                if self._target_point_active_list[i]:
                    active_robots_list.append(i)
            np.random.shuffle(active_robots_list)
            for i in range(len(active_robots_list)):
                self._add_target_point(robot=active_robots_list[i])

        if self._robot_scene.use_moving_objects:
            self._moving_object_check_counter = 0
            self._moving_object_active_list = [False] * self._robot_scene.num_robots
            self._sample_new_moving_object_list = [False] * self._robot_scene.num_robots
            self._moving_object_distance_list = [[] for _ in range(self._robot_scene.num_robots)]
            self._moving_object_fade_out_list = [[] for _ in range(self._robot_scene.num_robots)]
            self._moving_object_id_reuse_list = []
            self._moving_object_deterministic_list = []
            self._moving_object_deterministic_list_index = 0
            if self._moving_object_sequence == MOVING_OBJECT_SIMULTANEOUS:
                self._moving_object_active_list = [True] * self._robot_scene.num_robots
            elif self._moving_object_sequence == MOVING_OBJECT_SINGLE:
                for i in range(self._moving_object_active_number_single):
                    available_robots = []
                    for j in range(self._robot_scene.num_robots):
                        if not self._moving_object_active_list[j]:
                            available_robots.append(j)
                    moving_object_active_robot_index = available_robots[np.random.randint(0, len(available_robots))]
                    self._moving_object_active_list[moving_object_active_robot_index] = True
            else:
                # alternating
                moving_object_active_robot_index = np.random.randint(0, self._robot_scene.num_robots)
                self._moving_object_active_list[moving_object_active_robot_index] = True
            self._moving_object_initial_active_list = self._moving_object_active_list.copy()

            if self._moving_object_sequence == MOVING_OBJECT_SINGLE:
                self._moving_object_observation_robot_indices = \
                    np.full(self._moving_object_active_number_single, fill_value=-1)
            else:
                self._moving_object_observation_robot_indices = np.arange(0, self._robot_scene.num_robots)

            active_robots_list = []
            for i in range(self._robot_scene.num_robots):
                if self._moving_object_active_list[i]:
                    active_robots_list.append(i)
            np.random.shuffle(active_robots_list)
            for i in range(len(active_robots_list)):
                self._add_moving_object(robot=active_robots_list[i])
                if self._robot_scene.moving_object_random_initial_position:
                    moving_object = self._moving_object_list[active_robots_list[i]][-1]
                    valid_initial_position_found = False

                    max_update_step_counter = moving_object.max_time_update_step_counter
                    if moving_object.obstacle_hit_data is not None \
                            and moving_object.obstacle_hit_data['obstacle_hit_time'] < max_update_step_counter:
                        max_update_step_counter = moving_object.obstacle_hit_data['obstacle_hit_time']

                    max_distance_time_step = int(np.floor(max_update_step_counter /
                                                          self._obstacle_client_update_steps_per_action))

                    while not valid_initial_position_found:
                        initial_time_step = np.random.randint(low=0, high=max_distance_time_step + 1)
                        initial_update_step = initial_time_step * self._obstacle_client_update_steps_per_action
                        moving_object.set_position_update_step_counter(initial_update_step)
                        # check for collisions
                        p.performCollisionDetection(physicsClientId=self._simulation_client_id)
                        collision_detected = moving_object.check_if_object_is_colliding()

                        if not collision_detected:
                            valid_initial_position_found = True


        if self._robot_scene.planet_mode:
            self._planet_check_counter = 0
            if self._planet_list:
                for i in range(len(self._planet_list)):
                    self._planet_list[i].reset()

        if compute_initial_braking_trajectory and self._use_braking_trajectory_method:
            # compute a braking trajectory even for the first time step (e.g. if start velocity / acceleration != 0)
            self._braking_trajectory = {'acceleration': [start_acceleration],
                                        'velocity': [],
                                        'position': [],
                                        'min_distance': [],
                                        'max_torque': []}

            robot_stopped, braking_timeout = self._compute_next_braking_trajectory_time_step(
                start_position=start_position,
                start_velocity=start_velocity)

            while not (robot_stopped or braking_timeout):
                robot_stopped, braking_timeout = self._compute_next_braking_trajectory_time_step(
                    start_position=self._braking_trajectory['position'][-1])

            if len(self._braking_trajectory['acceleration']) > 1:
                self._valid_braking_trajectories['current'] = self._braking_trajectory
                self._valid_braking_trajectories['current']['acceleration'] = \
                    self._valid_braking_trajectories['current']['acceleration'][1:]  # remove start acceleration

    def get_info_and_print_stats(self, print_stats_every_n_episodes=1):
        episode_mean_time_influenced_by_braking_trajectory_collision = np.mean(
            self._time_influenced_by_braking_trajectory_collision_list)
        episode_mean_time_influenced_by_braking_trajectory_torque = np.mean(
            self._time_influenced_by_braking_trajectory_torque_list)
        info = {'obstacles_time_influenced_by_braking_trajectory':
                episode_mean_time_influenced_by_braking_trajectory_collision +
                episode_mean_time_influenced_by_braking_trajectory_torque,
                'obstacles_time_influenced_by_braking_trajectory_collision':
                    episode_mean_time_influenced_by_braking_trajectory_collision,
                'obstacles_time_influenced_by_braking_trajectory_torque':
                    episode_mean_time_influenced_by_braking_trajectory_torque,
                'obstacles_num_target_points_reached': float(self.get_num_target_points_reached()),
                'obstacles_starting_point_sampling_attempts': float(self._starting_point_sampling_attempts)
                }

        if self._use_target_points:
            info['obstacles_initial_target_point_distance'] = self._get_initial_target_point_distance()
            if self._robot_scene.num_robots > 1:
                for i in range(self._robot_scene.num_robots):
                    info['obstacles_num_target_points_reached_r' + str(i)] = \
                        float(self.get_num_target_points_reached(robot=i))
                    info['obstacles_initial_target_point_distance_r' + str(i)] = \
                        self._get_initial_target_point_distance(robot=i)

            for i in range(self._robot_scene.num_robots):
                if len(self._target_point_sampling_attempts_list[i]) > 0:
                    mean_sampling_attempts = float(np.mean(self._target_point_sampling_attempts_list[i]))
                else:
                    mean_sampling_attempts = 0.0
                info['obstacles_target_point_sampling_attempts_r' + str(i)] = mean_sampling_attempts

            if self._log_target_point_relative_pos_min_max:
                info['obstacles_target_point_relative_pos_min_x'] = self._target_point_relative_pos_min_max_log[0][0]
                info['obstacles_target_point_relative_pos_min_y'] = self._target_point_relative_pos_min_max_log[0][1]
                info['obstacles_target_point_relative_pos_min_z'] = self._target_point_relative_pos_min_max_log[0][2]
                info['obstacles_target_point_relative_pos_max_x'] = self._target_point_relative_pos_min_max_log[1][0]
                info['obstacles_target_point_relative_pos_max_y'] = self._target_point_relative_pos_min_max_log[1][1]
                info['obstacles_target_point_relative_pos_max_z'] = self._target_point_relative_pos_min_max_log[1][2]

        if self._robot_scene.use_moving_objects:

            hit_robot_list, hit_obstacle_list, hit_robot_or_obstacle_list, missed_robot_list, \
                no_final_status_list = self.get_moving_object_info()
            for i in range(self._robot_scene.num_robots):
                info['moving_object_hit_robot_r' + str(i)] = \
                    float(hit_robot_list[i])
                info['moving_object_hit_obstacle_r' + str(i)] = \
                    float(hit_obstacle_list[i])
                info['moving_object_hit_robot_or_obstacle_r' + str(i)] = \
                    float(hit_robot_or_obstacle_list[i])
                info['moving_object_missed_robot_r' + str(i)] = \
                    float(missed_robot_list[i])
                info['moving_object_no_final_status_r' + str(i)] = \
                    float(no_final_status_list[i])

            info['moving_object_hit_robot_total'] = \
                float(np.sum(hit_robot_list))
            info['moving_object_hit_obstacle_total'] = \
                float(np.sum(hit_obstacle_list))
            info['moving_object_hit_robot_or_obstacle_total'] = \
                float(np.sum(hit_robot_or_obstacle_list))
            info['moving_object_missed_robot_total'] = \
                float(np.sum(missed_robot_list))
            info['moving_object_no_final_status_total'] = \
                float(np.sum(no_final_status_list))

            moving_object_final_status_total = \
                info['moving_object_hit_robot_or_obstacle_total'] + \
                info['moving_object_missed_robot_total']

            if moving_object_final_status_total > 0:
                info['moving_object_hit_robot_fraction'] = \
                    info['moving_object_hit_robot_total'] / moving_object_final_status_total
                info['moving_object_hit_obstacle_fraction'] = \
                    info['moving_object_hit_obstacle_total'] / moving_object_final_status_total
                info['moving_object_hit_robot_or_obstacle_fraction'] = \
                    info['moving_object_hit_robot_or_obstacle_total'] / moving_object_final_status_total
                info['moving_object_missed_robot_fraction'] = \
                    info['moving_object_missed_robot_total'] / moving_object_final_status_total

            moving_object_distance_list_all_robots = []
            for i in range(self._robot_scene.num_robots):
                if self._moving_object_distance_list[i]:
                    info['moving_object_distance_r' + str(i)] = np.mean(self._moving_object_distance_list[i])
                    moving_object_distance_list_all_robots.extend(self._moving_object_distance_list[i])
                else:
                    info['moving_object_distance_r' + str(i)] = np.nan

            if moving_object_distance_list_all_robots:
                moving_object_distance_array_all_robots = np.array(moving_object_distance_list_all_robots)
                info['moving_object_distance_mean'] = np.mean(moving_object_distance_array_all_robots)
                info['moving_object_distance_max'] = np.max(moving_object_distance_array_all_robots)

        if self._braking_timeout:
            info['obstacles_episodes_with_braking_timeout'] = 1.0
        else:
            info['obstacles_episodes_with_braking_timeout'] = 0.0

        if self._braking_duration_list:
            info['obstacles_braking_duration_mean'] = np.mean(self._braking_duration_list)
            info['obstacles_braking_duration_max'] = np.max(self._braking_duration_list)

        if self._active_braking_duration_list:
            info['obstacles_active_braking_duration_mean'] = np.mean(self._active_braking_duration_list)
            info['obstacles_active_braking_duration_max'] = np.max(self._active_braking_duration_list)
            if self._active_braking_influence_time_list:
                info['obstacles_active_braking_influence_time_mean'] = np.mean(self._active_braking_influence_time_list)
                info['obstacles_active_braking_influence_time_max'] = np.max(self._active_braking_influence_time_list)
        else:
            # episode without braking influence;
            # Note: To compute the mean of the active_braking_duration (execution time of activated braking
            # trajectories) and the braking_influence_time (actual influence time, equals active_braking_duration if
            # the robot is completely stopped), episodes with 0 values need to be neglected
            info['obstacles_active_braking_duration_mean'] = 0
            info['obstacles_active_braking_duration_max'] = 0
            info['obstacles_active_braking_influence_time_mean'] = 0
            info['obstacles_active_braking_influence_time_max'] = 0

        if info['obstacles_time_influenced_by_braking_trajectory'] == 0.0:
            info['obstacles_episodes_without_influence_by_braking_trajectory'] = 1.0
        else:
            info['obstacles_episodes_without_influence_by_braking_trajectory'] = 0.0
        if info['obstacles_time_influenced_by_braking_trajectory_collision'] == 0.0:
            info['obstacles_episodes_without_influence_by_braking_trajectory_collision'] = 1.0
        else:
            info['obstacles_episodes_without_influence_by_braking_trajectory_collision'] = 0.0
        if info['obstacles_time_influenced_by_braking_trajectory_torque'] == 0.0:
            info['obstacles_episodes_without_influence_by_braking_trajectory_torque'] = 1.0
        else:
            info['obstacles_episodes_without_influence_by_braking_trajectory_torque'] = 0.0

        if self._robot_scene.collision_avoidance_mode:
            info['collision_avoidance_starting_point_selected_step'] = \
                float(self._collision_avoidance_starting_point_selected_step)
            info['collision_avoidance_starting_point_computation_steps'] = \
                float(self._collision_avoidance_starting_point_computation_steps)

        if self._log_obstacle_data:
            for obstacle in self._obstacle_list:
                info['obstacles_link_data_' + obstacle.name] = [obstacle.link_data[i].export_metrics() for i in
                                                                range(len(obstacle.link_data))]

            for i in range(len(self._links)):
                export_link_pair = [self._links[i].closest_point_active or
                                    self._links[self._links[i].self_collision_links[j]].closest_point_active
                                    for j in range(len(self._links[i].self_collision_links))]
                info['obstacles_self_collision_data_link_' + str(i) + '_' + self._links[i].name] = \
                    self._links[i].self_collision_data.export_metrics(export_link_pair=export_link_pair)

        if self._print_stats:
            if (self._episode_counter % print_stats_every_n_episodes) == 0:
                logging.info("Sampling attempts starting point: " + str(self._starting_point_sampling_attempts))
                if self._braking_duration_list:
                    logging.info("Mean braking duration: " + str(info['obstacles_braking_duration_mean']))
                    logging.info("Max braking duration: " + str(info['obstacles_braking_duration_max']))
                if self._use_target_points:
                    logging.info("Number of target points reached: " +
                                 str(info['obstacles_num_target_points_reached']))
                    if self._robot_scene.num_robots > 1:
                        for i in range(self._robot_scene.num_robots):
                            logging.info("    - Robot " + str(i) + ": " + str(
                                info['obstacles_num_target_points_reached_r' + str(i)]))
                    logging.info("Number of attempts to find a valid target point")
                    for i in range(self._robot_scene.num_robots):
                        logging.info("    - Robot " + str(i) + ": " + str(
                            info['obstacles_target_point_sampling_attempts_r' + str(i)]))

        return info

    def _specify_observed_points(self, link_name, link_index):
        observed_points = []

        link_state = p.getLinkState(bodyUniqueId=self._robot_scene.robot_id, linkIndex=link_index,
                                    computeLinkVelocity=False,
                                    computeForwardKinematics=True)
        com_offset = link_state[2]

        safety_distance = self._observed_point_safety_distance

        if self._observed_link_point_scene == 1:
            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_6"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.0, 0.02], bounding_sphere_radius=0.12, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

        if self._observed_link_point_scene == 2:

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_3"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, 0.01, 0.06], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

                observed_points.append(
                    LinkPointBase(name="P1", offset=[0.00, 0.03, 0.19], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_4"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.04, 0.02], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

                observed_points.append(
                    LinkPointBase(name="P1", offset=[0, -0.015, 0.16], bounding_sphere_radius=0.105, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_5"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.015, -0.11], bounding_sphere_radius=0.105, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_6"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.0, 0.02], bounding_sphere_radius=0.12, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))
        if self._observed_link_point_scene == 3:

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_3"):
                observed_points.append(
                    LinkPointBase(name="P1", offset=[0.00, 0.03, 0.19], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

        return observed_points

    def _get_robot_index_from_link_name(self, link_name):
        # returns the robot index extracted from the link name, e.g. 1 for iiwa_link_4_r1
        # returns -1 if no link index is found and if multiple robots are in use, 0 otherwise
        if self._robot_scene.num_robots > 1:
            if re.match('^.*_r[0-9]+$', link_name):
                # e.g. extract 1 from linkname_r1
                return int(link_name.rsplit('_', 1)[1][1:])
            else:
                return -1
        else:
            return 0

    def _specify_self_collision_links(self, link_name, link_name_list):
        self_collision_link_names = []

        if self._robot_scene.num_robots > 1:
            if self._robot_scene.robot_name == "iiwa7":
                collision_between_robots_link_names = ["iiwa_base_adapter", "iiwa_link_0", "iiwa_link_1", "iiwa_link_2",
                                                       "iiwa_link_3", "iiwa_link_4", "iiwa_link_5", "iiwa_link_6",
                                                       "iiwa_link_7"]
            elif self._robot_scene.robot_name.startswith("human"):
                collision_between_robots_link_names = ["upper_arm", "forearm", "hand"]
            else:
                collision_between_robots_link_names = []

            if self._robot_scene.ball_machine_mode:
                collision_between_robots_link_names += [self._target_link_name]

            for i in range(self._robot_scene.num_robots - 1):
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        collision_between_robots_link_names,
                        robot_indices=[i]):
                    self_collision_link_names += (self._robot_scene.get_link_names_for_multiple_robots(
                        collision_between_robots_link_names, robot_indices=np.arange(i + 1,
                                                                                     self._robot_scene.num_robots)))

            if self._robot_scene.robot_name.startswith("human"):
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        collision_between_robots_link_names):
                    self_collision_link_names += ["body", "head"]

        if self._robot_scene.ball_machine_mode:
            ball_machine_link_list = [self._target_link_name]
            for i in range(self._robot_scene.num_robots):
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(ball_machine_link_list,
                                                                                     robot_indices=[i]):
                    if self._robot_scene.robot_name.startswith("iiwa7"):
                        self_collision_link_names += self._robot_scene.get_link_names_for_multiple_robots(
                            ["iiwa_base_adapter", "iiwa_link_0", "iiwa_link_1", "iiwa_link_2",
                             "iiwa_link_3", "iiwa_link_4", "iiwa_link_5"], robot_indices=[i])

        self_collision_link_indices = []
        for i in range(len(self_collision_link_names)):
            link_index = None
            for j in range(len(link_name_list)):
                if link_name_list[j] == self_collision_link_names[i]:
                    link_index = j
                    self_collision_link_indices.append(link_index)
                    break
            if link_index is None:
                raise ValueError(self_collision_link_names[i] + " is not a valid link name")

        return self_collision_link_indices

    def get_starting_point_joint_pos_vel_acc(self, minimum_initial_distance_to_obstacles=None,
                                             minimum_distance_self_collision=None):

        # set vel and acc to zero, if not set otherwise
        starting_point_joint_vel = np.array([0.0] * self._robot_scene.num_manip_joints)
        starting_point_joint_acc = np.array([0.0] * self._robot_scene.num_manip_joints)

        if self._robot_scene.collision_avoidance_mode:
            if minimum_initial_distance_to_obstacles is None:
                minimum_initial_distance_to_obstacles = 0.001
            if minimum_distance_self_collision is None:
                minimum_distance_self_collision = 0.001
        else:
            if minimum_initial_distance_to_obstacles is None:
                minimum_initial_distance_to_obstacles = self._closest_point_safety_distance + 0.09
            if minimum_distance_self_collision is None:
                minimum_distance_self_collision = self._closest_point_safety_distance + 0.04

        starting_point_joint_pos, _, attempts_counter = self._get_collision_free_robot_position(
            minimum_initial_distance_to_obstacles=
            minimum_initial_distance_to_obstacles,
            minimum_distance_self_collision=
            minimum_distance_self_collision,
            cartesian_range=self._starting_point_cartesian_range,
            euler_angle_range=None,
            attempts=100000)

        self._starting_point_sampling_attempts = attempts_counter

        if self._robot_scene.collision_avoidance_mode or \
                self._robot_scene.always_use_collision_avoidance_starting_point_sampling:
            starting_point_found = False

            if self._robot_scene.collision_avoidance_kinematic_state_sampling_mode and \
                    np.random.uniform(low=0.0, high=1.0) < \
                    self._robot_scene.collision_avoidance_kinematic_state_sampling_probability:
                self._collision_avoidance_starting_point_selected_step = 0
                self._collision_avoidance_starting_point_computation_steps = 0
                while not starting_point_found:
                    starting_point_joint_vel = []
                    starting_point_joint_acc = []
                    self._collision_avoidance_starting_point_selected_step += 1
                    for i in range(len(starting_point_joint_pos)):
                        joint_vel_and_acc_found = False
                        for _ in range(5):
                            joint_vel = np.random.uniform(low=-1*self._robot_scene.max_velocities[i],
                                                          high=self._robot_scene.max_velocities[i],
                                                          size=(1,))
                            for _ in range(10):
                                joint_acc = np.random.uniform(low=-1*self._robot_scene.max_accelerations[i],
                                                              high=self._robot_scene.max_accelerations[i],
                                                              size=(1,))
                                violation_code = self._violation_code_function(joint_index=i,
                                                                               start_position=
                                                                               starting_point_joint_pos[i:i+1],
                                                                               start_velocity=joint_vel,
                                                                               start_acceleration=joint_acc)
                                self._collision_avoidance_starting_point_computation_steps += 1
                                if violation_code == 0:
                                    joint_vel_and_acc_found = True
                                    starting_point_joint_vel.append(joint_vel[0])
                                    starting_point_joint_acc.append(joint_acc[0])
                                    break
                            if joint_vel_and_acc_found:
                                break

                        if joint_vel_and_acc_found and i == len(starting_point_joint_pos) - 1:
                            starting_point_joint_vel = np.array(starting_point_joint_vel)
                            starting_point_joint_acc = np.array(starting_point_joint_acc)

                            execute_braking_trajectory = False
                            if self._use_braking_trajectory_method:
                                end_acceleration, _ = self._compute_braking_acceleration(
                                    start_position=starting_point_joint_pos,
                                    start_velocity=starting_point_joint_vel,
                                    start_acceleration=
                                    starting_point_joint_acc)
                                execute_braking_trajectory = \
                                    self.check_braking_trajectory_method(starting_point_joint_acc,
                                                                         starting_point_joint_vel,
                                                                         starting_point_joint_pos,
                                                                         end_acceleration,
                                                                         time_step_counter=0)
                            if not execute_braking_trajectory:
                                starting_point_found = True

                        if not joint_vel_and_acc_found:
                            starting_point_joint_pos, _, _ = self._get_collision_free_robot_position(
                                minimum_initial_distance_to_obstacles=
                                minimum_initial_distance_to_obstacles,
                                minimum_distance_self_collision=
                                minimum_distance_self_collision,
                                cartesian_range=self._starting_point_cartesian_range,
                                euler_angle_range=None,
                                attempts=100000)
                            break

            else:
                trajectory_setpoints = {'position': [starting_point_joint_pos],
                                        'velocity': [starting_point_joint_vel],
                                        'acceleration': [starting_point_joint_acc]}

                self._collision_avoidance_starting_point_computation_steps = 0

                while not starting_point_found:
                    starting_point_found = \
                        np.random.uniform(low=0.0, high=1.0) < \
                        self._robot_scene.collision_avoidance_stay_in_state_probability

                    if not starting_point_found:
                        self._collision_avoidance_starting_point_computation_steps += 1
                        joint_acc_min, joint_acc_max = \
                            self._acc_range_function(start_position=trajectory_setpoints['position'][-1],
                                                     start_velocity=trajectory_setpoints['velocity'][-1],
                                                     start_acceleration=trajectory_setpoints['acceleration'][-1])
                        safe_acc_range_min_max = np.asarray([joint_acc_min, joint_acc_max])
                        random_action = np.random.uniform(-1, 1, self._robot_scene.num_manip_joints)
                        end_acceleration = denormalize(random_action, safe_acc_range_min_max)

                        if self._use_braking_trajectory_method:
                            execute_braking_trajectory = self.check_braking_trajectory_method(
                                current_acc=trajectory_setpoints['acceleration'][-1],
                                current_vel=trajectory_setpoints['velocity'][-1],
                                current_pos=trajectory_setpoints['position'][-1],
                                target_acc=end_acceleration,
                                time_step_counter=0)

                            if execute_braking_trajectory:
                                end_acceleration, _ = self._compute_braking_acceleration(
                                    start_position=trajectory_setpoints['position'][-1],
                                    start_velocity=trajectory_setpoints['velocity'][-1],
                                    start_acceleration=trajectory_setpoints['acceleration'][-1])

                        end_velocity = interpolate_velocity(trajectory_setpoints['acceleration'][-1],
                                                            end_acceleration,
                                                            trajectory_setpoints['velocity'][-1],
                                                            self._trajectory_time_step, self._trajectory_time_step)
                        end_position = interpolate_position(trajectory_setpoints['acceleration'][-1],
                                                            end_acceleration,
                                                            trajectory_setpoints['velocity'][-1],
                                                            trajectory_setpoints['position'][-1],
                                                            self._trajectory_time_step,
                                                            self._trajectory_time_step)

                        if not self._use_braking_trajectory_method:
                            # check if the end_position is collision-free
                            self.set_robot_position_in_obstacle_client(target_position=end_position)

                            end_position_is_collision_free, _ = \
                                self._check_if_minimum_distance_between_robot_and_obstacles_exceeds_threshold(
                                    robot=None,
                                    distance_threshold=minimum_initial_distance_to_obstacles)

                            if end_position_is_collision_free:
                                end_position_is_collision_free, _ = \
                                    self._check_if_minimum_distance_between_robot_links_exceeds_threshold(
                                        robot=None,
                                        distance_threshold=minimum_distance_self_collision,
                                        soft_self_collision_constraints=False)

                            if end_position_is_collision_free:
                                if self._human is not None:
                                    self._human.set_position_in_obstacle_client_to_setpoints()
                                    obstacle_list = [self._human]

                                    minimum_distance = self._get_minimum_distance_to_obstacles(
                                        obstacle_list, maximum_relevant_distance=minimum_initial_distance_to_obstacles)
                                    if minimum_distance < minimum_initial_distance_to_obstacles:
                                        end_position_is_collision_free = False
                        else:
                            end_position_is_collision_free = True

                        if end_position_is_collision_free:
                            trajectory_setpoints['position'].append(end_position)
                            trajectory_setpoints['velocity'].append(end_velocity)
                            trajectory_setpoints['acceleration'].append(end_acceleration)
                        else:
                            # found a collision,
                            # randomly select one of the three latest previous states
                            starting_point_found = True
                            selected_previous_state = \
                                np.random.randint(low=1, high=min(len(trajectory_setpoints['position']), 3) + 1)
                            starting_point_joint_pos = trajectory_setpoints['position'][-1 * selected_previous_state]
                            starting_point_joint_vel = trajectory_setpoints['velocity'][-1 * selected_previous_state]
                            starting_point_joint_acc = \
                                trajectory_setpoints['acceleration'][-1 * selected_previous_state]
                            self._collision_avoidance_starting_point_selected_step = len(
                                trajectory_setpoints['position']) - selected_previous_state
                    else:
                        starting_point_joint_pos = trajectory_setpoints['position'][-1]
                        starting_point_joint_vel = trajectory_setpoints['velocity'][-1]
                        starting_point_joint_acc = trajectory_setpoints['acceleration'][-1]
                        self._collision_avoidance_starting_point_selected_step = \
                            len(trajectory_setpoints['position']) - 1

        return starting_point_joint_pos, starting_point_joint_vel, starting_point_joint_acc

    def _add_target_point(self, robot=0, minimum_initial_distance_to_obstacles=None,
                          minimum_distance_self_collision=None):

        if minimum_initial_distance_to_obstacles is None:
            minimum_initial_distance_to_obstacles = self._closest_point_safety_distance + 0.09
        if minimum_distance_self_collision is None:
            minimum_distance_self_collision = self._closest_point_safety_distance + 0.00

        euler_angle_range = None
        attempts = 25000

        self._target_point_joint_pos_list[robot], target_point_pos, attempts_counter = \
            self._get_collision_free_robot_position(
                minimum_initial_distance_to_obstacles=minimum_initial_distance_to_obstacles,
                minimum_distance_self_collision=minimum_distance_self_collision,
                cartesian_range=self._target_point_cartesian_range,
                check_initial_torque=True,
                euler_angle_range=euler_angle_range,
                robot=robot,
                attempts=attempts)

        self._target_point_sampling_attempts_list[robot].append(attempts_counter)
        logging.debug("Target point position robot %s: %s", robot, target_point_pos)

        color_index = robot if self._target_point_sequence != TARGET_POINT_SINGLE else 0
        if self._robot_scene.robot_name.startswith("human"):
            color_index = color_index + 1

        if len(self._target_point_id_reuse_list):
            id_to_reuse = self._target_point_id_reuse_list.pop(0)
            if self._simulation_client_id != self._backup_client_id:
                raise
        else:
            id_to_reuse = None
            if self._simulation_client_id == self._backup_client_id:
                raise

        if self._simulation_client_id != self._backup_client_id and len(self._target_point_deterministic_list) > 0:
            target_point_dict = self._target_point_deterministic_list.pop(0)
        elif self._simulation_client_id == self._backup_client_id and \
                self._robot_scene.risk_state_deterministic_backup_trajectory and \
                self._target_point_deterministic_list_index < len(self._target_point_deterministic_list):
            target_point_dict = self._target_point_deterministic_list[self._target_point_deterministic_list_index]
        else:
            target_point_dict = dict(enable_collisions=False,
                                     create_collision_shape=False,
                                     pos=target_point_pos,
                                     shape=p.GEOM_SPHERE, radius=self._target_point_radius,
                                     observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                         self._target_link_name),
                                     name="Target-" + str(robot) + "-" + str(
                                         len(self._target_point_list)),
                                     is_static=True,
                                     color=self.get_target_point_color(color_index))

        self._target_point_list[robot].append(self._add_obstacle(**target_point_dict,
                                                                 id_to_reuse=id_to_reuse))

        if self._simulation_client_id == self._backup_client_id and \
                self._robot_scene.risk_state_deterministic_backup_trajectory:
            if self._target_point_deterministic_list_index >= len(self._target_point_deterministic_list):
                self._target_point_deterministic_list.append(target_point_dict)
            self._target_point_deterministic_list_index = self._target_point_deterministic_list_index + 1


    def _add_moving_object(self, robot=0, minimum_initial_distance_to_obstacles=None,
                           minimum_distance_self_collision=None):
        if minimum_initial_distance_to_obstacles is None:
            minimum_initial_distance_to_obstacles = self._closest_point_safety_distance + 0.09
        if minimum_distance_self_collision is None:
            minimum_distance_self_collision = self._closest_point_safety_distance + 0.00

        moving_object_release_point = self._get_moving_object_release_point()
        moving_object_target_pos = None

        if self._robot_scene.moving_object_aim_at_current_robot_position:
            # aim at the current position of the robot
            # set robot in obstacle client to the actual position of the robot
            self.set_robot_position_in_obstacle_client(set_to_actual_values=True)

        if len(self._moving_object_id_reuse_list):
            id_to_reuse = self._moving_object_id_reuse_list.pop(0)
            if self._simulation_client_id != self._backup_client_id:
                raise
        else:
            id_to_reuse = None
            if self._simulation_client_id == self._backup_client_id:
                raise

        if self._simulation_client_id != self._backup_client_id and len(self._moving_object_deterministic_list) > 0:
            moving_object_dict = self._moving_object_deterministic_list.pop(0)
        elif self._simulation_client_id == self._backup_client_id and \
                self._robot_scene.risk_state_deterministic_backup_trajectory and \
                self._moving_object_deterministic_list_index < len(self._moving_object_deterministic_list):
            moving_object_dict = self._moving_object_deterministic_list[self._moving_object_deterministic_list_index]
        else:
            moving_object_dict = None

        if moving_object_dict is None:

            # ball settings
            initial_speed_vector = None

            euler_angle_range = None
            attempts = 25000

            observed_links = []
            num_observed_points_per_link = []

            if self._robot_scene.moving_object_observed_link_names:
                observed_links, num_observed_points_per_link = \
                    self.get_link_indices_in_link_name_list(self._robot_scene.moving_object_observed_link_names)

            for _ in range(attempts):
                valid_target_pos = True
                if self._robot_scene.moving_object_aim_at_current_robot_position:
                    # aim at the current position of the robot
                    # randomly select a target link
                    target_link_index = np.random.randint(low=0, high=len(self._moving_object_target_link_point_list))
                    moving_object_target_pos = \
                        self._moving_object_target_link_point_list[target_link_index].get_position(actual=True)

                else:
                    # set the robot to a random position and aim at the target link point position
                    _, moving_object_target_pos, attempts_counter = \
                        self._get_collision_free_robot_position(
                            minimum_initial_distance_to_obstacles=minimum_initial_distance_to_obstacles,
                            minimum_distance_self_collision=minimum_distance_self_collision,
                            cartesian_range=self._target_point_cartesian_range,
                            check_initial_torque=True,
                            euler_angle_range=euler_angle_range,
                            robot=robot,
                            attempts=attempts)

                ball_delta = np.array(moving_object_target_pos) - moving_object_release_point
                delta_xy = np.linalg.norm(ball_delta[:2])
                delta_height = ball_delta[2]
                g = 9.81
                v = self._robot_scene.moving_object_speed_meter_per_second
                square_root_numerator = v ** 4 - g * (g * delta_xy ** 2 + 2 * delta_height * v ** 2)
                if square_root_numerator >= 0:
                    select_theta_0 = np.random.uniform(0, 1) <= \
                                     self._robot_scene.moving_object_high_launch_angle_probability
                    if select_theta_0:
                        theta = np.arctan((v ** 2 + np.sqrt(square_root_numerator))
                                          / (g * delta_xy))
                    else:
                        theta = np.arctan((v ** 2 - np.sqrt(square_root_numerator))
                                          / (g * delta_xy))
                    ball_velocity_z = v * np.sin(theta)
                    ball_velocity_xy = v * np.cos(theta) * ball_delta[:2] / delta_xy
                    initial_speed_vector = np.hstack((ball_velocity_xy, ball_velocity_z))
                else:
                    valid_target_pos = False

                if valid_target_pos and self._robot_scene.moving_object_check_invalid_target_link_point_positions:
                    # check if target pos is valid based on moving_object_invalid_target_link_point_position_min_max
                    hit_time_invalid_area_top, hit_time_invalid_area_top_update_steps = \
                        Ball.get_target_height_time(
                            initial_height=moving_object_release_point[2],
                            initial_z_speed=initial_speed_vector[2],
                            target_height=self._moving_object_invalid_target_link_point_position_min_max[1][2]
                            + self._robot_scene.ball_radius,
                            update_time_step=self._update_time_step)

                    hit_time_invalid_area_bottom, hit_time_invalid_area_bottom_update_steps = \
                        Ball.get_target_height_time(
                            initial_height=moving_object_release_point[2],
                            initial_z_speed=initial_speed_vector[2],
                            target_height=self._moving_object_invalid_target_link_point_position_min_max[0][2]
                            + self._robot_scene.ball_radius,
                            update_time_step=self._update_time_step)

                    ball_position_top = \
                        moving_object_release_point[:2] + initial_speed_vector[:2] * hit_time_invalid_area_top
                    ball_position_bottom = \
                        moving_object_release_point[:2] + initial_speed_vector[:2] * hit_time_invalid_area_bottom

                    if (not np.isnan(hit_time_invalid_area_top)
                        and self._moving_object_invalid_target_link_point_position_min_max[0][0] <= ball_position_top[0]
                        <= self._moving_object_invalid_target_link_point_position_min_max[1][0]
                        and self._moving_object_invalid_target_link_point_position_min_max[0][1] <= ball_position_top[1]
                        <= self._moving_object_invalid_target_link_point_position_min_max[1][1]) or \
                            (not np.isnan(hit_time_invalid_area_bottom)
                             and self._moving_object_invalid_target_link_point_position_min_max[0][0]
                             <= ball_position_bottom[0] <=
                             self._moving_object_invalid_target_link_point_position_min_max[1][0]
                             and self._moving_object_invalid_target_link_point_position_min_max[0][1]
                             <= ball_position_bottom[1]
                             <= self._moving_object_invalid_target_link_point_position_min_max[1][1]):
                       valid_target_pos = False

                if valid_target_pos:
                    break
                else:
                    moving_object_release_point = self._get_moving_object_release_point()

            create_collision_shape_in_obstacle_client = True if self._reward_consider_moving_obstacles else False

        if moving_object_dict is None:

            final_ball_position_min_max = None if self._robot_scene.moving_object_sphere_center is None else \
                self._moving_object_final_object_position_min_max

            moving_object_dict = dict(pos=moving_object_release_point,
                                      robot_id=self._robot_scene.robot_id,
                                      initial_speed_vector=initial_speed_vector,
                                      update_time_step=self._update_time_step,
                                      update_steps_per_action=self._obstacle_client_update_steps_per_action,
                                      radius=self._robot_scene.ball_radius,
                                      update_position_in_obstacle_client=False,
                                      ball_behind_the_robot_x_value=
                                      self._moving_object_behind_the_robot_x_value,
                                      final_ball_position_min_max=final_ball_position_min_max,
                                      reload_as_dynamic_object_on_hit=True,
                                      update_steps_for_fade_out=60,
                                      obstacle_client_id=self._obstacle_client_id,
                                      backup_client_id=self._backup_client_id,
                                      use_real_robot=self._use_real_robot,
                                      observed_links=observed_links,
                                      num_observed_points_per_link=num_observed_points_per_link,
                                      visual_mode=self._visual_mode,
                                      create_collision_shape_in_simulation_client=True,
                                      create_collision_shape_in_obstacle_client=
                                      create_collision_shape_in_obstacle_client,
                                      num_clients=self._robot_scene.num_clients)

        moving_object = Ball(**moving_object_dict,
                             simulation_client_id=self._simulation_client_id,
                             id_to_reuse=id_to_reuse)

        obstacle_hit_data = None
        if self._robot_scene.robot_name == "iiwa7" and self._obstacle_scene != 0:
            # compute the time at which the ball would reach the table (height=0)
            hit_time_table, hit_time_table_update_steps = Ball.get_target_height_time(
                initial_height=moving_object_dict["pos"][2],
                initial_z_speed=moving_object_dict["initial_speed_vector"][2],
                target_height=self._robot_scene.ball_radius,
                update_time_step=self._update_time_step)

            # check if the ball x and y coordinates are within the table area
            ball_position_table = moving_object_dict["pos"][:2] \
                                  + moving_object_dict["initial_speed_vector"][:2] * hit_time_table
            # the table's dimensions are x: [-0.6, 0.6]; y: [-0.8, 0.8]
            if -0.6 <= ball_position_table[0] <= 0.6 and -0.8 <= ball_position_table[1] <= 0.8:
                obstacle_hit_data = {'obstacle_hit_index': -1,
                                     'obstacle_hit_time': hit_time_table_update_steps - 1}

        if obstacle_hit_data is None:
            # compute the time at which the ball reaches the ground
            hit_time_ground, hit_time_ground_update_steps = Ball.get_target_height_time(
                initial_height=moving_object_dict["pos"][2],
                initial_z_speed=moving_object_dict["initial_speed_vector"][2],
                target_height=self._robot_scene.plane_z_offset + self._robot_scene.ball_radius,
                update_time_step=self._update_time_step)

            obstacle_hit_data = {'obstacle_hit_index': -1,
                                 'obstacle_hit_time': hit_time_ground_update_steps - 1}

        moving_object.obstacle_hit_data = obstacle_hit_data

        self._moving_object_list[robot].append(moving_object)

        if self._simulation_client_id == self._backup_client_id and \
                self._robot_scene.risk_state_deterministic_backup_trajectory:
            if self._moving_object_deterministic_list_index >= len(self._moving_object_deterministic_list):
                self._moving_object_deterministic_list.append(moving_object_dict)
            self._moving_object_deterministic_list_index = self._moving_object_deterministic_list_index + 1

        if self._moving_object_sequence == MOVING_OBJECT_SINGLE:
            for i in range(len(self._moving_object_observation_robot_indices)):
                if self._moving_object_observation_robot_indices[i] == -1:
                    self._moving_object_observation_robot_indices[i] = robot
                    break

    def _get_moving_object_release_point(self):
        # define a release point for the moving object
        if self._robot_scene.moving_object_sphere_center is None:
            moving_object_release_point = np.array(self._robot_scene.moving_object_area_center) \
                                         - np.array([0, self._robot_scene.moving_object_area_width_height[0] / 2,
                                                     self._robot_scene.moving_object_area_width_height[1] / 2]) \
                                         + np.array([0, np.random.uniform(low=0.0, high=1.0) *
                                                     self._robot_scene.moving_object_area_width_height[0],
                                                     np.random.uniform(low=0.0, high=1.0) *
                                                     self._robot_scene.moving_object_area_width_height[1]])
        else:
            release_height = np.random.uniform(low=self._robot_scene.moving_object_sphere_height_min_max[0],
                                               high=self._robot_scene.moving_object_sphere_height_min_max[1])
            release_radius = np.sqrt(self._robot_scene.moving_object_sphere_radius ** 2 - release_height ** 2)
            release_angle = np.random.uniform(low=self._robot_scene.moving_object_sphere_angle_min_max[0],
                                              high=self._robot_scene.moving_object_sphere_angle_min_max[1])
            moving_object_release_point = self._robot_scene.moving_object_sphere_center \
                                         + np.array([release_radius * np.cos(release_angle),
                                                     release_radius * np.sin(release_angle),
                                                     release_height])

        return moving_object_release_point

    def get_target_point_color(self, robot=0, transparency=0.5):
        if robot == 0:
            target_point_color = (0, 1, 0, transparency)
        elif robot == 1:
            target_point_color = (126 / 255, 47 / 255, 142 / 255, transparency)
        elif robot == 2:
            target_point_color = (23 / 255, 190 / 255, 207 / 255, transparency)  # dark turquoise
        else:
            target_point_color = (255 / 255, 153 / 255, 0 / 255, transparency)  # orange

        return target_point_color

    def get_moving_object_color(self, robot=0, transparency=0.5):
        if robot == 0:
            moving_object_color = (0, 1, 0, transparency)  # green
        elif robot == 1:
            moving_object_color = (212 / 255, 21 / 255, 20 / 255, transparency)  # red
        elif robot == 2:
            moving_object_color = (23 / 255, 28 / 255, 243 / 255, transparency)  # dark blue
        else:
            moving_object_color = (252 / 255, 239 / 255, 113 / 255, transparency)  # yellow

        return moving_object_color

    def _get_collision_free_robot_position(self, minimum_initial_distance_to_obstacles, minimum_distance_self_collision,
                                           cartesian_range, euler_angle_range=None, check_initial_torque=True,
                                           robot=None, static_joint_pos=None, attempts=10000):

        valid_pos_found = False
        attempts_counter = 0
        if robot is None:  # all joints
            manip_joint_indices_robot = self._manip_joint_indices
        else:
            manip_joint_indices_robot = np.array(self._robot_scene.get_manip_joint_indices_per_robot(robot_index=robot))

        joint_limit_indices_robot = []
        static_joint_index_counter = -1
        static_joint_indices = []
        if static_joint_pos is None:
            use_static_joint_pos_from_argument = False
            static_joint_pos = []
        else:
            use_static_joint_pos_from_argument = True

        for i in range(len(self._manip_joint_indices)):
            if self._manip_joint_indices[i] not in manip_joint_indices_robot:
                static_joint_index_counter += 1
                static_joint_indices.append(self._manip_joint_indices[i])
                if not use_static_joint_pos_from_argument:
                    target_pos = self._target_position[i]
                    static_joint_pos.append(target_pos)
                    target_vel = 0
                else:
                    # use positions given by the argument static_joint_pos to generate a starting pos in multiple steps
                    target_pos = static_joint_pos[static_joint_index_counter]
                    target_vel = 0

                p.resetJointState(bodyUniqueId=self._robot_scene.robot_id,
                                  jointIndex=self._manip_joint_indices[i],
                                  targetValue=target_pos,
                                  targetVelocity=target_vel,
                                  physicsClientId=self._obstacle_client_id)

            else:
                joint_limit_indices_robot.append(i)

        joint_limit_indices_robot = np.array(joint_limit_indices_robot)

        if check_initial_torque and static_joint_pos:
            # set motor control for static joints
            self._robot_scene.set_motor_control(target_positions=static_joint_pos,
                                                physics_client_id=self._obstacle_client_id,
                                                manip_joint_indices=static_joint_indices)

        while not valid_pos_found:
            valid_pos_found = True
            self_collision_ignored_links = None
            reason = None
            attempts_counter += 1
            random_pos = np.random.uniform(
                np.array(self._robot_scene.joint_lower_limits_continuous)[joint_limit_indices_robot],
                np.array(self._robot_scene.joint_upper_limits_continuous)[joint_limit_indices_robot])

            # set position of affected links
            self.set_robot_position_in_obstacle_client(manip_joint_indices=manip_joint_indices_robot,
                                                       target_position=random_pos)

            if self._target_link_name is not None:
                for i in range(self._robot_scene.num_robots):
                    if robot is None or robot == i:
                        if isinstance(self._target_link_name, list):
                            target_link_point_index = i * len(self._target_link_name) \
                                                      + np.random.randint(low=0, high=len(self._target_link_name))
                        else:
                            target_link_point_index = i
                        target_link_pos, target_link_orn = \
                            self._target_link_point_list[target_link_point_index].get_position(actual=None,
                                                                                               return_orn=True)
                        if target_link_pos[0] < cartesian_range[0][0] \
                                or target_link_pos[0] > cartesian_range[0][1] or \
                                target_link_pos[1] < cartesian_range[1][0] \
                                or target_link_pos[1] > cartesian_range[1][1] or \
                                target_link_pos[2] < cartesian_range[2][0] \
                                or target_link_pos[2] > cartesian_range[2][1]:
                            valid_pos_found = False
                            break

                        if euler_angle_range is not None:
                            # check orientation of the end effector
                            # euler_angle_range e.g. [[alpha_min, alpha_max], [beta_min, beta_max],
                            # [gamma_min, gamma_max]]
                            target_link_orn_euler = p.getEulerFromQuaternion(target_link_orn)
                            if target_link_orn_euler[0] < euler_angle_range[0][0] \
                                    or target_link_orn_euler[0] > euler_angle_range[0][1] or \
                                    target_link_orn_euler[1] < euler_angle_range[1][0] \
                                    or target_link_orn_euler[1] > euler_angle_range[1][1] or \
                                    target_link_orn_euler[2] < euler_angle_range[2][0] \
                                    or target_link_orn_euler[2] > euler_angle_range[2][1]:
                                valid_pos_found = False
            else:
                raise ValueError("Target link is not defined")

            if valid_pos_found:
                valid_pos_found, reason = \
                    self._check_if_minimum_distance_between_robot_and_obstacles_exceeds_threshold(
                        robot=robot,
                        distance_threshold=minimum_initial_distance_to_obstacles)

            if valid_pos_found:
                valid_pos_found, reason = \
                    self._check_if_minimum_distance_between_robot_links_exceeds_threshold(
                        robot=robot,
                        distance_threshold=minimum_distance_self_collision,
                        soft_self_collision_constraints=attempts_counter > attempts * 0.75)

            if valid_pos_found and check_initial_torque:
                self._robot_scene.set_motor_control(target_positions=random_pos,
                                                    physics_client_id=self._obstacle_client_id,
                                                    manip_joint_indices=manip_joint_indices_robot)
                p.stepSimulation(physicsClientId=self._obstacle_client_id)

                actual_joint_torques = self._robot_scene.get_actual_joint_torques(
                    physics_client_id=self._obstacle_client_id,
                    manip_joint_indices=manip_joint_indices_robot)

                normalized_joint_torques = normalize(actual_joint_torques,
                                                     self._torque_limits[:, joint_limit_indices_robot])

                if np.any(np.abs(normalized_joint_torques) > 1):
                    if robot is not None and attempts_counter > attempts / 2:
                        # ignore torque violation to find at least a collision-free position for a target point
                        logging.warning("Ignored torque violation to find a collision-free target point. Robot: %s, "
                                        "Normalized torques: %s", robot, normalized_joint_torques)
                    else:
                        valid_pos_found = False
                        reason = "Torque violation: " + str(normalized_joint_torques)

            if valid_pos_found and self._human is not None:
                self._human.set_position_in_obstacle_client_to_setpoints()
                obstacle_list = [self._human]

                minimum_distance = self._get_minimum_distance_to_obstacles(
                    obstacle_list, maximum_relevant_distance=minimum_initial_distance_to_obstacles)
                if minimum_distance < minimum_initial_distance_to_obstacles:
                    valid_pos_found = False
                    reason = "Collision with human."

            if valid_pos_found and self_collision_ignored_links is not None:
                logging.warning("Ignored self_collision to find a collision-free target point. Robot: %s, "
                                "Ignored links: %s", robot, self_collision_ignored_links)

            if not valid_pos_found and attempts is not None and reason is not None and attempts_counter >= attempts:
                raise ValueError("Could not find a valid collision-free robot position. "
                                 + "Reason: " + reason
                                 + ", minimum_initial_distance_to_obstacles=" +
                                 str(minimum_initial_distance_to_obstacles)
                                 + ", minimum_distance_self_collision=" + str(minimum_distance_self_collision)
                                 + ", cartesian_range=" + str(cartesian_range)
                                 + ", check_initial_torque=" + str(check_initial_torque)
                                 + ", robot=" + str(robot)
                                 + ", attempts=" + str(attempts))

        return random_pos, target_link_pos, attempts_counter

    def _check_if_minimum_distance_between_robot_and_obstacles_exceeds_threshold(self, robot, distance_threshold):
        # checks are performed in the obstacle client
        # set the robot to the desired position in the obstacle client before calling this function
        distance_exceeds_threshold = True
        reason = ""
        for i in range(len(self._obstacle_list)):
            for j in range(len(self._obstacle_list[i].observed_links)):
                link_index = self._obstacle_list[i].observed_links[j]
                for obstacle_link in self._obstacle_list[i].obstacle_links:
                    if self._links[link_index].closest_point_active and \
                            (robot is None or self._links[link_index].robot_index == robot):
                        pos_obs, pos_rob, distance = self._compute_closest_points(
                            p.getClosestPoints(bodyA=self._obstacle_list[i].id,
                                               bodyB=self._robot_scene.robot_id,
                                               distance=distance_threshold + 0.005,
                                               linkIndexA=obstacle_link,
                                               linkIndexB=link_index,
                                               physicsClientId=self._obstacle_client_id))

                        if distance is not None and distance < distance_threshold:
                            distance_exceeds_threshold = False
                            reason = "Collision: LinkIndex: " + str(link_index) + ", ObstacleIndex: " + str(i)
                            break
                if not distance_exceeds_threshold:
                    break
            if not distance_exceeds_threshold:
                break

        return distance_exceeds_threshold, reason

    def _check_if_minimum_distance_between_robot_links_exceeds_threshold(self, robot, distance_threshold,
                                                                         soft_self_collision_constraints=False):
        # checks are performed in the obstacle client
        # set the robot to the desired position in the obstacle client before calling this function
        distance_exceeds_threshold = True
        reason = ""
        for i in range(len(self._links)):
            for j in range(len(self._links[i].self_collision_links)):
                if (self._links[i].closest_point_active or self._links[
                    self._links[i].self_collision_links[j]].closest_point_active) \
                        and (robot is None or (self._links[i].robot_index == robot or
                                               self._links[self._links[i].self_collision_links[j]].robot_index
                                               == robot)):
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=distance_threshold + 0.005,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    if distance is not None and distance < distance_threshold:
                        if robot is not None and soft_self_collision_constraints and (
                                (self._links[i].robot_index == robot and self._links[
                                    self._links[i].self_collision_links[j]].closest_point_active) or (
                                        self._links[self._links[i].self_collision_links[j]].robot_index
                                        == robot and self._links[i].closest_point_active)):
                            self_collision_ignored_links = [i, self._links[i].self_collision_links[j]]
                            # ignore self-collisions when finding target point positions if attempts_counter
                            # is high and if the colliding link can be actively moved (closest_point_actice)
                        else:
                            distance_exceeds_threshold = False
                            reason = "Self-collision: " \
                                     "[" + str(i) + ", " + str(self._links[i].self_collision_links[j]) + "]"
                            break
            if not distance_exceeds_threshold:
                break

        return distance_exceeds_threshold, reason

    def get_target_point_observation(self, compute_relative_pos_norm=False, compute_target_point_joint_pos_norm=False):
        relative_pos_norm_chain = []
        target_point_active_observation = []
        target_point_joint_pos_norm_chain = []
        target_point_pos_norm_chain = []

        if self._use_target_points:
            for i in range(self._robot_scene.num_robots):
                self._target_point_reached_list[i] = False
                if self._sample_new_target_point_list[i]:
                    self._add_target_point(robot=i, minimum_distance_self_collision=self._closest_point_safety_distance)
                    self._sample_new_target_point_list[i] = False
                    self._target_point_active_list[i] = True
                    self._target_point_pos_norm_list[i] = None
                    self._target_point_joint_pos_norm_list[i] = None
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        for j in range(self._robot_scene.num_robots):
                            self._initial_target_point_distance_list[j].append(np.nan)
                    else:
                        self._initial_target_point_distance_list[i].append(np.nan)

            for i in range(self._robot_scene.num_robots):
                if self._target_point_sequence == TARGET_POINT_SINGLE:
                    target_point_index = np.where(self._target_point_active_list)[0][0]
                    # return index of first true element -> active target point
                else:
                    target_point_index = i
                if self._target_point_active_list[i] or self._target_point_sequence == TARGET_POINT_SINGLE:
                    if self._target_point_sequence == TARGET_POINT_ALTERNATING:
                        target_point_active_observation.append(1.0)  # to indicate that the target point is active
                    target_point_pos = \
                        np.array(self._target_point_list[target_point_index][-1].get_position(actual=False))
                    self._last_target_point_distance_list[i] = np.linalg.norm(
                        target_point_pos - self._target_link_pos_list[i])
                    if np.isnan(self._initial_target_point_distance_list[i][-1]):
                        self._initial_target_point_distance_list[i][-1] = self._last_target_point_distance_list[i]

                    if self._target_point_active_list[i]:
                        if self._target_point_pos_norm_list[i] is None:
                            self._target_point_pos_norm_list[i] = \
                                list(normalize(target_point_pos, self._target_point_cartesian_range_min_max))
                        if compute_target_point_joint_pos_norm and self._target_point_joint_pos_norm_list[i] is None:
                            joint_indices = np.array(self._robot_scene.get_manip_joint_indices_per_robot(robot_index=i))
                            joint_limit_indices_robot = []
                            for k in range(len(self._manip_joint_indices)):
                                if self._manip_joint_indices[k] in joint_indices:
                                    joint_limit_indices_robot.append(k)
                            joint_limit_indices_robot = np.array(joint_limit_indices_robot)
                            joint_lower_limits = self._robot_scene.joint_lower_limits_continuous[
                                joint_limit_indices_robot]
                            joint_upper_limits = self._robot_scene.joint_upper_limits_continuous[
                                joint_limit_indices_robot]
                            self._target_point_joint_pos_norm_list[i] = [
                                -1 + 2 * (self._target_point_joint_pos_list[i][j] -
                                          joint_lower_limits[j]) /
                                (joint_upper_limits[j] - joint_lower_limits[j])
                                for j in range(len(self._target_point_joint_pos_list[i]))]

                    if compute_relative_pos_norm:
                        relative_pos = target_point_pos - self._target_link_pos_list[i]
                        relative_pos_norm = normalize(relative_pos, self._target_point_relative_pos_min_max)
                        relative_pos_norm_clip = [min(max(relative_pos_norm[i], -1), 1) for i in range(3)]
                        if self._log_target_point_relative_pos_min_max:
                            if not list(relative_pos_norm) == relative_pos_norm_clip and \
                                    self._simulation_client_id != self._backup_client_id:
                                logging.warning("Target point relative pos norm %s clipped", relative_pos_norm)
                            for j in range(3):
                                self._target_point_relative_pos_min_max_log[0][j] = \
                                    min(self._target_point_relative_pos_min_max_log[0][j], relative_pos[j])
                                self._target_point_relative_pos_min_max_log[1][j] = \
                                    max(self._target_point_relative_pos_min_max_log[1][j], relative_pos[j])
                        relative_pos_norm_chain.extend(relative_pos_norm_clip)
                else:
                    if self._target_point_sequence == TARGET_POINT_ALTERNATING:
                        target_point_active_observation.append(0.0)
                    self._target_point_pos_norm_list[i] = [0, 0, 0]
                    if compute_relative_pos_norm:
                        relative_pos_norm_chain.extend([0, 0, 0])
                    if compute_target_point_joint_pos_norm:
                        self._target_point_joint_pos_norm_list[i] = [0.0] * len(self._target_point_joint_pos_list[i])

            if self._target_point_sequence == TARGET_POINT_SINGLE:
                target_point_pos_norm_chain = self._target_point_pos_norm_list[target_point_index]
            else:
                target_point_pos_norm_chain = list(
                    itertools.chain.from_iterable(self._target_point_pos_norm_list))
                # merge all list entries to a single list
            if compute_target_point_joint_pos_norm:
                if self._target_point_sequence == TARGET_POINT_SINGLE:
                    target_point_joint_pos_norm_chain = list(self._target_point_joint_pos_norm_list[target_point_index])
                else:
                    target_point_joint_pos_norm_chain = list(itertools.chain.from_iterable(
                        self._target_point_joint_pos_norm_list))
            else:
                target_point_joint_pos_norm_chain = []

        return target_point_pos_norm_chain, relative_pos_norm_chain, target_point_joint_pos_norm_chain, \
            target_point_active_observation

    def get_target_point_reward(self, normalize_distance_reward_to_initial_target_point_distance=False):
        reward = 0
        if self._use_target_points:
            for i in range(self._robot_scene.num_robots):
                if (self._target_point_active_list[i] or self._target_point_reached_list[i]) and \
                        normalize_distance_reward_to_initial_target_point_distance:
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        initial_distance = []
                        for j in range(self._robot_scene.num_robots):
                            initial_distance.append(self._initial_target_point_distance_list[j][-1])
                        distance_normalization = min(initial_distance)
                    else:
                        distance_normalization = self._initial_target_point_distance_list[i][-1]
                    if distance_normalization == 0:
                        distance_normalization += 0.0000001
                else:
                    distance_normalization = 1

                if self._target_point_active_list[i]:
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        current_distance_list = []
                        for j in range(self._robot_scene.num_robots):
                            current_distance_list.append(np.linalg.norm(self._target_point_pos_list[i] -
                                                                        self._target_link_pos_list[j]))
                        current_distance = min(current_distance_list)
                        last_distance = min(self._last_target_point_distance_list)
                    else:
                        current_distance = np.linalg.norm(self._target_point_pos_list[i] -
                                                          self._target_link_pos_list[i])
                        last_distance = self._last_target_point_distance_list[i]

                    reward = reward + (last_distance - current_distance) / (self._trajectory_time_step
                                                                            * distance_normalization)
                elif self._target_point_reached_list[i]:
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        last_distance = min(self._last_target_point_distance_list)
                    else:
                        last_distance = self._last_target_point_distance_list[i]
                    reward = reward + last_distance / (self._trajectory_time_step * distance_normalization)
                    reward += self._target_point_reached_reward_bonus

        return reward

    def get_moving_object_observation(self, forecast_next_step=False, no_side_effects=False):
        moving_object_observation = []
        if self._robot_scene.use_moving_objects and not no_side_effects:
            for i in range(self._robot_scene.num_robots):
                if self._sample_new_moving_object_list[i]:
                    self._add_moving_object(robot=i,
                                            minimum_distance_self_collision=self._closest_point_safety_distance)
                    self._sample_new_moving_object_list[i] = False
                    self._moving_object_active_list[i] = True

        self._moving_object_initial_active_list = self._moving_object_active_list.copy()

        for j in range(len(self._moving_object_observation_robot_indices)):
            robot_index = self._moving_object_observation_robot_indices[j]
            if self._moving_object_active_list[robot_index]:
                # should always be active if moving object sequence is not alternating
                moving_object_observation.extend(
                    list(normalize(self._moving_object_list[robot_index][-1].get_position(forecast_next_step=
                                                                                          forecast_next_step),
                                   self._moving_object_position_min_max)))

                # add current velocity of the ball
                moving_object_observation.extend(
                    list(normalize(self._moving_object_list[robot_index][-1].get_current_velocity_vector(
                        forecast_next_step=forecast_next_step),
                        self._moving_object_ball_velocity_min_max)))


                if self._moving_object_sequence == MOVING_OBJECT_ALTERNATING:
                    # add moving object active signal
                    moving_object_observation.append(1.0)
            else:
                if self._moving_object_sequence == MOVING_OBJECT_ALTERNATING:
                    # add moving object inactive signal
                    moving_object_observation.append(-1.0)
                else:
                    # inactive moving object not allowed if sequence != MOVING_OBJECT_ALTERNATING
                    raise ValueError("Moving object is not active")

        return moving_object_observation

    def get_braking_trajectory_punishment(self, minimum_distance_max_threshold, maximum_torque_min_threshold):
        # computes a punishment factor within [0, 1] based on the minimum distance and the maximum torque
        # occurring during the braking trajectory
        minimum_distance_punishment = 0
        maximum_torque_punishment = 0

        if self._use_braking_trajectory_method:
            if self._braking_trajectory_minimum_distance < self._closest_point_safety_distance:
                minimum_distance_punishment = 1.0
            else:
                if self._braking_trajectory_minimum_distance < minimum_distance_max_threshold:
                    minimum_distance_punishment = max(
                        min((minimum_distance_max_threshold - self._braking_trajectory_minimum_distance) /
                            (minimum_distance_max_threshold - self._closest_point_safety_distance), 1), 0) ** 2

                if self._check_braking_trajectory_torque_limits:
                    maximum_torque_punishment = max(
                        min((self._braking_trajectory_maximum_rel_torque
                             - maximum_torque_min_threshold) /
                            (1 - maximum_torque_min_threshold), 1), 0) ** 2

        return minimum_distance_punishment, maximum_torque_punishment

    def _add_obstacles(self):
        if self._robot_scene.robot_name == "iiwa7":

            observed_link_names = ["iiwa_link_1", "iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                   "iiwa_link_6", "iiwa_link_7"]


            if self._robot_scene.ball_machine_mode:
                observed_link_names += [self._target_link_name]

            if self._obstacle_scene > 0:
                # load table mesh
                observed_link_names_table = observed_link_names if self._obstacle_scene == 5 else []

                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=[0.0, 0.0, 0.0],
                                       urdf_file_name="table",
                                       observed_link_names=observed_link_names_table, name="Table_mesh", is_static=True,
                                       color=[0.3, 0.3, 0.3, 1]))

            if self._obstacle_scene == 1:
                table_color = (0.02, 0.02, 0.4, 0.5)
                plane_points = [[-0.6, -0.8, 0], [0.6, -0.8, 0], [-0.6, 0.8, 0]]
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Table", plane_collision_shape_factor=2.0,
                                       is_static=True, color=table_color))

            if 2 <= self._obstacle_scene <= 4:
                # table
                table_color = (0.02, 0.02, 0.4, 0.5)
                plane_points = [[-0.6, -0.8, 0], [0.6, -0.8, 0], [-0.6, 0.8, 0]]
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Table", plane_collision_shape_factor=1.0,
                                       is_static=True, color=table_color))

                # virtual walls
                wall_color = (0.8, 0.8, 0.8, 0.1)
                wall_height = 1.2  # meter
                # left wall
                plane_points = [[-0.6, -0.8, 0], [0.6, -0.8, 0], [-0.6, -0.8, wall_height]]
                plane_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall left", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))
                # front wall
                plane_points = [[0.6, -0.8, 0], [0.6, -0.8, wall_height], [0.6, 0.8, 0]]
                plane_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions,
                                       pos=None, shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall front", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))
                # right wall
                plane_points = [[-0.6, 0.8, 0], [0.6, 0.8, 0.0], [-0.6, 0.8, wall_height]]
                plane_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall right", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))

                # back wall
                plane_points = [[-0.6, -0.8, 0], [-0.6, -0.8, wall_height], [-0.6, 0.8, 0]]
                plane_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall back", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))


            if self._obstacle_scene >= 6:
                logging.warning("Obstacle scene %s not defined for %s", self._obstacle_scene,
                                self._robot_scene.robot_name)

        if self._robot_scene.robot_name.startswith("human"):
            if self._obstacle_scene > 0:
                # load table mesh
                observed_link_names_table = \
                    self._robot_scene.get_link_names_for_multiple_robots(link_names=["forearm", "hand"])

                no_visual_shape = True if self._robot_scene.do_not_execute_robot_movement else False
                # do not load visual shape again if used as a second environment, where the visual shape of the
                # obstacle was already loaded by the first environment

                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=[0.0, 0.0, 0.0],
                                       urdf_file_name="table",
                                       observed_link_names=observed_link_names_table, name="Table_mesh", is_static=True,
                                       no_visual_shape=no_visual_shape,
                                       color=[0.3, 0.3, 0.3, 1]))

            if self._obstacle_scene > 1:
                # load iiwa mesh
                observed_link_names = []

                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=[0.0, 0.0, 0.0],
                                       urdf_file_name="../robot",
                                       observed_link_names=observed_link_names, name="robot_mesh", is_static=True,
                                       color=[1, 1, 1, 1]))

        self._update_links_in_use()

    def get_link_indices_in_link_name_list(self, link_name_list):
        link_indices = []
        observed_points_per_link = []
        for i in range(len(self._links)):
            if self._links[i].name in link_name_list:
                link_indices.append(i)
                observed_points_per_link.append(self._links[i].num_observed_points)

        return link_indices, observed_points_per_link

    def _add_obstacle(self, enable_collisions=False, create_collision_shape=True,
                      observed_link_names=[], *vargs, **kwargs):
        observed_links, num_observed_points_per_link = self.get_link_indices_in_link_name_list(observed_link_names)

        obstacle = ObstacleSim(observed_links=observed_links, num_observed_points_per_link=num_observed_points_per_link,
                               simulation_client_id=self._simulation_client_id,
                               obstacle_client_id=self._obstacle_client_id,
                               backup_client_id=self._backup_client_id,
                               use_real_robot=self._use_real_robot,
                               create_collision_shape_in_simulation_client=create_collision_shape,
                               create_collision_shape_in_obstacle_client=create_collision_shape,
                               num_clients=self._robot_scene.num_clients, *vargs, **kwargs)
        if not enable_collisions and create_collision_shape:
            self._deactivate_collision_detection(obstacle.id)
        return obstacle

    def _update_links_in_use(self):
        self._links_in_use = []
        for i in range(len(self._obstacle_list)):
            for j in range(len(self._obstacle_list[i].observed_links)):
                if self._obstacle_list[i].observed_links[j] not in self._links_in_use:
                    self._links_in_use.append(self._obstacle_list[i].observed_links[j])
                    if not self._links[self._obstacle_list[i].observed_links[j]].observed_points and not \
                            self._links[self._obstacle_list[i].observed_links[j]].observe_closest_point:
                        raise ValueError("No points to observe for link " +
                                         self._links[self._obstacle_list[i].observed_links[j]].name)

        self._links_in_use.sort()

    def _delete_all_target_points(self):
        for target_point in list(itertools.chain.from_iterable(self._target_point_list)):
            for j in range(self._robot_scene.num_clients):
                p.removeBody(target_point.id, physicsClientId=j)

        self._target_point_list = [[] for _ in range(self._robot_scene.num_robots)]

    def _delete_all_moving_objects(self):
        for moving_object in list(itertools.chain.from_iterable(self._moving_object_list)):
            moving_object.delete()

        self._moving_object_list = [[] for _ in range(self._robot_scene.num_robots)]

    def _deactivate_collision_detection(self, obstacle_id):
        for j in range(self._robot_scene.num_clients):
            for i in range(p.getNumJoints(self._robot_scene.robot_id)):
                p.setCollisionFilterPair(self._robot_scene.robot_id, obstacle_id, i,
                                         p.getNumJoints(obstacle_id) - 1, enableCollision=0, physicsClientId=j)

    def update(self, target_position, target_velocity, target_acceleration,
               actual_position, actual_velocity, update_after_reset=False):

        pos_obs_debug = []
        pos_rob_debug = []

        self._target_position = target_position
        self._target_velocity = target_velocity

        self._last_actual_position = self._actual_position
        self._last_actual_velocity = self._actual_velocity
        self._actual_position = actual_position
        self._actual_velocity = actual_velocity

        for i in range(len(self._links)):
            self._links[i].clear_previous_timestep()

        self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION

        for obstacle in self._obstacle_list:
            obstacle.update()
            obstacle.clear_previous_timestep()

        if self._human is not None and not update_after_reset:
            self._human.update()
            self._human.check_if_object_is_colliding()

        if self._robot_scene.use_moving_objects and not update_after_reset:

            for i in range(self._robot_scene.num_robots):
                if self._moving_object_active_list[i]:
                    self._moving_object_list[i][-1].update()
                if self._visual_mode:
                    for j in range(len(self._moving_object_fade_out_list[i])):
                        self._moving_object_fade_out_list[i][j].update(visual_mode=True)

        if self._robot_scene.planet_mode:
            if self._planet_list:
                for i in range(len(self._planet_list)):
                    self._planet_list[i].update()

                if self._robot_scene.terminate_on_collision_with_moving_obstacle:
                    planet_updates_per_check = 1
                    self._planet_check_counter = self._planet_check_counter + 1
                    if self._planet_check_counter == planet_updates_per_check:
                        self._planet_check_counter = 0
                        for i in range(len(self._planet_list)):
                            self._planet_list[i].check_if_object_is_colliding()

        if self._log_obstacle_data:
            # first step: actual values
            self.set_robot_position_in_obstacle_client(set_to_actual_values=True)
            obstacle_counter = - 1
            for obstacle in self._obstacle_list:
                obstacle_counter = obstacle_counter + 1
                for j in range(len(obstacle.observed_links)):
                    link_index = obstacle.observed_links[j]
                    # compute actual distance to closest point if enabled
                    if self._links[link_index].observe_closest_point:

                        pos_obs, pos_rob, distance = self._compute_closest_points(
                            p.getClosestPoints(bodyA=obstacle.id, bodyB=self._robot_scene.robot_id,
                                               distance=10,
                                               linkIndexA=obstacle.last_link,
                                               linkIndexB=link_index,
                                               physicsClientId=self._obstacle_client_id))
                        if obstacle_counter == self._debug_line_obstacle and j == self._debug_line_link and \
                                self._debug_line_point == 0:
                            pos_obs_debug = pos_obs
                            pos_rob_debug = pos_rob

                        obstacle.link_data[j].closest_point_distance_actual.append(distance)

                        if len(self._links[link_index].observed_points) > 0:
                            for k in range(len(self._links[link_index].observed_points)):
                                pos_rob = self._links[link_index].observed_points[k].get_position(actual=True)
                                pos_obs = obstacle.get_position(actual=True, pos_rob=pos_rob)
                                obstacle.link_data[j].observed_point_distance_actual[k].append(
                                    self._compute_distance(pos_obs, pos_rob,
                                                           radius_a=obstacle.bounding_sphere_radius,
                                                           radius_b=self._links[link_index].observed_points[
                                                               k].bounding_sphere_radius))
                                debug_line_point = k + 1 if self._links[link_index].observe_closest_point else k
                                if obstacle_counter == self._debug_line_obstacle and j == self._debug_line_link and \
                                        self._debug_line_point == debug_line_point:
                                    pos_obs_debug, pos_rob_debug = \
                                        self._consider_bounding_sphere(pos_obs, pos_rob,
                                                                       radius_a=obstacle.bounding_sphere_radius,
                                                                       radius_b=self._links[link_index].observed_points[
                                                                           k].bounding_sphere_radius)

            # self-collision
            for i in range(len(self._links)):
                for j in range(len(self._links[i].self_collision_links)):
                    # distance actual values
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=10,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    self._links[i].self_collision_data.closest_point_distance_actual[j].append(distance)

            # second step: set points
            self.set_robot_position_in_obstacle_client(set_to_setpoints=True)
            for obstacle in self._obstacle_list:

                for j in range(len(obstacle.observed_links)):
                    link_index = obstacle.observed_links[j]

                    if self._links[link_index].observe_closest_point:
                        pos_obs, pos_rob, distance = \
                            self._compute_closest_points(p.getClosestPoints(bodyA=obstacle.id,
                                                                            bodyB=self._robot_scene.robot_id,
                                                                            distance=10,
                                                                            linkIndexA=obstacle.last_link,
                                                                            linkIndexB=link_index,
                                                                            physicsClientId=self._obstacle_client_id))
                        obstacle.link_data[j].closest_point_distance_set.append(distance)

                    if len(self._links[link_index].observed_points) > 0:
                        for k in range(len(self._links[link_index].observed_points)):
                            pos_rob = self._links[link_index].observed_points[k].get_position(actual=False)
                            pos_obs = obstacle.get_position(actual=False, pos_rob=pos_rob)

                            obstacle.link_data[j].observed_point_distance_set[k].append(
                                self._compute_distance(pos_obs, pos_rob,
                                                       radius_a=obstacle.bounding_sphere_radius,
                                                       radius_b=self._links[link_index].observed_points[
                                                           k].bounding_sphere_radius))

            # self-collision
            for i in range(len(self._links)):
                for j in range(len(self._links[i].self_collision_links)):
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=10,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    self._links[i].self_collision_data.closest_point_distance_set[j].append(distance)

        if self._log_obstacle_data:
            if list(pos_obs_debug):
                line_color = [1, 0, 0]
                line_width = 2
                if self._debug_line is not None:
                    self._debug_line = p.addUserDebugLine(pos_obs_debug, pos_rob_debug, lineColorRGB=line_color,
                                                          lineWidth=line_width,
                                                          replaceItemUniqueId=self._debug_line,
                                                          physicsClientId=self._simulation_client_id)
                else:
                    self._debug_line = p.addUserDebugLine(pos_obs_debug, pos_rob_debug, lineColorRGB=line_color,
                                                          lineWidth=line_width,
                                                          physicsClientId=self._simulation_client_id)
            else:
                if self._debug_line is not None:
                    p.removeUserDebugItem(self._debug_line, physicsClientId=self._simulation_client_id)
                    self._debug_line = None

        if self._visualize_bounding_spheres:
            for j in range(len(self._links)):
                sphere_color = [0] * len(self._links[j].observed_points)
                for i in range(len(self._obstacle_list)):
                    for m in range(len(self._obstacle_list[i].observed_links)):
                        link_index = self._obstacle_list[i].observed_links[m]
                        if link_index == j:
                            for k in range(len(self._links[j].observed_points)):
                                if self._visualize_bounding_sphere_actual:
                                    distance = self._obstacle_list[i].link_data[m].observed_point_distance_actual[k][-1]
                                else:
                                    distance = self._obstacle_list[i].link_data[m].observed_point_distance_set[k][-1]

                                if distance < self._links[link_index].observed_points[k].safety_distance:
                                    sphere_color[k] = 2

                for k in range(len(self._links[j].observed_points)):
                    if sphere_color[k] == 2:
                        rgba_color = OBSERVED_POINT_VIOLATION_COLOR
                    else:
                        rgba_color = None

                    self._links[j].observed_points[k].update_bounding_sphere_position(
                        actual=self._visualize_bounding_sphere_actual)

                    if rgba_color is not None:
                        self._links[j].observed_points[k].set_bounding_sphere_color(rgba_color=rgba_color)

            if self._target_link_point_list:
                for k in range(len(self._target_link_point_list)):
                    self._target_link_point_list[k].update_bounding_sphere_position(
                        actual=self._visualize_bounding_sphere_actual)

        if self._use_target_points:
            # check if the target link point is close to the target point
            for i in range(self._robot_scene.num_robots):
                self._target_link_pos_list[i] = np.array(self._target_link_point_list[i].get_position(
                    actual=self._target_point_use_actual_position))

            for i in range(self._robot_scene.num_robots):
                if self._target_point_active_list[i]:
                    self._target_point_pos_list[i] = np.array(self._target_point_list[i][-1].get_position(actual=False))
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        distance_list = []
                        for k in range(self._robot_scene.num_robots):
                            distance_list.append(self._compute_distance(self._target_link_pos_list[k],
                                                                        self._target_point_pos_list[i]))
                        distance = min(distance_list)
                        closest_robot = np.argmin(distance_list)
                    else:
                        distance = self._compute_distance(self._target_link_pos_list[i], self._target_point_pos_list[i])
                        closest_robot = i
                    if distance < self._target_point_list[i][-1].bounding_sphere_radius:
                        self._target_point_reached_list[i] = True
                        self._target_point_list[i][-1].make_invisible()
                        if self._simulation_client_id == self._backup_client_id:
                            self._target_point_id_reuse_list.append(self._target_point_list[i][-1].id)
                        self._target_point_active_list[i] = False
                        self._num_target_points_reached_list[closest_robot] = \
                            self._num_target_points_reached_list[closest_robot] + 1
                        if self._target_point_sequence == TARGET_POINT_SIMULTANEOUS:
                            self._sample_new_target_point_list[i] = True
                        if self._target_point_sequence == TARGET_POINT_ALTERNATING:
                            target_point_active_robot_index = (i + 1) % self._robot_scene.num_robots
                            self._sample_new_target_point_list[target_point_active_robot_index] = True
                        if self._target_point_sequence == TARGET_POINT_SINGLE:
                            target_point_active_robot_index = np.random.randint(0, self._robot_scene.num_robots)
                            self._sample_new_target_point_list[target_point_active_robot_index] = True

        if self._robot_scene.use_moving_objects and not update_after_reset:

            moving_object_updates_per_check = 1
            self._moving_object_check_counter = self._moving_object_check_counter + 1
            if self._moving_object_check_counter == moving_object_updates_per_check:
                self._moving_object_check_counter = 0
                visualize_active_area = False
                moving_object_in_active_area_xy_distance_to_origin = 1.25
                if self._robot_scene.robot_name.startswith("iiwa7"):
                    moving_object_in_active_area_x_value = 1
                else:
                    moving_object_in_active_area_x_value = 1

                for i in range(self._robot_scene.num_robots):
                    if self._moving_object_active_list[i]:
                        moving_object_position = self._moving_object_list[i][-1].get_position()
                        # check if the moving object is behind the robot
                        moving_object_behind_robot = self._moving_object_list[i][-1].check_if_object_is_behind_the_robot()
                        if moving_object_behind_robot or \
                                self._moving_object_list[i][-1].check_if_object_hit_obstacle():
                            if moving_object_behind_robot:
                                self._deactivate_moving_object(robot_index=i,
                                                               missed_robot=True)
                            else:
                                self._deactivate_moving_object(robot_index=i,
                                                               hit_obstacle=True)
                        else:
                            # check if moving object is in active area
                            if (self._robot_scene.moving_object_sphere_center is None and
                                    moving_object_position[0] < moving_object_in_active_area_x_value) or \
                                    (np.linalg.norm(moving_object_position[0:2]) <
                                     moving_object_in_active_area_xy_distance_to_origin):
                                if visualize_active_area:
                                    self._moving_object_list[i][-1].set_color(rgba_color=[1, 1, 1, 1])

                                collision_detected = self._moving_object_list[i][-1].check_if_object_is_colliding()
                                if collision_detected:
                                    self._deactivate_moving_object(robot_index=i,
                                                                   hit_robot=True)


    def _deactivate_moving_object(self, robot_index, object_index=None, missed_robot=False, hit_robot=False,
                                  hit_obstacle=False):


        moving_object = self._moving_object_list[robot_index][-1]  # moving object from outside

        if self._simulation_client_id == self._backup_client_id:
            reuse_id = True
            self._moving_object_id_reuse_list.append(moving_object.id)
        else:
            reuse_id = False

        final_state_options = 0
        if missed_robot:
            final_state_options += 1
        if hit_robot:
            final_state_options += 1
        if hit_obstacle:
            final_state_options += 1

        if final_state_options != 1:
            raise ValueError("A single final state option has to be selected.")

        moving_object.set_final_state(missed_robot=missed_robot,
                                      hit_robot=hit_robot,
                                      hit_obstacle=hit_obstacle)

        self._moving_object_active_list[robot_index] = False

        # moving object reached final state -> sample new moving objects
        if self._moving_object_sequence == MOVING_OBJECT_SIMULTANEOUS:
            self._sample_new_moving_object_list[robot_index] = True
        if self._moving_object_sequence == MOVING_OBJECT_ALTERNATING:
            moving_object_active_robot_index = \
                (robot_index + 1) % self._robot_scene.num_robots
            self._sample_new_moving_object_list[moving_object_active_robot_index] = True
        if self._moving_object_sequence == TARGET_POINT_SINGLE:
            for i in range(len(self._moving_object_observation_robot_indices)):
                if self._moving_object_observation_robot_indices[i] == robot_index:
                    self._moving_object_observation_robot_indices[i] = -1
                    break
            available_robots = []
            for j in range(self._robot_scene.num_robots):
                if not self._moving_object_active_list[j] and not self._sample_new_moving_object_list[j]:
                    available_robots.append(j)
            moving_object_active_robot_index = available_robots[np.random.randint(0, len(available_robots))]
            self._sample_new_moving_object_list[moving_object_active_robot_index] = True

        if missed_robot and not reuse_id:
            self._moving_object_fade_out_list[robot_index].append(moving_object)

    def _compute_closest_points(self, list_of_closest_points):
        pos_a = [0, 0, 0]
        pos_b = [0, 0, 0]
        closest_index = 0
        closest_distance = None
        if len(list_of_closest_points) > 0:
            closest_distance = list_of_closest_points[0][8]

            for i in range(1, len(list_of_closest_points)):
                if list_of_closest_points[i][8] < closest_distance:
                    closest_distance = list_of_closest_points[i][8]
                    closest_index = i

            pos_a = list_of_closest_points[closest_index][5]
            pos_b = list_of_closest_points[closest_index][6]

        return pos_a, pos_b, closest_distance

    def _compute_distance(self, pos_a, pos_b, radius_a=0, radius_b=0):
        return compute_distance_c(pos_a[0], pos_a[1], pos_a[2], pos_b[0], pos_b[1], pos_b[2], radius_a, radius_b)

    def clear_other_link_position_and_orn(self):
        for i in range(len(self._links)):
            self._links[i].clear_other_position_and_orn()

    def set_robot_position_in_obstacle_client(self, manip_joint_indices=None, target_position=None,
                                              target_velocity=None, set_to_setpoints=False,
                                              set_to_actual_values=False,
                                              set_to_default=False,
                                              capture_frame=False):
        # set robot with physicsClientId self._obstacle_client_id to a specified position
        if int(set_to_setpoints) + int(set_to_actual_values) + int(set_to_default) > 1:
            raise ValueError("set_to_setpoints, set_to_actual_values and set_to_default are not allowed to be True "
                             "at the same time")

        if set_to_setpoints:
            if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_TARGET_POSITION:
                return
            target_position = self._target_position
            target_velocity = self._target_velocity
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_TARGET_POSITION

        if set_to_actual_values:
            if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_ACTUAL_POSITION:
                return
            target_position = self._actual_position
            target_velocity = self._actual_velocity
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_ACTUAL_POSITION

        if set_to_default:
            if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_DEFAULT_POSITION:
                return
            target_position = self._robot_scene.default_position
            target_velocity = [0.0] * len(target_position)
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_DEFAULT_POSITION

        if not set_to_setpoints and not set_to_actual_values:
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION
            if target_velocity is None:
                target_velocity = [0.0] * len(target_position)

        self.clear_other_link_position_and_orn()

        if manip_joint_indices is None:
            manip_joint_indices = self._manip_joint_indices

        p.resetJointStatesMultiDof(bodyUniqueId=self._robot_scene.robot_id,
                                   jointIndices=manip_joint_indices,
                                   targetValues=[[pos] for pos in target_position],
                                   targetVelocities=[[vel] for vel in target_velocity],
                                   physicsClientId=self._obstacle_client_id)

        if capture_frame and self._robot_scene.capture_frame_function is not None:
            self._robot_scene.capture_frame_function()

    def _consider_bounding_sphere(self, pos_a, pos_b, radius_a, radius_b):
        if not np.array_equal(pos_a, pos_b):
            pos_diff = np.array(pos_b) - np.array(pos_a)
            pos_diff_norm = np.linalg.norm(pos_diff)
            pos_a_sphere = np.array(pos_a) + (radius_a / pos_diff_norm) * pos_diff
            pos_b_sphere = np.array(pos_a) + (1 - (radius_b / pos_diff_norm)) * pos_diff
            return pos_a_sphere, pos_b_sphere
        else:
            return [], []

    def get_braking_acceleration(self, last=False, no_shift=False):
        if last:
            self._valid_braking_trajectories['current'] = self._valid_braking_trajectories['last']
            self._valid_braking_trajectories['last'] = None
        min_distance = None
        max_torque = None
        if self._valid_braking_trajectories['current'] is not None:
            braking_acceleration = self._valid_braking_trajectories['current']['acceleration'][0]
            robot_stopped = False
            if self._valid_braking_trajectories['current']['min_distance']:
                min_distance = self._valid_braking_trajectories['current']['min_distance'][0]
            if self._valid_braking_trajectories['current']['max_torque']:
                max_torque = self._valid_braking_trajectories['current']['max_torque'][0]

            if not no_shift:
                if len(self._valid_braking_trajectories['current']['acceleration']) > 1:
                    for key, value in self._valid_braking_trajectories['current'].items():
                        self._valid_braking_trajectories['current'][key] = value[1:]
                else:
                    self._valid_braking_trajectories['current'] = None
        else:
            braking_acceleration = np.zeros(self._robot_scene.num_manip_joints)
            robot_stopped = True

        return braking_acceleration, robot_stopped, min_distance, max_torque

    def check_braking_trajectory_method(self, current_acc, current_vel, current_pos, target_acc,
                                        time_step_counter=0):

        self._braking_trajectory = {'acceleration': [current_acc, target_acc],
                                    'velocity': [current_vel],
                                    'position': [current_pos],
                                    'min_distance': [],
                                    'max_torque': []}

        self._affected_observed_point = None
        self._affected_link_index_list = None
        self._braking_trajectory_collision_free = True
        self._braking_trajectory_torque_limited = True

        if self._check_braking_trajectory_collisions:
            self._braking_trajectory_collision_free, self._affected_link_index_list, self._affected_observed_point \
                = self._check_if_braking_trajectory_is_collision_free(time_step_counter)

        if self._check_braking_trajectory_torque_limits and self._braking_trajectory_collision_free:
            self._braking_trajectory_torque_limited, self._affected_link_index_list = \
                self._check_if_braking_trajectory_is_torque_limited(time_step_counter)

        execute_braking_trajectory = False

        if not self._braking_trajectory_collision_free or not self._braking_trajectory_torque_limited:
            execute_braking_trajectory = True

        return execute_braking_trajectory

    def adapt_action(self, current_acc, current_vel, current_pos, target_acc,
                     action_considered_as_safe=False, time_step_counter=0):

        execute_braking_trajectory = False
        adapted_acc = target_acc
        adaptation_punishment = 0  # adaptation punishment: 0: no adaptation, 1: braking trajectory executed
        min_distance = None  # after executing the adapted action
        max_torque = None
        # min_distance and max_torque after executing the adapted action (for reward calculation),
        # Both are None if execute_braking_trajectory == True
        # in addition, min_distance is None if the braking trajectory is not checked for collisions and
        # max_torque is None if the braking trajectory is not checked for torque limit violations

        if self._use_braking_trajectory_method:

            if not action_considered_as_safe:
                execute_braking_trajectory = \
                    self.check_braking_trajectory_method(current_acc, current_vel, current_pos, target_acc,
                                                         time_step_counter=time_step_counter)
            else:
                execute_braking_trajectory = False

            if execute_braking_trajectory:
                adaptation_punishment = 1.0

            self._braking_duration_list.append(self._braking_duration)

            if execute_braking_trajectory:
                if self._active_braking_influence_time == 0:
                    if len(self._braking_duration_list) > 0:
                        self._active_braking_duration_list.append(self._braking_duration_list[-1])
                    else:
                        self._active_braking_duration_list.append(0)  # start braking from the initial position
                self._active_braking_influence_time += self._trajectory_time_step

            else:
                if self._active_braking_influence_time != 0:
                    # the last action was adapted by the braking trajectory method
                    self._active_braking_influence_time_list.append(self._active_braking_influence_time)
                self._active_braking_influence_time = 0

                self._valid_braking_trajectories['last'] = self._valid_braking_trajectories['current']
                self._valid_braking_trajectories['current'] = self._braking_trajectory

                if self._valid_braking_trajectories['current']['min_distance']:
                    min_distance = self._valid_braking_trajectories['current']['min_distance'][0]
                    self._valid_braking_trajectories['current']['min_distance'] = \
                        self._valid_braking_trajectories['current']['min_distance'][1:]
                else:
                    min_distance = None
                if self._valid_braking_trajectories['current']['max_torque']:
                    max_torque = self._valid_braking_trajectories['current']['max_torque'][0]
                    self._valid_braking_trajectories['current']['max_torque'] = \
                        self._valid_braking_trajectories['current']['max_torque'][1:]
                else:
                    max_torque = None

                if self._braking_trajectory_length > 0:
                    for key in ['position', 'velocity', 'acceleration']:
                        self._valid_braking_trajectories['current'][key] = \
                            self._valid_braking_trajectories['current'][key][2:]
                        # remove current and next kinematic state
                else:
                    self._valid_braking_trajectories['current'] = None

            if self._visual_mode and not self._robot_scene.no_link_coloring:
                # set colors for each link
                for i in range(len(self._links)):
                    if execute_braking_trajectory and i in self._affected_link_index_list:
                        if self._braking_trajectory_torque_limited:
                            if len(self._affected_link_index_list) == 1:
                                color = LINK_OBJECT_COLLISION_INFLUENCE_COLOR
                            else:
                                color = LINK_SELF_COLLISION_INFLUENCE_COLOR  # self collision
                        else:
                            color = LINK_TORQUE_INFLUENCE_COLOR  # potential torque violations
                        self._links[i].set_color(color)
                    else:
                        self._links[i].set_color(rgba_color=None)  # set color to default
                    if self._visualize_bounding_spheres and self._distance_calculation_check_observed_points:
                        for k in range(len(self._links[i].observed_points)):
                            if self._affected_observed_point is not None and self._affected_observed_point[0] == i \
                                    and self._affected_observed_point[1] == k:
                                self._links[i].observed_points[k].set_bounding_sphere_color(
                                    rgba_color=OBSERVED_POINT_INFLUENCE_COLOR)
                            else:
                                self._links[i].observed_points[k].set_bounding_sphere_color(rgba_color=None)

        if self._braking_trajectory_collision_free:
            self._time_influenced_by_braking_trajectory_collision_list.append(0.0)
        else:
            self._time_influenced_by_braking_trajectory_collision_list.append(1.0)

        if self._braking_trajectory_torque_limited:
            self._time_influenced_by_braking_trajectory_torque_list.append(0.0)
        else:
            self._time_influenced_by_braking_trajectory_torque_list.append(1.0)

        return adapted_acc, execute_braking_trajectory, adaptation_punishment, min_distance, max_torque

    def _check_if_braking_trajectory_is_collision_free(self, time_step_counter=0):
        # execute the following step with the target acceleration and compute a braking trajectory after that step.
        # check for each time step if a collision occurred
        self._braking_trajectory_minimum_distance = np.inf
        self._braking_trajectory_maximum_rel_torque = 0

        robot_stopped = False
        collision_found = False
        affected_link_index_list = []
        affected_observed_point = None  # if affected: [link_index, point_index]
        minimum_distance = None

        time_since_start = np.linspace(self._trajectory_time_step / self._collision_checks_per_time_step,
                                       self._trajectory_time_step,
                                       self._collision_checks_per_time_step)
        while not robot_stopped and not collision_found:
            start_acceleration = self._braking_trajectory['acceleration'][-2]
            end_acceleration = self._braking_trajectory['acceleration'][-1]
            start_velocity = self._braking_trajectory['velocity'][-1]
            start_position = self._braking_trajectory['position'][-1]

            interpolated_position_batch = interpolate_position_batch(start_acceleration,
                                                                     end_acceleration,
                                                                     start_velocity,
                                                                     start_position, time_since_start,
                                                                     self._trajectory_time_step)

            for m in range(self._collision_checks_per_time_step):
                minimum_distance, collision_found, affected_link_index_list, affected_observed_point, _, _ = \
                    self.get_minimum_distance(manip_joint_indices=self._manip_joint_indices,
                                              target_position=interpolated_position_batch[m])

                if minimum_distance < self._braking_trajectory_minimum_distance:
                    self._braking_trajectory_minimum_distance = minimum_distance

                if collision_found:
                    break

            if not collision_found:
                self._braking_trajectory['min_distance'].append(minimum_distance)
                # minimum distance at m == self._collision_checks_per_time_step - 1:

                # compute the target acceleration for the next decision step
                robot_stopped, braking_timeout = \
                    self._compute_next_braking_trajectory_time_step(start_position=interpolated_position_batch[-1])

                if braking_timeout:
                    collision_found = True

        if robot_stopped and not collision_found:
            return True, [], None
        else:
            return False, affected_link_index_list, affected_observed_point

    def check_if_moving_object_missed_robot(self):
        # returns true if final status of a moving object is missed robot or hit obstacle
        for i in range(len(self._moving_object_active_list)):
            if not self._moving_object_active_list[i] and self._moving_object_initial_active_list[i] and \
                    self._moving_object_list[i][-1].check_final_state(missed_robot=True, hit_obstacle=True):
                return True
        return False

    def get_minimum_distance_to_moving_obstacles(self, manip_joint_indices=None, target_position=None,
                                                 maximum_relevant_distance=None):

        moving_obstacle_list = []
        for i in range(len(self._moving_object_active_list)):
            if self._moving_object_active_list[i]:
                moving_obstacle_list.append(self._moving_object_list[i][-1])
            elif self._moving_object_initial_active_list[i] and \
                    self._moving_object_list[i][-1].check_final_state(hit_robot=True):
                minimum_distance = 0.0
                return minimum_distance

        if self._planet_list:
            for i in range(len(self._planet_list)):
                if self._planet_list[i].collision_detected:
                    # collisions are detected only if terminate_on_collision_with_moving_obstacle==True
                    minimum_distance = 0.0
                    return minimum_distance
                moving_obstacle_list.append(self._planet_list[i])

        if target_position is not None:
            self.set_robot_position_in_obstacle_client(manip_joint_indices=manip_joint_indices,
                                                       target_position=target_position,
                                                       target_velocity=None,
                                                       capture_frame=True)

        for i in range(len(moving_obstacle_list)):
            moving_obstacle_list[i].update_position_in_obstacle_client()

        if self._human is not None:
            if self._human.collision_detected:
                minimum_distance = 0.0
                return minimum_distance
            self._human.set_position_in_obstacle_client_to_setpoints()
            moving_obstacle_list.append(self._human)

        minimum_distance = self._get_minimum_distance_to_obstacles(moving_obstacle_list,
                                                                   maximum_relevant_distance=maximum_relevant_distance)

        return minimum_distance

    def _get_minimum_distance_to_obstacles(self, obstacle_list, maximum_relevant_distance=None):
        if maximum_relevant_distance is None:
            maximum_relevant_distance = self._closest_point_maximum_relevant_distance
        minimum_distance_to_obstacles = maximum_relevant_distance + 0.002

        for i in range(len(obstacle_list)):
            for link_index in obstacle_list[i].observed_links:
                for obstacle_link in obstacle_list[i].obstacle_links:
                    _, _, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=obstacle_list[i].id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=maximum_relevant_distance,
                                           linkIndexA=obstacle_link,
                                           linkIndexB=link_index,
                                           physicsClientId=self._obstacle_client_id))

                    if distance is not None:
                        if distance < minimum_distance_to_obstacles:
                            minimum_distance_to_obstacles = distance
                            if minimum_distance_to_obstacles <= 0.0:
                                return 0.0

        return minimum_distance_to_obstacles

    def get_minimum_distance(self, manip_joint_indices, target_position, check_safety_distance=True):
        # returns minimum_distance, collision_found, affected_link_index_list, affected_observed_point,
        # minimum_distance_to_obstacles, minimum_distance_self_collision
        self.set_robot_position_in_obstacle_client(manip_joint_indices=manip_joint_indices,
                                                   target_position=target_position,
                                                   target_velocity=None,
                                                   capture_frame=True)

        minimum_distance = self._closest_point_maximum_relevant_distance
        minimum_distance_to_obstacles = self._closest_point_maximum_relevant_distance
        minimum_distance_self_collision = self._closest_point_maximum_relevant_distance

        for i in range(len(self._obstacle_list)):
            for link_index in self._obstacle_list[i].observed_links:
                for obstacle_link in self._obstacle_list[i].obstacle_links:
                    if self._distance_calculation_check_closest_points and \
                            self._links[link_index].closest_point_active:
                        pos_obs, pos_rob, distance = self._compute_closest_points(
                            p.getClosestPoints(bodyA=self._obstacle_list[i].id,
                                               bodyB=self._robot_scene.robot_id,
                                               distance=self._closest_point_maximum_relevant_distance,
                                               linkIndexA=obstacle_link,
                                               linkIndexB=link_index,
                                               physicsClientId=self._obstacle_client_id))

                        safety_distance = self._links[link_index].closest_point_safety_distance

                        if distance is not None:

                            if distance < minimum_distance:
                                minimum_distance = distance

                            if distance < safety_distance and check_safety_distance:
                                collision_found = True
                                affected_link_index_list = [link_index]
                                return minimum_distance, collision_found, affected_link_index_list, None, None, None

                            if distance < minimum_distance_to_obstacles:
                                minimum_distance_to_obstacles = distance

                if self._distance_calculation_check_observed_points \
                        and len(self._links[link_index].observed_points) > 0:
                    for k in range(len(self._links[link_index].observed_points)):
                        if self._links[link_index].observed_points[k].is_active:
                            pos_rob = self._links[link_index].observed_points[k].get_position(actual=None)
                            pos_obs = self._obstacle_list[i].get_position(actual=False, pos_rob=pos_rob)

                            distance = self._compute_distance(pos_obs, pos_rob,
                                                              radius_a=self._obstacle_list[
                                                                  i].bounding_sphere_radius,
                                                              radius_b=self._links[link_index].observed_points[
                                                                  k].bounding_sphere_radius)
                            safety_distance = self._links[link_index].observed_points[k].safety_distance

                            if distance < minimum_distance:
                                minimum_distance = distance

                            if distance < safety_distance and check_safety_distance:
                                collision_found = True
                                affected_observed_point = [link_index, k]
                                return minimum_distance, collision_found, [], affected_observed_point, None, None

                            if distance < minimum_distance_to_obstacles:
                                minimum_distance_to_obstacles = distance

        for i in range(len(self._links)):
            for j in range(len(self._links[i].self_collision_links)):
                if (self._links[i].closest_point_active or self._links[
                    self._links[i].self_collision_links[j]].closest_point_active) \
                        and self._distance_calculation_check_closest_points:
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=self._closest_point_maximum_relevant_distance,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    safety_distance = self._links[i].closest_point_safety_distance

                    if distance is not None:
                        if distance < minimum_distance:
                            minimum_distance = distance

                        if distance < safety_distance and check_safety_distance:
                            collision_found = True
                            affected_link_index_list = [i, self._links[i].self_collision_links[j]]
                            return minimum_distance, collision_found, affected_link_index_list, None, None, None

                        if distance < minimum_distance_self_collision:
                            minimum_distance_self_collision = distance

        return minimum_distance, False, [], None, minimum_distance_to_obstacles, minimum_distance_self_collision

    def _check_if_braking_trajectory_is_torque_limited(self, time_step_counter=0):

        braking_time_step = 0

        self.set_robot_position_in_obstacle_client(target_position=self._last_actual_position,
                                                   target_velocity=self._last_actual_velocity)

        self._robot_scene.set_motor_control(target_positions=self._braking_trajectory['position'][braking_time_step],
                                            target_velocities=self._braking_trajectory['velocity'][braking_time_step],
                                            physics_client_id=self._obstacle_client_id)

        for i in range(1):
            p.stepSimulation(physicsClientId=self._obstacle_client_id)


        affected_link_index_list = []
        maximum_rel_torque = None

        time_since_start = np.linspace(self._trajectory_time_step / self._simulation_steps_per_action,
                                       self._trajectory_time_step,
                                       self._simulation_steps_per_action)

        if self._check_braking_trajectory_collisions:
            extra_time_steps = 0  # no additional time steps are required as a certain safety distance is guaranteed
        else:
            extra_time_steps = 1  # check additional time steps since the actual values lag behind the setpoints

        robot_stopped = False

        while not robot_stopped:
            interpolated_position_batch = \
                interpolate_position_batch(self._braking_trajectory['acceleration'][braking_time_step],
                                           self._braking_trajectory['acceleration'][braking_time_step + 1],
                                           self._braking_trajectory['velocity'][braking_time_step],
                                           self._braking_trajectory['position'][braking_time_step],
                                           time_since_start, self._trajectory_time_step)

            if self._robot_scene.use_controller_target_velocities:
                interpolated_velocity_batch = interpolate_velocity_batch(
                    self._braking_trajectory['acceleration'][braking_time_step],
                    self._braking_trajectory['acceleration'][braking_time_step + 1],
                    self._braking_trajectory['velocity'][braking_time_step],
                    time_since_start, self._trajectory_time_step)
            else:
                interpolated_velocity_batch = None
            for m in range(self._simulation_steps_per_action):
                self._robot_scene.set_motor_control(target_positions=interpolated_position_batch[m],
                                                    target_velocities=interpolated_velocity_batch[m]
                                                    if interpolated_velocity_batch is not None else None,
                                                    physics_client_id=self._obstacle_client_id)
                p.stepSimulation(physicsClientId=self._obstacle_client_id)
                actual_joint_torques = self._robot_scene.get_actual_joint_torques(
                    physics_client_id=self._obstacle_client_id)
                normalized_joint_torques = normalize(actual_joint_torques, self._torque_limits)
                normalized_joint_torques_abs = np.abs(normalized_joint_torques)
                maximum_rel_torque = np.max(normalized_joint_torques_abs)
                if maximum_rel_torque > self._braking_trajectory_maximum_rel_torque:
                    self._braking_trajectory_maximum_rel_torque = maximum_rel_torque

                joint_torque_exceeded = normalized_joint_torques_abs > 0.98

                if np.any(joint_torque_exceeded):
                    affected_link_index_list = np.array(self._manip_joint_indices)[joint_torque_exceeded]
                    return False, affected_link_index_list

            self._braking_trajectory['max_torque'].append(maximum_rel_torque)
            # max torque at m == self._simulation_steps_per_action - 1 -> end of the time step

            braking_time_step = braking_time_step + 1

            if not self._check_braking_trajectory_collisions:
                # compute the target acceleration for the next decision step
                robot_stopped, braking_timeout = \
                    self._compute_next_braking_trajectory_time_step(start_position=interpolated_position_batch[-1],
                                                                    extend_trajectory=extra_time_steps > 0)
                if robot_stopped and extra_time_steps > 0:
                    extra_time_steps = extra_time_steps - 1
                    robot_stopped = False

                if braking_timeout:
                    return False, affected_link_index_list
            else:
                # braking trajectory already calculated
                robot_stopped = braking_time_step > self._braking_trajectory_length
                if robot_stopped and extra_time_steps > 0:
                    self._compute_next_braking_trajectory_time_step(start_position=interpolated_position_batch[-1],
                                                                    extend_trajectory=True)
                    extra_time_steps = extra_time_steps - 1
                    robot_stopped = False

        return True, affected_link_index_list

    def _compute_next_braking_trajectory_time_step(self, start_position, start_velocity=None, extend_trajectory=False,
                                                   braking_timeout=2.0):

        if self._braking_duration > braking_timeout:
            self._braking_timeout = True
            return False, self._braking_timeout

        if start_velocity is None:
            start_velocity = interpolate_velocity(self._braking_trajectory['acceleration'][-2],
                                                  self._braking_trajectory['acceleration'][-1],
                                                  self._braking_trajectory['velocity'][-1],
                                                  self._trajectory_time_step, self._trajectory_time_step)
        start_acceleration = self._braking_trajectory['acceleration'][-1]

        end_acceleration, robot_stopped = self._compute_braking_acceleration(start_position=start_position,
                                                                             start_velocity=start_velocity,
                                                                             start_acceleration=
                                                                             start_acceleration)

        if not robot_stopped or extend_trajectory:
            self._braking_trajectory['acceleration'].append(end_acceleration)
            self._braking_trajectory['velocity'].append(start_velocity)
            self._braking_trajectory['position'].append(start_position)

        return robot_stopped, False

    def _compute_braking_acceleration(self, start_position, start_velocity, start_acceleration):
        joint_acc_min, joint_acc_max = self._acc_range_function(start_position=start_position,
                                                                start_velocity=start_velocity,
                                                                start_acceleration=
                                                                start_acceleration)

        end_acceleration, robot_stopped = self._acc_braking_function(start_velocity=start_velocity,
                                                                     start_acceleration=start_acceleration,
                                                                     next_acc_min=joint_acc_min,
                                                                     next_acc_max=joint_acc_max,
                                                                     index=0)

        return end_acceleration, robot_stopped


    def _interpolate_position(self, start_acceleration, end_acceleration, start_velocity, start_position,
                              time_since_start):
        interpolated_position = start_position + start_velocity * time_since_start + \
                                0.5 * start_acceleration * time_since_start ** 2 + \
                                1 / 6 * ((end_acceleration - start_acceleration)
                                         / self._trajectory_time_step) * time_since_start ** 3

        return interpolated_position

    def _interpolate_velocity(self, start_acceleration, end_acceleration, start_velocity, time_since_start):

        interpolated_velocity = start_velocity + start_acceleration * time_since_start + \
                                0.5 * ((end_acceleration - start_acceleration) /
                                       self._trajectory_time_step) * time_since_start ** 2

        return interpolated_velocity

    def _interpolate_acceleration(self, start_acceleration, end_acceleration, time_since_start):
        interpolated_acceleration = start_acceleration + ((end_acceleration - start_acceleration) /
                                                          self._trajectory_time_step) * time_since_start

        return interpolated_acceleration

    def _normalize(self, value, value_range):
        normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
        return normalized_value

    @property
    def num_links(self):
        return len(self._links)

    @property
    def _braking_duration(self):
        return self._braking_trajectory_length * self._trajectory_time_step

    @property
    def _braking_trajectory_length(self):
        return len(self._braking_trajectory['position']) - 1  # only the "braking" part (first time step excluded)

    @property
    def link_names(self):
        link_names = [link.name for link in self._links]
        return link_names

    @property
    def debug_line_obstacle(self):
        return self._debug_line_obstacle

    @debug_line_obstacle.setter
    def debug_line_obstacle(self, val):
        self._debug_line_obstacle = val

    @property
    def debug_line_link(self):
        return self._debug_line_link

    @debug_line_link.setter
    def debug_line_link(self, val):
        self._debug_line_link = val

    @property
    def debug_line_point(self):
        return self._debug_line_point

    @debug_line_point.setter
    def debug_line_point(self, val):
        self._debug_line_point = val

    @property
    def planet_list(self):
        return self._planet_list

    @property
    def moving_object_deterministic_list(self):
        return self._moving_object_deterministic_list

    @property
    def target_point_deterministic_list(self):
        return self._target_point_deterministic_list

    @property
    def human(self):
        return self._human


class ObstacleBase:
    def __init__(self,
                 name=None,
                 observed_links=None,
                 num_observed_points_per_link=None,
                 *vargs,
                 **kwargs):

        if num_observed_points_per_link is None:
            num_observed_points_per_link = []
        if observed_links is None:
            observed_links = []
        self._observed_links = observed_links
        if len(self._observed_links) != len(num_observed_points_per_link):
            raise ValueError("observed Points per Link not specified")

        self._link_data = []
        for i in range(len(observed_links)):
            self._link_data.append(LinkData(num_observed_points=num_observed_points_per_link[i]))

        self._name = name
        self._last_link = None
        self._obstacle_links = None

    def get_link_index(self, link_number):
        for i in range(len(self._observed_links)):
            if self._observed_links[i] == link_number:
                return i
        raise ValueError("Desired link is not observed")

    @property
    def observed_links(self):
        return self._observed_links

    @property
    def obstacle_links(self):
        if self._obstacle_links is None:
            return [self._last_link]
        else:
            return self._obstacle_links

    @property
    def name(self):
        return self._name

    @property
    def link_data(self):
        return self._link_data


class ObstacleSim(ObstacleBase):
    def __init__(self,
                 pos=(0, 0, 0),
                 orn=(0.0, 0.0, 0.0, 1.0),
                 create_collision_shape_in_simulation_client=True,
                 create_collision_shape_in_obstacle_client=True,
                 no_visual_shape=False,
                 is_static=False,
                 shape=p.GEOM_SPHERE,
                 urdf_file_name=None,
                 urdf_scaling_factor=1.0,
                 radius=None,
                 length=None,
                 half_extents=None,
                 plane_points=None,
                 color=(0, 0, 1, 0.5),
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 backup_client_id=None,
                 gui_client_id=None,
                 visual_mode=False,
                 update_position_in_obstacle_client=False,
                 use_real_robot=False,
                 num_clients=1,
                 is_target=False,
                 plane_collision_shape_factor=1,
                 id_to_reuse=None,
                 do_not_load_urdf=False,
                 *vargs,
                 **kwargs):

        super().__init__(*vargs, **kwargs)
        default_orn = p.getQuaternionFromEuler([0, 0, 0])

        self._shape = shape
        self._bounding_sphere_radius = 0 if radius is None else radius
        self._is_static = is_static
        self._radius = radius
        self._length = length
        self._base_pos = pos
        self._base_orn = orn
        self._color = np.asarray(color)
        self._target_pos = [0, 0, 0]
        self._target_vel = [0, 0, 0]
        self._target_acc = [0, 0, 0]

        self._position_actual = None
        self._position_set = None
        self._orn_actual = None
        self._orn_set = None
        self._is_target = is_target

        self._num_clients = num_clients
        self._simulation_client_id = simulation_client_id
        self._obstacle_client_id = obstacle_client_id
        self._backup_client_id = backup_client_id
        self._gui_client_id = gui_client_id
        self._update_position_in_obstacle_client = update_position_in_obstacle_client
        self._use_real_robot = use_real_robot
        self._create_collision_shape_in_simulation_client = create_collision_shape_in_simulation_client
        self._create_collision_shape_in_obstacle_client = create_collision_shape_in_obstacle_client

        self._visual_mode = visual_mode

        self._urdf_file_name = urdf_file_name
        if self._urdf_file_name is not None:
            self._urdf_file_name_no_collision = urdf_file_name + "_no_collision"
            self._urdf_file_name_no_visual = urdf_file_name + "_no_visual"
        else:
            self._urdf_file_name_no_collision = None
        self._urdf_scaling_factor = urdf_scaling_factor
        self._urdf_dir = None

        if id_to_reuse is None:
            for i in range(self._num_clients):
                create_collision_shape = (create_collision_shape_in_simulation_client and
                                          (i == self._simulation_client_id or i == self._backup_client_id)) or \
                                         (create_collision_shape_in_obstacle_client and i == self._obstacle_client_id)
                if urdf_file_name is None:
                    if shape == p.GEOM_SPHERE:
                        if radius is None:
                            raise ValueError("Radius required")
                        if create_collision_shape:
                            shape_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius,
                                                                     physicsClientId=i)
                        else:
                            shape_collision = -1

                        shape_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius,
                                                           rgbaColor=color, physicsClientId=i)

                    if shape == p.GEOM_CAPSULE:
                        if radius is None or length is None:
                            raise ValueError("Radius and length required")
                        if create_collision_shape:
                            shape_collision = p.createCollisionShape(shapeType=p.GEOM_CAPSULE, radius=radius,
                                                                     height=length, physicsClientId=i)
                        else:
                            shape_collision = -1

                        shape_visual = p.createVisualShape(shapeType=p.GEOM_CAPSULE, length=length, radius=radius,
                                                           rgbaColor=color, physicsClientId=i)

                    if shape == p.GEOM_BOX:
                        if half_extents is None:
                            raise ValueError("half_extents required")
                        if create_collision_shape:
                            shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                                     physicsClientId=i)
                        else:
                            shape_collision = -1

                        shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                           rgbaColor=color, physicsClientId=i)

                    if shape == p.GEOM_PLANE:
                        plane_points = np.array(plane_points)
                        plane_x = np.linalg.norm(plane_points[1] - plane_points[0])
                        plane_y = np.linalg.norm(plane_points[2] - plane_points[0])
                        self._base_pos = plane_points[0] + 0.5 * (plane_points[1] - plane_points[0]) + \
                            0.5 * (plane_points[2] - plane_points[0])
                        half_extents_visual = [plane_x / 2, plane_y / 2, 0.0025]
                        shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents_visual,
                                                           rgbaColor=color, physicsClientId=i)
                        half_extents_collision = [plane_x / 2 * plane_collision_shape_factor,
                                                  plane_y / 2 * plane_collision_shape_factor, 0.0025]
                        if create_collision_shape:
                            shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                                     halfExtents=half_extents_collision,
                                                                     physicsClientId=i)
                        else:
                            shape_collision = -1

                        self._plane_normal = np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[0])
                        self._plane_constant = np.dot(plane_points[0], self._plane_normal)

                    if shape == p.GEOM_PLANE or self._is_static:
                        self._last_link = -1
                        self._is_static = True

                        self.id = p.createMultiBody(baseMass=0,
                                                    basePosition=self._base_pos,
                                                    baseOrientation=orn,
                                                    baseVisualShapeIndex=shape_visual,
                                                    baseCollisionShapeIndex=shape_collision,
                                                    physicsClientId=i)
                    else:
                        self._last_link = 2
                        self.id = p.createMultiBody(baseMass=0,
                                                    basePosition=self._base_pos,
                                                    baseOrientation=orn,
                                                    linkMasses=[1, 1, 1],
                                                    linkVisualShapeIndices=[-1, -1, shape_visual],
                                                    linkCollisionShapeIndices=[-1, -1, shape_collision],
                                                    linkPositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                    linkOrientations=[default_orn, default_orn, default_orn],
                                                    linkParentIndices=[0, 1, 2],
                                                    linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC,
                                                                    p.JOINT_PRISMATIC],
                                                    linkJointAxis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                    linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                    linkInertialFrameOrientations=[default_orn, default_orn, default_orn],
                                                    physicsClientId=i
                                                    )
                else:
                    self._is_static = True
                    self._last_link = -1
                    if create_collision_shape:
                        if no_visual_shape:
                            urdf_file_name = self._urdf_file_name_no_visual
                        else:
                            urdf_file_name = self._urdf_file_name
                    else:
                        urdf_file_name = self._urdf_file_name_no_collision
                    urdf_file_name = urdf_file_name + ".urdf"
                    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
                    self._urdf_dir = os.path.join(module_dir, "description", "urdf", "obstacles")
                    # the flag URDF_ENABLE_CACHED_GRAPHICS_SHAPES is required to avoid pybullet stopping to visualize
                    # new obstacles after a while
                    # however, the flag can cause the visualization of shadows to be faulty.
                    # in particular, setting the transparency of one of the objects to a value different to 1 might
                    # cause the shadows of the other objects to disappear
                    # comment out the flag if this behavior is undesired
                    if not do_not_load_urdf:
                        flags = 0
                        if USE_CACHED_GRAPHICS_SHAPES:
                            flags = flags | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                        self.id = p.loadURDF(os.path.join(self._urdf_dir, urdf_file_name), basePosition=self._base_pos,
                                            baseOrientation=self._base_orn, globalScaling=urdf_scaling_factor,
                                            useFixedBase=True, flags=flags,
                                            physicsClientId=i)
                        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=i)
                    else:
                        self.id = None

        else:
            self.id = id_to_reuse
            self._is_static = True
            self._last_link = -1
            if self._base_pos is not None:
                p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self._base_pos,
                                                  ornObj=self._base_orn,
                                                  physicsClientId=self._simulation_client_id)

            p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=self._simulation_client_id)

        if not (shape == p.GEOM_PLANE or self._is_static):
            for i in range(3):
                p.setJointMotorControl2(self.id, i,
                                        p.POSITION_CONTROL,
                                        targetPosition=self._target_pos[i],
                                        targetVelocity=0,
                                        maxVelocity=0,
                                        positionGain=0.1,
                                        velocityGain=1)

    def update_position_in_obstacle_client(self):
        if self._obstacle_client_id is not None:
            p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self._position_set,
                                              ornObj=self._orn_set,
                                              physicsClientId=self._obstacle_client_id)

    def set_to_unreachable_position_in_backup_client(self):
        self._position_set = [0, 0, -1]
        if self._backup_client_id is not None:
            p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self._position_set,
                                              ornObj=self._orn_set,
                                              physicsClientId=self._backup_client_id)

    def set_simulation_client_to_backup_client(self):
        self._simulation_client_id = self._backup_client_id

    def update(self):
        pass

    def get_local_position(self, world_position, actual=True):
        return np.array(world_position) - np.array(self.get_position(actual=actual))

    def make_invisible(self):
        if self._simulation_client_id == self._backup_client_id:
            p.changeVisualShape(self.id, -1, rgbaColor=[1, 1, 1, 0], physicsClientId=self._simulation_client_id)
        else:
            for i in range(self._num_clients):
                p.changeVisualShape(self.id, -1, rgbaColor=[1, 1, 1, 0], physicsClientId=i)

    @property
    def target_pos(self):
        return self._target_pos

    @property
    def is_target(self):
        return self._is_target

    @property
    def target_vel(self):
        return self._target_vel

    @property
    def target_acc(self):
        return self._target_acc

    @property
    def is_static(self):
        return self._is_static

    def get_position_set(self):
        return self._position_set

    def get_orn(self, actual=True):
        if actual:
            if self._orn_actual is None:
                self.get_position(actual=actual)
            return self._orn_actual
        else:
            if self._orn_set is None:
                self.get_position(actual=actual)
            return self._orn_set

    def get_position(self, actual=True, pos_rob=None):
        if self._shape is not p.GEOM_PLANE:
            if actual:
                if self._position_actual is None:
                    if self._is_static:
                        self._position_actual, self._orn_actual = self._base_pos, self._base_orn
                    else:
                        if self._simulation_client_id is not None and not self._use_real_robot:
                            if self._last_link == -1:
                                self._position_actual, self._orn_actual = \
                                    p.getBasePositionAndOrientation(self.id, physicsClientId=self._simulation_client_id)
                            else:
                                link_state = p.getLinkState(self.id, self._last_link, computeLinkVelocity=False,
                                                            computeForwardKinematics=True,
                                                            physicsClientId=self._simulation_client_id)
                                self._position_actual = link_state[4]
                                self._orn_actual = link_state[5]
                        else:
                            raise NotImplementedError("Actual obstacle position not implemented for real robots")
                return self._position_actual
            else:
                if self._position_set is None:
                    if self._is_static:
                        self._position_set, self._orn_set = self._base_pos, self._base_orn
                    else:
                        if self._last_link == -1:
                            self._position_set, self._orn_set = \
                                p.getBasePositionAndOrientation(self.id, physicsClientId=self._obstacle_client_id)
                        else:
                            link_state = p.getLinkState(self.id, self._last_link, computeLinkVelocity=False,
                                                        computeForwardKinematics=True,
                                                        physicsClientId=self._obstacle_client_id)
                            self._position_set = link_state[4]
                            self._orn_set = link_state[5]
                return self._position_set
        else:
            pos_rob = np.array(pos_rob)
            x = (self._plane_constant - np.dot(pos_rob, self._plane_normal)) / (np.linalg.norm(self._plane_normal) ** 2)
            return pos_rob + x * self._plane_normal

    def reset(self):
        self.clear_previous_timestep()
        for i in range(len(self._link_data)):
            self._link_data[i].reset()

    def clear_previous_timestep(self):
        if not self._is_static:
            self._position_actual = None
            self._position_set = None
            self._orn_actual = None
            self._orn_set = None

    def delete(self):
        if self.id is not None:
            for i in range(self._num_clients):
                p.removeBody(self.id, physicsClientId=i)

    def set_color(self, rgba_color):
        if self.id is not None:
            for i in range(self._num_clients):
                p.changeVisualShape(self.id, -1, rgbaColor=rgba_color, physicsClientId=i)

    def get_mass(self):
        mass = p.getDynamicsInfo(self.id, linkIndex=-1)[0]
        print("mass", mass)
        return mass

    def set_mass(self, mass):
        p.changeDynamics(self.id, linkIndex=-1, mass=mass)

    @property
    def last_link(self):
        return self._last_link

    @property
    def bounding_sphere_radius(self):
        return self._bounding_sphere_radius


class Ball(ObstacleSim):
    FINAL_STATE_MISSED_ROBOT = 1
    FINAL_STATE_HIT_ROBOT = 2
    FINAL_STATE_HIT_OBSTACLE = 3

    FADE_OUT_MODE_A = 0
    FADE_OUT_MODE_B = 1
    FADE_OUT_MODE_C = 2

    def __init__(self,
                 initial_speed_vector,
                 update_time_step,
                 update_steps_per_action,
                 robot_id,
                 update_steps_per_pos_update=1,
                 update_steps_per_color_update=10,
                 update_steps_for_fade_out=60,
                 ball_behind_the_robot_x_value=-1,
                 final_ball_position_min_max=None,
                 fade_out_mode=FADE_OUT_MODE_B,
                 radius=0.12,
                 use_angular_velocity=True,
                 reload_as_dynamic_object_on_hit=False,
                 *vargs,
                 **kwargs):

        self._initial_speed_vector = initial_speed_vector
        self._robot_id = robot_id
        self._direction = initial_speed_vector / np.linalg.norm(initial_speed_vector)
        # adjust the initial orientation of the ball
        rotation_matrix = get_rotation_matrix_between_two_vectors(source_vector=np.array([0, 0, 1.0]),
                                                                  destination_vector=self._direction)
        orn = get_quaternion_from_rotation_matrix(rotation_matrix.T)
        self._initial_orn_euler = np.array(p.getEulerFromQuaternion(orn))
        self._invisible = False
        self._deleted = False

        self._final_ball_position_min_max = final_ball_position_min_max
        self._ball_behind_the_robot_x_value = ball_behind_the_robot_x_value
        self._moving_object_mode = True
        self._ball_was_within_cube = False

        urdf_file_name = "basketball_red"
        urdf_scaling_factor = radius / 0.12  # the normal radius of a basketball is 0.12 m

        super().__init__(urdf_file_name=urdf_file_name,
                         orn=orn,
                         urdf_scaling_factor=urdf_scaling_factor,
                         is_static=True,
                         color=[1, 1, 1, 1],
                         *vargs, **kwargs)

        self._t = 0
        self._gravity_vector = np.array([0, 0, -9.81])

        if use_angular_velocity:  # euler angles per second (in radians)
            self._angular_velocity_vector = np.array([0, np.random.uniform(low=0, high=2*np.pi), 0])
        else:
            self._angular_velocity_vector = None

        self._update_time_step = update_time_step
        self._update_steps_per_action = update_steps_per_action
        self._position_set = np.array(self._base_pos)
        self._update_steps_per_pos_update = update_steps_per_pos_update  # to reduce the computation overhead
        self._update_steps_per_color_update = update_steps_per_color_update
        self._update_steps_for_fade_out = update_steps_for_fade_out
        if fade_out_mode == self.FADE_OUT_MODE_A:
            self._fade_out_color = np.asarray([1, 1, 1, 0])
        elif fade_out_mode == self.FADE_OUT_MODE_B:
            self._fade_out_color = np.copy(self._color)
            self._fade_out_color[3] = 0
        else:
            self._fade_out_color = np.copy(self._color)
            self._fade_out_color[3] = 1

        self._position_update_step_counter = 0
        self._color_update_step_counter = 0
        self._final_state = None
        self._deflection_direction = None

        self._final_ball_position = None
        self._final_ball_time = None

        self._obstacle_hit_data = None

        self._velocity_set = None

        self._reload_as_dynamic_object_on_hit = reload_as_dynamic_object_on_hit

        if final_ball_position_min_max is None:
            # use ball_behind_the_robot_x_value
            self._final_ball_time = (self._ball_behind_the_robot_x_value - self._base_pos[0]) / \
                                    self._initial_speed_vector[0]
            self._final_ball_position = self._base_pos + self._initial_speed_vector * self._final_ball_time
        else:

            self._final_ball_position, self._final_ball_time = \
                self.get_final_ball_position(self._base_pos, self._initial_speed_vector,
                                             self._final_ball_position_min_max, raise_error=True)

        self._max_time_update_step_counter = int(np.floor(self._final_ball_time / self._update_time_step))

    def update(self, visual_mode=False):
        self._position_update_step_counter = self._position_update_step_counter + 1
        self._t = self._t + self._update_time_step

        if not self._invisible and not self._deleted:
            if self._is_static:
                self._position_set = self._base_pos + self._initial_speed_vector * self._t \
                    + 0.5 * self._gravity_vector * self._t ** 2

                if self._angular_velocity_vector is not None:
                    euler_set = self._initial_orn_euler + self._t * self._angular_velocity_vector
                    self._orn_set = p.getQuaternionFromEuler(euler_set)
                else:
                    self._orn_set = self._base_orn

                if self._position_update_step_counter % self._update_steps_per_pos_update == 0:
                    if self._simulation_client_id is not None:
                        p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self._position_set,
                                                          ornObj=self._orn_set,
                                                          physicsClientId=self._simulation_client_id)

                if self._position_update_step_counter % self._update_steps_per_pos_update == 0:
                    if self._update_position_in_obstacle_client:
                        self.update_position_in_obstacle_client()

        if self._moving_object_mode and self._is_static and visual_mode \
                and self._reload_as_dynamic_object_on_hit:
            if self._obstacle_hit_data is not None and self.check_if_object_hit_obstacle():
                self._reload_as_dynamic_object()

    def get_position(self, actual=False, pos_rob=None, forecast_next_step=False):
        if not actual:
            if not forecast_next_step:
                return self._position_set
            else:
                t = self._t + (self._update_steps_per_action * self._update_time_step)
                return self._base_pos + self._initial_speed_vector * t \
                    + 0.5 * self._gravity_vector * t ** 2
        else:
            if actual:
                raise NotImplementedError
            return super().get_position(actual=True)

    def get_current_velocity_vector(self, forecast_next_step=False):
        if forecast_next_step:
            delta_t = self._update_steps_per_action * self._update_time_step
        else:
            delta_t = 0

        return self._initial_speed_vector + self._gravity_vector * (self._t + delta_t)

    def _reload_as_dynamic_object(self):
        # reload as object with a collision shape and without staticBase=True

        if self._deleted:
            raise

        if self._simulation_client_id != self._backup_client_id and self._is_static:
            velocity_set = self.get_current_velocity_vector()
            last_temp_id = None
            for i in range(self._num_clients):
                urdf_file_name = self._urdf_file_name if \
                    (i == self._simulation_client_id or i == self._backup_client_id) \
                    else self._urdf_file_name_no_collision
                flags = 0
                if USE_CACHED_GRAPHICS_SHAPES:
                    flags = flags | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                temp_id = p.loadURDF(os.path.join(self._urdf_dir, urdf_file_name + ".urdf"),
                                     basePosition=self._position_set,
                                     baseOrientation=self._orn_set, globalScaling=self._urdf_scaling_factor,
                                     flags=flags,
                                     physicsClientId=i)
                p.changeVisualShape(temp_id, -1, rgbaColor=self._color, physicsClientId=i)
                p.resetBaseVelocity(temp_id, linearVelocity=velocity_set, physicsClientId=i)
                if last_temp_id is not None and last_temp_id != temp_id:
                    raise
                last_temp_id = temp_id
            # delete previous object
            self.delete()
            self._deleted = False
            self._is_static = False
            self.id = temp_id

    def check_if_object_is_colliding(self):

        contact_points = p.getContactPoints(bodyA=self.id, bodyB=self._robot_id,
                                            physicsClientId=self._simulation_client_id)

        if contact_points:
            return True
        else:
            return False

    def check_if_object_is_behind_the_robot(self):
        return self._position_update_step_counter > self._max_time_update_step_counter

    def _check_if_object_is_within_cube(self, cube_min_max):
        if cube_min_max[0][0] <= self._position_set[0] <= cube_min_max[1][0] \
                and cube_min_max[0][1] <= self._position_set[1] <= cube_min_max[1][1] \
                and cube_min_max[0][2] <= self._position_set[2] <= cube_min_max[1][2]:
            return True
        else:
            return False

    def check_if_object_hit_obstacle(self):
        if self._obstacle_hit_data is not None:
            if self._position_update_step_counter >= self._obstacle_hit_data["obstacle_hit_time"]:
                return True
        return False

    def set_final_state(self, missed_robot=False, hit_robot=False, hit_obstacle=False):
        self._final_state = None
        if missed_robot:
            self._final_state = self.FINAL_STATE_MISSED_ROBOT
        if hit_robot:
            self._final_state = self.FINAL_STATE_HIT_ROBOT
        if hit_obstacle:
            self._final_state = self.FINAL_STATE_HIT_OBSTACLE

        if self._simulation_client_id != self._backup_client_id:
            if (hit_robot or hit_obstacle) and \
                    (self._moving_object_mode and self._visual_mode and self._reload_as_dynamic_object_on_hit):
                # this call might be redundant but the reload is skipped if _is_static is False
                self._reload_as_dynamic_object()

            if not self._visual_mode:
                if self._create_collision_shape_in_simulation_client or self._create_collision_shape_in_obstacle_client:
                    self.delete()
                else:
                    self.make_invisible()

    def delete(self):
        if self._simulation_client_id == self._backup_client_id:
            raise

        if not self._deleted:
            super().delete()
        self._deleted = True

    def make_invisible(self):
        self._invisible = True
        super().make_invisible()

    def check_final_state(self, missed_robot=False, hit_robot=False, hit_obstacle=False):
        if self._final_state is not None:
            if missed_robot and self._final_state == self.FINAL_STATE_MISSED_ROBOT:
                return True
            if hit_robot and self._final_state == self.FINAL_STATE_HIT_ROBOT:
                return True
            if hit_obstacle and self._final_state == self.FINAL_STATE_HIT_OBSTACLE:
                return True
        else:
            return False

    def set_position_update_step_counter(self, position_update_step_counter):
        self._position_update_step_counter = position_update_step_counter
        self._t = self._position_update_step_counter * self._update_time_step

        if not self._invisible and not self._deleted and self._is_static:
            self._position_set = self._base_pos + self._initial_speed_vector * self._t \
                                 + 0.5 * self._gravity_vector * self._t ** 2

            if self._angular_velocity_vector is not None:
                euler_set = self._initial_orn_euler + self._t * self._angular_velocity_vector
                self._orn_set = p.getQuaternionFromEuler(euler_set)
            else:
                self._orn_set = self._base_orn

            if self._simulation_client_id is not None:
                p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self._position_set,
                                                  ornObj=self._orn_set,
                                                  physicsClientId=self._simulation_client_id)

            if self._update_position_in_obstacle_client:
                self.update_position_in_obstacle_client()

    @property
    def radius(self):
        return self._radius

    @property
    def direction(self):
        return self._direction

    @property
    def final_object_position(self):
        return self._final_ball_position

    @property
    def obstacle_hit_data(self):
        return self._obstacle_hit_data

    @obstacle_hit_data.setter
    def obstacle_hit_data(self, val):
        self._obstacle_hit_data = val

    @property
    def max_time_update_step_counter(self):
        return self._max_time_update_step_counter

    @staticmethod
    def get_final_ball_position(base_pos, initial_speed_vector, final_ball_position_min_max, raise_error=False):
        # compute the second intersection between the ball and the xz and yz plane

        # additionally check if the time of the second intersection is shorter than the time required to reach the
        # lower xy plane (minimum z height)

        final_ball_position = None
        final_ball_time = None

        for i in range(2):
            a = (i + 1) % 3
            b = (i + 2) % 3
            if initial_speed_vector[i] != 0:
                for j in range(2):
                    final_ball_time_temp = (final_ball_position_min_max[j][i] - base_pos[i]) / initial_speed_vector[i]
                    final_ball_position_temp = base_pos + initial_speed_vector * final_ball_time_temp
                    if final_ball_position_min_max[0][a] <= final_ball_position_temp[a] \
                            <= final_ball_position_min_max[1][a] \
                            and final_ball_position_min_max[0][b] <= \
                            final_ball_position_temp[b] <= final_ball_position_min_max[1][b]:
                        if final_ball_time is None or final_ball_time_temp > final_ball_time:
                            final_ball_time = final_ball_time_temp
                            final_ball_position = final_ball_position_temp

        # compute intersection with the lower xy plane
        min_height_time, _ = Ball.get_target_height_time(initial_height=base_pos[2],
                                                         initial_z_speed=initial_speed_vector[2],
                                                         target_height=final_ball_position_min_max[0][2],
                                                         no_update_steps=True)

        if not np.isnan(min_height_time):
            if final_ball_time is None or min_height_time < final_ball_time:
                final_ball_time = min_height_time
                final_ball_position = base_pos + initial_speed_vector * final_ball_time

        if final_ball_position is None and raise_error:
            raise ValueError("Could not found final ball position for base pos {}, initial_speed_vector {}, "
                             "final_ball_position_min_max {}".format(base_pos, initial_speed_vector,
                                                                     final_ball_position_min_max))

        return final_ball_position, final_ball_time

    @staticmethod
    def get_target_height_time(initial_height, initial_z_speed, target_height, update_time_step=None,
                               no_update_steps=False):
        g = 9.81
        hit_time = np.nan
        hit_time_update_steps = np.nan
        sqrt_value = initial_z_speed ** 2 + 2 * g * (initial_height - target_height)
        if sqrt_value >= 0:
            hit_time = (initial_z_speed + np.sqrt(sqrt_value)) / g
            if hit_time > 0:
                if not no_update_steps:
                    hit_time_update_steps = round(hit_time / update_time_step)
            else:
                hit_time = np.nan

        return hit_time, hit_time_update_steps


class Planet(ObstacleSim):
    def __init__(self,
                 center,
                 radius_xy,
                 euler_angles,
                 period,
                 update_time_step,
                 time_shift=None,
                 default_orn=(0, 0, 0, 1),
                 initial_time=0,  # -1: random time
                 urdf_file_name="ISS",
                 orbit_color=None,
                 update_steps_per_pos_update=1,
                 orbit_debug_line_buffer_size=500,
                 orbit_interpolation_steps=1000,
                 rotations_per_period=0,  # has to be a non-negative integer
                 *vargs,
                 **kwargs):

        self._center = np.asarray(center)
        self._radius_xy = radius_xy
        self._orn_shift = p.getQuaternionFromEuler(euler_angles)
        self._period = period
        self._update_time_step = update_time_step
        self._time_shift = time_shift
        self._default_orn = default_orn
        self._orbit_color = orbit_color
        self._update_steps_per_pos_update = update_steps_per_pos_update
        self._rotations_per_period = rotations_per_period
        self._initial_time = initial_time

        self._orbit_debug_line_buffer = None

        if self._period < 0:  # inverse direction if period is negative
            self._period = self._period * (-1)
            direction_sign = -1
        else:
            direction_sign = 1

        orbit_theta_interpolation = np.linspace(0, direction_sign * 2 * np.pi, orbit_interpolation_steps)
        orbit_local_pos_interpolation = np.array([self._radius_xy[0] * np.cos(orbit_theta_interpolation),
                                                  self._radius_xy[1] * np.sin(orbit_theta_interpolation),
                                                  np.zeros_like(orbit_theta_interpolation)]).T
        orbit_global_pos_interpolation, _ = self._local_to_global(local_pos=orbit_local_pos_interpolation)
        orbit_local_pos_interpolation_diff = np.diff(orbit_local_pos_interpolation, axis=0)
        orbit_local_pos_interpolation_diff_norm = np.linalg.norm(orbit_local_pos_interpolation_diff, axis=1)
        orbit_length_cum_sum = np.concatenate(([0.0], np.cumsum(orbit_local_pos_interpolation_diff_norm)))
        self._orbit_total_length = orbit_length_cum_sum[-1]
        orbit_update_time_steps = np.arange(0, self._period, self._update_time_step)
        orbit_update_time_steps_length = orbit_update_time_steps / self._period * self._orbit_total_length
        orbit_update_time_steps_indices = np.searchsorted(orbit_length_cum_sum, orbit_update_time_steps_length)
        self._orbit_global_time_steps_pos = orbit_global_pos_interpolation[orbit_update_time_steps_indices]
        self._orbit_local_time_steps_pos = orbit_local_pos_interpolation[orbit_update_time_steps_indices]

        self._orbit_global_time_steps_orn = None
        if self._rotations_per_period:
            orbit_update_time_steps_local_z_rotation = \
                direction_sign * orbit_update_time_steps / self._period * 2 * np.pi * self._rotations_per_period
            orbit_update_time_steps_local_euler_angles = \
                np.array([np.zeros_like(orbit_update_time_steps_local_z_rotation),
                          np.zeros_like(orbit_update_time_steps_local_z_rotation),
                          orbit_update_time_steps_local_z_rotation]).T
            self._orbit_global_time_steps_orn = []
            for i in range(len(orbit_update_time_steps_local_euler_angles)):
                orn = p.getQuaternionFromEuler(orbit_update_time_steps_local_euler_angles[i])
                if default_orn != (0, 0, 0, 1):
                    orn = self._multiply_quaternion(default_orn, orn)
                if self._orn_shift != (0, 0, 0, 1):
                    orn = self._multiply_quaternion(self._orn_shift, orn)
                self._orbit_global_time_steps_orn.append(orn)

        if self._time_shift is not None:
            # the movement of the planet is coupled to the movement of a different planet with the same period
            self._time_step_index_shift = int(self.num_time_steps * self._time_shift / self._period)
        else:
            self._time_step_index_shift = None

        self.id = None
        self._position_set = None
        self._orn_set = None
        self._position_update_step_counter = None
        self._current_time_step_index = None
        self._coupled_planet = None

        if urdf_file_name == "asteroid":
            color = [0.5, 0.5, 0.5, 1]
        else:
            color = [1, 1, 1, 1]

        super().__init__(urdf_file_name=urdf_file_name,
                         pos=self._orbit_global_time_steps_pos[0],
                         orn=self._default_orn,
                         is_static=True,
                         color=color,
                         *vargs, **kwargs)

        if self._orbit_color is not None:
            orbit_theta_debug_line = np.arange(0, 2 * np.pi, 2 * np.pi / orbit_debug_line_buffer_size)
            orbit_local_debug_line = np.array([self._radius_xy[0] * np.cos(orbit_theta_debug_line),
                                               self._radius_xy[1] * np.sin(orbit_theta_debug_line),
                                               np.zeros_like(orbit_theta_debug_line)]).T
            orbit_global_pos_debug_line, _ = self._local_to_global(local_pos=orbit_local_debug_line)
            self._visualize_orbit(orbit_global_pos_debug_line)

        self._collision_detected = False

    def reset(self):
        self._position_update_step_counter = 0

        if self._time_step_index_shift is None:
            if self._initial_time == -1:
                # random but collision free initial time
                collision_detected = True
                while collision_detected:
                    self._collision_detected = False
                    self._current_time_step_index = np.random.randint(0, self.num_time_steps)
                    self.set_initial_position_and_orientation()
                    if self.coupled_planet is not None:
                        self.coupled_planet.current_time_step_index = \
                            (self._current_time_step_index + self.coupled_planet.time_step_index_shift) \
                            % self.coupled_planet.num_time_steps
                        self.coupled_planet.set_initial_position_and_orientation()
                    p.performCollisionDetection()
                    collision_detected = self.check_if_object_is_colliding()
                    if not collision_detected and self.coupled_planet is not None:
                        collision_detected = self.coupled_planet.check_if_object_is_colliding()
                        self.coupled_planet.collision_detected = False

            else:
                self._current_time_step_index = int(self._initial_time / self._update_time_step)
                self.set_initial_position_and_orientation()
                if self.coupled_planet is not None:
                    self.coupled_planet.current_time_step_index = \
                        (self._current_time_step_index + self.coupled_planet.time_step_index_shift) \
                        % self.coupled_planet.num_time_steps
                    self.coupled_planet.set_initial_position_and_orientation()

        self._collision_detected = False

    def update(self, visual_mode=False):
        self._current_time_step_index = (self._current_time_step_index + 1) % len(self._orbit_global_time_steps_pos)
        self._position_update_step_counter = self._position_update_step_counter + 1
        self._position_set = self._orbit_global_time_steps_pos[self._current_time_step_index]
        if self._rotations_per_period:
            self._orn_set = self._orbit_global_time_steps_orn[self._current_time_step_index]
        else:
            self._orn_set = self._default_orn

        if self._position_update_step_counter % self._update_steps_per_pos_update == 0:
            self._set_position_and_orientation()

    def set_initial_position_and_orientation(self):
        self._position_set = self._orbit_global_time_steps_pos[self._current_time_step_index]

        if self._rotations_per_period:
            self._orn_set = self._orbit_global_time_steps_orn[self._current_time_step_index]
        else:
            self._orn_set = self._default_orn

        if self.id is not None:
            # reset position and orn
            self._set_position_and_orientation()

    def _set_position_and_orientation(self):
        if self._simulation_client_id is not None:
            p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=self._position_set,
                                              ornObj=self._orn_set,
                                              physicsClientId=self._simulation_client_id)

        if self._update_position_in_obstacle_client:
            self.update_position_in_obstacle_client()

    @staticmethod
    def _multiply_quaternion(a, b):  # xyzw
        return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]]

    def _local_to_global(self, local_pos, local_orn=None):
        local_pos = np.asarray(local_pos)
        default_orn = (0, 0, 0, 1)
        if local_pos.ndim == 2:
            global_pos = np.zeros_like(local_pos)
            global_orn = None if local_orn is None else np.zeros_like(local_orn)
            for i in range(len(local_pos)):
                orientation_b = local_orn[i] if local_orn is not None else default_orn
                pos, orn = p.multiplyTransforms(positionA=self._center,
                                                orientationA=self._orn_shift,
                                                positionB=local_pos[i],
                                                orientationB=orientation_b,
                                                )
                global_pos[i] = np.array(pos)
                if local_orn is not None:
                    global_orn[i] = np.array(orn)
        else:
            orientation_b = local_orn if local_orn is not None else default_orn
            pos, orn = p.multiplyTransforms(positionA=self._center,
                                            orientationA=self._orn_shift,
                                            positionB=local_pos,
                                            orientationB=orientation_b)
            global_pos = np.array(pos)
            global_orn = np.array(orn) if local_orn is not None else None

        return global_pos, global_orn

    def check_if_object_is_colliding(self):
        if not self._collision_detected:
            contact_points = p.getContactPoints(bodyA=self.id, physicsClientId=self._simulation_client_id)
            if contact_points:
                self._collision_detected = True
                return True
            else:
                return False
        else:
            return True

    def _visualize_orbit(self, orbit_global):
        if self._visual_mode and self._gui_client_id is not None:
            if self._orbit_debug_line_buffer is None:
                # initialize debug line buffer
                valid_debug_lines = 0
                self._orbit_debug_line_buffer = []
                while valid_debug_lines < len(orbit_global) + 1:
                    debug_line_id = p.addUserDebugLine([0, 0, 0],
                                                       [0, 0, 0],
                                                       physicsClientId=self._gui_client_id)
                    if debug_line_id != -1:
                        valid_debug_lines = valid_debug_lines + 1
                        self._orbit_debug_line_buffer.append(debug_line_id)

            for i in range(len(orbit_global)):
                self._orbit_debug_line_buffer[i] = \
                    p.addUserDebugLine(orbit_global[i-1], orbit_global[i],
                                       lineColorRGB=self._orbit_color, lineWidth=1.5,
                                       replaceItemUniqueId=self._orbit_debug_line_buffer[i],
                                       physicsClientId=self._gui_client_id)

    @property
    def collision_detected(self):
        return self._collision_detected

    @collision_detected.setter
    def collision_detected(self, val):
        self._collision_detected = val

    @property
    def time_step_index_shift(self):
        return self._time_step_index_shift

    @property
    def coupled_planet(self):
        return self._coupled_planet

    @coupled_planet.setter
    def coupled_planet(self, val):
        self._coupled_planet = val

    @property
    def current_time_step_index(self):
        return self._current_time_step_index

    @current_time_step_index.setter
    def current_time_step_index(self, val):
        self._current_time_step_index = val

    @property
    def num_time_steps(self):
        return len(self._orbit_global_time_steps_pos)

    @property
    def orbit_local_time_steps_pos(self):
        return self._orbit_local_time_steps_pos

    @property
    def orbit_global_time_steps_pos(self):
        return self._orbit_global_time_steps_pos

    @property
    def radius_xy(self):
        return self._radius_xy


class Human(ObstacleSim):
    def __init__(self,
                 human_network_checkpoint,
                 robot_id,
                 use_collision_avoidance_starting_point_sampling=False,
                 risk_state_deterministic_backup_trajectory=False,
                 collision_avoidance_kinematic_state_sampling_probability=1.0,
                 collision_avoidance_stay_in_state_probability=0.3,
                 trajectory_duration=None,
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 backup_client_id=None,
                 use_gui=False,
                 no_link_coloring=False,
                 plot_trajectory=False,
                 use_fixed_seed=None,
                 *vargs,
                 **kwargs):

        self._human_network_checkpoint = human_network_checkpoint
        self._robot_id = robot_id
        self._use_collision_avoidance_starting_point_sampling = use_collision_avoidance_starting_point_sampling
        self._risk_state_deterministic_backup_trajectory = risk_state_deterministic_backup_trajectory
        self._collision_avoidance_kinematic_state_sampling_probability = \
            collision_avoidance_kinematic_state_sampling_probability
        self._collision_avoidance_stay_in_state_probability = collision_avoidance_stay_in_state_probability
        self._trajectory_duration = trajectory_duration
        self._use_gui = use_gui
        self._no_link_coloring = no_link_coloring
        self._plot_trajectory = plot_trajectory
        self._use_fixed_seed = use_fixed_seed

        # load env with human as robot
        if not os.path.isfile(self._human_network_checkpoint):
            self._human_network_checkpoint = os.path.join(current_dir, "trained_networks",
                                                          self._human_network_checkpoint)
            if not os.path.isfile(self._human_network_checkpoint):
                raise ValueError("Could not find human_network_checkpoint {}".format(self._human_network_checkpoint))

        import ray
        from ray import tune
        from ray.rllib import rollout
        from ray.rllib.models import ModelCatalog
        from ray.rllib.agents.callbacks import DefaultCallbacks
        from safemotions.model.keras_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation

        params_dir = os.path.dirname(os.path.dirname(self._human_network_checkpoint))
        params_path = os.path.join(params_dir, "params.json")

        with open(params_path) as params_file:
            checkpoint_config = json.load(params_file)
        checkpoint_config['evaluation_interval'] = None
        checkpoint_config['callbacks'] = DefaultCallbacks
        checkpoint_config['num_workers'] = 0
        checkpoint_config['explore'] = True  # the action generation is allowed to be stochastic

        env_config = checkpoint_config['env_config']

        physic_clients_dict = {'main_client_id': simulation_client_id,
                               'obstacle_client_id': obstacle_client_id,
                               'backup_client_id': backup_client_id}

        env_config['physic_clients_dict'] = physic_clients_dict
        env_config['do_not_execute_robot_movement'] = True
        env_config['risk_state_deterministic_backup_trajectory'] = risk_state_deterministic_backup_trajectory
        env_config['logging_level'] = None
        env_config['use_gui'] = self._use_gui
        env_config['no_link_coloring'] = self._no_link_coloring
        env_config['plot_trajectory'] = self._plot_trajectory

        # reduce computational effort by avoiding unnecessary reward calculations
        env_config['punish_end_min_distance'] = False
        env_config['collision_avoidance_self_collision_max_reward'] = 0.0
        env_config['collision_avoidance_static_obstacles_max_reward'] = 0.0
        env_config['collision_avoidance_moving_obstacles_max_reward'] = 0.0

        if self._use_collision_avoidance_starting_point_sampling:
            env_config['always_use_collision_avoidance_starting_point_sampling'] = True
            env_config['collision_avoidance_kinematic_state_sampling_mode'] = True
            env_config['collision_avoidance_kinematic_state_sampling_probability'] = \
                self._collision_avoidance_kinematic_state_sampling_probability
            env_config['collision_avoidance_stay_in_state_probability'] = \
                self._collision_avoidance_stay_in_state_probability

        if "seed" in checkpoint_config:
            checkpoint_config["seed"] = None

        if self._use_fixed_seed:
            checkpoint_config["seed"] = np.random.randint(low=0, high=np.iinfo(np.int32).max)
            # seeds numpy, random and the tf session
            # (required for a deterministic action generation if checkpoint_config['explore'] == True)
            # the selected seed is sampled deterministically if the base environment is seeded

        if "seed" in env_config:
            env_config["seed"] = None

        if self._trajectory_duration is not None:
            # prevent termination of the human environment
            env_config["trajectory_duration"] = self._trajectory_duration + \
                                                       env_config["trajectory_time_step"]

        ModelCatalog.register_custom_model('keras_fcnet_last_layer_activation',
                                           FullyConnectedNetworkLastLayerActivation)

        import safemotions.envs.safe_motions_env as safe_motions_env
        Env = getattr(safe_motions_env, checkpoint_config['env'])

        tune.register_env(Env.__name__,
                          lambda config_args: Env(**config_args))

        ray.init(dashboard_host="0.0.0.0", include_dashboard=False, ignore_reinit_error=True)

        cls = rollout.get_trainable_cls("PPO")

        self._agent = cls(env=Env.__name__, config=checkpoint_config)
        self._agent.restore(self._human_network_checkpoint)
        self._env = self._agent.workers.local_worker().env

        super().__init__(simulation_client_id=simulation_client_id,
                         obstacle_client_id=obstacle_client_id,
                         backup_client_id=backup_client_id,
                         pos=None,
                         id_to_reuse=self._env.robot_scene.robot_id,
                         *vargs, **kwargs)

        obstacle_link_names = self._env.robot_scene.get_link_names_for_multiple_robots(["upper_arm", "forearm", "hand"])
        self._obstacle_links, _ = \
            self._env.robot_scene.obstacle_wrapper.get_link_indices_in_link_name_list(obstacle_link_names)

        self._last_observation = None
        self._control_step_counter = None
        self._obstacle_update_step_counter = None
        self._collision_detected = False

        self._end_acceleration = None
        self._controller_setpoints = None
        self._obstacle_client_update_setpoints = None
        self._movement_info = None
        self._action_info = None
        self._robot_stopped = None

        self._stored_env_variables = None
        self._stored_human_variables = None
        self._deterministic_action_list = None
        self._deterministic_action_list_index = None
        self._step_lock = None

        self._do_not_copy_keys = ['_env', '_agent', '_stored_env_variables', '_stored_human_variables']

    def reset(self):
        self._last_observation = self._env.reset()
        self._control_step_counter = None
        self._obstacle_update_step_counter = None
        self._collision_detected = False
        # set the human to the setpoint position in the obstacle client
        self.set_position_in_obstacle_client_to_setpoints()

        self._end_acceleration = None
        self._controller_setpoints = None
        self._obstacle_client_update_setpoints = None
        self._movement_info = None
        self._action_info = None
        self._robot_stopped = None

        self._stored_env_variables = None
        self._stored_human_variables = None
        self._deterministic_action_list = []
        self._deterministic_action_list_index = None

        self._step_lock = False

    def step(self):
        if self._simulation_client_id != self._backup_client_id and self._step_lock:
            return None  # step lock is used for the forecasting of the kinematic observation
        if self._simulation_client_id != self._backup_client_id and len(self._deterministic_action_list) > 0:
            action = self._deterministic_action_list.pop(0)
        elif self._simulation_client_id == self._backup_client_id and self._risk_state_deterministic_backup_trajectory \
                and self._deterministic_action_list_index < len(self._deterministic_action_list):
            action = self._deterministic_action_list[self._deterministic_action_list_index]
        else:
            action = np.array(self._agent.compute_action(self._last_observation, full_fetch=False), dtype=np.float64)

        if self._simulation_client_id == self._backup_client_id and \
                self._risk_state_deterministic_backup_trajectory:
            if self._deterministic_action_list_index >= len(self._deterministic_action_list):
                self._deterministic_action_list.append(action)
            self._deterministic_action_list_index = self._deterministic_action_list_index + 1

        self._end_acceleration, self._controller_setpoints, self._obstacle_client_update_setpoints, \
            self._movement_info, self._action_info, self._robot_stopped = self._env.step(action=action)

        self._control_step_counter = 0
        self._obstacle_update_step_counter = 0
        self._step_lock = True

    def process_step_outcome(self):
        self._last_observation, reward, done, info = self._env.process_step_outcome(
            end_acceleration=self._end_acceleration,
            obstacle_client_update_setpoints=self._obstacle_client_update_setpoints,
            robot_stopped=self._robot_stopped,
            movement_info=self._movement_info,
            action_info=self._action_info)
        self._step_lock = False

    def prepare_sim_step(self):
        if self._control_step_counter < len(self._controller_setpoints['positions']):
            self._env.robot_scene.set_motor_control(
                self._controller_setpoints['positions'][self._control_step_counter],
                target_velocities=self._controller_setpoints['velocities'][self._control_step_counter],
                target_accelerations=self._controller_setpoints['accelerations'][self._control_step_counter])

            self._control_step_counter = self._control_step_counter + 1
        else:
            raise ValueError("Call to prepare_sim_step without valid controller setpoint")

    def update(self):
        # update position in obstacle client
        if self._obstacle_update_step_counter < len(self._obstacle_client_update_setpoints['positions']):
            actual_position, actual_velocity = self._env.robot_scene.get_actual_joint_position_and_velocity(
                physics_client_id=self._simulation_client_id)
            self._env.robot_scene.obstacle_wrapper.update(
                target_position=
                self._obstacle_client_update_setpoints['positions'][self._obstacle_update_step_counter],
                target_velocity=
                self._obstacle_client_update_setpoints['velocities'][self._obstacle_update_step_counter],
                target_acceleration=
                self._obstacle_client_update_setpoints['accelerations'][self._obstacle_update_step_counter],
                actual_position=actual_position,
                actual_velocity=actual_velocity)

            self._obstacle_update_step_counter = self._obstacle_update_step_counter + 1

            # set the human to the setpoint position in the obstacle client
            self.set_position_in_obstacle_client_to_setpoints()
        else:
            raise ValueError("Call update without valid obstacle_client_update_setpoints")

    def process_end_of_episode(self):
        if self._plot_trajectory:
            self._env.display_plot()

    def check_if_object_is_colliding(self):
        if not self._collision_detected:
            contact_points = p.getContactPoints(bodyA=self.id, bodyB=self._robot_id,
                                                physicsClientId=self._simulation_client_id)
            if contact_points:
                self._collision_detected = True
                return True
            else:
                return False
        else:
            return True

    def switch_to_backup_client(self):
        self._stored_env_variables = self._env.switch_to_backup_client(nested_env=True)
        self._stored_human_variables = self._env.copy_variables_to_dict(object=self)
        self._simulation_client_id = self._backup_client_id
        self._deterministic_action_list_index = 0
        self._env.robot_scene.no_link_coloring = True

    def switch_back_to_main_client(self):
        self._env.switch_back_to_main_client(stored_variables=self._stored_env_variables)
        self._env.switch_back_to_main_client(stored_variables=self._stored_human_variables,
                                             object=self)

    def copy_deterministic_lists_to_main_client(self):
        self._stored_env_variables["_robot_scene._obstacle_wrapper._target_point_deterministic_list"] = \
            self._env.robot_scene.obstacle_wrapper.target_point_deterministic_list
        self._stored_human_variables["_deterministic_action_list"] = self._deterministic_action_list

    @property
    def collision_detected(self):
        return self._collision_detected

    @property
    def observation(self):
        return self._last_observation

    @property
    def observation_size(self):
        return self._env.observation_size

    @property
    def kinematic_observation(self):
        return self._env.kinematic_observation

    @property
    def kinematic_observation_forecast(self):
        if not self._step_lock:
            raise ValueError

        prev_joint_accelerations_rel, curr_joint_position_rel_obs, curr_joint_velocity_rel_obs, \
            curr_joint_acceleration_rel_obs = self._env.get_kinematic_observation_components(
                prev_joint_accelerations=[],
                curr_joint_position=self._obstacle_client_update_setpoints['positions'][-1],
                curr_joint_velocity=self._obstacle_client_update_setpoints['velocities'][-1],
                curr_joint_acceleration=self._end_acceleration)

        kinematic_observation_forecast_not_clipped = \
            np.array([item for sublist in prev_joint_accelerations_rel for item in sublist]
                     + curr_joint_position_rel_obs + curr_joint_velocity_rel_obs
                     + curr_joint_acceleration_rel_obs, dtype=np.float32)

        kinematic_observation_forecast = np.core.umath.clip(kinematic_observation_forecast_not_clipped, -1, 1)

        return kinematic_observation_forecast

    @property
    def kinematic_observation_size(self):
        return self._env.kinematic_observation_size

    def set_position_in_obstacle_client_to_setpoints(self):
        self._env.robot_scene.obstacle_wrapper.set_robot_position_in_obstacle_client(set_to_setpoints=True)


class LinkPointBase(object):
    def __init__(self,
                 name="tbd",
                 offset=(0, 0, 0),
                 bounding_sphere_radius=0.0,
                 active=False,
                 safety_distance=0.00,
                 visualize_bounding_sphere=False,
                 default_bounding_sphere_color=(0, 1, 0, 0.5),
                 num_clients=1,
                 *vargs,
                 **kwargs):
        self._name = name
        self._offset = offset
        self._bounding_sphere_radius = bounding_sphere_radius
        self._default_bounding_sphere_color = default_bounding_sphere_color
        self._bounding_sphere_color = self._default_bounding_sphere_color
        self._active = active
        self._link_object = None
        self._visualize_bounding_sphere = visualize_bounding_sphere
        self._safetyDistance = safety_distance
        self._num_clients = num_clients
        self._debug_line = None

        if self._visualize_bounding_sphere and self._bounding_sphere_radius > 0:
            for i in range(self._num_clients):
                shape_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self._bounding_sphere_radius,
                                                   rgbaColor=self._default_bounding_sphere_color, physicsClientId=i)

                self._bounding_sphere_id = p.createMultiBody(baseMass=0,
                                                             basePosition=[0, 0, 0],
                                                             baseOrientation=[0, 0, 0, 1],
                                                             baseVisualShapeIndex=shape_visual,
                                                             physicsClientId=i)
        else:
            self._bounding_sphere_id = None

    def update_bounding_sphere_position(self, actual=True):
        if self._bounding_sphere_id is not None:
            pos = self.get_position(actual=actual)
            def_orn = p.getQuaternionFromEuler([0, 0, 0])
            for i in range(self._num_clients):
                p.resetBasePositionAndOrientation(bodyUniqueId=self._bounding_sphere_id, posObj=pos, ornObj=def_orn,
                                                  physicsClientId=i)

    def set_bounding_sphere_color(self, rgba_color=None):
        if self._bounding_sphere_id is not None:
            if rgba_color is None:
                rgba_color = self._default_bounding_sphere_color
            if rgba_color != self._bounding_sphere_color:
                self._bounding_sphere_color = rgba_color
                if self._link_object.simulation_client_id is not None:
                    p.changeVisualShape(self._bounding_sphere_id, -1, rgbaColor=self._bounding_sphere_color,
                                        physicsClientId=self._link_object.simulation_client_id)

    def get_position(self, actual=None, return_orn=False, additional_offset=None):
        if additional_offset is not None:
            offset = self._offset + additional_offset
        else:
            offset = self._offset

        def_orn = p.getQuaternionFromEuler([0, 0, 0])
        pos, orn = p.multiplyTransforms(positionA=self._link_object.get_position(actual=actual),
                                        orientationA=self._link_object.get_orn(actual=actual),
                                        positionB=offset,
                                        orientationB=def_orn)

        if return_orn:
            return pos, orn
        else:
            return pos

    def add_debug_line(self, actual, from_position_local, to_position_local):
        line_color = [1, 0, 0]
        line_width = 2
        from_position_global = self.get_position(actual=actual, additional_offset=from_position_local)
        to_position_global = self.get_position(actual=actual, additional_offset=to_position_local)
        if self._debug_line is not None:
            self._debug_line = p.addUserDebugLine(from_position_global, to_position_global, lineColorRGB=line_color,
                                                  lineWidth=line_width,
                                                  replaceItemUniqueId=self._debug_line,
                                                  physicsClientId=self._link_object.simulation_client_id)
        else:
            self._debug_line = p.addUserDebugLine(from_position_global, to_position_global, lineColorRGB=line_color,
                                                  lineWidth=line_width,
                                                  physicsClientId=self._link_object.simulation_client_id)

    @property
    def name(self):
        return self._name

    @property
    def is_active(self):
        return self._active

    @property
    def safety_distance(self):
        return self._safetyDistance

    @property
    def offset(self):
        return self._offset

    @property
    def link_object(self):
        return self._link_object

    @link_object.setter
    def link_object(self, val):
        self._link_object = val

    @property
    def bounding_sphere_radius(self):
        return self._bounding_sphere_radius


class LinkBase(object):
    def __init__(self,
                 name=None,
                 observe_closest_point=True,
                 closest_point_active=False,
                 closest_point_safety_distance=0.1,
                 observed_points=None,
                 index=None,
                 robot_id=None,
                 robot_index=None,
                 self_collision_links=None,
                 default_color=None,
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 use_real_robot=False,
                 set_robot_position_in_obstacle_client_function=None,
                 *vargs,
                 **kwargs):

        if default_color is None:
            default_color = [0.9, 0.9, 0.9, 1]
        if self_collision_links is None:
            self_collision_links = []
        if observed_points is None:
            observed_points = []

        self._observe_closest_point = observe_closest_point
        self._closest_point_active = closest_point_active
        self._closest_point_safety_distance = closest_point_safety_distance

        self._simulation_client_id = simulation_client_id
        self._obstacle_client_id = obstacle_client_id
        self._use_real_robot = use_real_robot

        if self._closest_point_active:
            self._observe_closest_point = True
        self._name = name
        self._observed_points = observed_points
        for i in range(len(self._observed_points)):
            self._observed_points[i].link_object = self
        self._index = index
        self._robot_id = robot_id
        self._robot_index = robot_index

        self._position_actual = None
        self._position_set = None
        self._orn_set = None
        self._orn_actual = None
        self._position_other = None
        self._orn_other = None
        self._self_collision_links = self_collision_links
        self._self_collision_data = SelfCollisionData(num_self_collision_links=len(self._self_collision_links))

        self._color = None
        self._default_color = default_color

        self._set_robot_position_in_obstacle_client_function = set_robot_position_in_obstacle_client_function
        self.set_color(rgba_color=None, obstacle_client=True)

    def get_local_position(self, world_position, actual=True):
        def_orn = p.getQuaternionFromEuler([0, 0, 0])
        com_pos_inv, com_orn_inv = p.invertTransform(position=self.get_position(actual=actual),
                                                     orientation=self.get_orn(actual=actual))
        pos, _ = p.multiplyTransforms(positionA=com_pos_inv,
                                      orientationA=com_orn_inv,
                                      positionB=world_position,
                                      orientationB=def_orn)

        return pos

    def get_orn(self, actual=True):
        if actual is None:
            if self._orn_other is None:
                self.get_position(actual=actual)
            return self._orn_other
        else:
            if actual:
                if self._orn_actual is None:
                    self.get_position(actual=actual)
                return self._orn_actual
            else:
                if self._orn_set is None:
                    self.get_position(actual=actual)
                return self._orn_set

    def get_position(self, actual=None):
        if actual is None:
            if self._position_other is None:
                link_state = p.getLinkState(bodyUniqueId=self._robot_id, linkIndex=self._index,
                                            computeLinkVelocity=False, computeForwardKinematics=True,
                                            physicsClientId=self._obstacle_client_id)
                self._position_other = link_state[4]
                self._orn_other = link_state[5]

            return self._position_other

        else:
            if actual:
                if self._position_actual is None:
                    self._set_robot_position_in_obstacle_client_function(set_to_actual_values=True)
                    link_state = p.getLinkState(bodyUniqueId=self._robot_id, linkIndex=self._index,
                                                computeLinkVelocity=False,
                                                computeForwardKinematics=True, physicsClientId=self._obstacle_client_id)
                    self._position_actual = link_state[4]
                    self._orn_actual = link_state[5]

                return self._position_actual

            else:
                # set point
                if self._position_set is None:
                    self._set_robot_position_in_obstacle_client_function(set_to_setpoints=True)
                    link_state = p.getLinkState(bodyUniqueId=self._robot_id, linkIndex=self._index,
                                                computeLinkVelocity=False,
                                                computeForwardKinematics=True, physicsClientId=self._obstacle_client_id)
                    self._position_set = link_state[4]
                    self._orn_set = link_state[5]
                return self._position_set

    def set_color(self, rgba_color, obstacle_client=False):
        if rgba_color is None:
            rgba_color = self._default_color
        if rgba_color != self._color:
            self._color = rgba_color
            if self._simulation_client_id is not None:
                p.changeVisualShape(self._robot_id, self._index, -1,
                                    rgbaColor=rgba_color, physicsClientId=self._simulation_client_id)
            if obstacle_client and self._obstacle_client_id is not None:
                p.changeVisualShape(self._robot_id, self._index, -1,
                                    rgbaColor=rgba_color, physicsClientId=self._obstacle_client_id)

    def clear_previous_timestep(self):
        self._position_actual = None
        self._position_set = None
        self._orn_set = None
        self._orn_actual = None
        self._position_other = None
        self._orn_other = None

    def clear_other_position_and_orn(self):
        self._position_other = None
        self._orn_other = None

    def reset(self):
        self.set_color(None)
        self._self_collision_data.reset()

    @property
    def self_collision_links(self):
        return self._self_collision_links

    @property
    def default_color(self):
        return self._default_color

    @property
    def self_collision_data(self):
        return self._self_collision_data

    @property
    def robot_index(self):
        return self._robot_index

    @property
    def observe_closest_point(self):
        return self._observe_closest_point

    @property
    def closest_point_active(self):
        return self._closest_point_active

    @property
    def closest_point_safety_distance(self):
        return self._closest_point_safety_distance

    @property
    def observed_points(self):
        return self._observed_points

    @property
    def num_observed_points(self):
        return len(self._observed_points)

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._index

    @property
    def simulation_client_id(self):
        return self._simulation_client_id

    def __deepcopy__(self, memo):
        # see
        # https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object
        # exclude _set_robot_position_in_obstacle_client_function from deepcopy
        __deepcopy__function = self.__deepcopy__
        self.__deepcopy__ = None
        memo[id(self._set_robot_position_in_obstacle_client_function)] = \
            self._set_robot_position_in_obstacle_client_function
        output = copy.deepcopy(self, memo)
        self.__deepcopy__ = __deepcopy__function
        output.__deepcopy__ = types.MethodType(__deepcopy__function.__func__, output)
        return output


class SelfCollisionData(object):
    def __init__(self,
                 num_self_collision_links):
        self._num_self_collision_links = num_self_collision_links
        self._closest_point_distance_actual = None
        self._closest_point_distance_set = None
        self.reset()

    def reset(self):
        self._closest_point_distance_actual = [[] for _ in range(self._num_self_collision_links)]
        self._closest_point_distance_set = [[] for _ in range(self._num_self_collision_links)]

    def export_metrics(self, export_link_pair=None):
        export_dict = {}
        if export_link_pair is None:
            export_link_pair = [True] * len(self._closest_point_distance_actual)
        export_dict['closest_point_distance_actual_min'] = [np.min(
            self._closest_point_distance_actual[i]) if self._closest_point_distance_actual[i] else None
                                                            for i in range(len(self._closest_point_distance_actual))
                                                            if export_link_pair[i]]
        export_dict['closest_point_distance_set_min'] = [np.min(
            self._closest_point_distance_set[i]) if self._closest_point_distance_set[i] else None for i in
                                                         range(len(self._closest_point_distance_set))
                                                         if export_link_pair[i]]

        return export_dict

    @property
    def closest_point_distance_actual(self):
        return self._closest_point_distance_actual

    @property
    def closest_point_distance_set(self):
        return self._closest_point_distance_set


class LinkData(object):
    def __init__(self,
                 num_observed_points,
                 *vargs,
                 **kwargs):
        self._num_observed_points = num_observed_points
        self._closest_point_distance_actual = None
        self._closest_point_distance_set = None
        self._closest_point_velocity_set = None
        self._observed_point_distance_actual = None
        self._observed_point_distance_set = None
        self._observed_point_velocity_set = None
        self.reset()

    def reset(self):
        self._closest_point_distance_actual = []
        self._closest_point_distance_set = []
        self._closest_point_velocity_set = []
        self._observed_point_distance_actual = [[] for _ in range(self._num_observed_points)]
        self._observed_point_distance_set = [[] for _ in range(self._num_observed_points)]
        self._observed_point_velocity_set = [[] for _ in range(self._num_observed_points)]

    def export_metrics(self):
        export_dict = {}
        export_dict['closest_point_distance_actual_min'] = np.min(
            self._closest_point_distance_actual) if self._closest_point_distance_actual else None
        export_dict['closest_point_distance_set_min'] = np.min(
            self._closest_point_distance_set) if self._closest_point_distance_set else None
        export_dict['observed_point_distance_actual_min'] = [np.min(
            self._observed_point_distance_actual[i]) if self._observed_point_distance_actual[i] else None for i in
                                                             range(len(self._observed_point_distance_actual))]
        export_dict['observed_point_distance_set_min'] = [np.min(
            self._observed_point_distance_set[i]) if self._observed_point_distance_set[i] else None for i in
                                                          range(len(self._observed_point_distance_set))]

        return export_dict

    @property
    def closest_point_distance_actual(self):
        return self._closest_point_distance_actual

    @property
    def closest_point_velocity_set(self):
        return self._closest_point_velocity_set

    @property
    def closest_point_distance_set(self):
        return self._closest_point_distance_set

    @property
    def observed_point_velocity_set(self):
        return self._observed_point_velocity_set

    @property
    def observed_point_distance_actual(self):
        return self._observed_point_distance_actual

    @property
    def observed_point_distance_set(self):
        return self._observed_point_distance_set

    @property
    def num_observed_points(self):
        return len(self._observed_point_distance_actual)


def get_rotation_matrix_between_two_vectors(source_vector, destination_vector):
    a = (source_vector / np.linalg.norm(source_vector)).reshape(3)
    b = (destination_vector / np.linalg.norm(destination_vector)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_quaternion_from_rotation_matrix(rotation_matrix):
    if rotation_matrix[2, 2] < 0:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1]:
            t = 1 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]
            q = [t, rotation_matrix[0, 1] + rotation_matrix[1, 0], rotation_matrix[2, 0] + rotation_matrix[0, 2],
                 rotation_matrix[1, 2] - rotation_matrix[2, 1]]
        else:
            t = 1 - rotation_matrix[0, 0] + rotation_matrix[1, 1] - rotation_matrix[2, 2]
            q = [rotation_matrix[0, 1] + rotation_matrix[1, 0], t, rotation_matrix[1, 2] + rotation_matrix[2, 1],
                 rotation_matrix[2, 0] - rotation_matrix[0, 2]]
    else:
        if rotation_matrix[0, 0] < -rotation_matrix[1, 1]:
            t = 1 - rotation_matrix[0, 0] - rotation_matrix[1, 1] + rotation_matrix[2, 2]
            q = [rotation_matrix[2, 0] + rotation_matrix[0, 2], rotation_matrix[1, 2] + rotation_matrix[2, 1], t,
                 rotation_matrix[0, 1] - rotation_matrix[1, 0]]
        else:
            t = 1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
            q = [rotation_matrix[1, 2] - rotation_matrix[2, 1], rotation_matrix[2, 0] - rotation_matrix[0, 2],
                 rotation_matrix[0, 1] - rotation_matrix[1, 0], t]

    q = np.array(q) * 0.5 / np.sqrt(t)
    return q

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from abc import ABC

import numpy as np
from gymnasium.spaces import Box
from klimits import normalize as normalize_array
from klimits import interpolate_acceleration_batch as interpolate_acceleration_batch
from klimits import interpolate_position_batch as interpolate_position_batch
from klimits import interpolate_velocity_batch as interpolate_velocity_batch
from safemotions.envs.safe_motions_base import SafeMotionsBase

def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))


def normalize(value, min_value, max_value):
    return -1 + 2 * (value - min_value) / (max_value - min_value)


def _normalize_joint_values_min_max(values, joint_limit_ranges):
    normalized_values = -1 + 2 * (values - joint_limit_ranges[0]) / (joint_limit_ranges[1] - joint_limit_ranges[0])
    continuous_joint_indices = np.isnan(joint_limit_ranges[0]) | np.isnan(joint_limit_ranges[1])
    if np.any(continuous_joint_indices):
        # continuous joint -> map [-np.pi, np.pi] and all values shifted by 2 * np.pi to [-1, 1]
        normalized_values[continuous_joint_indices] = \
            -1 + 2 * (((values[continuous_joint_indices] + np.pi)/(2 * np.pi)) % 1)
    return normalized_values


TARGET_POINT_SIMULTANEOUS = 0
TARGET_POINT_ALTERNATING = 1
TARGET_POINT_SINGLE = 2

MOVING_OBJECT_SIMULTANEOUS = 0
MOVING_OBJECT_ALTERNATING = 1
MOVING_OBJECT_SINGLE = 2


class SafeObservation(ABC, SafeMotionsBase):

    def __init__(self,
                 *vargs,
                 obs_planet_size_per_planet=1,  # 1: time step, 2: xy position in orbit plane, 3: xyz in global space
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        self._next_joint_acceleration_mapping = None

        obs_current_size = 3  # pos, vel, acc

        obs_target_point_size = 0

        obs_moving_object_size = 0

        obs_planet_size = 0
        self._obs_planet_size_per_planet = obs_planet_size_per_planet

        if self._risk_config is not None:
            self._load_risk_obs_config()

        if self._robot_scene.obstacle_wrapper.use_target_points:
            if self._obs_add_target_point_pos:
                if self._robot_scene.obstacle_wrapper.target_point_sequence == TARGET_POINT_SINGLE:
                    obs_target_point_size += 3  # one target point only
                else:
                    obs_target_point_size += 3 * self._robot_scene.num_robots  # one target point per robot

            if self._obs_add_target_point_relative_pos:
                obs_target_point_size += 3 * self._robot_scene.num_robots

            if self._robot_scene.obstacle_wrapper.target_point_sequence == TARGET_POINT_ALTERNATING:
                obs_target_point_size += self._robot_scene.num_robots  # target point active signal for each robot

        if self._robot_scene.use_moving_objects:
            obs_size_per_moving_object_direction = 3
            obs_size_per_moving_object = 3 + obs_size_per_moving_object_direction
            # current position [x, y, z] and direction [x, y, z]
            if self._robot_scene.obstacle_wrapper.moving_object_sequence == MOVING_OBJECT_ALTERNATING:
                obs_size_per_moving_object += 1  # moving object active signal

            if self._robot_scene.obstacle_wrapper.target_point_sequence == MOVING_OBJECT_SINGLE:
                obs_moving_object_number = self._robot_scene.obstacle_wrapper.moving_object_active_number
            else:
                obs_moving_object_number = self._robot_scene.num_robots

            obs_moving_object_size = obs_size_per_moving_object * obs_moving_object_number

        self._obs_planet_num_independent_planets = None
        if self._planet_mode:
            self._obs_planet_num_independent_planets = len(self._robot_scene.obstacle_wrapper.planet_list)
            if self._planet_two_time_shift is not None:
                self._obs_planet_num_independent_planets = 1  # planet 2 is coupled with planet 1
            obs_planet_size = self._obs_planet_num_independent_planets * self._obs_planet_size_per_planet

        if self._human_network_checkpoint is not None:
            if self._human_network_use_full_observation:
                obs_human_size = self._robot_scene.obstacle_wrapper.human.observation_size
            else:
                obs_human_size = self._robot_scene.obstacle_wrapper.human.kinematic_observation_size
        else:
            obs_human_size = 0

        self._observation_size = obs_current_size * self._num_manip_joints  \
            + obs_target_point_size + obs_moving_object_size + obs_planet_size \
            + obs_human_size

        self._kinematic_observation_size = obs_current_size * self._num_manip_joints

        self._kinematic_observation = None

        self.observation_space = Box(low=np.float32(-1), high=np.float32(1), shape=(self._observation_size,),
                                     dtype=np.float32)

        if self._risk_config is not None:
            risk_observation_size = obs_current_size * self._num_manip_joints \
                                    + obs_moving_object_size \
                                    + obs_planet_size \
                                    + obs_human_size

            if self._risk_config["observation_size"] is None:
                self._risk_config["observation_size"] = risk_observation_size

            if risk_observation_size != self._risk_config["observation_size"]:
                raise ValueError("The observation size of the risk network ({}) does not match with the "
                                 "configuration of the environment ({}).".format(
                                    self._risk_config["observation_size"], risk_observation_size))

            self._init_risk_network_and_backup_agent()

        logging.info("Observation size: " + str(self._observation_size))

    def _load_risk_obs_config(self):
        overwrite_parameters = ["obs_planet_size_per_planet"]

        for parameter in overwrite_parameters:
            if parameter in self._risk_config["config"]["env_config"]:
                setattr(self, "_{}".format(parameter), self._risk_config["config"]["env_config"][parameter])

    def reset(self, **kwargs):

        super().reset(**kwargs)
        self._robot_scene.prepare_for_start_of_episode()

        observation, observation_info = self._get_observation()

        if self._control_rate is not None and hasattr(self._control_rate, 'reset'):
            # reset control rate timer
            self._control_rate.reset()

        if self._risk_check_initial_backup_trajectory:
            # check if a valid backup trajectory exists
            stored_variables = self.switch_to_backup_client()
            self._trajectory_manager.trajectory_length = self._risk_state_initial_backup_trajectory_steps + 2
            initial_state_safe = True

            for _ in range(self._risk_state_initial_backup_trajectory_steps):
                backup_action = np.array(
                    self._backup_agent.compute_action(self._risk_observation, full_fetch=False),
                    dtype=np.float64)
                simulated_action = self._get_action_from_backup_action(backup_action)

                _, _, simulation_termination, simulation_truncation, _ = self.step(simulated_action)
                simulation_done = simulation_termination or simulation_truncation

                if simulation_done:
                    if self.termination_reason != self.TERMINATION_TRAJECTORY_LENGTH:
                        initial_state_safe = False
                    break

            if initial_state_safe and \
                    self._risk_state_config == self.RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY \
                    and self._risk_state_deterministic_backup_trajectory:
                stored_variables["_robot_scene._obstacle_wrapper._moving_object_deterministic_list"] = \
                    self._robot_scene.obstacle_wrapper.moving_object_deterministic_list
                if self.robot_scene.obstacle_wrapper.human is not None:
                    self.robot_scene.obstacle_wrapper.human.copy_deterministic_lists_to_main_client()

            self.switch_back_to_main_client(stored_variables)

            if not initial_state_safe:
                self._episode_counter -= 1
                observation, _ = self.reset(repeated_reset=True)

        info = {}

        return observation, info

    def _prepare_for_next_action(self):
        super()._prepare_for_next_action()

    def _get_next_risk_observation(self, end_acceleration, forecast_non_kinematic_components=False):
        # compute the (expected) observation of the following time step

        # compute the kinematic state at the beginning of the following time step
        sample_time = np.array([self._trajectory_time_step])
        next_joint_position = interpolate_position_batch(self._start_acceleration,
                                                         end_acceleration,
                                                         self._start_velocity,
                                                         self._start_position, sample_time,
                                                         self._trajectory_time_step)[-1]

        next_joint_velocity = interpolate_velocity_batch(self._start_acceleration,
                                                         end_acceleration,
                                                         self._start_velocity,
                                                         sample_time,
                                                         self._trajectory_time_step)[-1]

        next_joint_acceleration = interpolate_acceleration_batch(self._start_acceleration,
                                                                 end_acceleration,
                                                                 sample_time,
                                                                 self._trajectory_time_step)[-1]

        curr_joint_position_rel_obs, curr_joint_velocity_rel_obs, curr_joint_acceleration_rel_obs, \
            moving_object_rel_obs, planet_rel_obs, human_rel_obs = self._get_shared_observation_components(
            curr_joint_position=next_joint_position,
            curr_joint_velocity=next_joint_velocity,
            curr_joint_acceleration=next_joint_acceleration,
            no_side_effects=True,
            # prevent side effects normally triggered by receiving the observation, e.g. sampling of new moving objects
            forecast_non_kinematic_components=forecast_non_kinematic_components)

        next_risk_observation = self._compose_and_clip_risk_observation(
            curr_joint_position_rel_obs=curr_joint_position_rel_obs,
            curr_joint_velocity_rel_obs=curr_joint_velocity_rel_obs,
            curr_joint_acceleration_rel_obs=curr_joint_acceleration_rel_obs,
            moving_object_rel_obs=moving_object_rel_obs,
            planet_rel_obs=planet_rel_obs,
            human_rel_obs=human_rel_obs)

        return next_risk_observation

    def get_kinematic_observation_components(self, curr_joint_position, curr_joint_velocity,
                                             curr_joint_acceleration):

        curr_joint_position_rel_obs = list(_normalize_joint_values_min_max(curr_joint_position,
                                                                           self.pos_limits_min_max))
        curr_joint_velocity_rel_obs = normalize_joint_values(curr_joint_velocity, self._robot_scene.max_velocities)
        curr_joint_acceleration_rel_obs = normalize_joint_values(curr_joint_acceleration,
                                                                 self._robot_scene.max_accelerations)

        return curr_joint_position_rel_obs, curr_joint_velocity_rel_obs, curr_joint_acceleration_rel_obs

    def _get_shared_observation_components(self, curr_joint_position, curr_joint_velocity,
                                           curr_joint_acceleration, forecast_non_kinematic_components=False,
                                           no_side_effects=False):

        curr_joint_position_rel_obs, curr_joint_velocity_rel_obs, \
            curr_joint_acceleration_rel_obs = self.get_kinematic_observation_components(curr_joint_position,
                                                                                        curr_joint_velocity,
                                                                                        curr_joint_acceleration)

        # moving objects (ball from outside)
        moving_object_rel_obs = []
        if self._robot_scene.use_moving_objects:
            # the function needs to be called even if the return value is not used.
            # Otherwise, new moving objects are not generated
            moving_object_rel_obs = self._robot_scene.obstacle_wrapper.get_moving_object_observation(
                forecast_next_step=forecast_non_kinematic_components,
                no_side_effects=no_side_effects)

        # planet position
        planet_rel_obs = []
        if self._planet_mode:
            for i in range(self._obs_planet_num_independent_planets):
                planet = self._robot_scene.obstacle_wrapper.planet_list[i]

                if not forecast_non_kinematic_components:
                    current_planet_time_step_index = planet.current_time_step_index
                else:
                    current_planet_time_step_index = (planet.current_time_step_index +
                                                      self._obstacle_client_update_steps_per_action) % \
                                                     len(planet.orbit_local_time_steps_pos)

                if self._obs_planet_size_per_planet == 1:
                    planet_rel_obs.append(
                        normalize(value=current_planet_time_step_index,
                                  min_value=0,
                                  max_value=planet.num_time_steps))
                elif self._obs_planet_size_per_planet == 2:
                    planet_rel_obs.append(
                        normalize(value=planet.orbit_local_time_steps_pos[current_planet_time_step_index][0],
                                  min_value=-1.05 * planet.radius_xy[0],
                                  max_value=1.05 * planet.radius_xy[0]))
                    planet_rel_obs.append(
                        normalize(value=planet.orbit_local_time_steps_pos[current_planet_time_step_index][1],
                                  min_value=-1.05 * planet.radius_xy[1],
                                  max_value=1.05 * planet.radius_xy[1]))
                elif self._obs_planet_size_per_planet == 3:
                    planet_rel_obs.extend(list(normalize_array(
                        planet.orbit_global_time_steps_pos[current_planet_time_step_index],
                        self._robot_scene.planet_obs_global_pos_min_max)))

        human_rel_obs = []
        # human observation
        if self._human_network_checkpoint is not None:
            if self._human_network_use_full_observation:
                if not forecast_non_kinematic_components:
                    human_rel_obs.extend(list(self._robot_scene.obstacle_wrapper.human.observation))
                else:
                    raise ValueError("Forecasting of the full human observation not implemented")
            else:
                if not forecast_non_kinematic_components:
                    human_rel_obs.extend(list(self._robot_scene.obstacle_wrapper.human.kinematic_observation))
                else:
                    self._robot_scene.obstacle_wrapper.human.step()
                    human_rel_obs.extend(list(self._robot_scene.obstacle_wrapper.human.kinematic_observation_forecast))

        return curr_joint_position_rel_obs, curr_joint_velocity_rel_obs, \
            curr_joint_acceleration_rel_obs, moving_object_rel_obs, planet_rel_obs, \
            human_rel_obs

    def _get_observation(self):
        curr_joint_position = self._get_generated_trajectory_point(-1)
        curr_joint_velocity = self._get_generated_trajectory_point(-1, key='velocities')
        curr_joint_acceleration = self._get_generated_trajectory_point(-1, key='accelerations')

        curr_joint_position_rel_obs, curr_joint_velocity_rel_obs, \
            curr_joint_acceleration_rel_obs, moving_object_rel_obs, planet_rel_obs, \
            human_rel_obs = \
            self._get_shared_observation_components(curr_joint_position=curr_joint_position,
                                                    curr_joint_velocity=curr_joint_velocity,
                                                    curr_joint_acceleration=curr_joint_acceleration)

        # target point for reaching tasks
        target_point_rel_obs = []
        if self._robot_scene.obstacle_wrapper.use_target_points:
            # the function needs to be called even if the return value is not used.
            # Otherwise, new target points are not generated
            target_point_pos, target_point_relative_pos, _, target_point_active_obs = \
                self._robot_scene.obstacle_wrapper.get_target_point_observation(
                    compute_relative_pos_norm=self._obs_add_target_point_relative_pos,
                    compute_target_point_joint_pos_norm=False)
            if self._obs_add_target_point_pos:
                target_point_rel_obs = target_point_rel_obs + list(target_point_pos)

            if self._obs_add_target_point_relative_pos:
                target_point_rel_obs = target_point_rel_obs + list(target_point_relative_pos)

            target_point_rel_obs = target_point_rel_obs + list(target_point_active_obs)
            # to indicate if the target point is active (1.0) or inactive (0.0); the list is empty if not required

        observation_not_clipped = np.array(curr_joint_position_rel_obs
                                           + curr_joint_velocity_rel_obs
                                           + curr_joint_acceleration_rel_obs
                                           + target_point_rel_obs
                                           + moving_object_rel_obs
                                           + planet_rel_obs
                                           + human_rel_obs, dtype=np.float32)

        observation = np.clip(observation_not_clipped, -1, 1)

        self._kinematic_observation = observation[:self._kinematic_observation_size]

        if self._risk_config is not None:

            self._risk_observation = self._compose_and_clip_risk_observation(
                curr_joint_position_rel_obs=curr_joint_position_rel_obs,
                curr_joint_velocity_rel_obs=curr_joint_velocity_rel_obs,
                curr_joint_acceleration_rel_obs=curr_joint_acceleration_rel_obs,
                moving_object_rel_obs=moving_object_rel_obs,
                planet_rel_obs=planet_rel_obs,
                human_rel_obs=human_rel_obs)

        info = {'mean': {},
                'max': {},
                'min': {}}

        pos_violation = 0.0
        vel_violation = 0.0
        acc_violation = 0.0

        for j in range(self._num_manip_joints):

            info['mean']['joint_{}_pos'.format(j)] = curr_joint_position_rel_obs[j]
            info['mean']['joint_{}_pos_abs'.format(j)] = abs(curr_joint_position_rel_obs[j])
            info['max']['joint_{}_pos'.format(j)] = curr_joint_position_rel_obs[j]
            info['min']['joint_{}_pos'.format(j)] = curr_joint_position_rel_obs[j]
            if abs(curr_joint_position_rel_obs[j]) > 1.001:
                logging.warning("Position violation: t = %s Joint: %s Rel position %s",
                                self._episode_length * self._trajectory_time_step, j,
                                curr_joint_position_rel_obs[j])
                pos_violation = 1.0

            info['mean']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            info['mean']['joint_{}_vel_abs'.format(j)] = abs(curr_joint_velocity_rel_obs[j])
            info['max']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            info['min']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            if abs(curr_joint_velocity_rel_obs[j]) > 1.001:
                logging.warning("Velocity violation: t = %s Joint: %s Rel velocity %s",
                                self._episode_length * self._trajectory_time_step, j,
                                curr_joint_velocity_rel_obs[j])
                vel_violation = 1.0

            info['mean']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            info['mean']['joint_{}_acc_abs'.format(j)] = abs(curr_joint_acceleration_rel_obs[j])
            info['max']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            info['min']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            if abs(curr_joint_acceleration_rel_obs[j]) > 1.001:
                logging.warning("Acceleration violation: t = %s Joint: %s Rel acceleration %s",
                                self._episode_length * self._trajectory_time_step, j,
                                curr_joint_acceleration_rel_obs[j])
                acc_violation = 1.0

        info['mean']['joint_vel_norm'] = np.linalg.norm(curr_joint_velocity_rel_obs)
        info['max']['joint_pos_violation'] = pos_violation
        info['max']['joint_vel_violation'] = vel_violation
        info['max']['joint_acc_violation'] = acc_violation

        if np.array_equal(observation, observation_not_clipped):
            info['mean']['observation_clipping_rate'] = 0.0
        else:
            info['mean']['observation_clipping_rate'] = 1.0

        logging.debug("Observation %s: %s", self._episode_length, np.asarray(observation))

        return observation, info

    def _compose_and_clip_risk_observation(self, curr_joint_position_rel_obs,
                                           curr_joint_velocity_rel_obs,
                                           curr_joint_acceleration_rel_obs,
                                           moving_object_rel_obs, planet_rel_obs, human_rel_obs):
        risk_observation_not_clipped = \
            np.array(curr_joint_position_rel_obs
                     + curr_joint_velocity_rel_obs
                     + curr_joint_acceleration_rel_obs
                     + moving_object_rel_obs
                     + planet_rel_obs
                     + human_rel_obs, dtype=np.float32)

        return np.clip(risk_observation_not_clipped, -1, 1)

    @property
    def kinematic_observation(self):
        return self._kinematic_observation

    @property
    def kinematic_observation_size(self):
        return self._kinematic_observation_size

    @property
    def observation_size(self):
        return self._observation_size


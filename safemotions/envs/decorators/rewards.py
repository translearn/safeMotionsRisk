# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import logging
from abc import ABC

import numpy as np

from safemotions.envs.safe_motions_base import SafeMotionsBase
from klimits import normalize


def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))


def compute_quadratic_punishment(a, b, c, d):
    # returns max(min((a - b) / (c - d), 1), 0) ** 2
    punishment = (a - b) / (c - d)
    return max(min(punishment, 1), 0) ** 2


class RewardBase(ABC, SafeMotionsBase):
    # optional action penalty
    ACTION_THRESHOLD = 0.9
    ACTION_MAX_PUNISHMENT = 1.0

    # collision avoidance
    # 1. self-collision avoidance
    SELF_COLLISION_MAX_REWARD = 0.0
    SELF_COLLISION_MAX_REWARD_DISTANCE = 0.05
    # 2. avoidance of collisions with static obstacles
    STATIC_OBSTACLES_COLLISION_MAX_REWARD = 0.0
    STATIC_OBSTACLES_COLLISION_MAX_REWARD_DISTANCE = 0.1
    # 3. avoidance of collisions with moving obstacles
    MOVING_OBSTACLES_COLLISION_MAX_REWARD = 0.0
    MOVING_OBSTACLES_COLLISION_MAX_REWARD_DISTANCE = 0.30

    def __init__(self,
                 *vargs,
                 normalize_reward_to_frequency=False,
                 punish_action=False,
                 action_punishment_min_threshold=ACTION_THRESHOLD,
                 action_max_punishment=ACTION_MAX_PUNISHMENT,
                 collision_avoidance_self_collision_max_reward=SELF_COLLISION_MAX_REWARD,
                 collision_avoidance_self_collision_max_reward_distance=SELF_COLLISION_MAX_REWARD_DISTANCE,
                 collision_avoidance_static_obstacles_max_reward=STATIC_OBSTACLES_COLLISION_MAX_REWARD,
                 collision_avoidance_static_obstacles_max_reward_distance=
                 STATIC_OBSTACLES_COLLISION_MAX_REWARD_DISTANCE,
                 collision_avoidance_moving_obstacles_max_reward=MOVING_OBSTACLES_COLLISION_MAX_REWARD,
                 collision_avoidance_moving_obstacles_max_reward_distance=
                 MOVING_OBSTACLES_COLLISION_MAX_REWARD_DISTANCE,
                 **kwargs):
        # reward settings
        self.reward_range = [0, 1]  # dummy settings
        self._normalize_reward_to_frequency = normalize_reward_to_frequency

        self._punish_action = punish_action
        self._action_punishment_min_threshold = action_punishment_min_threshold
        self._action_max_punishment = action_max_punishment

        self._collision_avoidance_self_collision_max_reward = collision_avoidance_self_collision_max_reward
        self._collision_avoidance_self_collision_max_reward_distance = \
            collision_avoidance_self_collision_max_reward_distance
        self._collision_avoidance_static_obstacles_max_reward = collision_avoidance_static_obstacles_max_reward
        self._collision_avoidance_static_obstacles_max_reward_distance = \
            collision_avoidance_static_obstacles_max_reward_distance
        self._collision_avoidance_moving_obstacles_max_reward = collision_avoidance_moving_obstacles_max_reward
        self._collision_avoidance_moving_obstacles_max_reward_distance = \
            collision_avoidance_moving_obstacles_max_reward_distance

        self._reward_maximum_relevant_distance = self._compute_reward_maximum_relevant_distance()

        super().__init__(*vargs, **kwargs)

    def _compute_reward_maximum_relevant_distance(self):
        # can be overwritten by other classes
        return None

    def _compute_reward_maximum_relevant_distance_collision_avoidance(self):
        reward_maximum_relevant_distance = -1
        if self._collision_avoidance_self_collision_max_reward != 0:
            reward_maximum_relevant_distance = self._collision_avoidance_self_collision_max_reward_distance

        if self._collision_avoidance_static_obstacles_max_reward != 0 \
                and self._collision_avoidance_static_obstacles_max_reward_distance > \
                reward_maximum_relevant_distance:
            reward_maximum_relevant_distance = self._collision_avoidance_static_obstacles_max_reward_distance

        if reward_maximum_relevant_distance == -1:
            reward_maximum_relevant_distance = None
        return reward_maximum_relevant_distance

    def _compute_collision_avoidance_reward(self):
        self_collision_reward = 0
        static_obstacles_collision_reward = 0
        moving_obstacles_collision_reward = 0
        self_collision_detected = False
        collision_with_static_obstacle_detected = False
        collision_with_moving_obstacle_detected = False

        position_in_obstacle_client_set = False

        if self._collision_avoidance_self_collision_max_reward != 0 or \
                self._collision_avoidance_static_obstacles_max_reward != 0 or self._terminate_on_self_collision \
                or self._terminate_on_collision_with_static_obstacle:
            position_in_obstacle_client_set = True
            _, _, _, _, minimum_distance_to_static_obstacles, minimum_distance_self_collision = \
                self._robot_scene.obstacle_wrapper.get_minimum_distance(
                    manip_joint_indices=self._robot_scene.manip_joint_indices,
                    target_position=self._start_position,
                    check_safety_distance=False)

            if minimum_distance_to_static_obstacles < 0.001:
                minimum_distance_to_static_obstacles = 0
                collision_with_static_obstacle_detected = True

            if minimum_distance_self_collision < 0.001:
                minimum_distance_self_collision = 0
                self_collision_detected = True

            if self._collision_avoidance_self_collision_max_reward != 0:
                rel_distance = min(1.0, minimum_distance_self_collision /
                                   self._collision_avoidance_self_collision_max_reward_distance)
                self_collision_reward = rel_distance ** 2

            if self._collision_avoidance_static_obstacles_max_reward != 0:
                rel_distance = min(1.0, minimum_distance_to_static_obstacles /
                                   self._collision_avoidance_static_obstacles_max_reward_distance)
                static_obstacles_collision_reward = rel_distance ** 2

        if self._collision_avoidance_moving_obstacles_max_reward != 0 or \
                self._terminate_on_collision_with_moving_obstacle:
            if not position_in_obstacle_client_set:
                manip_joint_indices = self._robot_scene.manip_joint_indices
                target_position = self._start_position
            else:
                manip_joint_indices = None
                target_position = None

            if self._collision_avoidance_moving_obstacles_max_reward == 0:
                maximum_relevant_distance = 0.001
            else:
                maximum_relevant_distance = self._collision_avoidance_moving_obstacles_max_reward_distance

            minimum_distance_to_moving_obstacles = \
                self._robot_scene.obstacle_wrapper.get_minimum_distance_to_moving_obstacles(
                    manip_joint_indices=manip_joint_indices,
                    target_position=target_position,
                    maximum_relevant_distance=maximum_relevant_distance)

            if minimum_distance_to_moving_obstacles < 0.001:
                minimum_distance_to_moving_obstacles = 0
                collision_with_moving_obstacle_detected = True

            rel_distance = min(1.0, minimum_distance_to_moving_obstacles /
                               self._collision_avoidance_moving_obstacles_max_reward_distance)
            moving_obstacles_collision_reward = rel_distance ** 2

        return self_collision_reward, static_obstacles_collision_reward, moving_obstacles_collision_reward, \
            self_collision_detected, collision_with_static_obstacle_detected, collision_with_moving_obstacle_detected

    def _compute_action_punishment(self):
        # The aim of the action punishment is to avoid the action being too close to -1 or 1.
        action_abs = np.abs(self._last_action)
        max_action_abs = max(action_abs)
        return compute_quadratic_punishment(max_action_abs, self._action_punishment_min_threshold,
                                            1, self._action_punishment_min_threshold)


    def _common_reward_function(self, reward, info):
        if self._normalize_reward_to_frequency:
            # Baseline: 10 Hz
            reward = reward * self._trajectory_time_step / 0.1

        for key in ['average', 'min', 'max']:
            info[key].update(reward=reward)

        # add information about the jerk as custom metric
        curr_joint_jerk = \
            (np.array(self._get_generated_trajectory_point(-1, key='accelerations'))
             - np.array(self._get_generated_trajectory_point(-2, key='accelerations'))) \
            / self._trajectory_time_step

        curr_joint_jerk_rel = normalize_joint_values(curr_joint_jerk, self._robot_scene.max_jerk_linear_interpolation)
        jerk_violation = 0.0

        for j in range(self._num_manip_joints):
            info['average']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]
            info['average']['joint_{}_jerk_abs'.format(j)] = abs(curr_joint_jerk_rel[j])
            info['max']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]
            info['min']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]

        max_normalized_jerk = np.max(np.abs(curr_joint_jerk_rel))
        if max_normalized_jerk > 1.002:
            jerk_violation = 1.0
            logging.warning("Jerk violation: t = %s Joint: %s Rel jerk %s",
                            (self._episode_length - 1) * self._trajectory_time_step,
                            np.argmax(np.abs(curr_joint_jerk_rel)),
                            max_normalized_jerk)

        info['max']['joint_jerk_violation'] = jerk_violation

        logging.debug("Reward %s: %s", self._episode_length - 1, reward)

        return reward, info

    @property
    def reward_maximum_relevant_distance(self):
        return self._reward_maximum_relevant_distance

    @property
    def reward_consider_moving_obstacles(self):
        return True if self._collision_avoidance_moving_obstacles_max_reward != 0 else False


class TargetPointReachingReward(RewardBase):
    ADAPTATION_MAX_PUNISHMENT = 1.0
    END_MIN_DISTANCE_MAX_PUNISHMENT = 1.0
    END_MAX_TORQUE_MAX_PUNISHMENT = 1.0
    END_MAX_TORQUE_MIN_THRESHOLD = 0.9

    # braking trajectory max punishment (either collision or torque -> max)
    BRAKING_TRAJECTORY_MAX_PUNISHMENT = 1.0
    # braking trajectory torque penalty
    BRAKING_TRAJECTORY_MAX_TORQUE_MIN_THRESHOLD = 0.9  # rel. abs. torque threshold

    def __init__(self,
                 *vargs,
                 normalize_reward_to_initial_target_point_distance=False,
                 punish_adaptation=False,
                 adaptation_max_punishment=ADAPTATION_MAX_PUNISHMENT,
                 punish_end_min_distance=False,
                 end_min_distance_max_punishment=END_MIN_DISTANCE_MAX_PUNISHMENT,
                 end_min_distance_max_threshold=None,
                 punish_end_max_torque=False,
                 end_max_torque_max_punishment=END_MAX_TORQUE_MAX_PUNISHMENT,
                 end_max_torque_min_threshold=END_MAX_TORQUE_MIN_THRESHOLD,
                 braking_trajectory_max_punishment=BRAKING_TRAJECTORY_MAX_PUNISHMENT,
                 punish_braking_trajectory_min_distance=False,
                 braking_trajectory_min_distance_max_threshold=None,
                 punish_braking_trajectory_max_torque=False,
                 braking_trajectory_max_torque_min_threshold=BRAKING_TRAJECTORY_MAX_TORQUE_MIN_THRESHOLD,
                 target_point_reward_factor=1.0,
                 **kwargs):

        self._normalize_reward_to_initial_target_point_distance = normalize_reward_to_initial_target_point_distance

        self._punish_adaptation = punish_adaptation
        self._adaptation_max_punishment = adaptation_max_punishment

        self._punish_end_min_distance = punish_end_min_distance
        self._end_min_distance_max_punishment = end_min_distance_max_punishment
        self._end_min_distance_max_threshold = end_min_distance_max_threshold
        self._punish_end_max_torque = punish_end_max_torque
        self._end_max_torque_max_punishment = end_max_torque_max_punishment
        self._end_max_torque_min_threshold = end_max_torque_min_threshold

        self._punish_braking_trajectory_min_distance = punish_braking_trajectory_min_distance
        self._braking_trajectory_min_distance_max_threshold = braking_trajectory_min_distance_max_threshold
        self._punish_braking_trajectory_max_torque = punish_braking_trajectory_max_torque
        self._braking_trajectory_max_punishment = braking_trajectory_max_punishment
        self._max_torque_min_threshold = braking_trajectory_max_torque_min_threshold

        self._target_point_reward_factor = target_point_reward_factor

        super().__init__(*vargs, **kwargs)

    def _compute_reward_maximum_relevant_distance(self):
        reward_maximum_relevant_distance = None

        if self._punish_braking_trajectory_min_distance or self._punish_end_min_distance:
            if self._punish_braking_trajectory_min_distance and \
                    self._braking_trajectory_min_distance_max_threshold is None:
                raise ValueError("punish_braking_trajectory_min_distance requires "
                                 "braking_trajectory_min_distance_max_threshold to be specified")
            if self._punish_end_min_distance and \
                    self._end_min_distance_max_threshold is None:
                raise ValueError("punish_end_min_distance requires "
                                 "end_min_distance_max_threshold to be specified")

            if self._punish_braking_trajectory_min_distance and self._punish_end_min_distance:
                reward_maximum_relevant_distance = max(self._braking_trajectory_min_distance_max_threshold,
                                                       self._end_min_distance_max_threshold)
            elif self._punish_braking_trajectory_min_distance:
                reward_maximum_relevant_distance = self._braking_trajectory_min_distance_max_threshold
            else:
                reward_maximum_relevant_distance = self._end_min_distance_max_threshold

        reward_maximum_relevant_distance_collision_avoidance = \
            self._compute_reward_maximum_relevant_distance_collision_avoidance()

        if reward_maximum_relevant_distance is not None and \
                reward_maximum_relevant_distance_collision_avoidance is not None:
            reward_maximum_relevant_distance = max(reward_maximum_relevant_distance,
                                                   reward_maximum_relevant_distance_collision_avoidance)
        elif reward_maximum_relevant_distance is None:
            reward_maximum_relevant_distance = reward_maximum_relevant_distance_collision_avoidance

        return reward_maximum_relevant_distance

    def _get_reward(self):
        info = {'average': {}, 'min': {}, 'max': {}}

        target_point_reward = 0
        action_punishment = 0
        adaptation_punishment = 0
        end_min_distance_punishment = 0
        end_max_torque_punishment = 0


        braking_trajectory_min_distance_punishment = 0
        braking_trajectory_max_torque_punishment = 0

        if self._punish_action:
            action_punishment = self._compute_action_punishment()  # action punishment

        if self._punish_adaptation:
            adaptation_punishment = self._adaptation_punishment

        if self._punish_end_min_distance:
            if self._end_min_distance is None:
                self._end_min_distance, _, _, _, _, _ = self._robot_scene.obstacle_wrapper.get_minimum_distance(
                    manip_joint_indices=self._robot_scene.manip_joint_indices,
                    target_position=self._start_position)

            end_min_distance_punishment = compute_quadratic_punishment(
                a=self._end_min_distance_max_threshold,
                b=self._end_min_distance,
                c=self._end_min_distance_max_threshold,
                d=self._robot_scene.obstacle_wrapper.closest_point_safety_distance)

        if self._punish_end_max_torque:
            if self._end_max_torque is not None:
                # None if check_braking_trajectory is False and asynchronous movement execution is active
                # in this case, no penalty is computed, but the penalty is not required anyways
                end_max_torque_punishment = compute_quadratic_punishment(
                    a=self._end_max_torque,
                    b=self._end_max_torque_min_threshold,
                    c=1,
                    d=self._end_max_torque_min_threshold)

        if self._robot_scene.obstacle_wrapper.use_target_points:

            target_point_reward = self._robot_scene.obstacle_wrapper.get_target_point_reward(
                normalize_distance_reward_to_initial_target_point_distance=
                self._normalize_reward_to_initial_target_point_distance)

        self_collision_reward, static_obstacles_collision_reward, moving_obstacles_collision_reward, \
            self._self_collision_detected, self._collision_with_static_obstacle_detected, \
            self._collision_with_moving_obstacle_detected = self._compute_collision_avoidance_reward()

        reward = target_point_reward * self._target_point_reward_factor \
            - action_punishment * self._action_max_punishment \
            - adaptation_punishment * self._adaptation_max_punishment \
            - end_min_distance_punishment * self._end_min_distance_max_punishment \
            - end_max_torque_punishment * self._end_max_torque_max_punishment  \
            + self_collision_reward * self._collision_avoidance_self_collision_max_reward \
            + static_obstacles_collision_reward * self._collision_avoidance_static_obstacles_max_reward \
            + moving_obstacles_collision_reward * self._collision_avoidance_moving_obstacles_max_reward

        if self._punish_braking_trajectory_min_distance or self._punish_braking_trajectory_max_torque:
            braking_trajectory_min_distance_punishment, braking_trajectory_max_torque_punishment = \
                self._robot_scene.obstacle_wrapper.get_braking_trajectory_punishment(
                    minimum_distance_max_threshold=self._braking_trajectory_min_distance_max_threshold,
                    maximum_torque_min_threshold=self._max_torque_min_threshold)

            if self._punish_braking_trajectory_min_distance and self._punish_braking_trajectory_max_torque:
                braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                max(braking_trajectory_min_distance_punishment,
                                                    braking_trajectory_max_torque_punishment)
            elif self._punish_braking_trajectory_min_distance:
                braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                braking_trajectory_min_distance_punishment
            else:
                braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                braking_trajectory_max_torque_punishment

            reward = reward - braking_trajectory_punishment

        for key in ['average', 'min', 'max']:
            info[key].update(action_punishment=action_punishment,
                             adaptation_punishment=adaptation_punishment,
                             end_min_distance_punishment=end_min_distance_punishment,
                             end_max_torque_punishment=end_max_torque_punishment,
                             target_point_reward=target_point_reward,
                             self_collision_reward=self_collision_reward,
                             static_obstacles_collision_reward=static_obstacles_collision_reward,
                             moving_obstacles_collision_reward=moving_obstacles_collision_reward,
                             braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                             braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment)

        reward, info = self._common_reward_function(reward=reward, info=info)

        return reward, info


class CollisionAvoidanceReward(RewardBase):

    # low acceleration reward
    LOW_ACCELERATION_MAX_REWARD = 1.0
    LOW_ACCELERATION_THRESHOLD = 0.1

    # low velocity reward
    LOW_VELOCITY_MAX_REWARD = 1.0
    LOW_VELOCITY_THRESHOLD = 0.1

    def __init__(self,
                 *vargs,
                 collision_avoidance_low_acceleration_max_reward=LOW_ACCELERATION_MAX_REWARD,
                 collision_avoidance_low_acceleration_threshold=LOW_ACCELERATION_THRESHOLD,
                 collision_avoidance_low_velocity_max_reward=LOW_VELOCITY_MAX_REWARD,
                 collision_avoidance_low_velocity_threshold=LOW_VELOCITY_THRESHOLD,
                 collision_avoidance_episode_termination_bonus=0,
                 collision_avoidance_episode_early_termination_punishment=-0,
                 **kwargs):

        self._collision_avoidance_low_acceleration_max_reward = collision_avoidance_low_acceleration_max_reward
        self._collision_avoidance_low_acceleration_threshold = collision_avoidance_low_acceleration_threshold
        self._collision_avoidance_low_velocity_max_reward = collision_avoidance_low_velocity_max_reward
        self._collision_avoidance_low_velocity_threshold = collision_avoidance_low_velocity_threshold
        self._collision_avoidance_episode_termination_bonus = collision_avoidance_episode_termination_bonus
        self._collision_avoidance_episode_early_termination_punishment = \
            collision_avoidance_episode_early_termination_punishment

        super().__init__(*vargs, **kwargs)

    def _compute_reward_maximum_relevant_distance(self):
        return self._compute_reward_maximum_relevant_distance_collision_avoidance()

    def _get_reward(self):
        info = {'average': {}, 'min': {}, 'max': {}}

        reward = 0
        action_punishment = 1.0

        low_acceleration_reward = 0
        low_velocity_reward = 0

        episode_termination_bonus = 0
        episode_early_termination_punishment = 0

        self_collision_reward, static_obstacles_collision_reward, moving_obstacles_collision_reward, \
            self._self_collision_detected, self._collision_with_static_obstacle_detected, \
            self._collision_with_moving_obstacle_detected = self._compute_collision_avoidance_reward()

        if self._collision_avoidance_low_acceleration_max_reward:
            rel_acceleration = normalize(self._start_acceleration, self.acc_limits_min_max)
            max_acceleration_abs = max(np.abs(rel_acceleration))
            rel_deviation = min(1.0, max_acceleration_abs /
                                self._collision_avoidance_low_acceleration_threshold)
            low_acceleration_reward = (rel_deviation - 1) ** 2

        if self._collision_avoidance_low_velocity_max_reward:
            rel_velocity = normalize(self._start_velocity, self.vel_limits_min_max)
            max_velocity_abs = max(np.abs(rel_velocity))
            rel_deviation = min(1.0, max_velocity_abs /
                                self._collision_avoidance_low_velocity_threshold)
            low_velocity_reward = (rel_deviation - 1) ** 2

        if self._punish_action:
            action_punishment = self._compute_action_punishment()  # action punishment

        if (self._trajectory_manager.is_trajectory_finished(self._current_trajectory_point_index)[0]) \
                and not \
                (self._terminate_on_self_collision and self._self_collision_detected) \
                and not \
                (self._terminate_on_collision_with_static_obstacle and self._collision_with_static_obstacle_detected) \
                and not \
                (self._terminate_on_collision_with_moving_obstacle and self._collision_with_moving_obstacle_detected):
            episode_termination_bonus = self._collision_avoidance_episode_termination_bonus

        if (self._terminate_on_self_collision and self._self_collision_detected) \
                or \
                (self._terminate_on_collision_with_static_obstacle and self._collision_with_static_obstacle_detected) \
                or \
                (self._terminate_on_collision_with_moving_obstacle and self._collision_with_moving_obstacle_detected):
            episode_early_termination_punishment = self._collision_avoidance_episode_early_termination_punishment

        reward = (1 - action_punishment) * self._action_max_punishment \
            + self_collision_reward * self._collision_avoidance_self_collision_max_reward \
            + static_obstacles_collision_reward * self._collision_avoidance_static_obstacles_max_reward \
            + moving_obstacles_collision_reward * self._collision_avoidance_moving_obstacles_max_reward \
            + low_acceleration_reward * self._collision_avoidance_low_acceleration_max_reward \
            + low_velocity_reward * self._collision_avoidance_low_velocity_max_reward \
            + episode_termination_bonus \
            + episode_early_termination_punishment

        for key in ['average', 'min', 'max']:
            info[key].update(action_punishment=action_punishment,
                             self_collision_reward=self_collision_reward,
                             static_obstacles_collision_reward=static_obstacles_collision_reward,
                             moving_obstacles_collision_reward=moving_obstacles_collision_reward,
                             low_acceleration_reward=low_acceleration_reward,
                             low_velocity_reward=low_velocity_reward,
                             episode_termination_bonus=episode_termination_bonus,
                             episode_early_termination_punishment=episode_early_termination_punishment)

        reward, info = self._common_reward_function(reward=reward, info=info)

        return reward, info





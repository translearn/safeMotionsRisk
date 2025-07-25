# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os.path
import json
import numpy as np
import logging

TERMINATION_TRAJECTORY_LENGTH = 2

def clip_index(index, list_length):
    if index < 0 and abs(index) > list_length:
        return 0
    if index > 0 and index > list_length - 1:
        return -1
    else:
        return index


class TrajectoryManager(object):

    def __init__(self,
                 trajectory_time_step,
                 trajectory_duration,
                 obstacle_wrapper,
                 env=None,
                 **kwargs):

        self._trajectory_time_step = trajectory_time_step
        self._trajectory_duration = trajectory_duration
        self._num_time_steps = int(trajectory_duration / trajectory_time_step)
        self._obstacle_wrapper = obstacle_wrapper

        self._trajectory_start_position = None
        self._trajectory_start_velocity = None
        self._trajectory_start_acceleration = None
        self._trajectory_length = None
        self._num_manip_joints = None
        self._zero_joint_vector = None
        self._generated_trajectory = None
        self._measured_actual_trajectory_control_points = None
        self._computed_actual_trajectory_control_points = None
        self._generated_trajectory_control_points = None

        self._controller_model_coefficient_a = None
        self._controller_model_coefficient_b = None

        self._env = env

        self._do_not_copy_keys = ['_obstacle_wrapper', '_env']

    @property
    def generated_trajectory_control_points(self):
        return self._generated_trajectory_control_points

    @property
    def measured_actual_trajectory_control_points(self):
        return self._measured_actual_trajectory_control_points

    @property
    def computed_actual_trajectory_control_points(self):
        return self._computed_actual_trajectory_control_points

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @property
    def trajectory_length(self):
        return self._trajectory_length

    @trajectory_length.setter
    def trajectory_length(self, val):
        self._trajectory_length = val

    def reset(self, get_new_trajectory=True, duration_multiplier=None):
        if get_new_trajectory:

            self._trajectory_start_position, self._trajectory_start_velocity, self._trajectory_start_acceleration \
                = self.get_new_trajectory_start_position_velocity_acceleration()
            trajectory_duration = self._trajectory_duration

            if duration_multiplier is not None:
                trajectory_duration = trajectory_duration * duration_multiplier

            self._trajectory_length = round(trajectory_duration / self._trajectory_time_step) + 1
        self._num_manip_joints = len(self._trajectory_start_position)
        self._zero_joint_vector = np.array([0.0] * self._num_manip_joints)

        if self._trajectory_start_velocity is None:
            self._trajectory_start_velocity = self._zero_joint_vector

        if self._trajectory_start_acceleration is None:
            self._trajectory_start_acceleration = self._zero_joint_vector

        self._generated_trajectory = {'positions': [self._trajectory_start_position],
                                      'velocities': [self._trajectory_start_velocity],
                                      'accelerations': [self._trajectory_start_acceleration]}
        self._measured_actual_trajectory_control_points = {'positions': [self._trajectory_start_position],
                                                           'velocities': [self._trajectory_start_velocity],
                                                           'accelerations': [self._trajectory_start_acceleration]}
        self._computed_actual_trajectory_control_points = {'positions': [self._trajectory_start_position],
                                                           'velocities': [self._trajectory_start_velocity],
                                                           'accelerations': [self._trajectory_start_acceleration]}
        self._generated_trajectory_control_points = {'positions': [self._trajectory_start_position],
                                                     'velocities': [self._trajectory_start_velocity],
                                                     'accelerations': [self._trajectory_start_acceleration]}


    def get_trajectory_start_position(self):
        return self._trajectory_start_position

    def get_trajectory_start_velocity(self):
        return self._trajectory_start_velocity

    def get_trajectory_start_acceleration(self):
        return self._trajectory_start_acceleration

    def get_generated_trajectory_point(self, index, key='positions'):
        i = clip_index(index, len(self._generated_trajectory[key]))

        return self._generated_trajectory[key][i]

    def get_measured_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        i = clip_index(index, len(self._measured_actual_trajectory_control_points[key]))

        if not start_at_index:
            return self._measured_actual_trajectory_control_points[key][i]
        else:
            return self._measured_actual_trajectory_control_points[key][i:]

    def get_computed_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        i = clip_index(index, len(self._computed_actual_trajectory_control_points[key]))

        if not start_at_index:
            return self._computed_actual_trajectory_control_points[key][i]
        else:
            return self._computed_actual_trajectory_control_points[key][i:]

    def get_generated_trajectory_control_point(self, index, key='positions'):
        i = clip_index(index, len(self._generated_trajectory_control_points[key]))

        return self._generated_trajectory_control_points[key][i]

    def add_generated_trajectory_point(self, positions, velocities, accelerations):
        self._generated_trajectory['positions'].append(positions)
        self._generated_trajectory['velocities'].append(velocities)
        self._generated_trajectory['accelerations'].append(accelerations)

    def add_measured_actual_trajectory_control_point(self, positions, velocities, accelerations):
        self._measured_actual_trajectory_control_points['positions'].append(positions)
        self._measured_actual_trajectory_control_points['velocities'].append(velocities)
        self._measured_actual_trajectory_control_points['accelerations'].append(accelerations)

    def add_computed_actual_trajectory_control_point(self, positions, velocities, accelerations):
        self._computed_actual_trajectory_control_points['positions'].append(positions)
        self._computed_actual_trajectory_control_points['velocities'].append(velocities)
        self._computed_actual_trajectory_control_points['accelerations'].append(accelerations)

    def add_generated_trajectory_control_point(self, positions, velocities, accelerations):
        self._generated_trajectory_control_points['positions'].append(positions)
        self._generated_trajectory_control_points['velocities'].append(velocities)
        self._generated_trajectory_control_points['accelerations'].append(accelerations)

    def compute_controller_model_coefficients(self, time_constants, sampling_time):
        self._controller_model_coefficient_a = 1 + (2 * np.array(time_constants) / sampling_time)
        self._controller_model_coefficient_b = 1 - (2 * np.array(time_constants) / sampling_time)

    def model_position_controller_to_compute_actual_values(self, current_setpoint, last_setpoint, key='positions'):
        # models the position controller as a discrete transfer function and returns the
        # computed actual position, given the next position setpoint and previous computed actual positions
        # the controller is modelled as a first order low-pass with a (continuous) transfer function of
        #  G(s) = 1 / (1 + T * s)
        # the transfer function is discretized using Tustins approximation: s = 2 / Ta * (z - 1) / (z + 1)
        # the following difference equation can be derived:
        # y_n = 1/a * (x_n + x_n_minus_one - b * y_n_minus_one) with a = 1 + (2 * T / Ta) and b = 1 - (2 * T / Ta)

        x_n = np.asarray(current_setpoint)
        x_n_minus_one = np.asarray(last_setpoint)
        y_n_minus_one = self.get_computed_actual_trajectory_control_point(-1, key=key)
        computed_actual_position = 1 / self._controller_model_coefficient_a * \
                                   (x_n + x_n_minus_one - self._controller_model_coefficient_b * y_n_minus_one)

        return computed_actual_position

    def is_trajectory_finished(self, index):

        if index >= self._trajectory_length - 1:  # trajectory duration
            return True, TERMINATION_TRAJECTORY_LENGTH

        return False, None

    def get_new_trajectory_start_position_velocity_acceleration(self):
        return self._obstacle_wrapper.get_starting_point_joint_pos_vel_acc()


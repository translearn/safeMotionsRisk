# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import errno
import json
import logging
import os
import pickle
from pathlib import Path
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
from klimits import normalize as normalize_fast
from klimits import normalize_batch as normalize_batch
from klimits import interpolate_position_batch as interpolate_position_batch
from klimits import interpolate_velocity_batch as interpolate_velocity_batch
from klimits import interpolate_acceleration_batch as interpolate_acceleration_batch


class TrajectoryPlotter:
    def __init__(self,
                 time_step=None,
                 control_time_step=None,
                 simulation_time_step=None,
                 pos_limits=None,
                 vel_limits=None,
                 acc_limits=None,
                 jerk_limits=None,
                 torque_limits=None,
                 plot_joint=None,
                 plot_acc_limits=False,
                 plot_time_limits=None,
                 plot_actual_values=False,
                 plot_computed_actual_values=False,
                 plot_actual_torques=False,
                 plot_value=False,
                 evaluation_dir=None):

        self._time_step = time_step
        self._control_time_step = control_time_step
        self._simulation_time_step = simulation_time_step
        self._plot_acc_limits = plot_acc_limits
        self._plot_time_limits = plot_time_limits  # e.g. [1, 1.5] -> time limits for plotting
        self._plot_actual_values = plot_actual_values
        self._plot_computed_actual_values = plot_computed_actual_values
        self._plot_actual_torques = plot_actual_torques
        self._plot_joint = plot_joint
        self._plot_value = plot_value

        self._target_pos = None
        self._trajectory_counter = 0

        self._obstacle_wrapper = None
        self._fig = None
        self._ax = None
        self._ax_indices = None
        self._slider_joint_cartesian = None
        self._ax_slider_link_point = None
        self._joint_cartesian_value = 0
        self._obstacle_index = None
        self._link_index = None
        self._point_index = None

        self._entry_width = 0.05
        self._computed_actual_pos = None
        self._computed_actual_vel = None
        self._computed_actual_acc = None
        self._actual_torque = None

        self._baseline_pos = None
        self._baseline_vel = None
        self._baseline_acc = None

        self._value = None

        if time_step:
            self._plot_num_sub_time_steps = int(1000 * time_step)
            self._time_step_counter = None

            self._current_jerk = None
            self._current_acc = None
            self._current_vel = None
            self._current_pos = None

            self._time = None
            self._pos = None
            self._vel = None
            self._acc = None
            self._jerk = None

            self._sub_time = None
            self._sub_pos = None
            self._sub_vel = None
            self._sub_acc = None
            self._sub_jerk = None

            self._pos_limits = np.array(pos_limits)
            self._pos_limits_min_max = np.swapaxes(self._pos_limits, 0, 1)
            self._vel_limits = np.array(vel_limits)
            self._vel_limits_min_max = np.swapaxes(self._vel_limits, 0, 1)
            self._acc_limits = np.array(acc_limits)
            self._acc_limits_min_max = np.swapaxes(self._acc_limits, 0, 1)
            self._jerk_limits = np.array(jerk_limits)
            self._jerk_limits_min_max = np.swapaxes(self._jerk_limits, 0, 1)
            if torque_limits is not None:
                self._torque_limits = np.array(torque_limits)
                self._torque_limits_min_max = np.swapaxes(self._torque_limits, 0, 1)

            self._num_joints = len(self._pos_limits)

            if plot_joint is None:
                self._plot_joint = [True for i in range(self._num_joints)]

            self._episode_counter = 0
            self._zero_vector = np.zeros(self._num_joints)

            self._current_acc_limits = None
            self._actual_pos = None

            self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        # you can select the backend for matplotlib with matplotlib.use(backend_str)
        # potential values for backend_str with GUI support: 'QT5Agg', 'TkAgg'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['text.usetex'] = False

        self._evaluation_dir = evaluation_dir

    @property
    def trajectory_time(self):
        return self._time[-1]

    def reset_plotter(self, initial_joint_position, initial_joint_velocity, initial_joint_acceleration):

        self._time_step_counter = 0
        self._trajectory_counter = self._trajectory_counter + 1  # for naming files

        self._episode_counter = self._episode_counter + 1

        self._current_acc = np.array(initial_joint_acceleration)
        self._current_vel = np.array(initial_joint_velocity)
        self._current_pos = np.array(initial_joint_position)

        self._pos = []
        self._vel = []
        self._acc = []
        self._jerk = []

        self._current_acc_limits = []
        self._current_acc_limits.append([[0, 0] for i in range(self._num_joints)])

        self._sub_pos = []
        self._sub_vel = []
        self._sub_acc = []
        self._sub_jerk = []

        self._actual_pos = []  # not normalized
        self._target_pos = []  # not normalized

        self._computed_actual_pos = []
        self._computed_actual_vel = []
        self._computed_actual_acc = []

        self._actual_torque = []

        self._baseline_pos = []
        self._baseline_vel = []
        self._baseline_acc = []

        self._value = []

        self._time = [0]
        self._pos.append(normalize(self._current_pos, self._pos_limits_min_max))
        self._vel.append(normalize_fast(self._current_vel, self._vel_limits_min_max))
        self._acc.append(normalize_fast(self._current_acc, self._acc_limits_min_max))
        self._jerk.append(self._zero_vector.copy())  # zero jerk as initial value

        self._sub_time = [0]
        self._sub_pos = self._pos.copy()
        self._sub_vel = self._vel.copy()
        self._sub_acc = self._acc.copy()
        self._sub_jerk = self._jerk.copy()

    def display_plot(self, max_time=None, obstacle_wrapper=None, blocking=True):
        if obstacle_wrapper is not None:
            self._obstacle_wrapper = obstacle_wrapper

        num_subplots = 4
        if self._plot_actual_torques:
            num_subplots += 1
        if self._plot_value:
            num_subplots += 1

        fig, ax = plt.subplots(num_subplots, 1, sharex=True)

        if self._obstacle_wrapper is None or self._obstacle_wrapper.obstacle_scene == 0 \
                or not self._obstacle_wrapper.log_obstacle_data:
            plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.98, wspace=0.15, hspace=0.15)
        else:
            plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.94, wspace=0.15, hspace=0.15)
            self._add_slider(fig)

        ax_pos = 0
        ax_vel = 1
        ax_acc = 2
        ax_jerk = 3
        ax_offset = 4

        if self._plot_actual_torques:
            ax_torque = ax_offset
            ax_offset += 1
        else:
            ax_torque = None

        if self._plot_value:
            ax_value = ax_offset
            ax_offset += 1
        else:
            ax_value = None

        self._fig = fig
        self._ax = ax
        self._ax_indices = [ax_pos, ax_vel, ax_acc, ax_jerk, ax_torque, ax_value]

        if self._joint_cartesian_value == 0:
            self._plot_joint_data(max_time)
        else:
            self._plot_obstacle_data(max_time)

        fig.set_size_inches((24.1, 13.5), forward=False)
        logging.info("Trajectory plotted. Close plot to continue")
        plt.show(block=blocking)

    def _add_slider(self, fig):
        ax_slider_joint_cartesian = fig.add_axes([0.03, 0.96, 0.10, 0.02])
        self._slider_joint_cartesian = ButtonArray(ax_slider_joint_cartesian, label='',
                                                   entries_str=["Joint", "Cartesian"])

        num_obstacles = self._obstacle_wrapper.num_obstacles
        ax_slider_obstacle = fig.add_axes([0.15, 0.96, self._entry_width * num_obstacles, 0.02])
        entries_str_obstacles = ["Obstacle " + str(i + 1) if self._obstacle_wrapper.obstacle[i].name is None else
                                 self._obstacle_wrapper.obstacle[i].name
                                 for i in range(self._obstacle_wrapper.num_obstacles)]
        entries_str = entries_str_obstacles

        self._slider_obstacle = ButtonArray(ax_slider_obstacle, label='', entries_str=entries_str)

        ax_slider_link = fig.add_axes([0.15 + self._entry_width * num_obstacles + 0.03, 0.96,
                                       len(self._obstacle_wrapper.links_in_use) * self._entry_width, 0.02])
        entries_str = []
        for i in range(len(self._obstacle_wrapper.links_in_use)):
            entries_str.append(self._obstacle_wrapper.link_names[self._obstacle_wrapper.links_in_use[i]])
        self._slider_link = ButtonArray(ax_slider_link, label='', entries_str=entries_str)
        self._slider_link.set_active_values(self._obstacle_wrapper.get_indices_of_observed_links_in_use(
            obstacle_index=0))

        if self._obstacle_index is not None:
            if self._obstacle_index >= num_obstacles:
                self._obstacle_index = num_obstacles - 1
            self._slider_joint_cartesian.set_val(self._joint_cartesian_value)
            self._slider_joint_cartesian.update()
            self._slider_obstacle.set_val(self._obstacle_index)
            self._slider_obstacle.update()
            indices_of_observed_links_in_use = self._obstacle_wrapper.get_indices_of_observed_links_in_use(
                obstacle_index=self._obstacle_index)
            self._slider_link.set_active_values(indices_of_observed_links_in_use)
            slider_link_val = self._obstacle_wrapper.get_index_of_link_in_use(self._obstacle_wrapper.obstacle[
                                                                                  self._obstacle_index].
                                                                              observed_links[self._link_index])
            self._slider_link.set_val(slider_link_val)
            self._slider_link.update()
            self._add_slider_link_point(fig, slider_link_index=slider_link_val, point_index=self._point_index)
            self._slider_link_point.update()

        else:
            self._add_slider_link_point(fig, slider_link_index=self._slider_link.last_val_int)

        self._slider_joint_cartesian.on_changed(self._update_slider_joint_cartesian)
        self._slider_obstacle.on_changed(self._update_slider_obstacle)
        self._slider_link.on_changed(self._update_slider_link)

    def _add_slider_link_point(self, fig, slider_link_index, point_index=0):
        if self._ax_slider_link_point:
            self._ax_slider_link_point.remove()
        link_index = self._obstacle_wrapper.links_in_use[slider_link_index]
        link_point_names = []
        if self._obstacle_wrapper.links[link_index].observe_closest_point:
            link_point_names.append("Closest")
        for i in range(self._obstacle_wrapper.links[link_index].num_observed_points):
            link_point_names.append(self._obstacle_wrapper.links[link_index].observed_points[i].name)
        self._ax_slider_link_point = fig.add_axes([0.15 +
                                                   self._entry_width * (self._obstacle_wrapper.num_obstacles +
                                                                        self._obstacle_wrapper.num_target_points
                                                                        + len(self._obstacle_wrapper.links_in_use))
                                                   + 0.03, 0.96,
                                                   len(link_point_names) * self._entry_width, 0.02])
        self._slider_link_point = ButtonArray(self._ax_slider_link_point, label='', entries_str=link_point_names)
        self._slider_link_point.set_val(point_index)
        self._slider_link_point.on_changed(self._update_slider_link_point)

    def _update_slider_joint_cartesian(self, val):
        int_val = int(self._slider_joint_cartesian.val)
        self._joint_cartesian_value = int_val
        if int_val != self._slider_joint_cartesian.last_val_int:
            if int_val == 0:  # joint
                self._plot_joint_data()
            else:
                self._update_obstacle_indices()
                self._plot_obstacle_data()

    def _update_slider_obstacle(self, val):
        int_val = int(self._slider_obstacle.val)
        if int_val != self._slider_obstacle.last_val_int:
            indices_of_observed_links_in_use = self._obstacle_wrapper.get_indices_of_observed_links_in_use(
                obstacle_index=int_val)
            if self._slider_link.last_val_int not in indices_of_observed_links_in_use:
                self._add_slider_link_point(self._fig, slider_link_index=indices_of_observed_links_in_use[0])
            self._slider_link.set_active_values(indices_of_observed_links_in_use)
            self._update_obstacle_indices(obstacle_index=int_val)
            if self._slider_joint_cartesian.last_val_int == 1:
                self._plot_obstacle_data()

    def _update_slider_link(self, val):
        int_val = int(self._slider_link.val)
        if int_val != self._slider_link.last_val_int and int_val in self._slider_link.active_values:
            self._add_slider_link_point(self._fig, slider_link_index=int_val)
            self._update_obstacle_indices(link_index=int_val)
            if self._slider_joint_cartesian.last_val_int == 1:
                self._plot_obstacle_data()

    def _update_slider_link_point(self, val):
        int_val = int(self._slider_link_point.val)
        if int_val != self._slider_link_point.last_val_int:
            self._update_obstacle_indices(point_index=int_val)
            if self._slider_joint_cartesian.last_val_int == 1:
                self._plot_obstacle_data()

    def _clear_axes(self):
        for i in range(len(self._ax)):
            self._ax[i].clear()

    def _update_obstacle_indices(self, **kwargs):
        self._obstacle_index = kwargs.get('obstacle_index', self._slider_obstacle.last_val_int)
        slider_link_value = kwargs.get('link_index', self._slider_link.last_val_int)
        link_val = self._obstacle_wrapper.links_in_use[slider_link_value]

        self._link_index = self._obstacle_wrapper.obstacle[self._obstacle_index].get_link_index(link_val)
        self._point_index = kwargs.get('point_index', self._slider_link_point.last_val_int)

        self._obstacle_wrapper.debug_line_obstacle = self._obstacle_index
        self._obstacle_wrapper.debug_line_link = self._link_index
        self._obstacle_wrapper.debug_line_point = self._point_index

    def _plot_obstacle_data(self, max_time=None):

        self._clear_axes()
        fig = self._fig
        ax = self._ax
        ax_pos = self._ax_indices[0]
        ax_vel = self._ax_indices[1]
        ax_acc = self._ax_indices[2]
        ax_jerk = self._ax_indices[3]
        ax_torque = self._ax_indices[4]
        ax_value = self._ax_indices[5]

        if ax_torque is not None or ax_value is not None:
            if ax_torque is not None:
                ax[ax_torque].set_visible(False)
            if ax_value is not None:
                ax[ax_value].set_visible(False)
            plt.subplots_adjust(left=0.05, bottom=-0.2, right=0.95, top=0.94, wspace=0.15, hspace=0.15)

        for i in range(len(ax)):
            ax[i].grid(True)
            ax[i].set_xlabel('Time [s]')

        if ax_pos is not None:
            ax[ax_pos].set_ylabel('Distance')

        if ax_vel is not None:
            ax[ax_vel].set_ylabel('Velocity')

        if ax_jerk is not None:
            ax[ax_jerk].set_ylabel('Jerk')

        if ax_acc is not None:
            ax[ax_acc].set_ylabel('Acceleration')

        link_num = self._obstacle_wrapper.obstacle[self._obstacle_index].observed_links[self._link_index]
        point_index = self._point_index
        if self._point_index == 0 and self._obstacle_wrapper.links[link_num].observe_closest_point:
            actual_dis = self._obstacle_wrapper.obstacle[self._obstacle_index].link_data[
                self._link_index].closest_point_distance_actual
            set_dis = self._obstacle_wrapper.obstacle[self._obstacle_index].link_data[
                self._link_index].closest_point_distance_set
        else:
            if self._obstacle_wrapper.links[link_num].observe_closest_point:
                point_index = point_index - 1
            actual_dis = self._obstacle_wrapper.obstacle[self._obstacle_index].link_data[
                self._link_index].observed_point_distance_actual[point_index]
            set_dis = self._obstacle_wrapper.obstacle[self._obstacle_index].link_data[
                self._link_index].observed_point_distance_set[point_index]

        if actual_dis:
            actual_time = np.arange(0, len(actual_dis)) * self._control_time_step
            actual_vel = np.diff(actual_dis) / self._control_time_step
            actual_acc = np.diff(actual_vel) / self._control_time_step
            actual_jerk = np.diff(actual_acc) / self._control_time_step
            if max_time is None or max_time >= actual_time[-1]:
                actual_time_max_index = len(actual_time)
            else:
                actual_time_max_index = np.argmin(np.asarray(actual_time) <= max_time)

            logging.info('Actual Distance: ' + str(np.min(actual_dis)) + ' / ' + str(np.max(actual_dis)) +
                         '; Actual Vel: ' + str(np.min(actual_vel)) + ' / ' + str(np.max(actual_vel)) +
                         '; Actual Acc: ' + str(np.min(actual_acc)) + ' / ' + str(np.max(actual_acc)) +
                         '; Actual Jerk: ' + str(np.min(actual_jerk)) + ' / ' + str(np.max(actual_jerk)))

        if set_dis:
            set_time = np.arange(0, len(set_dis)) * self._control_time_step
            set_vel = np.diff(set_dis) / self._control_time_step
            set_acc = np.diff(set_vel) / self._control_time_step
            set_jerk = np.diff(set_acc) / self._control_time_step
            if max_time is None or max_time >= set_time[-1]:
                set_time_max_index = len(set_time)
            else:
                set_time_max_index = np.argmin(np.asarray(set_time) <= max_time)

            logging.info('Setpoint Distance: ' + str(np.min(set_dis)) + ' / ' + str(np.max(set_dis)) +
                         '; Setpoint Vel: ' + str(np.min(set_vel)) + ' / ' + str(np.max(set_vel)) +
                         '; Setpoint Acc: ' + str(np.min(set_acc)) + ' / ' + str(np.max(set_acc)) +
                         '; Setpoint Jerk: ' + str(np.min(set_jerk)) + ' / ' + str(np.max(set_jerk)))

        line_style = '-'
        marker = '.'
        line_style_actual_value = '--'
        color_actual = 'C0'
        color_set = 'C1'

        timesteps_per_control_step = int(round(self._time_step / self._control_time_step))

        if ax_pos is not None:

            if set_dis:
                ax[ax_pos].plot(set_time[:set_time_max_index], set_dis[:set_time_max_index], color=color_set,
                                linestyle=line_style, label='_nolegend_')
                sub_time = set_time[0:set_time_max_index:timesteps_per_control_step]
                sub_set_dis = set_dis[0:set_time_max_index:timesteps_per_control_step]
                ax[ax_pos].plot(sub_time, sub_set_dis, color=color_set, marker=marker,
                                linestyle='None', label='_nolegend_')

            if actual_dis and self._plot_actual_values:
                ax[ax_pos].plot(actual_time[:actual_time_max_index], actual_dis[:actual_time_max_index],
                                color=color_actual,
                                linestyle=line_style_actual_value, label='_nolegend_')
                sub_time = actual_time[0:actual_time_max_index:timesteps_per_control_step]
                sub_actual_dis = actual_dis[0:actual_time_max_index:timesteps_per_control_step]
                ax[ax_pos].plot(sub_time, sub_actual_dis, color=color_actual, marker=marker,
                                linestyle='None', label='_nolegend_')

        if ax_vel is not None:
            if set_dis:
                vel_time = np.array(
                    set_time[:min(len(set_time) - 1, set_time_max_index)]) + 0.5 * self._control_time_step
                vel_plot = list(set_vel[:min(len(set_time) - 1, set_time_max_index)])
                ax[ax_vel].plot(vel_time, vel_plot, color=color_set,
                                linestyle=line_style, label='_nolegend_')
                sub_time = actual_time[timesteps_per_control_step:set_time_max_index:timesteps_per_control_step]
                sub_set_vel = np.interp(sub_time, vel_time, vel_plot)
                ax[ax_vel].plot(sub_time, sub_set_vel, color=color_set, marker=marker,
                                linestyle='None', label='_nolegend_')

            if actual_dis and self._plot_actual_values:
                vel_time = np.array(
                    actual_time[:min(len(actual_time) - 1, actual_time_max_index)]) + 0.5 * self._control_time_step
                vel_plot = list(actual_vel[:min(len(actual_time) - 1, actual_time_max_index)])
                ax[ax_vel].plot(vel_time, vel_plot, color=color_actual,
                                linestyle=line_style_actual_value, label='_nolegend_')
                sub_time = actual_time[timesteps_per_control_step:actual_time_max_index:timesteps_per_control_step]
                sub_actual_vel = np.interp(sub_time, vel_time, vel_plot)
                ax[ax_vel].plot(sub_time, sub_actual_vel, color=color_actual, marker=marker,
                                linestyle='None', label='_nolegend_')

        if ax_acc is not None:

            if set_dis:
                acc_time = set_time[1:min(len(set_time) - 1, set_time_max_index)]
                acc_plot = list(set_acc[:min(len(set_acc), set_time_max_index - 1)])
                ax[ax_acc].plot(acc_time, acc_plot,
                                color=color_set, linestyle=line_style, label='_nolegend_')
                sub_time = actual_time[timesteps_per_control_step:actual_time_max_index:timesteps_per_control_step]
                sub_set_acc = np.interp(sub_time, acc_time, acc_plot)
                ax[ax_acc].plot(sub_time, sub_set_acc, color=color_set, marker=marker,
                                linestyle='None', label='_nolegend_')

            if actual_dis and self._plot_actual_values:
                acc_time = actual_time[1:min(len(actual_time) - 1, actual_time_max_index)]
                acc_plot = list(actual_acc[:min(len(actual_acc), actual_time_max_index - 1)])
                ax[ax_acc].plot(acc_time, acc_plot,
                                color=color_actual, linestyle=line_style_actual_value, label='_nolegend_')
                sub_time = actual_time[timesteps_per_control_step:actual_time_max_index:timesteps_per_control_step]
                sub_actual_acc = np.interp(sub_time, acc_time, acc_plot)
                ax[ax_acc].plot(sub_time, sub_actual_acc, color=color_actual, marker=marker,
                                linestyle='None', label='_nolegend_')

        if ax_jerk is not None:
            if set_dis:
                jerk_time = np.array(set_time[:min(len(set_time) - 3,
                                                   set_time_max_index - 2)] + 1.5 * self._control_time_step)
                jerk_plot = list(set_jerk[:min(len(set_jerk), set_time_max_index - 2)])
                ax[ax_jerk].plot(jerk_time, jerk_plot,
                                 color=color_set, linestyle=line_style, label='_nolegend_')
                if timesteps_per_control_step == 1:
                    sub_time = set_time[2:set_time_max_index:timesteps_per_control_step]
                else:
                    sub_time = set_time[timesteps_per_control_step:set_time_max_index:timesteps_per_control_step]
                sub_set_jerk = np.interp(sub_time, jerk_time, jerk_plot)
                ax[ax_jerk].plot(sub_time, sub_set_jerk, color=color_set, marker=marker,
                                 linestyle='None', label='_nolegend_')

            if actual_dis and self._plot_actual_values:

                jerk_time = np.array(actual_time[:min(len(actual_time) - 3,
                                                      actual_time_max_index - 2)] + 1.5 * self._control_time_step)
                jerk_plot = list(actual_jerk[:min(len(actual_jerk), actual_time_max_index - 2)])
                ax[ax_jerk].plot(jerk_time, jerk_plot,
                                 color=color_actual, linestyle=line_style_actual_value, label='_nolegend_')
                if timesteps_per_control_step == 1:
                    sub_time = actual_time[2:actual_time_max_index:timesteps_per_control_step]
                else:
                    sub_time = actual_time[timesteps_per_control_step:actual_time_max_index:timesteps_per_control_step]
                sub_actual_jerk = np.interp(sub_time, jerk_time, jerk_plot)
                ax[ax_jerk].plot(sub_time, sub_actual_jerk, color=color_actual, marker=marker,
                                 linestyle='None', label='_nolegend_')

        for i in range(len(ax)):

            if self._plot_time_limits is None:
                ax[i].set_xlim([0, self._time[-1]])
            else:
                ax[i].set_xlim([0, self._plot_time_limits[1] - self._plot_time_limits[0]])

        fig.align_ylabels(ax)

    def _plot_joint_data(self, max_time=None, clear_axes=True, line_style='-'):
        if clear_axes:
            self._clear_axes()
        fig = self._fig
        ax = self._ax
        ax_pos = self._ax_indices[0]
        ax_vel = self._ax_indices[1]
        ax_acc = self._ax_indices[2]
        ax_jerk = self._ax_indices[3]
        ax_torque = self._ax_indices[4]
        ax_value = self._ax_indices[5]

        if ax_torque is not None or ax_value is not None:
            adjust_plot = False
            if ax_torque is not None and not ax[ax_torque].get_visible():
                ax[ax_torque].set_visible(True)
                adjust_plot = True
            if ax_value is not None and not ax[ax_value].get_visible():
                ax[ax_value].set_visible(True)
                adjust_plot = True
            if adjust_plot:
                plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.94, wspace=0.15, hspace=0.15)

        for i in range(len(ax)):
            ax[i].grid(True)
        ax[-1].set_xlabel('Time [s]')

        if ax_pos is not None:
            ax[ax_pos].set_ylabel('Position')

        if ax_vel is not None:
            ax[ax_vel].set_ylabel('Velocity')

        if ax_jerk is not None:
            ax[ax_jerk].set_ylabel('Jerk')

        if ax_acc is not None:
            ax[ax_acc].set_ylabel('Acceleration')

        if ax_torque is not None:
            ax[ax_torque].set_ylabel('Torque')

        if ax_value is not None:
            ax[ax_value].set_ylabel('Value')

        joint_pos = np.swapaxes(self._pos, 0, 1)  # swap time and joint index dimension for plotting
        joint_vel = np.swapaxes(self._vel, 0, 1)
        joint_acc = np.swapaxes(self._acc, 0, 1)
        joint_jerk = np.swapaxes(self._jerk, 0, 1)

        if self._plot_acc_limits:
            # [time][joint_index][min/max] -> [joint_index][min/max][time]
            joint_acc_limits = np.swapaxes(self._current_acc_limits, 0, 1)
            joint_acc_limits = np.swapaxes(joint_acc_limits, 1, 2)

        joint_sub_pos = np.swapaxes(self._sub_pos, 0, 1)  # swap time and joint index dimension for plotting
        joint_sub_vel = np.swapaxes(self._sub_vel, 0, 1)
        joint_sub_acc = np.swapaxes(self._sub_acc, 0, 1)
        joint_sub_jerk = np.swapaxes(self._sub_jerk, 0, 1)

        actual_time = [0.0]
        computed_actual_time = [0.0]
        baseline_time = [0.0]
        actual_time_torque = [0.0]

        if self._actual_pos:
            actual_time = np.arange(0, len(self._actual_pos)) * self._control_time_step
            joint_actual_pos = np.swapaxes(self._actual_pos, 0, 1)

        if self._plot_computed_actual_values and self._computed_actual_pos:
            computed_actual_time = np.arange(0, len(self._computed_actual_pos)) * self._simulation_time_step
            joint_computed_actual_pos = np.swapaxes(self._computed_actual_pos, 0, 1)
            joint_computed_actual_vel = np.swapaxes(self._computed_actual_vel, 0, 1)
            joint_computed_actual_acc = np.swapaxes(self._computed_actual_acc, 0, 1)

        if self._baseline_pos:
            baseline_time = np.arange(0, len(self._baseline_pos)) * self._simulation_time_step
            joint_baseline_pos = np.swapaxes(normalize(self._baseline_pos, self._pos_limits_min_max), 0, 1)
            joint_baseline_vel = np.swapaxes(normalize(self._baseline_vel, self._vel_limits_min_max), 0, 1)
            joint_baseline_acc = np.swapaxes(normalize(self._baseline_acc, self._acc_limits_min_max), 0, 1)
            baseline_jerk = (np.diff(np.swapaxes(self._baseline_acc, 0, 1)) / self._simulation_time_step).T
            joint_baseline_jerk = np.swapaxes(normalize(baseline_jerk, self._jerk_limits_min_max), 0, 1)

        if self._plot_actual_torques and self._actual_torque:
            actual_time_torque = np.arange(0, len(self._actual_torque)) * self._control_time_step
            joint_actual_torque = np.swapaxes(self._actual_torque, 0, 1)

        marker = '.'
        line_style_actual_value = '--'
        line_style_computed_actual_value = '-.'

        use_computed_actual_value_setup_color = True

        if self._plot_time_limits is not None:
            # shift time so that the plot starts at t=0
            self._time = np.asarray(self._time) - self._plot_time_limits[0]
            self._sub_time = np.asarray(self._sub_time) - self._plot_time_limits[0]
            if self._actual_pos:
                actual_time = actual_time - self._plot_time_limits[0]

        if max_time is None or max_time >= self._time[-1]:
            time_max_index = len(self._time)
        else:
            time_max_index = np.argmin(np.asarray(self._time) <= max_time)
        if max_time is None or max_time >= self._sub_time[-1]:
            sub_time_max_index = len(self._sub_time)
        else:
            sub_time_max_index = np.argmin(np.asarray(self._sub_time) <= max_time)

        if self._actual_pos:
            if max_time is None or max_time >= actual_time[-1]:
                actual_time_max_index = len(actual_time)
            else:
                actual_time_max_index = np.argmin(np.asarray(actual_time) <= max_time)

        if self._plot_actual_torques and self._actual_torque:
            if max_time is None or max_time >= actual_time_torque[-1]:
                actual_time_torque_max_index = len(actual_time_torque)
            else:
                actual_time_torque_max_index = np.argmin(np.asarray(actual_time_torque) <= max_time)

        if len(self._pos) > 1:
            for j in range(self._num_joints):
                torque_string = ' Actual Torque: ' + str(normalize(np.min(joint_actual_torque[j]),
                                                                   self._torque_limits[j])) \
                                + ' / ' + str(normalize(np.max(joint_actual_torque[j]), self._torque_limits[j])) \
                                + ';' if self._plot_actual_torques and self._actual_torque else ""
                logging.info('Joint ' + str(j + 1) + ' (min/max)' + torque_string +
                             ' Jerk: ' + str(np.min(joint_sub_jerk[j])) + ' / ' + str(np.max(joint_sub_jerk[j])) +
                             '; Acc: ' + str(np.min(joint_sub_acc[j])) + ' / ' + str(np.max(joint_sub_acc[j])) +
                             '; Vel: ' + str(np.min(joint_sub_vel[j])) + ' / ' + str(np.max(joint_sub_vel[j])) +
                             '; Pos: ' + str(np.min(joint_sub_pos[j])) + ' / ' + str(np.max(joint_sub_pos[j])))

        if len(self._baseline_pos) > 1:
            for j in range(self._num_joints):
                logging.info('Joint ' + str(j + 1) +
                             ' Jerk: ' + str(np.min(joint_baseline_jerk[j])) + ' / '
                             + str(np.max(joint_baseline_jerk[j])) +
                             '; Acc: ' + str(np.min(joint_baseline_acc[j])) + ' / '
                             + str(np.max(joint_baseline_acc[j])) +
                             '; Vel: ' + str(np.min(joint_baseline_vel[j])) + ' / '
                             + str(np.max(joint_baseline_vel[j])) +
                             '; Pos: ' + str(np.min(joint_baseline_pos[j])) + ' / '
                             + str(np.max(joint_baseline_pos[j])))

        for j in range(self._num_joints):
            color = 'C' + str(j)  # "C0", "C1" -> index to the default color cycle
            color_actual = color
            color_limits = color
            color_computed_actual_value_setup = color

            if self._plot_joint[j]:
                label = 'Joint ' + str(j + 1)
                if ax_pos is not None:
                    ax[ax_pos].plot(self._time[:time_max_index], joint_pos[j][:time_max_index], color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_pos].plot(self._sub_time[:sub_time_max_index], joint_sub_pos[j][:sub_time_max_index],
                                    color=color, linestyle=line_style, label=label)

                    if self._actual_pos and self._plot_actual_values:
                        actual_pos_plot = normalize(joint_actual_pos[j], self._pos_limits[j])
                        ax[ax_pos].plot(actual_time[:actual_time_max_index], actual_pos_plot[:actual_time_max_index],
                                        color=color_actual, linestyle=line_style_actual_value, label='_nolegend_')

                    if self._plot_computed_actual_values and self._computed_actual_pos:
                        computed_actual_pos_plot = normalize(joint_computed_actual_pos[j], self._pos_limits[j])
                        computed_actual_value_color = color_computed_actual_value_setup \
                            if use_computed_actual_value_setup_color else color
                        ax[ax_pos].plot(computed_actual_time, computed_actual_pos_plot,
                                        color=computed_actual_value_color,
                                        linestyle=line_style_computed_actual_value, label='_nolegend_')

                    if self._baseline_pos:
                        ax[ax_pos].plot(baseline_time, joint_baseline_pos[j],
                                        color=color,
                                        linestyle=line_style, label='_nolegend_')

                if ax_vel is not None:
                    ax[ax_vel].plot(self._time[:time_max_index], joint_vel[j][:time_max_index], color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_vel].plot(self._sub_time[:sub_time_max_index], joint_sub_vel[j][:sub_time_max_index],
                                    color=color, linestyle=line_style, label=label)

                if self._actual_pos and self._plot_actual_values:
                    actual_vel = np.diff(joint_actual_pos[j]) / self._control_time_step
                    actual_vel_plot = normalize(actual_vel, self._vel_limits[j])
                    if ax_vel is not None:
                        ax[ax_vel].plot(np.array(
                            actual_time[:min(len(actual_time) - 1, actual_time_max_index)])
                                        + 0.5 * self._control_time_step,
                                        list(actual_vel_plot[:min(len(actual_time) - 1, actual_time_max_index)]),
                                        color=color_actual, linestyle=line_style_actual_value, label='_nolegend_')

                if self._plot_computed_actual_values and self._computed_actual_pos:
                    computed_actual_vel_plot = normalize(joint_computed_actual_vel[j], self._vel_limits[j])
                    if ax_vel is not None:
                        computed_actual_value_color = color_computed_actual_value_setup \
                            if use_computed_actual_value_setup_color else color
                        ax[ax_vel].plot(computed_actual_time, computed_actual_vel_plot,
                                        color=computed_actual_value_color,
                                        linestyle=line_style_computed_actual_value, label='_nolegend_')

                if self._baseline_pos:
                    if ax_vel is not None:
                        ax[ax_vel].plot(baseline_time, joint_baseline_vel[j],
                                        color=color,
                                        linestyle=line_style, label='_nolegend_')

                if ax_acc is not None:
                    ax[ax_acc].plot(self._time[:time_max_index], joint_acc[j][:time_max_index], color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_acc].plot(self._sub_time[:sub_time_max_index], joint_sub_acc[j][:sub_time_max_index],
                                    color=color, linestyle=line_style, label=label)

                if self._actual_pos and self._plot_actual_values:
                    actual_acc = np.diff(actual_vel) / self._control_time_step
                    actual_acc_plot = normalize(actual_acc, self._acc_limits[j])
                    if ax_acc is not None:
                        ax[ax_acc].plot(actual_time[1:min(len(actual_time) - 1, actual_time_max_index)],
                                        list(actual_acc_plot[:min(len(actual_acc_plot), actual_time_max_index - 1)]),
                                        color=color_actual, linestyle=line_style_actual_value, label='_nolegend_')

                if self._plot_computed_actual_values and self._computed_actual_pos:
                    computed_actual_acc_plot = normalize(joint_computed_actual_acc[j], self._acc_limits[j])
                    if ax_acc is not None:
                        computed_actual_value_color = color_computed_actual_value_setup \
                            if use_computed_actual_value_setup_color else color
                        ax[ax_acc].plot(computed_actual_time, computed_actual_acc_plot,
                                        color=computed_actual_value_color, linestyle=line_style_computed_actual_value,
                                        label='_nolegend_')

                if self._baseline_pos:
                    if ax_acc is not None:
                        ax[ax_acc].plot(baseline_time, joint_baseline_acc[j],
                                        color=color, linestyle=line_style,
                                        label='_nolegend_')

                if ax_acc is not None:
                    if self._plot_acc_limits and len(joint_acc_limits[0][0]) > 1:
                        for i in range(2):
                            ax[ax_acc].plot(self._time[:time_max_index], joint_acc_limits[j][i][:time_max_index],
                                            color=color_limits, linestyle='--', label='_nolegend_')

                if ax_jerk is not None:
                    ax[ax_jerk].plot(self._time[:time_max_index], joint_jerk[j][:time_max_index], color=color,
                                     marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_jerk].plot(self._sub_time[:sub_time_max_index], joint_sub_jerk[j][:sub_time_max_index],
                                     color=color, linestyle=line_style, label=label)

                if self._actual_pos and self._plot_actual_values:
                    actual_jerk = np.diff(actual_acc) / self._control_time_step
                    actual_jerk_plot = normalize(actual_jerk, self._jerk_limits[j])
                    if ax_jerk is not None:
                        ax[ax_jerk].plot(np.array(actual_time[:min(len(actual_time) - 3,
                                                                   actual_time_max_index - 2)]
                                                  + 1.5 * self._control_time_step),
                                         list(actual_jerk_plot[:min(len(actual_jerk_plot), actual_time_max_index - 2)]),
                                         color=color_actual, linestyle=line_style_actual_value, label='_nolegend_')

                if self._plot_computed_actual_values and self._computed_actual_pos:
                    computed_actual_jerk = np.diff(joint_computed_actual_acc[j]) / self._simulation_time_step
                    computed_actual_jerk_plot = normalize(computed_actual_jerk, self._jerk_limits[j])
                    if ax_jerk is not None:
                        computed_actual_value_color = color_computed_actual_value_setup \
                            if use_computed_actual_value_setup_color else color
                        ax[ax_jerk].plot(
                            np.asarray(computed_actual_time[:-1]) + 0.5 * self._simulation_time_step,
                            computed_actual_jerk_plot, color=computed_actual_value_color,
                            linestyle=line_style_computed_actual_value,
                            label='_nolegend_')

                if self._baseline_pos:
                    if ax_jerk is not None:
                        ax[ax_jerk].plot(
                            np.asarray(baseline_time[:-1]) + 0.5 * self._simulation_time_step,
                            joint_baseline_jerk[j], color=color,
                            linestyle=line_style,
                            label='_nolegend_')

                if ax_torque is not None and self._plot_actual_torques and self._actual_torque:
                    actual_torque_plot = normalize(joint_actual_torque[j], self._torque_limits[j])
                    ax[ax_torque].plot(actual_time_torque[:actual_time_torque_max_index],
                                       actual_torque_plot[:actual_time_torque_max_index], color=color_actual,
                                       linestyle=line_style_actual_value, label=label)

        if ax_value is not None:
            last_value_index = min(time_max_index, len(self._value))
            ax[ax_value].plot(self._time[:last_value_index], self._value[:last_value_index], color="black",
                              marker=marker, linestyle=line_style, label='_nolegend_')

        if len(ax) > 1:
            # use solid lines for the legend, although the torques in ax[-1] are visualized with dashed lines
            if ax_value is not None:
                ax_index = -2
            else:
                ax_index = -1
            handles, labels = ax[ax_index - 1].get_legend_handles_labels()
            ax[ax_index].legend(handles, labels, loc='lower right')
        else:
            ax[-1].legend(loc='lower right')

        for i in range(len(ax)):
            if self._plot_time_limits is None:
                ax[i].set_xlim([0, max(self._time[-1], actual_time[-1], computed_actual_time[-1],
                                       baseline_time[-1], actual_time_torque[-1])])
            else:
                ax[i].set_xlim([0, self._plot_time_limits[1] - self._plot_time_limits[0]])

        if ax_pos is not None:
            ax[ax_pos].set_ylim([-1.05, 1.05])
        if ax_vel is not None:
            ax[ax_vel].set_ylim([-1.05, 1.05])
        if ax_acc is not None:
            ax[ax_acc].set_ylim([-1.05, 1.05])
        if ax_jerk is not None:
            ax[ax_jerk].set_ylim([-1.05, 1.05])
        if ax_torque is not None:
            ax[ax_torque].set_ylim([-1.05, 1.05])

        fig.align_ylabels(ax)

    def add_data_point(self, current_acc, acc_range=None, last_pos=None, last_vel=None, last_acc=None):
        self._time.append(self._time[-1] + self._time_step)
        if last_acc is None:
            last_acc = self._current_acc.copy()
        if last_vel is None:
            last_vel = self._current_vel.copy()
        if last_pos is None:
            last_pos = self._current_pos.copy()
        self._current_acc = current_acc
        self._current_jerk = (self._current_acc - last_acc) / self._time_step
        self._jerk.append(normalize_fast(self._current_jerk, self._jerk_limits_min_max))

        if acc_range is not None:
            self._current_acc_limits.append(normalize_batch(acc_range.T, self._acc_limits_min_max).T)

        for j in range(1, self._plot_num_sub_time_steps + 1):
            self._sub_jerk.append(self._jerk[-1])
        time_since_start = np.linspace(self._time_step / self._plot_num_sub_time_steps, self._time_step,
                                       self._plot_num_sub_time_steps)
        self._sub_time.extend(list(time_since_start + self._time_step_counter * self._time_step))
        sub_current_acc = interpolate_acceleration_batch(last_acc, self._current_acc, time_since_start, self._time_step)
        sub_current_vel = interpolate_velocity_batch(last_acc, self._current_acc, last_vel, time_since_start,
                                                     self._time_step)
        sub_current_pos = interpolate_position_batch(last_acc, self._current_acc, last_vel, last_pos, time_since_start,
                                                     self._time_step)

        self._current_vel = sub_current_vel[-1]
        self._current_pos = sub_current_pos[-1]

        self._sub_acc.extend(list(normalize_batch(sub_current_acc, self._acc_limits_min_max)))
        self._sub_vel.extend(list(normalize_batch(sub_current_vel, self._vel_limits_min_max)))
        self._sub_pos.extend(list(normalize(sub_current_pos, self._pos_limits_min_max)))

        self._acc.append(self._sub_acc[-1])
        self._vel.append(self._sub_vel[-1])
        self._pos.append(self._sub_pos[-1])

        self._time_step_counter = self._time_step_counter + 1

    def add_actual_position(self, actual_position):
        self._actual_pos.append(actual_position)

    def add_actual_torque(self, actual_torque):
        self._actual_torque.append(list(actual_torque))

    def add_computed_actual_value(self, computed_position, computed_velocity, computed_acceleration):
        self._computed_actual_pos.append(computed_position)
        self._computed_actual_vel.append(computed_velocity)
        self._computed_actual_acc.append(computed_acceleration)

    def add_baseline_value(self, baseline_position, baseline_velocity, baseline_acceleration):
        self._baseline_pos.append(baseline_position)
        self._baseline_vel.append(baseline_velocity)
        self._baseline_acc.append(baseline_acceleration)

    def add_value(self, value):
        self._value.append(value)

    def save_obstacle_data(self, class_name, experiment_name):
        if self._obstacle_wrapper is not None:
            if self._evaluation_dir is None:
                eval_dir = os.path.join(Path.home(), "safe_motions_risk_evaluation", "trajectory_plotter", "logs",
                                        class_name, experiment_name, self._timestamp)
            else:
                eval_dir = os.path.join(self._evaluation_dir, "trajectory_plotter")

            if not os.path.exists(eval_dir):
                try:
                    os.makedirs(eval_dir)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            with open(os.path.join(eval_dir, "episode_" + str(self._episode_counter) + "_obstacle_data.pkl"),
                      'wb') as f:
                pickle.dump(self._obstacle_wrapper, f)

    def load_obstacle_data(self, path_to_obstacle_data):
        if os.path.exists(path_to_obstacle_data):
            with open(path_to_obstacle_data, 'rb') as f:
                self._obstacle_wrapper = pickle.load(f)
        else:
            raise

    def save_trajectory(self, class_name, experiment_name):
        trajectory_dict = {}
        trajectory_dict['pos'] = np.asarray(self._pos).tolist()
        trajectory_dict['vel'] = np.asarray(self._vel).tolist()
        trajectory_dict['acc'] = np.asarray(self._acc).tolist()
        trajectory_dict['jerk'] = np.asarray(self._jerk).tolist()

        trajectory_dict['current_acc_limits'] = np.asarray(self._current_acc_limits).tolist()

        trajectory_dict['sub_pos'] = np.asarray(self._sub_pos).tolist()
        trajectory_dict['sub_vel'] = np.asarray(self._sub_vel).tolist()
        trajectory_dict['sub_acc'] = np.asarray(self._sub_acc).tolist()
        trajectory_dict['sub_jerk'] = np.asarray(self._sub_jerk).tolist()

        trajectory_dict['control_time_step'] = float(self._control_time_step)
        trajectory_dict['time_step'] = float(self._time_step)

        trajectory_dict['actual_pos'] = np.asarray(self._actual_pos).tolist()
        trajectory_dict['actual_torque'] = np.asarray(self._actual_torque).tolist()

        trajectory_dict['time'] = np.asarray(self._time).tolist()
        trajectory_dict['sub_time'] = np.asarray(self._sub_time).tolist()

        trajectory_dict['pos_limits'] = self._pos_limits.tolist()
        trajectory_dict['vel_limits'] = self._vel_limits.tolist()
        trajectory_dict['acc_limits'] = self._acc_limits.tolist()
        trajectory_dict['jerk_limits'] = self._jerk_limits.tolist()
        trajectory_dict['torque_limits'] = self._torque_limits.tolist()

        trajectory_dict['baseline_pos'] = np.asarray(self._baseline_pos).tolist()
        trajectory_dict['baseline_vel'] = np.asarray(self._baseline_vel).tolist()
        trajectory_dict['baseline_acc'] = np.asarray(self._baseline_acc).tolist()

        if self._evaluation_dir is None:
            eval_dir = os.path.join(Path.home(), "safe_motions_risk_evaluation", "trajectory_plotter", "logs",
                                    class_name, experiment_name, self._timestamp)
        else:
            eval_dir = os.path.join(self._evaluation_dir, "trajectory_plotter")

        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        with open(os.path.join(eval_dir, "episode_" + str(self._episode_counter) + ".json"), 'w') as f:
            f.write(json.dumps(trajectory_dict, default=np_encoder))
            f.flush()

    def load_trajectory(self, path_to_trajectory):

        if os.path.exists(path_to_trajectory):
            with open(path_to_trajectory) as f:
                trajectory_dict = json.load(f)
        else:
            raise

        self._pos = trajectory_dict['pos']
        self._vel = trajectory_dict['vel']
        self._acc = trajectory_dict['acc']
        self._jerk = trajectory_dict['jerk']

        self._current_acc_limits = trajectory_dict['current_acc_limits']

        self._sub_pos = trajectory_dict['sub_pos']
        self._sub_vel = trajectory_dict['sub_vel']
        self._sub_acc = trajectory_dict['sub_acc']
        self._sub_jerk = trajectory_dict['sub_jerk']

        self._control_time_step = trajectory_dict['control_time_step']
        if 'time_step' in trajectory_dict:
            self._time_step = trajectory_dict['time_step']
        self._actual_pos = trajectory_dict['actual_pos']

        if 'actual_torque' in trajectory_dict:
            self._actual_torque = trajectory_dict['actual_torque']

        if 'torque_limits' in trajectory_dict:
            self._torque_limits = trajectory_dict['torque_limits']

        self._baseline_pos = trajectory_dict['baseline_pos'] if 'baseline_pos' in trajectory_dict else []
        self._baseline_vel = trajectory_dict['baseline_vel'] if 'baseline_vel' in trajectory_dict else []
        self._baseline_acc = trajectory_dict['baseline_acc'] if 'baseline_acc' in trajectory_dict else []

        self._time = trajectory_dict['time']
        self._sub_time = trajectory_dict['sub_time']

        self._num_joints = len(self._pos[0])

        self._pos_limits = trajectory_dict['pos_limits']
        self._vel_limits = trajectory_dict['vel_limits']
        self._acc_limits = trajectory_dict['acc_limits']
        self._jerk_limits = trajectory_dict['jerk_limits']


def normalize(value, value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    continuous_joint_indices = np.isnan(value_range[0]) | np.isnan(value_range[1])
    if np.any(continuous_joint_indices):
        # continuous joint -> map [-np.pi, np.pi] and all values shifted by 2 * np.pi to [-1, 1]
        if np.array(value).ndim == 2:  # batch computation
            normalized_value[:, continuous_joint_indices] = \
                -1 + 2 * (((value[:, continuous_joint_indices] + np.pi) / (2 * np.pi)) % 1)
        else:
            normalized_value[continuous_joint_indices] = \
                -1 + 2 * (((value[continuous_joint_indices] + np.pi)/(2 * np.pi)) % 1)
    return normalized_value


def denormalize(norm_value, value_range):
    if np.isnan(value_range[0]) or np.isnan(value_range[1]):  # continuous joint
        value_range[0] = -np.pi
        value_range[1] = np.pi
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


class ButtonArray(matplotlib.widgets.Slider):

    def __init__(self, ax, label, entries_str, val_init=0, **kwargs):

        self.color_enabled = "white"
        self.color_active = "orange"
        self.color_disabled = "lightgray"
        self.font_size = 10

        self.num_buttons = len(entries_str)

        super(ButtonArray, self).__init__(ax, label, 0, self.num_buttons,
                                          valinit=val_init, valfmt='%1d', **kwargs)

        self._last_val_int = int(val_init)

        self.poly.set_visible(False)
        self.page_buttons = []
        self._active_values = []
        self.vline.set_visible(False)
        for i in range(self.num_buttons):
            button_color = self.color_active if i == val_init else self.color_enabled
            b = matplotlib.patches.Rectangle((float(i) / self.num_buttons, 0), 1. / self.num_buttons, 1,
                                             transform=ax.transAxes, facecolor=button_color)
            ax.add_artist(b)
            self.page_buttons.append(b)
            self._active_values.append(i)
            ax_text = entries_str[i]
            ax.text(float(i) / self.num_buttons + 0.5 / self.num_buttons, 0.45, ax_text,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.font_size)
        self.valtext.set_visible(False)

    def set_active_values(self, active_values):
        self._active_values = active_values
        self._colorize()

    def update(self):
        self._update(event=None)

    def _update(self, event):
        if event:
            super(ButtonArray, self)._update(event)
        i = int(self.val)
        if i >= self.valmax or i not in self._active_values:
            return
        self._last_val_int = i
        self._colorize()

    def _colorize(self):
        for j in range(self.num_buttons):
            if j in self._active_values:
                self.page_buttons[j].set_facecolor(self.color_enabled)
            else:
                self.page_buttons[j].set_facecolor(self.color_disabled)
        if self._last_val_int not in self._active_values:
            self._last_val_int = self._active_values[0]

        self.page_buttons[self._last_val_int].set_facecolor(self.color_active)

    @property
    def last_val_int(self):
        return self._last_val_int

    @property
    def active_values(self):
        return self._active_values

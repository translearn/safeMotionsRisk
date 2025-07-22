# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os
os.environ["LC_NUMERIC"] = "en_US.UTF-8"  # avoid wrong parsing of urdf files caused by localization (, vs .)
import re
import numpy as np
import pybullet as p

from safemotions.robot_scene.collision_torque_limit_prevention import ObstacleWrapperSim


class RobotSceneBase(object):
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    URDF_DIR = os.path.join(MODULE_DIR, "description", "urdf")
    JOINT_LIMITS_SAFETY_BUFFER_IIWA = 0.035
    MAX_ACCELERATION_IIWA = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
    MAX_JERK_IIWA = [7500, 3750, 5000, 6250, 7500, 10000, 10000]
    # Human
    MAX_ACCELERATION_HUMAN_ARM = [15.0, 15.0, 15.0, 15.0]
    MAX_JERK_HUMAN_ARM = [7500, 7500, 7500, 7500]

    FIFO_PATH = None

    def __init__(self,
                 simulation_client_id=None,
                 simulation_time_step=None,
                 obstacle_client_id=None,
                 backup_client_id=None,
                 gui_client_id=None,
                 trajectory_time_step=None,
                 trajectory_duration=None,
                 use_real_robot=False,
                 robot_scene=0,
                 obstacle_scene=0,
                 activate_obstacle_collisions=False,
                 observed_link_point_scene=0,
                 log_obstacle_data=False,
                 visual_mode=False,
                 capture_frame_function=None,
                 visualize_bounding_spheres=False,
                 visualize_debug_lines=False,
                 acc_range_function=None,
                 acc_braking_function=None,
                 violation_code_function=None,
                 collision_check_time=None,
                 check_braking_trajectory_collisions=False,
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
                 moving_object_high_launch_angle_probability=1.0,
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
                 terminate_on_collision_with_moving_obstacle=False,
                 ball_machine_mode=False,
                 collision_avoidance_mode=False,
                 collision_avoidance_kinematic_state_sampling_mode=False,
                 collision_avoidance_kinematic_state_sampling_probability=1.0,
                 collision_avoidance_stay_in_state_probability=0.3,
                 always_use_collision_avoidance_starting_point_sampling=False,
                 risk_state_deterministic_backup_trajectory=False,
                 human_network_checkpoint=None,
                 human_network_use_collision_avoidance_starting_point_sampling=False,
                 human_network_collision_avoidance_kinematic_state_sampling_probability=0.3,
                 human_network_collision_avoidance_stay_in_state_probability=0.3,
                 no_self_collision=False,
                 target_link_name=None,
                 target_link_offset=None,
                 pos_limit_factor=1,
                 vel_limit_factor=1,
                 acc_limit_factor=1,
                 jerk_limit_factor=1,
                 torque_limit_factor=1,
                 use_controller_target_velocities=False,
                 reward_maximum_relevant_distance=None,
                 reward_consider_moving_obstacles=False,
                 obstacle_client_update_steps_per_action=24,
                 no_link_coloring=False,
                 no_target_link_coloring=False,
                 static_robot=False,
                 print_link_infos=False,
                 do_not_execute_robot_movement=False,
                 use_fixed_seed=False,
                 **kwargs):

        self._simulation_client_id = simulation_client_id
        self._simulation_time_step = simulation_time_step
        self._obstacle_client_id = obstacle_client_id
        self._backup_client_id = backup_client_id
        self._gui_client_id = gui_client_id
        self._use_real_robot = use_real_robot
        self._trajectory_time_step = trajectory_time_step
        self._trajectory_duration = trajectory_duration
        self.capture_frame_function = capture_frame_function
        self._visual_mode = visual_mode
        self._collision_avoidance_mode = collision_avoidance_mode
        self._collision_avoidance_kinematic_state_sampling_mode = collision_avoidance_kinematic_state_sampling_mode
        self._collision_avoidance_kinematic_state_sampling_probability = \
            collision_avoidance_kinematic_state_sampling_probability
        self._collision_avoidance_stay_in_state_probability = collision_avoidance_stay_in_state_probability
        self._always_use_collision_avoidance_starting_point_sampling = \
            always_use_collision_avoidance_starting_point_sampling
        self._risk_state_deterministic_backup_trajectory = risk_state_deterministic_backup_trajectory
        self._human_network_checkpoint = human_network_checkpoint
        self._human_network_use_collision_avoidance_starting_point_sampling = \
            human_network_use_collision_avoidance_starting_point_sampling
        self._human_network_collision_avoidance_kinematic_state_sampling_probability = \
            human_network_collision_avoidance_kinematic_state_sampling_probability
        self._human_network_collision_avoidance_stay_in_state_probability = \
            human_network_collision_avoidance_stay_in_state_probability
        self._ball_machine_mode = ball_machine_mode
        self._do_not_execute_robot_movement = do_not_execute_robot_movement
        self._use_fixed_seed = use_fixed_seed

        self._num_clients = 0
        if self._simulation_client_id is not None:
            self._num_clients += 1
        if self._obstacle_client_id is not None:
            self._num_clients += 1
        if self._backup_client_id is not None:
            self._num_clients += 1

        self._no_self_collision = no_self_collision

        self._num_robots = None
        self._robot_scene = robot_scene
        self._shared_link_names = []

        robot_urdf = None
        robot_base_position = [0, 0, 0]
        robot_base_orientation = (0, 0, 0, 1)

        self._plane_z_offset = 0

        if robot_scene == 0:
            self._num_robots = 1
            robot_urdf = "robot_ball_machine" if self._ball_machine_mode else "robot"
            self._plane_z_offset = -0.94
            self._robot_name = "iiwa7"

        if robot_scene == 9:
            self._num_robots = 2
            self._plane_z_offset = -0.94
            robot_base_position = [0.7, 0, -0.94]
            robot_base_orientation = p.getQuaternionFromEuler([0, 0, np.pi])
            if robot_scene == 9:
                self._robot_name = "human"
                robot_urdf = "human"
            self._shared_link_names = ['shoes', 'lower_legs', 'upper_legs', 'body', 'head']

        self._target_link_name = target_link_name

        if self._target_link_name is None:
            if self._ball_machine_mode:
                self._target_link_name = "ball_machine"
            else:
                if self._robot_name == "iiwa7":
                    self._target_link_name = "iiwa_link_7"
                elif self._robot_name.startswith("human"):
                    self._target_link_name = "hand"

        if target_link_offset is None:
            if self._ball_machine_mode:
                if not use_target_points:
                    target_link_offset = [0, 0, 0]
                else:
                    target_link_offset = [0, 0, 0.10]
            else:
                if self._robot_name == "iiwa7":
                    target_link_offset = [0, 0, 0.126]
                elif self._robot_name.startswith("human"):
                    target_link_offset = [0.0, 0.0, -0.185]
                else:
                    target_link_offset = [0, 0, 0]

        self._risk_color_link_index = 0  # link index to visualize the risk if visualize_risk is True

        self._use_moving_objects = use_moving_objects
        self._moving_object_area_center = moving_object_area_center
        self._moving_object_area_width_height = moving_object_area_width_height
        self._moving_object_speed_meter_per_second = moving_object_speed_meter_per_second
        self._moving_object_aim_at_current_robot_position = moving_object_aim_at_current_robot_position
        self._moving_object_check_invalid_target_link_point_positions = \
            moving_object_check_invalid_target_link_point_positions
        self._moving_object_high_launch_angle_probability = moving_object_high_launch_angle_probability
        self._moving_object_random_initial_position = moving_object_random_initial_position

        self._moving_object_current_robot_position_target_link_list = None
        self._moving_object_observed_link_names = []
        self._planet_observed_link_names = []
        # observed links to calculate a minimum distance for the reward calculation
        if self._robot_name == "iiwa7":
            self._moving_object_observed_link_names = ["iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                                       "iiwa_link_6", "iiwa_link_7"]
            if self.ball_machine_mode:
                self._moving_object_observed_link_names += [self._target_link_name]

        if self._moving_object_aim_at_current_robot_position:
            if self._robot_name == "iiwa7":
                self._moving_object_current_robot_position_target_link_list = ["iiwa_link_3", "iiwa_link_4",
                                                                               "iiwa_link_5"]
            if self._moving_object_current_robot_position_target_link_list is None:
                raise ValueError("Robot position target link list needs to be defined")

        if self._moving_object_area_center is None:
            self._moving_object_area_center = [5, 0, 0.5]

        if self._moving_object_area_width_height is None:
            self._moving_object_area_width_height = [1, 1]

        self._moving_object_sphere_center = np.array(moving_object_sphere_center) \
            if moving_object_sphere_center is not None else None
        self._moving_object_sphere_radius = moving_object_sphere_radius
        if self._moving_object_sphere_radius is None:
            self._moving_object_sphere_radius = 5
        self._moving_object_sphere_height_min_max = moving_object_sphere_height_min_max
        self._moving_object_sphere_angle_min_max = moving_object_sphere_angle_min_max
        if self._moving_object_sphere_angle_min_max is None:
            self._moving_object_sphere_angle_min_max = [0, 2 * np.pi]

        if self._moving_object_sphere_height_min_max is None:
            self._moving_object_sphere_height_min_max = [-0.1 * self._moving_object_sphere_radius,
                                                         0.1 * self._moving_object_sphere_radius]

        self._planet_mode = planet_mode

        # center: "[-0.1, 0.0, 0.8]"
        self._planet_one_center = np.array(planet_one_center) if planet_one_center is not None else None
        self._planet_one_radius_xy = np.array(planet_one_radius_xy) \
            if planet_one_radius_xy is not None else np.array([0.65, 0.8])
        self._planet_one_euler_angles = np.array(planet_one_euler_angles) \
            if planet_one_euler_angles is not None else np.array([0.35, 0, 0])
        self._planet_one_period = planet_one_period \
            if planet_one_period is not None else 5.0

        self._planet_two_center = np.array(planet_two_center) if planet_two_center is not None else None
        self._planet_two_radius_xy = np.array(planet_two_radius_xy) \
            if planet_two_radius_xy is not None else np.array([0.75, 0.8])
        self._planet_two_euler_angles = np.array(planet_two_euler_angles) \
            if planet_two_euler_angles is not None else np.array([-0.35, 0, 0])

        self._planet_two_time_shift = planet_two_time_shift if self._planet_one_center is not None else None
        if self._planet_two_time_shift is not None:
            self._planet_two_period = self._planet_one_period  # periods are coupled
        else:
            self._planet_two_period = planet_two_period \
                if planet_two_period is not None else 5.0

        self._planet_obs_global_pos_min_max = \
            np.array([[-0.9, 0.7], [-0.85, 0.85], [0.45, 1.2]]).T  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        self._planet_observed_link_names = []
        # observed links to calculate a minimum distance for the reward calculation
        if self._robot_name == "iiwa7":
            self._planet_observed_link_names = ["iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                                "iiwa_link_6", "iiwa_link_7"]
            if self.ball_machine_mode:
                self._planet_observed_link_names += [self._target_link_name]

        self._terminate_on_collision_with_moving_obstacle = terminate_on_collision_with_moving_obstacle

        self._no_link_coloring = no_link_coloring
        self._no_target_link_coloring = no_target_link_coloring
        self._static_robot = static_robot

        urdf_path = os.path.join(self.URDF_DIR, robot_urdf + ".urdf")

        for i in range(self._num_clients):
            fixed_base = True
            if self._no_self_collision:
                self._robot_id = p.loadURDF(urdf_path,
                                            basePosition=robot_base_position,
                                            baseOrientation=robot_base_orientation,
                                            useFixedBase=fixed_base,
                                            physicsClientId=i)
            else:
                self._robot_id = p.loadURDF(urdf_path,
                                            basePosition=robot_base_position,
                                            baseOrientation=robot_base_orientation,
                                            useFixedBase=fixed_base,
                                            flags=p.URDF_USE_SELF_COLLISION,
                                            physicsClientId=i)

        if print_link_infos:
            self._print_link_infos()

        for i in range(self._num_clients):
            half_extents = [10, 15, 0.0025]
            shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                               rgbaColor=(0.9, 0.9, 0.9, 1), physicsClientId=i)
            shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                     physicsClientId=i)

            self._plane_id = p.createMultiBody(baseMass=0, basePosition=[0, 0, self._plane_z_offset],
                                               baseOrientation=(0.0, 0.0, 0.0, 1.0),
                                               baseVisualShapeIndex=shape_visual,
                                               baseCollisionShapeIndex=shape_collision,
                                               physicsClientId=i)

        joint_lower_limits, joint_upper_limits, force_limits, velocity_limits = [], [], [], []

        self._manip_joint_indices, self._manip_joint_indices_per_robot = self._get_manip_joint_indices()
        self._num_manip_joints = len(self._manip_joint_indices)
        self._default_position = [0.0] * self._num_manip_joints

        self._link_name_list = []

        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            self._link_name_list.append(
                str(joint_info[12])[2:-1])  # link name is loaded as  b'linkname' -> extract linkname

        for i in self._manip_joint_indices:
            joint_infos = p.getJointInfo(self._robot_id, i)
            if self._robot_name == "iiwa7":
                joint_limits_safety_buffer = self.JOINT_LIMITS_SAFETY_BUFFER_IIWA
            else:
                joint_limits_safety_buffer = 0
            if joint_infos[8] == 0 and joint_infos[9] == -1.0:
                # continuous joint
                joint_lower_limits.append(np.nan)
                joint_upper_limits.append(np.nan)
            else:
                joint_lower_limits.append(joint_infos[8] + joint_limits_safety_buffer)
                joint_upper_limits.append(joint_infos[9] - joint_limits_safety_buffer)
            force_limits.append(joint_infos[10])
            velocity_limits.append(joint_infos[11])
        self._initial_joint_lower_limits = np.array(joint_lower_limits)
        self._initial_joint_upper_limits = np.array(joint_upper_limits)
        self._initial_max_torques = np.array(force_limits)
        self._initial_max_velocities = np.array(velocity_limits)
        if self._robot_name == "iiwa7":
            self._initial_max_accelerations = np.array(self.MAX_ACCELERATION_IIWA * self._num_robots)
            self._initial_max_jerk = np.array(self.MAX_JERK_IIWA * self._num_robots)
        if self._robot_name.startswith("human"):
            self._initial_max_accelerations = np.array(self.MAX_ACCELERATION_HUMAN_ARM * self._num_robots)
            self._initial_max_jerk = np.array(self.MAX_JERK_HUMAN_ARM * self._num_robots)

        self._deactivate_self_collision_for_adjoining_links()

        self._obstacle_wrapper = \
            ObstacleWrapperSim(robot_scene=self,
                               simulation_client_id=self._simulation_client_id,
                               simulation_time_step=self._simulation_time_step,
                               obstacle_client_id=self._obstacle_client_id,
                               backup_client_id=self._backup_client_id,
                               gui_client_id=self._gui_client_id,
                               trajectory_time_step=self._trajectory_time_step,
                               obstacle_client_update_steps_per_action=obstacle_client_update_steps_per_action,
                               use_real_robot=self._use_real_robot,
                               visual_mode=self._visual_mode,
                               obstacle_scene=obstacle_scene,
                               activate_obstacle_collisions=activate_obstacle_collisions,
                               observed_link_point_scene=observed_link_point_scene,
                               log_obstacle_data=log_obstacle_data,
                               link_name_list=self._link_name_list,
                               manip_joint_indices=self._manip_joint_indices,
                               acc_range_function=acc_range_function,
                               acc_braking_function=acc_braking_function,
                               violation_code_function=violation_code_function,
                               check_braking_trajectory_collisions=check_braking_trajectory_collisions,
                               check_braking_trajectory_torque_limits=check_braking_trajectory_torque_limits,
                               collision_check_time=collision_check_time,
                               distance_calculation_check_observed_points=distance_calculation_check_observed_points,
                               closest_point_safety_distance=closest_point_safety_distance,
                               observed_point_safety_distance=observed_point_safety_distance,
                               starting_point_cartesian_range_scene=starting_point_cartesian_range_scene,
                               use_target_points=use_target_points,
                               target_point_cartesian_range_scene=target_point_cartesian_range_scene,
                               target_point_relative_pos_scene=target_point_relative_pos_scene,
                               target_point_radius=target_point_radius,
                               target_point_sequence=target_point_sequence,
                               target_point_reached_reward_bonus=target_point_reached_reward_bonus,
                               target_point_use_actual_position=target_point_use_actual_position,
                               target_link_name=self._target_link_name,
                               target_link_offset=target_link_offset,
                               moving_object_sequence=moving_object_sequence,
                               moving_object_active_number_single=moving_object_active_number_single,
                               visualize_bounding_spheres=visualize_bounding_spheres,
                               visualize_debug_lines=visualize_debug_lines,
                               reward_maximum_relevant_distance=reward_maximum_relevant_distance,
                               reward_consider_moving_obstacles=reward_consider_moving_obstacles,
                               )

        self._joint_lower_limits = None
        self._joint_lower_limits_continuous = None
        self._joint_upper_limits = None
        self._joint_upper_limits_continuous = None
        self._max_velocities = None
        self._max_accelerations = None
        self._max_jerk_linear_interpolation = None
        self._max_torques = None

        self._pos_limit_factor = pos_limit_factor
        self._vel_limit_factor = vel_limit_factor
        self._acc_limit_factor = acc_limit_factor
        self._jerk_limit_factor = jerk_limit_factor
        self._torque_limit_factor = torque_limit_factor

        self._use_controller_target_velocities = use_controller_target_velocities

        self._trajectory_index = -1

        self._do_not_copy_keys = ["capture_frame_function"]
        self._recursive_copy_keys = ["_obstacle_wrapper"]

    def compute_actual_joint_limits(self):
        self._joint_lower_limits = list(np.array(self._initial_joint_lower_limits) * self._pos_limit_factor)
        self._joint_lower_limits_continuous = np.where(np.isnan(self._joint_lower_limits), -np.pi,
                                                       self._joint_lower_limits)
        self._joint_upper_limits = list(np.array(self._initial_joint_upper_limits) * self._pos_limit_factor)
        self._joint_upper_limits_continuous = np.where(np.isnan(self._joint_upper_limits), np.pi,
                                                       self._joint_upper_limits)
        self._max_velocities = self._initial_max_velocities * self._vel_limit_factor
        self._max_accelerations = self._initial_max_accelerations * self._acc_limit_factor
        self._max_jerk_linear_interpolation = np.array([min(2 * self._max_accelerations[i] / self._trajectory_time_step,
                                                            self._initial_max_jerk[i]) * self._jerk_limit_factor
                                                        for i in range(len(self._max_accelerations))])
        self._max_torques = self._initial_max_torques * self._torque_limit_factor

        self._obstacle_wrapper.torque_limits = np.array([-1 * self._max_torques, self._max_torques])

        logging.info("Pos upper limits: %s", np.array(self._joint_upper_limits))
        logging.info("Pos lower limits: %s", np.array(self._joint_lower_limits))
        logging.info("Vel limits: %s", self._max_velocities)
        logging.info("Acc limits: %s", self._max_accelerations)
        logging.info("Jerk limits: %s", self._max_jerk_linear_interpolation)
        logging.info("Torque limits: %s", self._max_torques)

    def _print_link_infos(self):
        for i in range(p.getNumJoints(self._robot_id)):
            link_info = p.getLinkState(self._robot_id, i)
            joint_info = p.getJointInfo(self._robot_id, i)
            dynamics_info = p.getDynamicsInfo(self._robot_id, i)
            logging.debug("Link %s, name %s, joint name %s, joint type %s", i, joint_info[1], joint_info[12],
                          joint_info[2])
            logging.debug("Axis %s, parentFramePos %s, parentFrameOrn %s, parentFrameOrnEuler %s, parentIndex %s",
                          joint_info[13], joint_info[14], joint_info[15], p.getEulerFromQuaternion(joint_info[15]),
                          joint_info[16])
            logging.debug("jointLowerLimit %s, jointUpperLimit %s, jointMaxForce %s, jointMaxVelocity %s, "
                          "jointFriction %s, jointDamping %s",
                          joint_info[8], joint_info[9], joint_info[10], joint_info[11], joint_info[7], joint_info[6])
            logging.debug("localInertialFramePosition %s, localInertialFrameOrientation %s, "
                          "localInertialFrameOrientationEuler %s",
                          link_info[2], link_info[3], p.getEulerFromQuaternion(link_info[3]))
            logging.debug("mass %s, lateral_friction %s, local inertia diagonal %s, local inertial pos %s, "
                          "local inertial orn %s, local inertial orn euler %s, restitution %s, "
                          "rolling friction %s, spinning friction %s, contact damping %s, contact stiffness %s",
                          dynamics_info[0], dynamics_info[1], dynamics_info[2],
                          dynamics_info[3], dynamics_info[4], p.getEulerFromQuaternion(dynamics_info[4]),
                          dynamics_info[5], dynamics_info[6], dynamics_info[7], dynamics_info[8], dynamics_info[9])

    def prepare_for_end_of_episode(self):
        # overwritten by reality_wrapper
        pass

    def prepare_for_start_of_episode(self):
        # overwritten by reality_wrapper
        pass

    @property
    def trajectory_duration(self):
        return self._trajectory_duration

    @property
    def manip_joint_indices(self):
        return self._manip_joint_indices

    @property
    def num_manip_joints(self):
        return self._num_manip_joints

    @property
    def default_position(self):
        return self._default_position

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def use_controller_target_velocities(self):
        return self._use_controller_target_velocities

    @property
    def num_robots(self):
        return self._num_robots

    @property
    def plane_z_offset(self):
        return self._plane_z_offset

    @property
    def joint_lower_limits(self):
        return self._joint_lower_limits

    @property
    def joint_lower_limits_continuous(self):
        return self._joint_lower_limits_continuous

    @property
    def joint_upper_limits(self):
        return self._joint_upper_limits

    @property
    def joint_upper_limits_continuous(self):
        return self._joint_upper_limits_continuous

    @property
    def max_velocities(self):
        return self._max_velocities

    @property
    def max_accelerations(self):
        return self._max_accelerations

    @property
    def max_jerk_linear_interpolation(self):
        return self._max_jerk_linear_interpolation

    @property
    def max_torques(self):
        return self._max_torques

    @property
    def robot_scene_id(self):
        return self._robot_scene

    @property
    def robot_name(self):
        return self._robot_name

    @property
    def num_clients(self):
        return self._num_clients

    @property
    def obstacle_wrapper(self):
        return self._obstacle_wrapper

    @property
    def collision_avoidance_mode(self):
        return self._collision_avoidance_mode

    @property
    def collision_avoidance_kinematic_state_sampling_mode(self):
        return self._collision_avoidance_kinematic_state_sampling_mode

    @property
    def collision_avoidance_kinematic_state_sampling_probability(self):
        return self._collision_avoidance_kinematic_state_sampling_probability

    @property
    def collision_avoidance_stay_in_state_probability(self):
        return self._collision_avoidance_stay_in_state_probability

    @property
    def always_use_collision_avoidance_starting_point_sampling(self):
        return self._always_use_collision_avoidance_starting_point_sampling

    @property
    def ball_machine_mode(self):
        return self._ball_machine_mode

    @property
    def ball_radius(self):
        return 0.038

    @property
    def use_moving_objects(self):
        return self._use_moving_objects

    @property
    def moving_object_area_center(self):
        return self._moving_object_area_center

    @property
    def moving_object_area_width_height(self):
        return self._moving_object_area_width_height

    @property
    def moving_object_sphere_center(self):
        return self._moving_object_sphere_center

    @property
    def moving_object_sphere_radius(self):
        return self._moving_object_sphere_radius

    @property
    def moving_object_sphere_height_min_max(self):
        return self._moving_object_sphere_height_min_max

    @property
    def moving_object_sphere_angle_min_max(self):
        return self._moving_object_sphere_angle_min_max

    @property
    def moving_object_speed_meter_per_second(self):
        return self._moving_object_speed_meter_per_second

    @property
    def moving_object_aim_at_current_robot_position(self):
        return self._moving_object_aim_at_current_robot_position

    @property
    def moving_object_check_invalid_target_link_point_positions(self):
        return self._moving_object_check_invalid_target_link_point_positions

    @property
    def moving_object_high_launch_angle_probability(self):
        return self._moving_object_high_launch_angle_probability

    @property
    def moving_object_random_initial_position(self):
        return self._moving_object_random_initial_position

    @property
    def moving_object_current_robot_position_target_link_list(self):
        return self._moving_object_current_robot_position_target_link_list

    @property
    def moving_object_observed_link_names(self):
        return self._moving_object_observed_link_names

    @property
    def planet_mode(self):
        return self._planet_mode

    @property
    def planet_one_center(self):
        return self._planet_one_center

    @property
    def planet_one_radius_xy(self):
        return self._planet_one_radius_xy

    @property
    def planet_one_euler_angles(self):
        return self._planet_one_euler_angles

    @property
    def planet_one_period(self):
        return self._planet_one_period

    @property
    def planet_two_center(self):
        return self._planet_two_center

    @property
    def planet_two_radius_xy(self):
        return self._planet_two_radius_xy

    @property
    def planet_two_euler_angles(self):
        return self._planet_two_euler_angles

    @property
    def planet_two_period(self):
        return self._planet_two_period

    @property
    def planet_two_time_shift(self):
        return self._planet_two_time_shift

    @property
    def planet_obs_global_pos_min_max(self):
        return self._planet_obs_global_pos_min_max

    @property
    def planet_observed_link_names(self):
        return self._planet_observed_link_names

    @property
    def terminate_on_collision_with_moving_obstacle(self):
        return self._terminate_on_collision_with_moving_obstacle

    @property
    def no_target_link_coloring(self):
        return self._no_target_link_coloring

    @property
    def no_link_coloring(self):
        return self._no_link_coloring

    @no_link_coloring.setter
    def no_link_coloring(self, val):
        self._no_link_coloring = val

    @property
    def risk_color_link_index(self):
        return self._risk_color_link_index

    @property
    def risk_state_deterministic_backup_trajectory(self):
        return self._risk_state_deterministic_backup_trajectory

    @property
    def human_network_checkpoint(self):
        return self._human_network_checkpoint

    @property
    def human_network_use_collision_avoidance_starting_point_sampling(self):
        return self._human_network_use_collision_avoidance_starting_point_sampling

    @property
    def human_network_collision_avoidance_kinematic_state_sampling_probability(self):
        return self._human_network_collision_avoidance_kinematic_state_sampling_probability

    @property
    def human_network_collision_avoidance_stay_in_state_probability(self):
        return self._human_network_collision_avoidance_stay_in_state_probability

    @property
    def use_fixed_seed(self):
        return self._use_fixed_seed

    @property
    def do_not_execute_robot_movement(self):
        return self._do_not_execute_robot_movement

    def get_manip_joint_indices_per_robot(self, robot_index):
        return self._manip_joint_indices_per_robot[robot_index]

    def get_actual_joint_positions(self):
        return [s[0] for s in p.getJointStates(self._robot_id, self._manip_joint_indices,
                                               physicsClientId=self._simulation_client_id)]

    def _get_manip_joint_indices(self):
        joint_indices = []
        joint_indices_per_robot = [[] for _ in range(self._num_robots)]
        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            q_index = joint_info[3]  # to distinguish fixed from moving joints
            if q_index > -1:
                joint_indices.append(i)
                if self._num_robots == 1:
                    joint_indices_per_robot[0].append(i)
                else:
                    link_name = str(joint_info[12])[2:-1]

                    if re.match('^.*_r[0-9]+$', link_name):
                        # e.g. extract 1 from linkname_r1
                        robot_index = int(link_name.rsplit('_', 1)[1][1:])
                        if robot_index >= self._num_robots:
                            raise ValueError("Found link name " + link_name + ", but expected " + str(
                                self._num_robots) + " robots only.")
                        else:
                            joint_indices_per_robot[robot_index].append(i)

                    else:
                        raise ValueError("Could not find a robot suffix like _r0 for link " + link_name)

        return tuple(joint_indices), joint_indices_per_robot

    def set_motor_control(self, target_positions, target_velocities=None, target_accelerations=None,
                          mode=p.POSITION_CONTROL, physics_client_id=None,
                          manip_joint_indices=None, use_max_force=False,
                          velocity_gain=0.87,
                          initial_kinematic_state=False,
                          **kwargs):
        # overwritten by real robot scene
        if physics_client_id is None:
            physics_client_id = self._simulation_client_id
        if not self._static_robot or initial_kinematic_state:
            if manip_joint_indices is None:
                manip_joint_indices = self._manip_joint_indices

            if self._use_controller_target_velocities and target_velocities is not None:
                # velocity_gain_list = [velocity_gain] * len(target_velocities)
                velocity_gain_list = [1.0] * len(target_velocities)
                target_velocities = target_velocities * velocity_gain

            if use_max_force:
                if manip_joint_indices != self._manip_joint_indices:
                    joint_limit_indices = np.array([i for i in range(len(self._manip_joint_indices))
                                                    if self._manip_joint_indices[i] in manip_joint_indices])
                    max_forces = self._max_torques[joint_limit_indices]
                else:
                    max_forces = self._max_torques

                if self._use_controller_target_velocities and target_velocities is not None:
                    p.setJointMotorControlArray(self._robot_id, manip_joint_indices,
                                                mode, targetPositions=target_positions,
                                                forces=max_forces,
                                                targetVelocities=target_velocities,
                                                velocityGains=velocity_gain_list,
                                                physicsClientId=physics_client_id)
                else:
                    p.setJointMotorControlArray(self._robot_id, manip_joint_indices,
                                                mode, targetPositions=target_positions,
                                                forces=max_forces,
                                                physicsClientId=physics_client_id)
            else:
                if self._use_controller_target_velocities and target_velocities is not None:
                    p.setJointMotorControlArray(self._robot_id, manip_joint_indices,
                                                mode, targetPositions=target_positions,
                                                targetVelocities=target_velocities,
                                                velocityGains=velocity_gain_list,
                                                physicsClientId=physics_client_id)
                else:
                    p.setJointMotorControlArray(self._robot_id, manip_joint_indices,
                                                mode, targetPositions=target_positions,
                                                physicsClientId=physics_client_id)

    def get_actual_joint_torques(self, physics_client_id=None, manip_joint_indices=None):
        # overwritten by reality wrapper

        if physics_client_id is None:
            physics_client_id = self._simulation_client_id
        if manip_joint_indices is None:
            manip_joint_indices = self.manip_joint_indices

        joint_states = p.getJointStates(self._robot_id, manip_joint_indices, physicsClientId=physics_client_id)
        actual_torques = np.asarray([joint_state[3] for joint_state in joint_states])
        return actual_torques

    def _deactivate_self_collision_for_adjoining_links(self):
        # deactivate erroneous self-collisions resulting from inaccurate collision meshes
        if self._robot_name == "iiwa7":
            deactivate_self_collision_detection_link_name_pair_list = []
            deactivate_self_collision_detection_link_name_pair_list_per_robot = [["iiwa_link_5", "iiwa_link_7"]]

        elif self._robot_name.startswith("human"):
            deactivate_self_collision_detection_link_name_pair_list = []
            deactivate_self_collision_detection_link_name_pair_list_per_robot = [["body", "upper_arm"]]

        else:
            deactivate_self_collision_detection_link_name_pair_list = []
            deactivate_self_collision_detection_link_name_pair_list_per_robot = []

        for link_name_pair in deactivate_self_collision_detection_link_name_pair_list:
            self._deactivate_self_collision_detection(link_name_a=link_name_pair[0],
                                                      link_name_b=link_name_pair[1])

        for link_name_pair in deactivate_self_collision_detection_link_name_pair_list_per_robot:
            for j in range(self.num_robots):
                link_name_pair_robot = self.get_link_names_for_multiple_robots(link_name_pair, robot_indices=[j])
                self._deactivate_self_collision_detection(link_name_a=link_name_pair_robot[0],
                                                          link_name_b=link_name_pair_robot[1])

    def _deactivate_self_collision_detection(self, link_name_a, link_name_b):
        link_index_a = self.get_link_index_from_link_name(link_name_a)
        link_index_b = self.get_link_index_from_link_name(link_name_b)
        for j in range(self.num_clients):
            p.setCollisionFilterPair(self.robot_id, self.robot_id, link_index_a,
                                     link_index_b, enableCollision=0, physicsClientId=j)

    def get_link_names_for_multiple_robots(self, link_names, robot_indices=None):
        if isinstance(link_names, str):
            link_names = [link_names]

        if self._num_robots == 1:
            return link_names  # do nothing if there is one robot only

        link_names_multiple_robots = []

        if robot_indices is None:
            robot_indices = np.arange(self._num_robots)

        for j in range(len(link_names)):
            if link_names[j] in self._shared_link_names:
                link_names_multiple_robots.append(link_names[j])

        for i in range(len(robot_indices)):
            for j in range(len(link_names)):
                if link_names[j] not in self._shared_link_names:
                    link_names_multiple_robots.append(link_names[j] + "_r" + str(robot_indices[i]))

        return link_names_multiple_robots

    def get_robot_identifier_from_link_name(self, link_name):
        # returns _r1 for a link called iiwa_link_4_r1 and "" for a link called iiwa_link_4
        if re.match('^.*_r[0-9]+$', link_name):
            link_identifier = "_" + link_name.rsplit('_', 1)[1]
            # split string from the right side and choose the last element
        else:
            link_identifier = ""
        return link_identifier

    def get_link_index_from_link_name(self, link_name):
        for i in range(len(self._link_name_list)):
            if self._link_name_list[i] == link_name:
                return i
        return -1

    def get_robot_index_from_link_name(self, link_name):
        # returns the robot index extracted from the link name, e.g. 1 for iiwa_link_4_r1
        # returns -1 if no link index is found and if multiple robots are in use, 0 otherwise
        if self._num_robots > 1:
            if re.match('^.*_r[0-9]+$', link_name):
                # e.g. extract 1 from linkname_r1
                return int(link_name.rsplit('_', 1)[1][1:])
            else:
                return -1
        else:
            return 0

    def prepare_switch_to_backup_client(self, nested_env=False):
        self._simulation_client_id = self._backup_client_id
        self._obstacle_wrapper.prepare_switch_to_backup_client(nested_env=nested_env)

    def switch_back_to_main_client(self):
        self._obstacle_wrapper.switch_back_to_main_client()

    @staticmethod
    def is_sphere_on_board(sphere_pos_local_rel):
        return np.all(np.abs(sphere_pos_local_rel) < 1.0)

    def clear_last_action(self):
        pass

    def disconnect(self):
        pass

    @staticmethod
    def send_command_to_trajectory_controller(target_positions, **kwargs):
        raise NotImplementedError()

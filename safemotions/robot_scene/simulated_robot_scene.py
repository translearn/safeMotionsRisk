# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import pybullet as p

from safemotions.robot_scene.robot_scene_base import RobotSceneBase


class SimRobotScene(RobotSceneBase):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)


    def pose_manipulator(self, joint_positions, joint_velocities=None):
        if joint_velocities is None:
            target_velocities = [[0.0] for _ in joint_positions]
        else:
            target_velocities = [[vel] for vel in joint_velocities]

        p.resetJointStatesMultiDof(bodyUniqueId=self._robot_id,
                                   jointIndices=self._manip_joint_indices,
                                   targetValues=[[pos] for pos in joint_positions],
                                   targetVelocities=target_velocities,
                                   physicsClientId=self._simulation_client_id)


    def get_actual_joint_position_and_velocity(self, manip_joint_indices=None, physics_client_id=None):
        if physics_client_id is None:
            physics_client_id = self._simulation_client_id
        if manip_joint_indices is None:
            manip_joint_indices = self._manip_joint_indices
        # return the actual joint position and velocity for the specified joint indices from the physicsClient
        joint_states = p.getJointStates(self._robot_id, manip_joint_indices,
                                        physicsClientId=physics_client_id)

        joint_states_swap = np.swapaxes(np.array(joint_states, dtype=object), 0, 1)

        return joint_states_swap[0], joint_states_swap[1]

    @staticmethod
    def send_command_to_trajectory_controller(target_positions, **kwargs):
        pass

    def clear_last_action(self):
        super().clear_last_action()






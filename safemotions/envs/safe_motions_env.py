# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import inspect
import numpy as np
import logging


from safemotions.envs.decorators import actions, observations, rewards, video_recording
from safemotions.envs.safe_motions_base import resampling_decorator


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class SafeMotionsEnv(actions.AccelerationPredictionBoundedJerkAccVelPos,
                     observations.SafeObservation,
                     rewards.TargetPointReachingReward,
                     video_recording.VideoRecordingManager):
    def __init__(self,
                 *vargs,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

    def _process_end_of_episode(self, observation, reward, info):
        observation, reward, termination, truncation, info = super()._process_end_of_episode(observation, reward, info)

        if 'obstacles_num_target_points_reached' in info:
            info['obstacles_num_target_points_reached_per_time'] = info['obstacles_num_target_points_reached'] / \
                                                                   (self._episode_length * self._trajectory_time_step)

        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # distinguish here between termination and truncation if required

        if self._use_target_points and self._termination_reason == self.TERMINATION_TRAJECTORY_LENGTH:
            termination = False
            truncation = True

        return observation, reward, termination, truncation, info


class SafeMotionsEnvCollisionAvoidance(actions.AccelerationPredictionBoundedJerkAccVelPos,
                                       observations.SafeObservation,
                                       rewards.CollisionAvoidanceReward,
                                       video_recording.VideoRecordingManager):
    def __init__(self,
                 *vargs,
                 **kwargs):
        super().__init__(*vargs, **kwargs)


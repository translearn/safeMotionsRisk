#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
# limit the number of threads per worker for parallel execution to one
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)

os.environ['RAY_DEDUP_LOGS'] = str(0)  # no log deduplication
os.environ['RAY_AIR_NEW_OUTPUT'] = str(0)  # old (more detailed) cli logging

import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import logging

from safemotions.common_functions import register_envs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    args = parser.parse_args()
    ppo_config = PPOConfig()

    register_envs()

    ray.init(dashboard_host='0.0.0.0', include_dashboard=False,
             ignore_reinit_error=True, logging_level=logging.INFO)

    results = tune.Tuner.restore(
        path=args.experiment_dir,
        trainable=PPO,
        resume_errored=True,
    ).fit()

    ray.shutdown()

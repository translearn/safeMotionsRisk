# This file makes use of code from
# https://github.com/ray-project/ray/blob/master/rllib/examples/inference/policy_inference_after_training_w_connector.py
# and
# https://github.com/ray-project/ray/blob/master/rllib/connectors/module_to_env/get_actions.py
# Copyright 2023 Ray Authors
# Licensed under the Apache License, Version 2.0;

import os
import inspect
from gymnasium.spaces import Box
import numpy as np
from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
import torch

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

class RLAgent:
    def __init__(self, checkpoint_path, explore=False,
                 clip_actions=True,
                 use_module_env_pipeline=False,  # usually faster if False
                 observation_size=None, action_size=None, # only required if use_module_env_pipeline=True
                 policy_name="backup"):

        self._checkpoint_path = checkpoint_path
        self._explore = explore
        self._use_module_env_pipeline = use_module_env_pipeline
        self._clip_actions = clip_actions
        self._policy_name = policy_name

        rl_module_path = str(os.path.join(self._checkpoint_path, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER,
                                          COMPONENT_RL_MODULE, DEFAULT_MODULE_ID))

        if not os.path.isdir(self._checkpoint_path):
            raise NotADirectoryError("Could not find the checkpoint specified for the {} policy {}".format(
                self._policy_name, self._checkpoint_path))

        if not os.path.isdir(rl_module_path):
            raise NotADirectoryError("Could not find the rl module required"
                                     " for the {} policy {}".format(self._policy_name, rl_module_path))

        self._rl_module = RLModule.from_checkpoint(rl_module_path)

        if self._use_module_env_pipeline:
            env_to_module_path = str(os.path.join(self._checkpoint_path, COMPONENT_ENV_RUNNER,
                                                  COMPONENT_ENV_TO_MODULE_CONNECTOR))

            module_to_env_path = str(os.path.join(self._checkpoint_path, COMPONENT_ENV_RUNNER,
                                                  COMPONENT_MODULE_TO_ENV_CONNECTOR))

            if not os.path.isdir(env_to_module_path):
                raise NotADirectoryError("Could not find the env-to-module pipeline required"
                                         " for the {} policy {}".format(self._policy_name, env_to_module_path))

            if not os.path.isdir(module_to_env_path):
                raise NotADirectoryError("Could not find the rl module required"
                                         " for the {} policy {}".format(self._policy_name, module_to_env_path))

            self._env_to_module = EnvToModulePipeline.from_checkpoint(env_to_module_path)

            self._module_to_env = ModuleToEnvPipeline.from_checkpoint(module_to_env_path)

            if observation_size is None or action_size is None:
                raise ValueError("observation_size and action_size need to be specified if "
                                 "use_module_env_pipeline == True")

            self._observation_space = Box(low=np.float32(-1), high=np.float32(1), shape=(observation_size,),
                                          dtype=np.float32)

            if self._clip_actions:
                self._action_space = Box(low=np.float32(-1), high=np.float32(1),
                                         shape=(action_size,), dtype=np.float32)
            else:
                self._action_space = Box(low=-np.inf, high=np.inf, shape=(action_size,),
                                        dtype=np.float32)



    def compute_action(self, observation, **kwargs):

        if self._use_module_env_pipeline:
            episode = SingleAgentEpisode(
                observations=[observation],
                observation_space=self._observation_space,
                action_space=self._action_space,
            )

            shared_data = {}
            input_dict = self._env_to_module(
                episodes=[episode],
                rl_module=self._rl_module,
                explore=self._explore,
                shared_data=shared_data,
            )

            rl_module_out = self._rl_module.forward_inference(input_dict)

            output_dict = self._module_to_env(
                batch=rl_module_out,
                episodes=[episode],
                rl_module=self._rl_module,
                explore=self._explore,
                shared_data=shared_data,
            )

            action = output_dict.pop(Columns.ACTIONS_FOR_ENV)[0]

        else:
            # ignore the env_to_module and module_to_env pipeline
            # just sample and clip actions
            # if no complex pipeline is required, this option is faster
            # (ca. 0.15 ms per action compared to 0.3 ms per action)

            input_dict = {Columns.OBS: torch.from_numpy(np.asarray(observation)).unsqueeze(0)}
            rl_module_out = self._rl_module.forward_inference(input_dict)

            if self._explore:
                action_dist_class = self._rl_module.get_exploration_action_dist_cls()
            else:
                action_dist_class = self._rl_module.get_inference_action_dist_cls()

            action_dist = action_dist_class.from_logits(rl_module_out[Columns.ACTION_DIST_INPUTS])

            if not self._explore:
                action_dist = action_dist.to_deterministic()

            action_tensor = action_dist.sample()
            action_np_unclipped = action_tensor.detach().cpu().numpy().astype(np.float32)[0]

            if self._clip_actions:
                action = np.clip(action_np_unclipped, -1, 1)
            else:
                action = action_np_unclipped

        return action

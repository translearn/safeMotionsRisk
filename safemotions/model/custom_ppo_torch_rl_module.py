# This file reuses code from the following ray rllib classes:
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.models.torch.heads import TorchMLPHead
# Copyright 2023 Ray Authors
# Licensed under the Apache License, Version 2.0;

import torch

class CustomPPOTorchRLModule(DefaultPPOTorchRLModule):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = CustomPPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

class CustomPPOCatalog(PPOCatalog):
    def __init__(
            self,
            observation_space,
            action_space,
            model_config_dict):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        self.vf_head_hiddens = self._model_config_dict["vf_head_fcnet_hiddens"]
        self.pi_head_hiddens = self._model_config_dict["pi_head_fcnet_hiddens"]
        self.pi_head_last_layer_activation = self._model_config_dict["pi_head_last_layer_activation"]
        self.pi_head_no_log_std_activation = self._model_config_dict["pi_head_no_log_std_activation"]

        # overwrite vf_head_config using self.vf_head_hiddens instead of self.pi_and_vf_head_hiddens
        self.vf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.vf_head_hiddens,
            hidden_layer_activation=self.pi_and_vf_head_activation,
            output_layer_activation="linear",
            output_layer_dim=1,
        )

    def _get_pi_head_config(self, framework: str):
        if self._model_config_dict["free_log_std"]:
            raise ValueError("This custom RL module does not support free_log_std == True.")

        action_distribution_cls = self.get_action_dist_cls(framework=framework)
        required_output_dim = action_distribution_cls.required_input_dim(
            space=self.action_space, model_config=self._model_config_dict
        )

        self.pi_head_config = CustomTorchMLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.pi_and_vf_head_hiddens,
            hidden_layer_activation=self.pi_and_vf_head_activation,
            output_layer_dim=required_output_dim,
            output_layer_activation=self.pi_head_last_layer_activation,  # allow custom last layer activation
            clip_log_std=False if (self.pi_head_last_layer_activation == "tanh"
                                   and not self.pi_head_no_log_std_activation) else True,
            log_std_clip_param=self._model_config_dict.get("log_std_clip_param", 20),
            log_std_range=self._model_config_dict['log_std_range'],
            no_log_std_activation=self.pi_head_no_log_std_activation,
        )

        return self.pi_head_config

    def build_pi_head(self, framework: str):
        self.pi_head_config = self._get_pi_head_config(framework=framework)
        return self.pi_head_config.build(framework=framework)


class CustomTorchMLPHeadConfig(MLPHeadConfig):
    def __init__(self, *args, log_std_range=None, no_log_std_activation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std_range = log_std_range
        self.no_log_std_activation = no_log_std_activation

    def build(self, *args, **kwargs):
        self._validate(framework="torch")
        return CustomTorchMLPHead(self)

    def _validate(self, framework: str = "torch"):
        if self.clip_log_std and self.output_layer_activation == "tanh" and not self.no_log_std_activation:
            raise ValueError(
                    f"clip_log_std == True not supported with output_layer_activation == tanh"
                )
        super()._validate(framework=framework)

class CustomTorchMLPHead(TorchMLPHead):
    def __init__(self, config: CustomTorchMLPHeadConfig) -> None:
        self.output_layer_activation = config.output_layer_activation
        self.no_log_std_activation = config.no_log_std_activation
        self.log_std_range = config.log_std_range
        config.output_layer_activation = "linear"  # output layer activation is handled in forward method
        super().__init__(config)

    def _forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.net(inputs)
        means, log_stds = torch.chunk(logits, chunks=2, dim=-1)
        if self.output_layer_activation == "tanh":
                means = torch.tanh(means)
                if not self.no_log_std_activation:
                    log_stds = torch.tanh(log_stds)
                    if self.log_std_range is not None:
                        # map log_std tanh output range (-1 to 1) to self.log_std_range
                        log_stds = (
                                self.log_std_range[0] + 0.5 * (log_stds + 1) * (
                                    self.log_std_range[1] - self.log_std_range[0]))

        if self.clip_log_std:
            # Clip the log standard deviations.
            log_stds = torch.clamp(
                log_stds, -self.log_std_clip_param_const, self.log_std_clip_param_const
            )

        return torch.cat((means, log_stds), dim=-1)

from ray import tune
import os
import inspect
import json
import errno
import numpy as np

from ray.rllib.utils.from_config import NotProvided
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

safemotions_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Termination reason
TERMINATION_UNSET = -1
TERMINATION_SUCCESS = 0  # unused
TERMINATION_JOINT_LIMITS = 1
TERMINATION_TRAJECTORY_LENGTH = 2
TERMINATION_SELF_COLLISION = 3
TERMINATION_COLLISION_WITH_STATIC_OBSTACLE = 4
TERMINATION_COLLISION_WITH_MOVING_OBSTACLE = 5

termination_reasons_dict = {TERMINATION_JOINT_LIMITS: 'joint_limit_violation_termination_rate',
                            TERMINATION_TRAJECTORY_LENGTH: 'trajectory_length_termination_rate',
                            TERMINATION_SELF_COLLISION: 'collision_self_termination_rate',
                            TERMINATION_COLLISION_WITH_STATIC_OBSTACLE: 'collision_static_obstacles_termination_rate',
                            TERMINATION_COLLISION_WITH_MOVING_OBSTACLE: 'collision_moving_obstacles_termination_rate'
                            }

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def register_envs():
    from safemotions.envs.safe_motions_env import SafeMotionsEnvCollisionAvoidance
    from safemotions.envs.safe_motions_env import SafeMotionsEnv

    tune.register_env(SafeMotionsEnvCollisionAvoidance.__name__,
                      lambda config_args: SafeMotionsEnvCollisionAvoidance(**config_args))
    tune.register_env(SafeMotionsEnv.__name__,
                      lambda config_args: SafeMotionsEnv(**config_args))

def get_checkpoint_path_and_config(checkpoint_path: str):

    if os.path.basename(checkpoint_path) == "":  # remove tailing "/" from path if required
        checkpoint_path = os.path.dirname(checkpoint_path)

    if not os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(safemotions_dir, 'trained_networks', checkpoint_path)

    if os.path.isdir(checkpoint_path):
        params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")

        if not os.path.isfile(params_path):
            params_path = os.path.join(checkpoint_path, "params.json")
            if os.path.isfile(params_path):
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint')
                if not os.path.isdir(checkpoint_path):
                    raise NotADirectoryError('Could not find checkpoint directory {}'.format(checkpoint_path))
            else:
                raise FileNotFoundError("Could not find file {}", params_path)

        with open(params_path) as params_file:
            checkpoint_config = json.load(params_file)

    else:
        raise NotADirectoryError('Could not find checkpoint directory {}'.format(checkpoint_path))

    return checkpoint_path, checkpoint_config

def get_model_config_and_rl_module_spec_from_checkpoint_config(model_config_str, rl_module_spec_str,
                                                               inference_only=False):

    # returns the parameters model_config and rl_module_spec required to specify an RL Module

    from ast import literal_eval

    if rl_module_spec_str is None:
        use_custom_model = False
    else:
        use_custom_model = True

    if use_custom_model:
        from safemotions.model.custom_ppo_torch_rl_module import CustomPPOTorchRLModule

        # extract model_config from _rl_module_spec string
        # extract dictionary definition string for model_config from rl_module_spec_str
        # first step: extract string like
        # "{'key_a': value_a, 'key_b': value_b}, catalog_class=None, load_state_path=None, model_config_dict=None)"
        model_config_str_plus_tail = rl_module_spec_str[rl_module_spec_str.find("model_config={") + 13:]
        # second step: remove tail like ", catalog_class=None, load_state_path=None, model_config_dict=None)"
        # by searching for a "}" that corresponds to the initial "{"
        bracket_count = 1  # initial bracket
        model_config_str_end_index = -1

        for i in range(1, len(model_config_str_plus_tail)):
            if model_config_str_plus_tail[i] == "{":
                bracket_count += 1
            if model_config_str_plus_tail[i] == "}":
                bracket_count -= 1
            if bracket_count == 0:
                model_config_str_end_index = i
                break

        model_config_str = model_config_str_plus_tail[0:model_config_str_end_index + 1]
        model_config_dict = literal_eval(model_config_str)

        rl_module_spec = RLModuleSpec(
            module_class=CustomPPOTorchRLModule,
            model_config=model_config_dict,
            inference_only=inference_only
        )

        model_config = NotProvided

    else:

        default_model_config_str = model_config_str.split("DefaultModelConfig", 1)[1].strip('()')
        default_model_config_list = default_model_config_str.split("=")

        default_model_config_dict = {}
        for i in range(len(default_model_config_list) - 1):
            default_model_config_dict[default_model_config_list[i].rsplit(",", 1)[-1].strip()] = (
                literal_eval(default_model_config_list[i + 1].rsplit(",", 1)[0].strip()))

        if inference_only:
            default_model_config_dict['inference_only'] = True

        model_config = DefaultModelConfig(**default_model_config_dict)

        rl_module_spec = NotProvided

    return model_config, rl_module_spec


def betas_tensor_to_float(learner):
    # fix to avoid "beta1 as a Tensor is not supported for capturable=False and foreach=True" error when
    # restoring checkpoints using train.py with num_gpu != 0
    # see https://github.com/ray-project/ray/issues/51560
    param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
    param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])

def store_args(path, args):
    make_dir(path)

    with open(os.path.join(path, "arguments.json"), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True))
        f.flush()

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def store_metrics(base_dir, real_robot, pid, episode_counter, reward_total, last_info, episode_info):
    metric_file = "episode_{}_{}_{:.3f}.json".format(episode_counter, pid, reward_total)
    episode_info['reward'] = float(reward_total)
    episode_info['episode_length'] = int(last_info['episode_length'])
    episode_info['trajectory_length'] = int(last_info['trajectory_length'])
    episode_info['success_rate'] = last_info['trajectory_successful'] if 'trajectory_successful' in last_info else 0.0

    for key, value in last_info.items():
        if key.startswith("obstacles"):
            episode_info[key] = value

    for key, value in episode_info['sum'].items():
        episode_info['sum'][key] = float(np.sum(np.array(value)))
    for key, value in episode_info['max'].items():
        episode_info['max'][key] = float(np.max(np.array(value)))
    for key, value in episode_info['mean'].items():
        episode_info['mean'][key] = float(np.mean(np.array(value)))
    for key, value in episode_info['min'].items():
        episode_info['min'][key] = float(np.min(np.array(value)))

    for key, value in termination_reasons_dict.items():
        episode_info[value] = 1.0 if last_info['termination_reason'] == key else 0.0

    metrics_dir = get_metrics_dir(base_dir, real_robot)
    with open(os.path.join(metrics_dir, metric_file), 'w') as f:
        f.write(json.dumps(episode_info, default=np_encoder))
        f.flush()

def get_metrics_dir(base_dir, real_robot):
    if real_robot:
        metrics_dir = os.path.join(base_dir, "trajectory_logs_real")
    else:
        metrics_dir = os.path.join(base_dir, "trajectory_logs_sim")

    return metrics_dir

def make_metrics_dir_and_store_args(base_dir, real_robot, args):
    metrics_dir = get_metrics_dir(base_dir, real_robot)
    make_dir_and_store_args(metrics_dir, args)
    return metrics_dir

def get_network_data_dir(base_dir, real_robot):
    if real_robot:
        metrics_dir = os.path.join(base_dir, "network_data_real")
    else:
        metrics_dir = os.path.join(base_dir, "network_data_sim")

    return metrics_dir

def make_network_data_dir_and_and_store_args(base_dir, real_robot, args):
    network_data_dir = get_network_data_dir(base_dir, real_robot)
    make_dir_and_store_args(network_data_dir, args)

def make_dir_and_store_args(path, args):
    make_dir(path)
    with open(os.path.join(path, "arguments.json"), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, default=np_encoder))
        f.flush()

def store_env_config(eval_dir, env_config):
    make_dir(eval_dir)

    with open(os.path.join(eval_dir, "env_config.json"), 'w') as f:
        f.write(json.dumps(env_config, sort_keys=True))
        f.flush()
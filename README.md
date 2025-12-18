# Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles 
[![IEEE RAL 2024](https://img.shields.io/badge/IEEE_RAL-2024-%3C%3E)](https://ieeexplore.ieee.org/document/10738380)
[![arXiv](https://img.shields.io/badge/arXiv-2411.05784-B31B1B)](https://arxiv.org/abs/2411.05784)
[![GitHub issues](https://img.shields.io/github/issues/translearn/safemotionsRisk)](https://github.com/translearn/safeMotionsRisk/issues/)<br>

This branch contains a PyTorch port of the initial code.

<div align='center'>
    <img src="https://github.com/user-attachments/assets/c20f450c-15f8-4639-ab85-7663688cb9e7" width="750"/>
</div>

## Installation

This branch was tested using Python 3.12 and Ray 2.49.1. The required dependencies can be installed by running:

    pip install -r requirements.txt


## Trained networks 

We provide pretrained task networks, backup networks and risk networks. \
Rollouts for the task networks can be visualized in a simulator by running one of the commands below:  


### Reaching task

**Space environment, state-action-based risk**


```bash
python safemotions/evaluate.py --checkpoint=task_networks/reaching_task/space/state_action --no_exploration --visualize_risk --trajectory_duration=30 --use_gui
```

**Ball environment, state-action-based risk**


```bash
python safemotions/evaluate.py --checkpoint=task_networks/reaching_task/ball/state_action --no_exploration --visualize_risk --trajectory_duration=30 --use_gui
```

**Human environment, state-action-based risk**


```bash
python safemotions/evaluate.py --checkpoint=task_networks/reaching_task/human/state_action --no_exploration --visualize_risk --trajectory_duration=30 --use_gui
```

## Training

The training process involves three steps:

1. Training of a backup policy using reinforcement learning
2. Training of a risk estimator using supervised learning
3. Training of a task policy using reinforcement learning

The progress of each step can be observed using tensorboard:

```bash
python -m tensorboard.main --logdir=path_to_training_logs
```

### 1. Training of the backup policy

The backup policy is trained on avoiding collisions. Once trained, it can be used to learn different task policies. 
For the training commands below, you can additionally specify the number of workers with --num_workers 
(e.g. --num_workers=12, typically number of CPU cores - 1) and the number of GPUs with 
--num_gpus (e.g. --num_gpus=1). After N iterations (as specified by --iterations_per_checkpoint=N) 
a checkpoint is created in the directory specified by --logdir.

**Space environment**

```bash
python safemotions/train.py --logdir=specify_path_for_training_logs --name=backup_space --acc_limit_factor=1.0  --action_max_punishment=0.4 --action_punishment_min_threshold=0.95 --batch_size_factor=8.0 --closest_point_safety_distance=0.01 --collision_avoidance_episode_early_termination_punishment=-15 --collision_avoidance_episode_termination_bonus=15 --collision_avoidance_kinematic_state_sampling_mode --collision_avoidance_kinematic_state_sampling_probability=0.7 --collision_avoidance_low_acceleration_max_reward=0.0  --collision_avoidance_low_acceleration_threshold=1.0 --collision_avoidance_low_velocity_max_reward=0.0 --collision_avoidance_low_velocity_threshold=1.0 --collision_avoidance_mode  --collision_avoidance_moving_obstacles_max_reward_distance=0.6 --collision_avoidance_moving_obstacles_max_reward=3.0 --collision_avoidance_self_collision_max_reward_distance=0.05  --collision_avoidance_self_collision_max_reward=1.0 --collision_avoidance_static_obstacles_max_reward_distance=0.1 --collision_avoidance_static_obstacles_max_reward=1.0  --collision_avoidance_stay_in_state_probability=0.3 --collision_check_time=0.033 --episodes_per_simulation_reset=4000 --gamma=1.0 --hidden_layer_activation=swish --iterations_per_checkpoint=50 --jerk_limit_factor=1.0 --kl_target=0.005 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --no_use_gae --obs_planet_size_per_planet=2 --obstacle_scene=5 --planet_mode --planet_one_center="[-0.1, 0.0, 0.8]" --planet_one_euler_angles="[0.35, 0, 0]" --planet_one_period=5.0 --planet_one_radius_xy="[0.65, 0.8]" --planet_two_center="[-0.1, 0, 0.8]" --planet_two_euler_angles="[-0.35, 0, 0]" --planet_two_radius_xy="[0.75, 0.8]" --planet_two_time_shift=-2.0 --pos_limit_factor=1.0 --punish_action --robot_scene=0 --solver_iterations=50 --starting_point_cartesian_range_scene=1 --terminate_on_collision_with_moving_obstacle --terminate_on_collision_with_static_obstacle --terminate_on_self_collision --trajectory_duration=2.0  --trajectory_time_step=0.1 --use_controller_target_velocities --vel_limit_factor=1.0 --time=500 
```


**Ball environment**

```bash
python safemotions/train.py --logdir=specify_path_for_training_logs --name=backup_ball --acc_limit_factor=1.0 --action_max_punishment=0.4 --action_punishment_min_threshold=0.95  --batch_size_factor=8.0 --closest_point_safety_distance=0.01 --collision_avoidance_episode_early_termination_punishment=-15 --collision_avoidance_episode_termination_bonus=15  --collision_avoidance_kinematic_state_sampling_mode --collision_avoidance_kinematic_state_sampling_probability=0.7 --collision_avoidance_low_acceleration_max_reward=0.0 --collision_avoidance_low_acceleration_threshold=1.0 --collision_avoidance_low_velocity_max_reward=0.0 --collision_avoidance_low_velocity_threshold=1.0 --collision_avoidance_mode --collision_avoidance_moving_obstacles_max_reward_distance=0.6 --collision_avoidance_moving_obstacles_max_reward=3.0 --collision_avoidance_self_collision_max_reward_distance=0.05 --collision_avoidance_self_collision_max_reward=1.0 --collision_avoidance_static_obstacles_max_reward_distance=0.1 --collision_avoidance_static_obstacles_max_reward=1.0 --collision_avoidance_stay_in_state_probability=0.3 --collision_check_time=0.033 --episodes_per_simulation_reset=4000 --gamma=1.0 --hidden_layer_activation=swish --iterations_per_checkpoint=50 --jerk_limit_factor=1.0 --kl_target=0.005 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --moving_object_sphere_center="[0, 0, 0.5]" --moving_object_sphere_radius=2.5 --moving_object_sphere_height_min_max="[-0.5, 0.5]" --moving_object_sphere_angle_min_max="[0, 6.2831]" --moving_object_speed_meter_per_second=6.0 --moving_object_check_invalid_target_link_point_positions --moving_object_random_initial_position --no_use_gae --obstacle_scene=5 --pos_limit_factor=1.0 --punish_action --robot_scene=0 --solver_iterations=50 --starting_point_cartesian_range_scene=1 --terminate_on_collision_with_moving_obstacle --terminate_on_collision_with_static_obstacle --terminate_on_self_collision --trajectory_duration=2.0 --trajectory_time_step=0.1 --use_controller_target_velocities --use_moving_objects --vel_limit_factor=1.0 --time=500 
```


**Human environment**

```bash
python safemotions/train.py --logdir=specify_path_for_training_logs --name=backup_human --acc_limit_factor=1.0 --action_max_punishment=0.4 --action_punishment_min_threshold=0.95  --batch_size_factor=8.0 --closest_point_safety_distance=0.01 --collision_avoidance_episode_early_termination_punishment=-15 --collision_avoidance_episode_termination_bonus=15  --collision_avoidance_kinematic_state_sampling_mode --collision_avoidance_kinematic_state_sampling_probability=0.7 --collision_avoidance_low_acceleration_max_reward=0.0 --collision_avoidance_low_acceleration_threshold=1.0 --collision_avoidance_low_velocity_max_reward=0.0 --collision_avoidance_low_velocity_threshold=1.0 --collision_avoidance_mode --collision_avoidance_moving_obstacles_max_reward_distance=0.6 --collision_avoidance_moving_obstacles_max_reward=3.0 --collision_avoidance_self_collision_max_reward_distance=0.05 --collision_avoidance_self_collision_max_reward=1.0 --collision_avoidance_static_obstacles_max_reward_distance=0.1 --collision_avoidance_static_obstacles_max_reward=1.0 --collision_avoidance_stay_in_state_probability=0.3 --collision_check_time=0.033 --episodes_per_simulation_reset=4000 --gamma=1.0 --hidden_layer_activation=swish --human_network_checkpoint=human_network/checkpoint/checkpoint --human_network_collision_avoidance_kinematic_state_sampling_probability=0.3 --human_network_collision_avoidance_stay_in_state_probability=0.3 --human_network_use_collision_avoidance_starting_point_sampling --iterations_per_checkpoint=50 --jerk_limit_factor=1.0 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --no_use_gae --obstacle_scene=5 --pos_limit_factor=1.0 --punish_action --robot_scene=0 --solver_iterations=50 --starting_point_cartesian_range_scene=1 --terminate_on_collision_with_moving_obstacle --terminate_on_collision_with_static_obstacle --terminate_on_self_collision --trajectory_duration=3.0 --trajectory_time_step=0.1 --use_controller_target_velocities --vel_limit_factor=1.0 --time=500 
```

**Note:** The human in the human environment is controlled by a neural network. The checkpoint of this network is defined by --human_network_checkpoint. The *human network* used for our experiments was trained using the following command: 


```bash
python safemotions/train.py --logdir=specify_path_for_human_network --name=human_network --acc_limit_factor_braking=1.0 --acc_limit_factor=1.0  --action_max_punishment=0.4 --action_punishment_min_threshold=0.95 --check_braking_trajectory_collisions --closest_point_safety_distance=0.01 --collision_check_time=0.033 --end_min_distance_max_punishment=0.5 --end_min_distance_max_threshold=0.05 --hidden_layer_activation=swish --iterations_per_checkpoint=50 --jerk_limit_factor_braking=1.0 --jerk_limit_factor=1.0 --kl_target=0.005 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --normalize_reward_to_initial_target_point_distance --obs_add_target_point_pos --obs_add_target_point_relative_pos --obstacle_scene=1 --pos_limit_factor=1.0 --punish_action --punish_end_min_distance --robot_scene=9 --solver_iterations=50 --target_link_offset="[0.0, 0.0, -0.185]" --target_point_cartesian_range_scene=9 --target_point_radius=0.065 --target_point_reached_reward_bonus=5 --target_point_relative_pos_scene=0 --target_point_sequence=1 --torque_limit_factor=1.0 --trajectory_duration=8.0 --trajectory_time_step=0.1 --use_controller_target_velocities --use_target_points --vel_limit_factor=1.0 --time=500 
```


#### Visualization of the backup policy

After training, you can visualize rollouts of the backup policy by running:
```bash
python safemotions/evaluate.py --checkpoint=path_to_checkpoint --no_exploration --use_gui
```
You can add --plot_trajectory to plot the generated trajectory in joint space. 

**Example for the space environment**
```bash
python safemotions/evaluate.py --checkpoint=backup_networks/space --no_exploration --use_gui
```

**Example for the ball environment**
```bash
python safemotions/evaluate.py --checkpoint=backup_networks/ball --no_exploration --use_gui
```

**Example for the human environment**
```bash
python safemotions/evaluate.py --checkpoint=backup_networks/human --no_exploration --use_gui
```

### 2. Training of the risk estimator

The first step towards training the risk estimator is to generate training data.
Subsequently, a risk network can be trained via supervised learning. 

#### Generation of a dataset to train the risk estimator

**Example for the space environment**
```bash
python safemotions/evaluate.py --checkpoint=backup_networks/space --evaluation_dir=specify_path_for_risk_training_data --collision_avoidance_kinematic_state_sampling_probability=0.5 --collision_avoidance_stay_in_state_probability=1.0 --random_agent --risk_state_config=RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY --risk_store_ground_truth --risk_ground_truth_episodes_per_file=5 --risk_ignore_estimation_probability=0.35 --risk_state_deterministic_backup_trajectory --trajectory_duration=1000 --episodes_per_simulation_reset=20 --episodes=10000
```

**Example for the ball environment**
```bash
python safemotions/evaluate.py --checkpoint=backup_networks/ball --evaluation_dir=specify_path_for_risk_training_data --collision_avoidance_kinematic_state_sampling_probability=0.5 --collision_avoidance_stay_in_state_probability=1.0 --random_agent --risk_state_config=RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY --risk_store_ground_truth --risk_ground_truth_episodes_per_file=5 --risk_ignore_estimation_probability=0.35 --risk_state_deterministic_backup_trajectory --trajectory_duration=1000 --episodes_per_simulation_reset=20 --episodes=10000
```

**Example for the human environment**
```bash
python safemotions/evaluate.py --checkpoint=backup_networks/human --evaluation_dir=specify_path_for_risk_training_data --collision_avoidance_kinematic_state_sampling_probability=0.5 --collision_avoidance_stay_in_state_probability=1.0 --random_agent --risk_state_config=RISK_CHECK_NEXT_STATE_SIMULATE_NEXT_STEP_AND_BACKUP_TRAJECTORY --risk_store_ground_truth --risk_ground_truth_episodes_per_file=5 --risk_ignore_estimation_probability=0.35 --risk_state_deterministic_backup_trajectory --trajectory_duration=1000 --episodes_per_simulation_reset=20 --episodes=10000
```

The dataset generation can be accelerated by specifying the number of parallel workers with --num_workers=N. 
The risk data is stored in a subfolder of the directory specified with --evaluation_dir:
```
└── evaluation_dir
    └── safe_motions_risk_evaluation
        └── SafeMotionsEnvCollisionAvoidance
            └── name_Of_backup_policy
                └── timestamp
                    ├── state_action_risk
                    │   └── risk data for state-action-based risk estimation (*.csv)
                    ├── state_risk
                    │   └── risk data for state-based risk estimation (*.csv)
                    └── risk_config.json
```
The next step is to split the risk data into a training dataset and a test dataset. 
Use the following command to split the data for the state-action-based risk estimation so that 10% of the data is assigned to the test data set:

```bash
python safemotions/split_risk_data.py --risk_data_dir=evaluation_dir/safe_motions_risk_evaluation/SafeMotionsEnvCollisionAvoidance/name_Of_backup_policy/timestamp/state_action_risk --test_data_fraction=0.1
```

After executing the command, the state_action_risk directory will contain the subdirectories train and test with the corresponding data for training and testing the risk estimator. 

```
└── timestamp
    ├── state_action_risk
    │   ├── train
    │   │   └── training data for a state-action-based risk estimator (*.csv)        
    │   └── test
    │       └── test data for a state-action-based risk estimator (*.csv)  
    └── risk_config.json
```


#### Training the risk estimator via supervised learning

Use the following commands to train the risk estimator. The parameter --risk_data_dir needs to specify the path to the directory state_action_risk (for state-action-based risk estimation) or state_risk (for state-based risk estimation). 

**Example for the space environment**
```bash
python safemotions/train_risk_network.py --logdir=specify_path_for_risk_training_logs --batch_size=300000 --risk_data_dir=evaluation_dir/safe_motions_risk_evaluation/SafeMotionsEnvCollisionAvoidance/backup_space/timestamp/state_action_risk --dropout=0.075 --epochs=6000 --experiment_name=space --fcnet_hiddens="[512, 256, 128]" --hidden_layer_activation=selu --last_layer_activation=sigmoid --lr=0.005 --risky_state_class_weight=1.0 --shuffle
```

**Example for the ball environment**

```bash
python safemotions/train_risk_network.py --logdir=specify_path_for_risk_training_logs --batch_size=300000 --risk_data_dir=evaluation_dir/safe_motions_risk_evaluation/SafeMotionsEnvCollisionAvoidance/backup_ball/timestamp/state_action_risk --dropout=0.075 --epochs=6000 --experiment_name=ball --fcnet_hiddens="[512, 256, 128]" --hidden_layer_activation=selu --last_layer_activation=sigmoid --lr=0.005 --risky_state_class_weight=1.0 --shuffle
```

**Example for the human environment**

```bash
python safemotions/train_risk_network.py --logdir=specify_path_for_risk_training_logs --batch_size=300000 --risk_data_dir=evaluation_dir/safe_motions_risk_evaluation/SafeMotionsEnvCollisionAvoidance/backup_human/timestamp/state_action_risk --dropout=0.1 --epochs=6000 --experiment_name=human --fcnet_hiddens="[512, 256, 128]" --hidden_layer_activation=selu --last_layer_activation=sigmoid --lr=0.005  --risky_state_class_weight=1.0 --shuffle
```

The model weights of the risk estimator will be stored in a subdirectory of --logdir:
```
└── logdir
    └── state_action_risk
        └── experiment_name
            └── timestamp
                ├── model.pt
                └── risk_config.json
```


### 3. Training of the task policy

Once the backup policy and the corresponding risk estimator are trained, they can be used to avoid collisions while training a task policy. 
The commands below show how a task policy for a reaching task can be trained.
Use the parameter --risk_config_dir to specify the directory containing the file *model.pt* resulting from the training of the risk estimator.
The parameter --risk_threshold $\in [0.0, 1.0]$ controls the trade-off between the rate of adjusted actions and the occurrence of collisions.
Typically, smaller values lead to fewer collisions, but also to a lower final task performance.

**Space environment**

```bash
python safemotions/train.py --logdir=specify_path_for_training_logs --name=reaching_task_space --acc_limit_factor=1.0 --action_max_punishment=0.4  --action_punishment_min_threshold=0.95 --batch_size_factor=8 --closest_point_safety_distance=0.01 --collision_check_time=0.033 --episodes_per_simulation_reset=2000 --hidden_layer_activation=swish --iterations_per_checkpoint=50 --jerk_limit_factor=1.0 --kl_target=0.005 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --normalize_reward_to_initial_target_point_distance --obs_add_target_point_pos --obs_add_target_point_relative_pos --obstacle_scene=5 --pos_limit_factor=1.0 --punish_action --risk_check_initial_backup_trajectory --risk_config_dir=risk_networks/state_action/space --risk_state_initial_backup_trajectory_steps=20 --risk_threshold=0.01 --robot_scene=0 --solver_iterations=50 --starting_point_cartesian_range=1 --target_point_cartesian_range_scene=0 --target_point_radius=0.065 --target_point_reached_reward_bonus=5 --target_point_relative_pos_scene=0 --target_point_sequence=0 --terminate_on_collision_with_moving_obstacle --terminate_on_collision_with_static_obstacle --terminate_on_self_collision --trajectory_duration=8.0 --trajectory_time_step=0.1 --use_controller_target_velocities --use_target_points --vel_limit_factor=1.0 --time=500 
```

**Ball environment**

```bash
python safemotions/train.py --logdir=specify_path_for_training_logs --name=reaching_task_ball --acc_limit_factor=1.0 --action_max_punishment=0.4  --action_punishment_min_threshold=0.95 --batch_size_factor=8 --closest_point_safety_distance=0.01 --collision_check_time=0.033 --episodes_per_simulation_reset=2000 --hidden_layer_activation=swish --iterations_per_checkpoint=50 --jerk_limit_factor=1.0 --kl_target=0.005 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --normalize_reward_to_initial_target_point_distance --obs_add_target_point_pos --obs_add_target_point_relative_pos --obstacle_scene=5 --pos_limit_factor=1.0 --punish_action --risk_check_initial_backup_trajectory --risk_config_dir=risk_networks/state_action/ball --risk_state_initial_backup_trajectory_steps=20  --risk_threshold=0.105 --robot_scene=0 --solver_iterations=50 --starting_point_cartesian_range=1 --target_point_cartesian_range_scene=0 --target_point_radius=0.065 --target_point_reached_reward_bonus=5 --target_point_relative_pos_scene=0 --target_point_sequence=0 --terminate_on_collision_with_moving_obstacle --terminate_on_collision_with_static_obstacle --terminate_on_self_collision --trajectory_duration=8.0 --trajectory_time_step=0.1 --use_controller_target_velocities --use_target_points --vel_limit_factor=1.0 --time=500 
```

**Human environment**

```bash
python safemotions/train.py --logdir=specify_path_for_training_logs --name=reaching_task_human --acc_limit_factor=1.0 --action_max_punishment=0.4  --action_punishment_min_threshold=0.95 --batch_size_factor=2 --closest_point_safety_distance=0.01 --collision_check_time=0.033 --episodes_per_simulation_reset=2000 --hidden_layer_activation=swish --iterations_per_checkpoint=50 --jerk_limit_factor=1.0 --kl_target=0.005 --last_layer_activation=tanh --log_std_range="[-1.375, 0.0]" --normalize_reward_to_initial_target_point_distance --obs_add_target_point_pos --obs_add_target_point_relative_pos --obstacle_scene=5 --pos_limit_factor=1.0 --punish_action --risk_check_initial_backup_trajectory --risk_config_dir=risk_networks/state_action/human --risk_state_initial_backup_trajectory_steps=20  --risk_threshold=0.06 --robot_scene=0 --solver_iterations=50 --starting_point_cartesian_range=1 --target_point_cartesian_range_scene=0 --target_point_radius=0.065 --target_point_reached_reward_bonus=5 --target_point_relative_pos_scene=0 --target_point_sequence=0 --terminate_on_collision_with_moving_obstacle --terminate_on_collision_with_static_obstacle --terminate_on_self_collision --trajectory_duration=8.0 --trajectory_time_step=0.1 --use_controller_target_velocities --use_target_points --vel_limit_factor=1.0 --time=500 
```


## Publication
Further details can be found in the following [publication](https://arxiv.org/abs/2411.05784):
```
  title={Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles}, 
  journal={IEEE Robotics and Automation Letters}, 
  author={Kiemel, Jonas and Righetti, Ludovic and Kröger, Torsten and Asfour, Tamim},
  year={2024},
  volume={9},
  number={12},
  pages={11353-11360},
  doi={10.1109/LRA.2024.3488402}
```

## Video
<a href="https://youtu.be/fR9SByhCbDI"><img width="1832" alt="video_thumbnail" src="https://github.com/user-attachments/assets/787da584-4742-494a-8776-400b937ff463" /></a>


## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

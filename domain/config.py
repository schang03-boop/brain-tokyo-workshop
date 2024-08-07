from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
                           'input_size', 'output_size', 'layers', 'i_act', 'h_act',
                           'o_act', 'weightCap', 'noise_bias', 'output_noise', 'max_episode_length', 'in_out_labels'])

games = {}

# -- Cart-pole Swingup --------------------------------------------------- -- #

# > Slower reaction speed
cartpole_swingup = Game(
    env_name='CartPoleSwingUp_Hard',
    actionSelect='all',  # all, soft, hard
    input_size=5,
    output_size=1,
    time_factor=0,
    layers=[5, 5],
    i_act=np.full(5, 1),
    h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    o_act=np.full(1, 1),
    weightCap=2.0,
    noise_bias=0.0,
    output_noise=[False, False, False],
    max_episode_length=200,
    in_out_labels=['x', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot',
                   'force']
                        )
games['swingup_hard'] = cartpole_swingup

# > Normal reaction speed
cartpole_swingup = cartpole_swingup._replace( \
    env_name='CartPoleSwingUp', max_episode_length=1000)
games['swingup'] = cartpole_swingup

# -- Bipedal Walker ------------------------------------------------------ -- #

# > Flat terrain
biped = Game(
    env_name='BipedalWalker-v2',
    actionSelect='all',  # all, soft, hard
    input_size=24,
    output_size=4,
    time_factor=0,
    layers=[40, 40],
    i_act=np.full(24, 1),
    h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    o_act=np.full(4, 1),
    weightCap=2.0,
    noise_bias=0.0,
    output_noise=[False, False, False],
    max_episode_length=400,
    in_out_labels=[
        'hull_angle', 'hull_vel_angle', 'vel_x', 'vel_y',
        'hip1_angle', 'hip1_speed', 'knee1_angle', 'knee1_speed', 'leg1_contact',
        'hip2_angle', 'hip2_speed', 'knee2_angle', 'knee2_speed', 'leg2_contact',
        'lidar_0', 'lidar_1', 'lidar_2', 'lidar_3', 'lidar_4',
        'lidar_5', 'lidar_6', 'lidar_7', 'lidar_8', 'lidar_9',
        'hip_1', 'knee_1', 'hip_2', 'knee_2']
)
games['biped'] = biped

# > Hilly Terrain
bipedmed = biped._replace(env_name='BipedalWalkerMedium-v2')
games['bipedmedium'] = bipedmed

# > Obstacles, hills, and pits
bipedhard = biped._replace(env_name='BipedalWalkerHardcore-v2')
games['bipedhard'] = bipedhard

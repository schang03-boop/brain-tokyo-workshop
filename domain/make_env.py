import numpy as np
import gymnasium as gym
from matplotlib.pyplot import imread

from domain.slimevolley import SlimeVolley


def make_env(env_name, seed=123, render_mode=False):
    # -- SlimeVolley ---------------------------------------------------- -- #
    if env_name.startswith("SlimeVolley"):
        curr_env = SlimeVolley()

    return curr_env
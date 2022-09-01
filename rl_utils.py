import numpy as np
import matplotlib.pyplot as plt

def discount_rwds(r, gamma = 0.99):
    disc_rwds = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add*gamma + r[t]
        disc_rwds[t] = running_add
    return disc_rwds

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def oneD2twoD(env_shape, idx):
    return (int(idx / env_shape[1]),np.mod(idx,env_shape[1]))

def twoD2oneD(env_shape, coord_tuple):
    r,c = coord_tuple
    return (r * env_shape[1]) + c

# Visualizing the reward function another way
def plot_reward_map(gw_obs):
    R_map = plt.imshow(gw_obs.R.reshape(gw_obs.r,gw_obs.c))
    plt.colorbar(R_map)
    plt.show()

import os
import gym
import torch
from model import train
from net import es


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    chkpt_dir = 'checkpoints/CartPole/'
    os.makedirs(chkpt_dir, exist_ok=True)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    synced_model = es(observation_dim, action_dim)
    train(
        synced_model,
        env,
        chkpt_dir,
        max_gradient_updates=100000,
        model_num=40,
        sigma=0.05,
        max_episode_length=100000,
        learning_rate=1e-1,
        lr_decay=1,
        variable_ep_len=False,
        cpu_num=4
    )
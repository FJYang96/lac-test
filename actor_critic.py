import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import LinearSystem
from gymnasium.wrappers import FlattenObservation
from utils import *


class LinearTrackingWrapper(gym.Env):
    def __init__(self, A, B, Q, R, T):
        super().__init__()
        self.env = LinearSystem(A, B, T)
        self.T = T
        self.n = A.shape[0]
        self.m = B.shape[1]

        self.Q = Q
        self.R = R

        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(self.m,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.n,), dtype=np.float32
        )

        self.ref_traj = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.ref_traj = generate_sin_traj(self.T)
        self.t = 0
        self.env.t = 0
        self.env.x = self.env.x0

        return self._get_obs(), {}

    def step(self, action):
        u = np.array(action).astype(np.float32)
        x, _, _, _ = self.env.step(u)
        r = self.ref_traj[self.t]
        self.t += 1

        tracking_cost = (x - r).T @ self.Q @ (x - r)
        control_cost = u.T @ self.R @ u
        reward = -(tracking_cost + control_cost)

        done = self.t >= self.T
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        if self.t < len(self.ref_traj):
            r = self.ref_traj[self.t]
        else:
            r = np.zeros(self.env.state_space)
        return np.concatenate([self.env.x, r]).astype(np.float32)


import matplotlib.pyplot as plt

def plot_trajectory_vs_reference(env, model, T):
    obs, _ = env.reset()
    x_traj = []
    r_traj = []

    for _ in range(T):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        x_traj.append(env.env.x.copy())
        r_traj.append(env.ref_traj[env.t - 1])  # previous r_t

    x_traj = np.array(x_traj)
    r_traj = np.array(r_traj)

    for i in range(env.n):
        plt.plot(x_traj[:, i], label=f"x[{i}]")
        plt.plot(r_traj[:, i], '--', label=f"r[{i}]")

    plt.legend()
    plt.grid(True)
    plt.show()

# Define system
T = 20
n = 2
m = 1
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(n)
R = np.eye(m) * 0.1

env = LinearTrackingWrapper(A, B, Q, R, T)
check_env(env)

# model = PPO("MlpPolicy", env, verbose=1)

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128, 128, 128],
                   vf=[64, 64])]
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    batch_size=32,
    n_epochs=10,
    gamma=0.97
)

model.learn(total_timesteps=50000)

model.save("ppo_linear")

obs, _ = env.reset()
for _ in range(T):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(env.env.x, reward)

plot_trajectory_vs_reference(env,model, T)

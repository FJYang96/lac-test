import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LinearSystem(gym.Env):
    def __init__(self, A, B, T=50):
        super().__init__()

        self.state_space = A.shape[0]
        self.action_space = B.shape[1]  # control inputs

        # self.action_limit = 2

        self.x0 = np.random.normal(loc=0.0, scale=1.0, size=self.state_space)
        print('x0', self.x0)

        self.A = A
        self.B = B

        self.T = T  # seconds / time horizon
        self.t = 0  # start time

        self.x = self.x0

    def step(self, u):
        # forward dynamics step: x(t+1) = Ax(t) + Bu(t)

        # u = np.clip(u, -self.action_limit, self.action_limit)
        self.x = self.A @ self.x + self.B @ u

        self.t += 1

        cost = self.x.T @ self.x + 0.1 * (u.T @ u)
        done = self.t >= self.T  # or np.linalg.norm(self.x) < 1e-3

        return self.x, -cost, done, {}

    def reset(self):
        # go back to initial state and reset time
        self.x = self.x0
        self.t = 0
        return self.x

    def render(self, mode='human'):
        print(f"t={self.t}, state={self.x}")



class UnicycleTrackingEnv(gym.Env):
    def __init__(self, T):
        super().__init__()

        self.T = T
        self.dt = 1
        self.state_dim = 4
        self.action_dim = 2

        self.Q = np.eye(self.state_dim)
        self.R = 0.1 * np.eye(self.action_dim)

        self.x0 = np.random.normal(0, 1, size=self.state_dim)
        print("x0", self.x0)

        self.x = self.x0
        self.t = 0
        self.ref_traj = self._generate_unicycle_reference()

    def _generate_unicycle_reference(self):
        ref = []
        for t in range(self.T + 1):
            angle = 0.1 * t
            px = 2.0 * np.cos(angle)
            py = 2.0 * np.sin(angle)
            theta = angle + np.pi / 2
            v = 0.2
            ref.append([px, py, theta, v])
        return np.array(ref)

    def step(self, u):
        u = np.array(u).astype(np.float64)
        a, omega = u[0], u[1]

        px, py, theta, v = self.x
        dx = np.array([
            np.cos(theta) * v,
            np.sin(theta) * v,
            omega,
            a
        ])

        self.x = self.x + self.dt * dx
        self.t += 1

        if self.t < len(self.ref_traj):
            r = self.ref_traj[self.t]
        else:
            r = np.zeros(self.state_dim)

        tracking_error = self.x - r
        cost = tracking_error.T @ self.Q @ tracking_error + u.T @ self.R @ u
        done = self.t >= self.T

        return self.x, -cost, done, {}

    def reset(self):
        self.x0 = np.random.normal(0, 1, size=self.state_dim)
        self.x = self.x0
        self.t = 0
        self.ref_traj = self._generate_unicycle_reference()
        return self.x

    def render(self, mode='human'):
        print(f"t={self.t}, state={self.x}")

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class ClassicalRegulation:  # drives states to zero
    def __init__(self, env, Q, R):
        self.env = env
        self.Q = Q
        self.R = R

    def calc_cost(self, x, u):
        control_cost = 0.1 * (u @ self.R @ u)
        state_cost = x @ self.Q @ x
        return control_cost.sum(0) + state_cost.sum(0)

    def next_u(self, x_curr):
        A = self.env.A
        B = self.env.B
        Q = self.Q
        R = self.R

        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        x_curr = np.asarray(x_curr).reshape(-1, 1)
        u = -K @ x_curr
        return u.flatten()

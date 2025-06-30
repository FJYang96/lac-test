import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class iLQR_Unicycle:
    def __init__(self, env, Q, R, ref_traj):
        self.env = env
        self.Q = Q
        self.R = R
        self.ref_traj = ref_traj
        self.T = env.T
        self.n = 4  # state: [px, py, theta, v]
        self.m = 2  # control: [a, omega]
        self.dt = env.dt

        self.K_list = []
        self.kff_list = []
        self._compute_gains()

    def _dynamics_jacobians(self, x, u):
        px, py, theta, v = x
        a, omega = u
        dt = self.dt

        A = np.eye(self.n)
        A[0, 2] = -dt * v * np.sin(theta)
        A[0, 3] = dt * np.cos(theta)
        A[1, 2] = dt * v * np.cos(theta)
        A[1, 3] = dt * np.sin(theta)
        A[2, 2] = 1
        A[3, 3] = 1

        B = np.zeros((self.n, self.m))
        B[2, 1] = dt  # theta += omega * dt
        B[3, 0] = dt  # v += a * dt

        return A, B

    def _simulate_dynamics(self, x, u):
        theta = x[2]
        v = x[3]
        dx = np.array([
            np.cos(theta) * v,
            np.sin(theta) * v,
            u[1],
            u[0]
        ])
        return x + self.dt * dx

    def _compute_gains(self):
        T, n, m = self.T, self.n, self.m
        Q, R = self.Q, self.R

        P = Q.copy()
        K_list = []
        kff_list = []

        for t in reversed(range(T)):
            x_ref = self.ref_traj[t]
            u_ref = np.zeros(m)

            A, B = self._dynamics_jacobians(x_ref, u_ref)

            S = R + B.T @ P @ B
            F = B.T @ P @ A
            K = np.linalg.solve(S, F)
            A_cl = A - B @ K
            P = Q + K.T @ R @ K + A_cl.T @ P @ A_cl

            K_list.insert(0, K)
            kff_list.insert(0, -K @ x_ref)

        self.K_list = K_list
        self.kff_list = kff_list

    def next_u(self, x, t):
        K = self.K_list[t]
        kff = self.kff_list[t]
        u = -K @ x + kff
        return u

    def simulate(self, x0):
        traj = [x0.copy()]
        u_list = []
        cost_list = []

        x = x0.copy()

        for t in range(self.T):
            u = self.next_u(x, t)
            x_ref = self.ref_traj[t]
            e = x - x_ref

            cost = e.T @ self.Q @ e + u.T @ self.R @ u

            x = self._simulate_dynamics(x, u)

            traj.append(x.copy())
            u_list.append(u.copy())
            cost_list.append(cost)

        return np.array(traj), np.array(u_list), np.array(cost_list)

    def plot_results(self, traj, u_list, cost_list):
        ref_array = np.array(self.ref_traj[:len(traj)])
        time_steps = np.arange(len(traj))

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].plot(traj[:, 0], traj[:, 1], label="Trajectory", color="blue")
        axs[0].plot(ref_array[:, 0], ref_array[:, 1], '--', label="Reference",
                    color="red")
        axs[0].set_title("Position (px, py)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(u_list[:, 0], label="a", color="green")
        axs[1].plot(u_list[:, 1], label="omega", color="orange")
        axs[1].set_title("Control Inputs")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(cost_list, label="Instantaneous Cost", color="black")
        axs[2].set_title("Tracking Cost")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

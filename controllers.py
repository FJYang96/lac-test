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


class FiniteHorizonLQR:  # using augmented state x' = [x, r]
    def __init__(self, env, Q, R, ref_traj):
        self.env = env
        self.A = env.A
        self.B = env.B
        self.n = env.state_space
        self.m = env.action_space
        # print(self.n, self.m)

        self.Q = Q
        self.R = R
        self.ref_traj = ref_traj
        self.T = env.T

        self._build_aug_system()
        self._solve_finite_horizon_lqr()

    def _build_aug_system(self):
        n, m = self.n, self.m
        T = self.T

        Z = np.zeros((T * n, T * n))
        for i in range(T - 1):
            Z[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = np.eye(n)
        Z[-n:, -n:] = np.eye(n)  # keep r_T fixed in last row

        # augmented A and B

        self.A_aug = np.zeros((n + T * n, n + T * n))
        self.A_aug[:n, :n] = self.A
        self.A_aug[n:, n:] = Z
        self.B_aug = np.vstack([self.B, np.zeros((T * n, m))])

        # augmented Q for (x - r) Q (x - r)
        Q_bar = np.zeros((n + T * n, n + T * n))
        Q_bar[:n, :n] = self.Q
        Q_bar[:n, n:n + n] = -self.Q
        Q_bar[n:n + n, :n] = -self.Q
        Q_bar[n:n + n, n:n + n] = self.Q
        self.Q_bar = Q_bar

    def _solve_finite_horizon_lqr(self):

        # self.P = [None] * (self.T + 1)
        # self.K = [None] * self.T
        # self.P[self.T] = self.Q_bar.copy()
        #
        # for t in reversed(range(self.T)):
        #     S = self.B_aug.T @ self.P[t + 1] @ self.B_aug + self.R
        #     F = self.B_aug.T @ self.P[t + 1] @ self.A_aug
        #     self.K[t] = np.linalg.solve(S , F)
        #     A_cl = self.A_aug - self.B_aug @ self.K[t]
        #     self.P[t] = self.Q_bar + self.K[t].T @ self.R @ self.K[t] + A_cl.T @ self.P[t + 1] @ A_cl

        P = self.Q_bar.copy()

        for _ in range(self.T):
            K = np.linalg.inv(self.R + self.B_aug.T @ P @ self.B_aug) @ (
                    self.B_aug.T @ P @ self.A_aug)
            P = self.Q_bar + self.A_aug.T @ P @ self.A_aug - self.A_aug.T @ P @ self.B_aug @ K

        self.K = K

    def next_u(self, x_aug, t):
        # return (-self.K[-1] @ x_aug).flatten()
        return (-self.K @ x_aug).flatten()

    def simulate(self, x0):
        x = x0.copy()
        n = self.n
        T = self.T

        # initial augmented state: [x0, r_0, ..., r_{T-1}]
        r_window = np.concatenate([self.ref_traj[i] for i in range(T)], axis=0)
        x_aug = np.concatenate([x, r_window])
        traj = [x.copy()]
        u_list = []
        cost_list = []

        for t in range(self.T):
            u = self.next_u(x_aug, t)
            r = self.ref_traj[t]
            e = x - r
            cost = e.T @ self.Q @ e + u.T @ self.R @ u

            # update augmented state: [x, r_{t+1}, ..., r_{t+T}, r_{t+T}]

            x_aug = self.A_aug @ x_aug + self.B_aug @ u
            x = x_aug[:self.n]

            traj.append(x.copy())
            u_list.append(u.copy())
            cost_list.append(cost.item())

        traj = np.array(traj)
        u_list = np.array(u_list)
        cost_list = np.array(cost_list)

        return traj, u_list, cost_list

    def plot_results(self, traj, u_list, cost_list):
        ref_array = np.array(self.ref_traj[:len(traj)])
        time_steps = np.arange(len(traj))

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].plot(time_steps, traj[:, 0], label="x[0]", color="blue")
        axs[0].plot(time_steps, ref_array[:, 0], '--', label="r[0]",
                    color="red")
        axs[0].plot(time_steps, traj[:, 1], label="x[1]", color="green")
        axs[0].plot(time_steps, ref_array[:, 1], '--', label="r[1]",
                    color="orange")
        axs[0].set_title("State and Reference Trajectories")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time_steps[:-1], u_list, label="u", color="purple")
        axs[1].set_title("Control Inputs")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(time_steps[:-1], cost_list, label="Cost", color="black")
        axs[2].set_title("Instantaneous Tracking Cost")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

class FiniteHorizonLQRUnicycle:
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
        axs[0].plot(ref_array[:, 0], ref_array[:, 1], '--', label="Reference", color="red")
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
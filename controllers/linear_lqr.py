import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class FiniteHorizonLQR:  # using augmented state x' = [x, r]
    def __init__(self, env, Q, R, ref_traj=None):
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

        if ref_traj is None:
            print('generating traj')
            self._generate_ref_traj(env.x0)
        else:
            self.ref_traj = ref_traj

        # print(self.ref_traj)


    def _generate_ref_traj(self, x0):
        T = self.T
        x0 = np.atleast_1d(x0)
        n = self.n

        c = x0[0]  # x(0), shape (2,)
        b = x0[1]  # x_dot(0), shape (2,)
        a = np.random.uniform(-0.1, 0.1)
        print('a = ', a)
        print('c = ', c)
        print('b = ', b)

        traj = []
        for t in range(T + 1):
            # t = tn * 0.1
            x_t = a * t ** 2 + b * t + c  # now all vector math: shape (2,)
            v_t = 2 * a * t + b
            traj.append([x_t, v_t])

        self.ref_traj = np.array(traj)

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

    def _solve_finite_horizon_lqr_old(self):
        A = self.A
        B = self.B
        T = self.T
        R = self.R
        Q = self.Q

        P = [None] * (T + 1)
        K = [None] * (T)
        P[T] = np.eye(2)  # Final cost

        for t in range(T - 1, -1, -1):
            K[t] = -np.linalg.inv(R + B.T @ P[t + 1] @ B) @ B.T @ P[t + 1] @ A
            P[t] = Q + A.T @ P[t + 1] @ A + A.T @ P[t + 1] @ B @ K[t]

        self.P = P
        self.K = K
        print(K[0])
        # return P, K



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
        return (-self.K[-1] @ x_aug).flatten()
        # return (-self.K @ x_aug).flatten()

    def simulate(self):
        x = self.env.x0
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



import numpy as np
import matplotlib.pyplot as plt

class PID_Unicycle:
    def __init__(self, env, Q, R, ref_traj, kp_pos=1.0, ki_pos=0.0, kd_pos=0.1,
                 kp_theta=4.0, ki_theta=0.0, kd_theta=0.2):
        self.env = env
        self.Q = Q
        self.R = R
        self.ref_traj = ref_traj
        self.T = env.T
        self.dt = env.dt
        self.n = 4  # [px, py, theta, v]
        self.m = 2  # [a, omega]

        # PID gains for linear speed
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        # PID gains for heading
        self.kp_theta = kp_theta
        self.ki_theta = ki_theta
        self.kd_theta = kd_theta

        self.integral_error_pos = np.zeros(2)
        self.prev_error_pos = np.zeros(2)
        self.integral_error_theta = 0.0
        self.prev_error_theta = 0.0

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

    def next_u(self, x, t):
        x_ref = self.ref_traj[t]
        pos = x[:2]
        theta = x[2]
        v = x[3]

        # Position error
        pos_ref = x_ref[:2]
        pos_error = pos_ref - pos
        self.integral_error_pos += pos_error * self.dt
        derror_pos = (pos_error - self.prev_error_pos) / self.dt
        self.prev_error_pos = pos_error

        # Desired heading to target
        heading_ref = np.arctan2(pos_error[1], pos_error[0])
        heading_error = heading_ref - theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        self.integral_error_theta += heading_error * self.dt
        derror_theta = (heading_error - self.prev_error_theta) / self.dt
        self.prev_error_theta = heading_error

        # Compute control inputs
        a = (
            self.kp_pos * np.linalg.norm(pos_error) +
            self.ki_pos * np.linalg.norm(self.integral_error_pos) +
            self.kd_pos * np.linalg.norm(derror_pos)
        )

        omega = (
            self.kp_theta * heading_error +
            self.ki_theta * self.integral_error_theta +
            self.kd_theta * derror_theta
        )

        u = np.array([a, omega])
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

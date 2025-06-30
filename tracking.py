from controllers import *
from env import *
from utils import *
from controllers.linear_lqr import  FiniteHorizonLQR
import numpy as np


def simulate_linear_aug_lqr(A, B, Q, R, T, ref_traj = None):
    env = LinearSystem(A, B, T)
    controller = FiniteHorizonLQR(env, Q, R, ref_traj)
    x0 = env.x0
    traj, u_list, cost_list = controller.simulate()
    controller.plot_results(traj, u_list, cost_list)


# A1 = np.array([[1.0, 0.1], [0, 1.0]])
# B1 = np.array([[0],
#                [0.1]])
#
# A2 = np.array([[1.2, 0.2],
#                [0.1, 0.95]])
# B2 = np.array([[1],
#                [0]])
#
# A3 = np.array([[0.8, 0.2],
#                [-0.2, 1.1]])
# B3 = np.array([[0],
#                [1]])
#
# Q = np.array([[100.1, 0],
#               [0, 100.5]])
# R = 0.1 * np.eye(1)
# T = 100

# simulate_linear_regulation(A3, B3, Q, R, 20)
# ref_traj = np.array(generate_sin_traj(T))
ref_traj = None

# A = np.array([[1, 1],
#               [0, 1]])
# B = np.array([[0],
#               [1]])
A = np.array([[0.95, 1],
              [-0.1, 0.98]])
B = np.array([[0],
              [0.5]])
# A = np.array([[1.1, 1],
#               [0.0, 1.05]])
# B = np.array([[0],
#               [1]])
Q = 100*np.eye(2)
R = 0.1 * np.eye(1)
Qf = np.eye(2)  # terminal cost (can also try np.eye(2))
T = 20

simulate_linear_aug_lqr(A, B, Q, R, T,ref_traj)



#
# def simulate_unicycle_lqr(T, Q, R, ref_traj):
#     env = UnicycleTrackingEnv(T)
#     controller = FiniteHorizonLQRUnicycle(env, Q, R, ref_traj)
#     x0 = env.x0
#     traj, u_list, cost_list = controller.simulate(x0)
#     controller.plot_results(traj, u_list, cost_list)
#
# def generate_sin_traj_unicycle(T, dt=0.1, amplitude=2.0, frequency=0.05,
#                                velocity=0.05):
#     """
#     Generates a sine wave reference trajectory for a unicycle model.
#
#     Returns:
#         ref_traj: (T+1, 4) array with each row as [px, py, theta, v]
#     """
#     ref_traj = []
#     for t in range(T + 1):
#         time = t * dt
#         px = velocity * time
#         py = amplitude * np.sin(2 * np.pi * frequency * time)
#
#         dx = velocity
#         dy = amplitude * 2 * np.pi * frequency * np.cos(
#             2 * np.pi * frequency * time)
#         theta = np.arctan2(dy, dx)
#
#         ref_traj.append([px, py, theta, velocity])
#
#     return np.array(ref_traj, dtype=np.float32)
#
#
# T = 100
# Q = np.diag([10.0, 10.0, 1.0, 1.0])
# R = 0.1 * np.eye(2)
# ref_traj = generate_sin_traj_unicycle(T)
#
# simulate_unicycle_lqr(T, Q, R, ref_traj)

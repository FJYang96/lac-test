import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from controllers import FiniteHorizonLQR
from env import LinearSystem


def generate_sinusoidal_ref(T, n, amplitude=1.0, frequency=0.2):
    ref_traj = np.array([
        [np.sin(0.1 * t), np.cos(0.1 * t)] for t in range(T + 1)
    ])
    return ref_traj


def dual_ascent_with_lqr(env, Q, R, rho, xi, ref_true, max_iters=10):
    n, T = env.state_space, env.T
    print('n', n)
    nu = np.zeros((T + 1, n))
    r = ref_true.copy()

    residuals = []

    for _ in range(max_iters):
        ref_plus_nu = r + nu

        lqr = FiniteHorizonLQR(env, Q, R, ref_plus_nu)
        x_traj, u_traj, _ = lqr.simulate(xi)
        print(x_traj)

        r_var = cp.Variable((T + 1, n))
        cost = (cp.sum(
            [cp.quad_form(r_var[t] - ref_true[t], Q) for t in range(T + 1)]) +
                (rho / 2) * cp.sum_squares(r_var + nu - x_traj))
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        r = r_var.value

        nu = nu + (r - x_traj)

        residuals.append(np.linalg.norm(r - x_traj))

    return r, x_traj, u_traj, nu, residuals, ref_true


A = np.array([[1.0, 0.5], [0.0, 1.0]])
B = np.array([[0], [1.0]])
T = 50
dt = 1
env = LinearSystem(A, B, T)

Q = 10000 * np.eye(2)
R = 0.01 * np.eye(1)
rho = 2.0
xi = np.array([1.0, 1.0])
X_bounds = (np.full(2, -10.0), np.full(2, 10.0))

ref_sin = generate_sinusoidal_ref(T, n=2, amplitude=1.0, frequency=0.2)
print(ref_sin)

r_final, x_final, u_final, nu_final, residuals, ref_sin = dual_ascent_with_lqr(
    env, Q, R, rho, xi, ref_sin, max_iters=10)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(residuals, marker='o')
plt.title("Convergence: ||r - x|| over iterations")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_final[:, 0], label="x[0]")
plt.plot(r_final[:, 0], '--', label="r[0]")
plt.plot(ref_sin[:, 0], ':', label="r_ref[0]")
plt.title("Tracking Sinusoidal Reference")
plt.xlabel("Time Step")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_final[:, 1], label="x[1]")
plt.plot(r_final[:, 1], '--', label="r[1]")
plt.plot(ref_sin[:, 1], ':', label="r_ref[1]")
plt.title("Tracking Sinusoidal Reference")
plt.xlabel("Time Step")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

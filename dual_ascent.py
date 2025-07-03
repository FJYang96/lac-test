import cvxpy as cp
import pandas as pd
from controllers.linear_lqr import *
from env import *

def dual_ascent(A, B, Q, R, rho, T, xi, X_bounds, U_bounds, nu_init, max_iters=10):
    n, m = A.shape[0], B.shape[1]
    nu = nu_init.copy()
    r_sol = np.zeros((T + 1, n))

    for _ in range(max_iters):
        #  (x, u)
        x = cp.Variable((T + 1, n))
        u = cp.Variable((T, m))
        r_nu_sum = r_sol + nu
        cost_xu = (cp.sum([cp.quad_form(u[t], R) for t in range(T)]) +
                   (rho / 2) * cp.sum_squares(r_nu_sum - x))

        constraints = [x[0] == xi]
        for t in range(T):
            constraints += [x[t + 1] == A @ x[t] + B @ u[t],
                            u[t] >= U_bounds[0], u[t] <= U_bounds[1]]

        prob_xu = cp.Problem(cp.Minimize(cost_xu), constraints)
        prob_xu.solve()
        x_sol = x.value
        u_sol = u.value

        # r
        r = cp.Variable((T + 1, n))
        # cost_r = cp.sum([cp.quad_form(r[t], Q) for t in range(T + 1)]) +
        #           (rho / 2) * cp.sum_squares(r + nu - x_sol)

        ref_traj = np.array([
            [np.sin(0.2 * t), 0.0] for t in range(T + 1)
        ])
        cost_r = cp.sum(
            [cp.quad_form(r[t] - ref_traj[t], Q) for t in range(T + 1)]) + \
                 (rho / 2) * cp.sum_squares(r + nu - x_sol)

        prob_r = cp.Problem(cp.Minimize(cost_r), [r >= X_bounds[0], r <= X_bounds[1]])
        prob_r.solve()
        r_sol = r.value

        # ν
        nu = nu + (r_sol - x_sol)

    return r_sol, x_sol, u_sol, nu

# System setup
n, m = 2, 1
T = 10
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(n)
R = np.eye(m)
rho = 1.0
X_bounds = (np.full(n, -10.0), np.full(n, 10.0))
U_bounds = (np.array([-1.0]), np.array([1.0]))




initial_conditions = [
    np.array([0.0, 0.0]),
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([-1.0, 0.5]),
    np.array([2.0, -1.0])
]

results = []
trajectories = []

for xi in initial_conditions:
    nu_init = np.zeros((T + 1, n))
    r_sol, x_sol, u_sol, nu_final = dual_ascent(
        A, B, Q, R, rho, T, xi, X_bounds, U_bounds, nu_init, max_iters=10
    )
    residual_norm = np.linalg.norm(x_sol - r_sol)
    results.append({
        "initial_condition": xi,
        "residual_norm": residual_norm
    })
    trajectories.append((xi, r_sol, x_sol, u_sol, nu_final))

# Display results table
df_results = pd.DataFrame(results)
print(df_results)




# Plot one case
xi, r_sol, x_sol, u_sol, nu_sol = trajectories[3]

plt.figure(figsize=(12, 5))

# Plot state and reference
plt.subplot(1, 2, 1)
plt.plot(x_sol[:, 0], label="x1")
plt.plot(x_sol[:, 1], label="x2")
plt.plot(r_sol[:, 0], '--', label="r1")
plt.plot(r_sol[:, 1], '--', label="r2")
plt.title("State vs Reference")
plt.xlabel("Time Step")
plt.legend()

# Plot control
plt.subplot(1, 2, 2)
plt.step(range(T), u_sol[:, 0], where='post', label="u")
plt.title("Control Inputs")
plt.xlabel("Time Step")
plt.legend()

plt.tight_layout()
# plt.show()

plt.figure()
plt.plot(r_sol + nu_sol)
plt.plot(r_sol, '--')
plt.show()

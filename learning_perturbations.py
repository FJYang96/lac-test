import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def solve_r(A, B, Q, R, T, xi, nu):

    n = A.shape[0]

    r = cp.Variable((T + 1, n))

    ref_cost = 0
    for t in range(T + 1):
        ref_cost += cp.sum_squares(r[t])  # r_t^T I r_t where I is identity

    tracking_penalty = 0
    for t in range(T):
        if t < nu.shape[0]:
            perturbed_ref = r[t] + nu[t]
            tracking_penalty += cp.quad_form(perturbed_ref, Q)
        else:
            tracking_penalty += cp.quad_form(r[t], Q)

    objective = cp.Minimize(ref_cost + 0.1 * tracking_penalty)

    constraints = []
    # constraints.append(cp.norm(r, 'inf') <= 10.0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    return r.value


def solve_xu(A, B, Q, R, T, xi, r_plus_nu, rho=1.0):
    # Solve eq (3)
    n, m = A.shape[0], B.shape[1]

    x = cp.Variable((T + 1, n))
    u = cp.Variable((T, m))

    control_cost = 0
    for t in range(T):
        control_cost += 0.1 * cp.sum_squares(u[t])  # 0.1 * u_t^T I u_t

    tracking_cost = 0
    for t in range(T):
        if t < r_plus_nu.shape[0] - 1:
            tracking_error = r_plus_nu[t] - x[t]
            tracking_cost += (rho / 2) * cp.sum_squares(tracking_error)

    objective = cp.Minimize(control_cost + tracking_cost)


    constraints = []
    constraints.append(x[0] == xi)

    for t in range(T):
        constraints.append(x[t + 1] == A @ x[t] + B @ u[t])

    # constraints.append(cp.norm(u, 'inf') <= 5.0)
    # constraints.append(cp.norm(x, 'inf') <= 20.0)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value, u.value, problem.value


def dual_ascent(A, B, Q, R, T, xi, max_iterations=25, rho=1.0):
    n = A.shape[0]

    nu = np.zeros((T, n))

    convergence_data = {
        'iterations': [],
        'nu_norms': [],
        'primal_residuals': [],
        'nu_changes': [],
        'r_trajectories': [],
        'x_trajectories': [],
        'nu_trajectories': [],
        'objective_values': []
    }

    for k in range(max_iterations):
        r_plus = solve_r(A, B, Q, R, T, xi, nu)

        if r_plus is None:
            print(f"✗ r-subproblem failed at iteration {k}")
            break


        r_plus_nu = r_plus.copy()
        r_plus_nu[:-1] += nu

        x_plus, u_plus, obj_value = solve_xu(A, B, Q, R, T,
                                                              xi, r_plus_nu,
                                                              rho)

        if obj_value == float('inf'):
            print(f"✗ xu-subproblem failed at iteration {k}")
            break

        r_executable = r_plus[:-1]  # Remove final reference point
        x_executable = x_plus[:-1]  # Remove final state (not controllable)

        min_len = min(r_executable.shape[0], x_executable.shape[0],
                      nu.shape[0])
        r_executable = r_executable[:min_len]
        x_executable = x_executable[:min_len]

        primal_residual = r_executable - x_executable
        nu_new = nu.copy()
        nu_new[:min_len] = nu[:min_len] + primal_residual

        nu_norm = np.linalg.norm(nu_new)
        primal_residual_norm = np.linalg.norm(primal_residual)
        nu_change = np.linalg.norm(nu_new - nu)

        # Store data
        convergence_data['iterations'].append(k)
        convergence_data['nu_norms'].append(nu_norm)
        convergence_data['primal_residuals'].append(primal_residual_norm)
        convergence_data['nu_changes'].append(nu_change)
        convergence_data['r_trajectories'].append(r_plus.copy())
        convergence_data['x_trajectories'].append(x_plus.copy())
        convergence_data['nu_trajectories'].append(nu_new.copy())
        convergence_data['objective_values'].append(obj_value)

        # Status determination
        if primal_residual_norm < 1e-5 and nu_change < 1e-5:
            status = "CONVERGED"
        elif nu_norm > 100 or obj_value > 1e6:
            status = "DIVERGING"
        elif k > 10 and nu_change < 1e-3:
            status = "SLOW CONV"
        else:
            status = "ITERATING"

        print(
            f"{k:3d}  {nu_norm:7.4f}  {primal_residual_norm:9.6f}  {nu_change:9.6f}  {obj_value:9.2f}  {status}")

        nu = nu_new

    #     if status == "CONVERGED":
    #         print(f"\n✓ CVXPy algorithm converged after {k + 1} iterations!")
    #         break
    #     elif status == "DIVERGING":
    #         print(f"\n✗ CVXPy algorithm diverging at iteration {k + 1}")
    #         break
    #
    # if k == max_iterations - 1:
    #     print(f"\n⚠ Maximum iterations ({max_iterations}) reached")
    #     final_status = "MAX_ITER"
    # else:
    #     final_status = status

    # Final verification
    print(f"\nFinal CVXPy Results:")
    print(f"  Final ||ν||: {np.linalg.norm(nu):.6f}")
    print(f"  Final ||r-x||: {primal_residual_norm:.6f}")
    print(f"  Final objective: {obj_value:.4f}")

    return nu, convergence_data



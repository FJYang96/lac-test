from controllers import *
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import cvxpy as cp
from env import LinearSystem
from utils import generate_spline_traj
from controllers.linear_lqr import FiniteHorizonLQR
import matplotlib.pyplot as plt


def solve_r(A, B, Q, R, T, x_traj, nu):
    n = A.shape[0]
    r = cp.Variable((T + 1, n))

    cost = 0
    for t in range(T + 1):
        rt = r[t, :]
        nut = nu[t, :] if t < nu.shape[0] else np.zeros(n)
        cost += cp.quad_form(rt, Q) - nut @ rt

    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    return r.value

def dual_ascent(env, Q, R, rho, xi, r_init, max_iters=8):
    n, T = env.state_space, env.T
    r = r_init.copy()
    nu = np.zeros_like(r)
    trajs, nus, rs = [], [], []

    for i in range(max_iters):
        print(nu.shape)
        r_perturbed = r + nu
        controller = FiniteHorizonLQR(env, Q, R, r_perturbed)
        x_traj, u_traj, _ = controller.simulate()

        r = solve_r(env.A, env.B, Q, R, T, x_traj, nu)


        nu = nu + rho * (x_traj - r)

        trajs.append(x_traj.copy())
        nus.append(nu.copy())
        rs.append(r.copy())

    return trajs, rs, nus

# def dual_ascent_with_model(A, B, Q, R, T, rho, model, max_iters=1, ref_traj = None):
#     env = LinearSystem(A,B, T)
#     n, T = env.state_space, env.T
#     trajs, nus, rs = [], [], []
#
#     controller = FiniteHorizonLQR(env, Q, R)
#     if ref_traj is None:
#         r = controller.ref_traj
#     else:
#         r = ref_traj
#     nu = model.predict(r.flatten()[None])[0].reshape(T + 1, n)
#
#     x_traj, u_traj, _ = controller.simulate()
#
#     r = solve_r(env.A, env.B, Q, R, T, x_traj, nu)
#     nu = nu + rho * (x_traj - r)
#     r_perturbed = r + nu
#
#     trajs.append(x_traj.copy())
#     nus.append(nu.copy())
#     rs.append(r_perturbed.copy())
#
#     traj = trajs[-1]
#     ref = rs[-1]
#     plt.figure(figsize=(10, 4))
#     for i in range(n):
#         plt.plot(traj[:, i], label=f"r[{i}]")
#         plt.plot(ref[:, i], '--', label=f"r + v[{i}]")
#     plt.legend()
#     plt.title("Final Trajectory vs Reference with Predicted ν")
#     plt.grid(True)
#     plt.show()
#
#     return trajs, rs, nus

def dual_ascent_with_model(A, B, Q, R, T, rho, model, ref_traj):
    env = LinearSystem(A, B, T)
    n = env.state_space

    r = ref_traj
    nu = model.predict(r.flatten()[None])[0].reshape(T + 1, n)

    r_perturbed = r + nu
    controller = FiniteHorizonLQR(env, Q, R, r_perturbed)
    x_traj, u_traj, _ = controller.simulate()

    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.plot(x_traj[:, i], label=f"x[{i}]")
        plt.plot(r_perturbed[:, i], '--', label=f"r + v[{i}]")
    plt.legend()
    plt.title("Trajectory with Predicted ν")
    plt.grid(True)
    plt.show()

    return x_traj, r_perturbed, nu

def generate_dual_data(env, Q, R, rho, num_samples=1000, max_iters=10):
    r_list = []
    nu_star_list = []

    for _ in range(num_samples):
        controller = FiniteHorizonLQR(env, Q, R)
        r_init = controller.ref_traj

        # r_init =   # shape (T+1, n)
        trajs, rs, nus = dual_ascent(env, Q, R, rho, env.x0, r_init, max_iters)

        r_final = rs[-1].flatten()
        nu_final = nus[-1].flatten()

        r_list.append(r_final)
        nu_star_list.append(nu_final)

    return np.array(r_list), np.array(nu_star_list)

def build_dual_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(input_dim)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def train_dual_network(A, B, Q, R, T, rho=1.0, num_samples=1000):
    env = LinearSystem(A, B, T)
    r_data, nu_data = generate_dual_data(env, Q, R, rho, num_samples)
    X_train, X_val, y_train, y_val = train_test_split(r_data, nu_data, test_size=0.1)

    model = build_dual_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_data=(X_val, y_val))

    loss = model.evaluate(X_val, y_val)
    print(f"Validation MSE: {loss:.4f}")

    preds = model.predict(X_val)
    idx = 0
    plt.plot(y_val[idx], label='True ν*')
    plt.plot(preds[idx], label='Predicted ν')
    plt.legend()
    plt.title("Sample ν Prediction vs Ground Truth")
    plt.grid(True)
    plt.show()

    return model

if __name__ == "__main__":
    T = 20
    n = 2
    m = 1
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(n)
    R = np.eye(m) * 0.1


    trained_model = train_dual_network(A, B, Q, R,T, rho=1.0, num_samples=1000)

    r_test = generate_spline_traj(T)
    dual_ascent_with_model(A, B, Q, R,T, rho=1.0, model=trained_model, ref_traj = r_test)

    trained_model.save("dual_network_modelnew.h5")
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from tracking import *
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from env import *
from utils import *
from controllers import *
import matplotlib.pyplot as plt


def collect_rollouts(A, B, Q, R, T, num_rollouts=100):
    ref_traj_list = []
    cumulative_costs = []
    env = LinearSystem(A, B, T)

    for _ in range(num_rollouts):
        ref_traj = generate_sin_traj(T)
        controller = FiniteHorizonLQR(env, Q, R, ref_traj)
        traj, u_list, cost_list = controller.simulate(env.x0)

        cumulative_costs.append(np.sum(cost_list))
        ref_traj_list.append(ref_traj)

    controller.plot_results(traj, u_list, cost_list)
    np.savez('rollout_data_1.npz', refs=ref_traj_list, costs=cumulative_costs)
    return np.array(ref_traj_list), np.array(cumulative_costs)


def flatten_refs(ref_traj_list):
    flat_refs = np.array([r.flatten() for r in ref_traj_list])
    return flat_refs


def build_quadratic_features(X):
    """
    X: shape (N, d), where each row is a flattened ref traj
    Returns: Phi: (N, num_features), with quadratic + linear + constant features
    """
    N, d = X.shape
    quad_terms = np.einsum('ni,nj->nij', X, X)  # shape: (N, d, d)
    # Take only upper triangular part (flattened)
    triu_idx = np.triu_indices(d)
    quad_flat = quad_terms[:, triu_idx[0],
                triu_idx[1]]  # shape: (N, d*(d+1)/2)
    linear = X  # (N, d)
    bias = np.ones((N, 1))  # (N, 1)
    return np.hstack([quad_flat, linear, bias])


def fit_quadratic_model(ref_traj_list, cumulative_costs):
    flat_refs = flatten_refs(ref_traj_list)
    Phi = build_quadratic_features(flat_refs)
    reg = LinearRegression().fit(Phi, cumulative_costs)
    # reg = Ridge(alpha=1.0).fit(Phi, cumulative_costs)
    return reg


def plot_training_quad(reg, cumulative_costs, ref_traj_list):
    flat_refs = flatten_refs(ref_traj_list)
    Phi_all = build_quadratic_features(flat_refs)
    pred_costs = reg.predict(Phi_all)

    # plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(cumulative_costs, pred_costs, alpha=0.6, s=20,
                label='Rollouts')
    plt.plot([min(cumulative_costs), max(cumulative_costs)],
             [min(cumulative_costs), max(cumulative_costs)],
             'r--', label='Ideal Prediction')

    plt.xlabel("True Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Quadratic Model Predictions on Training Rollouts")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_testing_quad(reg, env, Q, R, num_tests=50):
    true_costs = []
    pred_costs = []

    print("\nEvaluating on 50 unseen sine wave references:")
    for i in range(num_tests):
        ref_traj = generate_sin_traj(T)
        controller = FiniteHorizonLQR(env, Q, R, ref_traj)
        traj, _, cost_list = controller.simulate(env.x0)

        true_cost = np.sum(cost_list)
        flat_ref = np.array(ref_traj).flatten()[None]
        phi_ref = build_quadratic_features(flat_ref)

        pred_cost = reg.predict(phi_ref)[0]

        true_costs.append(true_cost)
        pred_costs.append(pred_cost)

    true_costs = np.array(true_costs)
    pred_costs = np.array(pred_costs)

    # plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(true_costs, pred_costs, alpha=0.7, label='Trajectories')
    plt.plot([0, true_costs.max()],
             [0, true_costs.max()],
             'r--', label='Ideal Prediction')
    plt.xlabel("True Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Quadratic Model Predictions on Unseen Sine References")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def fit_neural_network(ref_traj_list, costs, T):
    ref_seqs = np.array(ref_traj_list)  # Shape: (N, T+1, n)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(T + 1, n)),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Training neural network...")

    model.fit(ref_seqs, costs, epochs=50, batch_size=32, verbose=1)
    return model

def plot_training_nn(nn_model, cumulative_costs, ref_traj_list):
    ref_seqs = np.array(ref_traj_list)  # shape: (N, T+1, n)
    pred_costs = nn_model.predict(ref_seqs).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(cumulative_costs, pred_costs, alpha=0.6, s=20, label='Rollouts')
    plt.plot([min(cumulative_costs), max(cumulative_costs)],
             [min(cumulative_costs), max(cumulative_costs)],
             'r--', label='Ideal Prediction')

    plt.xlabel("True Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Neural Network Predictions on Training Rollouts")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_testing_nn(nn_model, env, Q, R, T, num_tests=50):
    true_costs = []
    pred_costs = []

    print("\nEvaluating on unseen sine wave references:")
    for _ in range(num_tests):
        ref_traj = generate_sin_traj(T)  # shape (T+1, n)
        ref_traj = np.array(ref_traj)
        controller = FiniteHorizonLQR(env, Q, R, ref_traj)
        traj, _, cost_list = controller.simulate(env.x0)

        true_cost = np.sum(cost_list)
        pred_cost = nn_model.predict(ref_traj[None])[0][0]

        true_costs.append(true_cost)
        pred_costs.append(pred_cost)

    true_costs = np.array(true_costs)
    pred_costs = np.array(pred_costs)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_costs, pred_costs, alpha=0.7, label='Trajectories')
    plt.plot([0, true_costs.max()],
             [0, true_costs.max()],
             'r--', label='Ideal Prediction')
    plt.xlabel("True Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Neural Network Predictions on Unseen Sine References")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def predict_cost(model, r, model_type='quad'):
    """
    model_type: 'quad' or 'nn'
    r: shape (T+1, n)
    """
    if model_type == 'quad':
        r_flat = r.flatten()[None]
        phi = build_quadratic_features(r_flat)
        return model.predict(phi)[0]
    elif model_type == 'nn':
        r_seq = r[None]  # shape: (1, T+1, n)
        return model.predict(r_seq, verbose=0)[0][0]
    else:
        raise ValueError("model_type must be 'quad' or 'nn'")


def optimize_trajectory_gradient_descent(model, r_init, T, n,
                                         model_type='quad', lr=1e-2,
                                         max_iters=100):
    def cost_fn(r_flat):
        r = r_flat.reshape((T + 1, n))
        return predict_cost(model, r, model_type)

    res = minimize(cost_fn, r_init.flatten(), method='L-BFGS-B',
                   options={'maxiter': 100, 'disp': True})

    # return res.x.reshape((T + 1, n))
    return res.x.reshape(r_init.shape)


def optimize_trajectory_cem(model, T, n, model_type='quad', num_samples=100, elite_frac=0.1, iterations=10):
    d = (T + 1) * n
    mu = np.zeros(d)
    std = np.ones(d) * 0.5

    for _ in range(iterations):
        samples = np.random.randn(num_samples, d) * std + mu
        costs = np.array([
            predict_cost(model, s.reshape((T + 1, n)), model_type)
            for s in samples
        ])

        elite_idxs = np.argsort(costs)[:int(num_samples * elite_frac)]
        elite_samples = samples[elite_idxs]
        mu = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

    return mu.reshape((T + 1, n))


A = np.array([[2.0, 3], [0, 2.0]])
B = np.array([[0],
              [1]])

Q = np.array([[10.1, 0],
              [0, 10.5]])
R = 0.1 * np.eye(1)
T = 100
n = A.shape[0]
env = LinearSystem(A, B, T)

### COLLECTING ROLLOUTS ###

# load_data = False
load_data = True
if load_data:
    data = np.load('rollout_data_1.npz', allow_pickle=True)
    ref_traj_list = data['refs']
    cumulative_costs = data['costs']
else:
    ref_traj_list, cumulative_costs = collect_rollouts(A, B, Q, R, T,
                                                       num_rollouts=1000)

### TRAINING QUADRATIC MODEL ###

train_quad = False
if train_quad:
    quadratic_model = fit_quadratic_model(ref_traj_list, cumulative_costs)
    joblib.dump(quadratic_model, 'quad_model.pkl')
else:
    quadratic_model = joblib.load('quad_model.pkl')

plot_training_quad(quadratic_model, cumulative_costs, ref_traj_list)
plot_testing_quad(quadratic_model, env, Q, R)

### TRAINING NEURAL NETWORK MODEL ###
train_nn = False
if train_nn:
    nn_model = fit_neural_network(ref_traj_list, cumulative_costs, T)
    nn_model.save('nn_model_1.keras')
else:
    nn_model = load_model('nn_model_1.keras')

plot_training_nn(nn_model, cumulative_costs, ref_traj_list)
plot_testing_nn(nn_model, env, Q, R, T, num_tests=50)

### GENERATE NEW TRAJECTORIES ###

r_init = np.array(generate_sin_traj(T))
traj_gd = optimize_trajectory_gradient_descent(quadratic_model, r_init, T, n, model_type='quad')
traj_cem = optimize_trajectory_cem(quadratic_model, T, n, model_type='quad')
simulate_linear_aug_lqr(A, B, Q, R, T, traj_cem)

traj_gd_nn = optimize_trajectory_gradient_descent(nn_model, r_init, T, n, model_type='nn')
traj_cem_nn = optimize_trajectory_cem(nn_model, T, n, model_type='nn')
simulate_linear_aug_lqr(A, B, Q, R, T, traj_cem_nn)

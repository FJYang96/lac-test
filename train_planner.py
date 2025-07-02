from env import *
from controllers.linear_lqr import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM, Dropout, Linear


def simulate_linear_aug_lqr(A, B, Q, R, T, ref_traj=None):
    env = LinearSystem(A, B, T)
    controller = FiniteHorizonLQR(env, Q, R, ref_traj)
    x0 = env.x0
    traj, u_list, cost_list = controller.simulate()
    controller.plot_results(traj, u_list, cost_list)


def collect_rollouts(A, B, Q, R, T, num_rollouts=100):
    ref_traj_list = []
    cumulative_costs = []

    for i in range(num_rollouts):
        # print(i)
        env = LinearSystem(A, B, T)
        ref_traj = None
        controller = FiniteHorizonLQR(env, Q, R, ref_traj)
        traj, u_list, cost_list = controller.simulate()

        cumulative_costs.append(np.sum(cost_list))
        print(controller.ref_traj)
        ref_traj_list.append(controller.ref_traj)
        if i % 100 == 0:
            controller.plot_results(traj, u_list, cost_list)

    np.savez('rollout_data_new_ref.npz', refs=ref_traj_list, costs=cumulative_costs)
    return np.array(ref_traj_list), np.array(cumulative_costs)


def flatten_refs(traj_list):
    flat_refs = np.array([r.flatten() for r in traj_list])
    return flat_refs


def build_quadratic_features(X):
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


def plot_testing_quad(reg, A, B, Q, R, T, num_tests=50):
    true_costs = []
    pred_costs = []

    print("\nEvaluating on 50 unseen references:")
    for i in range(num_tests):
        env = LinearSystem(A, B, T)
        controller = FiniteHorizonLQR(env, Q, R)
        traj, _, cost_list = controller.simulate()

        true_cost = np.sum(cost_list)
        ref_traj = controller.ref_traj
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


import torch
import torch.nn as nn
import torch.optim as optim


class CostPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        return self.fc(x[:, -1, :])  # use final timestep


from torch.utils.data import TensorDataset, DataLoader


def fit_neural_network(ref_traj_list, costs, T, batch_size=32, epochs=50, lr=1e-3):
    ref_seqs = np.array(ref_traj_list, dtype=np.float32)  # (N, T+1, n)
    costs = np.array(costs, dtype=np.float32).reshape(-1, 1)

    N, seq_len, n = ref_seqs.shape
    model = CostPredictorLSTM(n)

    X_tensor = torch.tensor(ref_seqs)
    y_tensor = torch.tensor(costs)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training neural network...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss / len(dataset):.4f}")

    return model


def plot_training_nn(nn_model, cumulative_costs, ref_traj_list):
    ref_seqs = torch.tensor(np.array(ref_traj_list), dtype=torch.float32)
    nn_model.eval()
    with torch.no_grad():
        pred_costs = nn_model(ref_seqs).squeeze().numpy()

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


def plot_testing_nn(nn_model, A, B, Q, R, T, num_tests=50):
    true_costs = []
    pred_costs = []

    print("\nEvaluating on unseen sine wave references:")
    for _ in range(num_tests):
        env = LinearSystem(A, B, T)
        controller = FiniteHorizonLQR(env, Q, R)
        ref_traj = controller.ref_traj  # shape (T+1, n)
        ref_traj_np = np.array(ref_traj, dtype=np.float32)
        traj, _, cost_list = controller.simulate()
        true_cost = np.sum(cost_list)

        ref_input = torch.tensor(ref_traj_np[None], dtype=torch.float32)
        with torch.no_grad():
            pred_cost = nn_model(ref_input).item()

        true_costs.append(true_cost)
        pred_costs.append(pred_cost)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_costs, pred_costs, alpha=0.7, label='Trajectories')
    plt.plot([0, max(true_costs)],
             [0, max(true_costs)],
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
        r_seq = torch.tensor(r[None], dtype=torch.float32)  # shape: (1, T+1, n)
        model.eval()
        with torch.no_grad():
            return model(r_seq).item()
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


# def optimize_trajectory_cem(model, T, n, model_type='quad', num_samples=100, elite_frac=0.1, iterations=10):
#     d = (T + 1) * n
#     mu = np.zeros(d)
#     std = np.ones(d) * 0.5
#
#     for _ in range(iterations):
#         samples = np.random.randn(num_samples, d) * std + mu
#         costs = np.array([
#             predict_cost(model, s.reshape((T + 1, n)), model_type)
#             for s in samples
#         ])
#
#         elite_idxs = np.argsort(costs)[:int(num_samples * elite_frac)]
#         elite_samples = samples[elite_idxs]
#         mu = elite_samples.mean(axis=0)
#         std = elite_samples.std(axis=0)
#
#     return mu.reshape((T + 1, n))

def optimize_trajectory_cem(model, T, n, model_type='quad',
                            num_samples=100, elite_frac=0.1,
                            iterations=10, min_std=1e-2):
    d = (T + 1) * n
    mu = np.zeros(d)
    std = np.ones(d) * 0.5

    for _ in range(iterations):
        samples = np.random.randn(num_samples, d) * std + mu
        # Optional: clip or bound sample space
        # samples = np.clip(samples, -5, 5)

        costs = np.array([
            predict_cost(model, s.reshape((T + 1, n)), model_type)
            for s in samples
        ])
        costs = np.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=1e6)

        num_elites = max(2, int(num_samples * elite_frac))
        elite_idxs = np.argsort(costs)[:num_elites]
        elite_samples = samples[elite_idxs]

        mu = elite_samples.mean(axis=0)
        std = np.maximum(elite_samples.std(axis=0), min_std)

    return mu.reshape((T + 1, n))


A = np.array([[1, 1], [0, 1]])
n = 2
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.eye(1)
Qf = np.eye(2)  # terminal cost (can also try np.eye(2))
T = 20
env = LinearSystem(A, B, T)
controller = FiniteHorizonLQR(env, Q, R)

### COLLECTING ROLLOUTS ###

# load_data = False
load_data = True
if load_data:
    data = np.load('rollout_data_new_ref.npz', allow_pickle=True)
    ref_traj_list = data['refs']
    cumulative_costs = data['costs']
    print(ref_traj_list)
else:
    ref_traj_list, cumulative_costs = collect_rollouts(A, B, Q, R, T,
                                                       num_rollouts=1000)

### TRAINING QUADRATIC MODEL ###

train_quad = True
if train_quad:
    quadratic_model = fit_quadratic_model(ref_traj_list, cumulative_costs)
    joblib.dump(quadratic_model, 'quad_model.pkl')
else:
    quadratic_model = joblib.load('quad_model.pkl')

plot_training_quad(quadratic_model, cumulative_costs, ref_traj_list)
plot_testing_quad(quadratic_model, A, B, Q, R, T)

### TRAINING NEURAL NETWORK MODEL ###
train_nn = False
if train_nn:
    nn_model = fit_neural_network(ref_traj_list, cumulative_costs, T)
    torch.save(nn_model.state_dict(), 'lstm_cost_model.pt')
else:
    nn_model = CostPredictorLSTM(n)
    nn_model.load_state_dict(torch.load('lstm_cost_model.pt'))
    nn_model.eval()

# plot_training_nn(nn_model, cumulative_costs, ref_traj_list)
# plot_testing_nn(nn_model, A, B, Q, R, T, num_tests=50)

### GENERATE NEW TRAJECTORIES ###
#
r_init = controller.ref_traj

traj_gd = optimize_trajectory_gradient_descent(quadratic_model, r_init, T, n, model_type='quad')
traj_cem = optimize_trajectory_cem(quadratic_model, T, n, model_type='quad')
simulate_linear_aug_lqr(A, B, Q, R, T, traj_cem)
simulate_linear_aug_lqr(A, B, Q, R, T, traj_gd)

# traj_gd_nn = optimize_trajectory_gradient_descent(nn_model, r_init, T, n, model_type='nn')
# traj_cem_nn = optimize_trajectory_cem(nn_model, T, n, model_type='nn')
# simulate_linear_aug_lqr(A, B, Q, R, T, traj_cem_nn)
# simulate_linear_aug_lqr(A, B, Q, R, T, traj_gd_nn)

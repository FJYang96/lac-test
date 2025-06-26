import numpy as np
from scipy.interpolate import CubicSpline

def generate_sin_traj(T):
    dt = 1
    time_steps = np.arange(T + 1) * dt
    amp1 = np.random.uniform(0.1, 1)
    amp2 = np.random.uniform(0.1, 1)
    phase1 = np.random.uniform(0, np.pi)
    phase2 = np.random.uniform(0, np.pi)

    return [np.array([
        amp1 * np.sin(0.05 * t + phase1),
        amp2 * np.cos(0.05 * t + phase2)]) for t in time_steps]


def generate_spline_traj(T, dt=1, num_waypoints=4, scale=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    T = T +1
    total_time = T * dt
    time_steps = np.linspace(0, total_time, T)

    waypoint_times = np.linspace(0, total_time, num_waypoints)

    waypoints_x = np.random.randn(num_waypoints) * scale
    waypoints_y = np.random.randn(num_waypoints) * scale

    spline_x = CubicSpline(waypoint_times, waypoints_x)
    spline_y = CubicSpline(waypoint_times, waypoints_y)

    reference_traj = []
    for t in time_steps:
        r = np.array([spline_x(t), spline_y(t)])
        reference_traj.append(r)

    return np.array(reference_traj)

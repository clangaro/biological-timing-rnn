import numpy as np
from timedpred.dynamics import Parameters, Weights
from timedpred.simulate import SimConfig, make_trial_inputs_minimal, simulate_trial

import matplotlib.pyplot as plt

def init_weights(rng: np.random.Generator, n_e: int, n_i: int, w_scale: float=0.05) -> Weights:
    """Initialize weights with small random values."""
    return Weights(
        w_ee=rng.normal(0.0, w_scale, size=(n_e, n_e)),
        w_ei=rng.normal(0.0, w_scale, size=(n_e, n_i)),
        w_ie=rng.normal(0.0, w_scale, size=(n_i, n_e)),
        w_ii=rng.normal(0.0, w_scale, size=(n_i, n_i)),
    )

def main():
    n_e = 20
    n_i = 10
    fraction_a = 0.5
    n_a = int(np.round(fraction_a * n_e))


    cs_onset = 0.050
    cs_duration = 0.020
    us_duration = 0.020
    
    training_intervals = [0.25, 0.30, 0.35]  # seconds
    random_seeds = [0, 1, 2, 3, 4, 5]

    N_train = 20
    epsilon_bias = 0.01

    # Learning strenght 
    eta_ee = 1e-4
    w_ee_max = 1.0

    C = SimConfig(dt=0.001, max_time=0.6, record_every=1)

    # Bias terms and initial state
    b_e = np.zeros(n_e)
    b_i = np.zeros(n_i)
    r_e0 = np.zeros(n_e)
    r_i0 = np.zeros(n_i)
    s_e0 = np.zeros(n_e)
    e_e0 = np.zeros(n_e)

    # Result bins 
    mean_peak_times = []
    std_peak_times = []
    expected_us_times = []

    

    for interval in training_intervals:
        for seed in random_seeds:
            np.random.seed(seed)

            # Run simulation with current interval and seed
            traj = simulate_trial(
                r_e0, r_i0, s_e0, e_e0, inputs_cs_only, b_e, b_i, W, P_test, C
            )

            # Compute peak time error for this trial
            peak_time_B = traj.t[np.argmax(traj.r_e[:, n_a:])]
            expected_us_time = interval + 0.250  # CS-US interval
            error_B = peak_time_B - expected_us_time
            peak_time_errors.append(error_B)

    # Plot results
    plt.figure()
    plt.bar(training_intervals, peak_time_errors)
    plt.xlabel("Training interval (s)")
    plt.ylabel("Peak time error (s)")
    plt.title("Peak time error vs. training interval")
    plt.show()

if __name__ == "__main__":
    main()
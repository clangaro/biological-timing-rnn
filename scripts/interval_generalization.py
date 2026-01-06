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

    # Outer loop over training intervals

    for interval in training_intervals:
        us_onset = cs_onset + interval 
        expected_us_times.append(us_onset)

        peak_times_this_interval = []

        # Inner loop over random seeds
        for seed in random_seeds:
            rng = np.random.default_rng(seed)

            #New network per each seed 
            W = init_weights(rng, n_e=n_e, n_i=n_i, w_scale=0.05)

            # Weak bias from E_A to E_B
            W.w_ee[n_a:, :n_a] += epsilon_bias

            # Training Parameters
            P_train = Parameters(
                tau_e=0.020,
                tau_i=0.010,
                tau_s=0.200,
                tau_elig=0.25,
                eta_ee=eta_ee,
                w_ee_max=w_ee_max
            )

            # Training inputs
            inputs_train = make_trial_inputs_minimal(
                n_e=n_e,
                n_i=n_i,
                dt=C.dt,
                max_time=C.max_time,
                cs_onset=cs_onset,
                cs_duration=cs_duration,
                cs_amp=1.0,
                us_onset=us_onset,
                us_duration=us_duration,
                us_amp_e=1.0,
                us_amp_i=1.0,
                fraction_a=fraction_a
            )

            #Train for N trials
            for trial in range(N_train):
                simulate_trial(
                    r_e0, r_i0, s_e0, e_e0, inputs_train, b_e, b_i, W, P_train, C
                )

            # Testing Parameters (no learning)
            P_test = Parameters(
                tau_e=0.020,
                tau_i=0.010,
                tau_s=0.200,
                tau_elig=0.2,
                eta_ee=eta_ee,
                w_ee_max=w_ee_max
            )

            # Testing inputs (CS only)
            inputs_test = make_trial_inputs_minimal(
                n_e=n_e,
                n_i=n_i,
                dt=C.dt,
                max_time=C.max_time,
                cs_onset=cs_onset,
                cs_duration=cs_duration,
                cs_amp=1.0,
                us_amp_e=0.0,
                us_amp_i=0.0,
                fraction_a=fraction_a
            )

            # Simulate test trial
            traj = simulate_trial(
                r_e0, r_i0, s_e0, e_e0, inputs_test, b_e, b_i, W, P_test, C
            )

            mean_r_r_B = traj.r_e[:, n_a:].mean(axis=1)
            peak_time_B = traj.t[np.argmax(mean_r_r_B)]
            peak_times_this_interval.append(peak_time_B)

        mean_peak_times.append(float(np.mean(peak_times_this_interval)))
        std_peak_times.append(float(np.std(peak_times_this_interval)))

    # Plot results
    expected_us_times = np.array(expected_us_times)
    mean_peak_times = np.array(mean_peak_times)
    std_peak_times = np.array(std_peak_times)

    plt.figure()
    plt.errorbar(
        expected_us_times, mean_peak_times,
        yerr=std_peak_times,
        fmt="o",
        capsize=4,
        label="EB peak time (mean ± sd across seeds)",
    )

    # Reference line: perfect timing
    t_min = min(expected_us_times.min(), mean_peak_times.min())
    t_max = max(expected_us_times.max(), mean_peak_times.max())
    plt.plot([t_min, t_max], [t_min, t_max], linestyle="--", color="gray", label="y = x (perfect)")

    plt.xlabel("Expected US time (cs_onset + trained interval) [s]")
    plt.ylabel("Measured EB peak time in CS-only test [s]")
    plt.title("Training-interval generalisation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
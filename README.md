# Project Goal

The goal of this project is to construct a minimal, biologically motivated **rate-based** recurrent network in which:

- Timed predictions emerge intrinsically from network dynamics, without explicit clock variables or delay lines
- Timed prediction errors (including omission responses exceeding presence responses) arise from circuit interactions, without dedicated error units
- Learning depends on a local, interpretable plasticity rule
- Network behaviour maps onto the cortical timing phenomena reported by **Liu & Buonomano (2025)**

The emphasis is on **dynamical systems and circuit mechanisms**, not performance-optimised machine learning. This is a firing-rate model, not a spiking network.

---

# Minimal Network Architecture

## Populations

- Two excitatory populations, **EA** and **EB**, driven by temporally ordered inputs (CS-like and US-like)
- A single inhibitory population, **I**, providing feedforward inhibition and stabilisation

This is the minimal circuit motif capable of expressing learned sequential activation, asymmetric coupling, and inhibition-stabilised dynamics.

## Rate Dynamics

Population firing rates obey continuous-time rate equations, integrated by forward Euler:

```math
\tau_E \, \dot{\mathbf{r}}_E = -\mathbf{r}_E + \phi\!\left( W_{EE}\,\mathbf{s}_E - W_{EI}\,\mathbf{r}_I + \mathbf{u}_E(t) + \mathbf{b}_E \right)
```

```math
\tau_I \, \dot{r}_I = -r_I + \phi\!\left( W_{IE}\,\mathbf{r}_E - W_{II}\,r_I + u_I(t) + b_I \right)
```

where **φ(·)** is a threshold-linear (ReLU) nonlinearity. Recurrent excitatory input is mediated by the slow trace **sE** (below), not by instantaneous rates.

## Stability Regime

Initial weights are small and non-negative, with inhibition sufficient to keep the network in a stable, damped-transient regime.

---

# Emergent Timing Mechanism

Timing is not encoded explicitly. The intrinsic timescale arises from a **slow synaptic current** (NMDA-like filtering of excitatory activity):

```math
\tau_s \, \dot{\mathbf{s}}_E = -\mathbf{s}_E + \mathbf{r}_E
```

Recurrent excitatory input is mediated via **sE**, so the slow time constant `tau_s` sets the circuit's internal timescale — producing temporal structure without delay lines or clocks.

---

# Learning Rule

Only excitatory-to-excitatory (**E → E**) synapses learn. They follow a local, temporally asymmetric Hebbian rule driven by an eligibility trace:

```math
\tau_{\text{elig}} \, \dot{e}_j = -e_j + r_j(t)
```

```math
\Delta w_{ij} \propto \eta_{EE}\, r_i(t)\, e_j(t)
```

Because the eligibility trace keeps the presynaptic (CS-driven) population tagged when the postsynaptic (US-driven) population is active, this rule strengthens **EA → EB** coupling specifically — the directional asymmetry is emergent, not hard-coded. Weights obey Dale's law (clipped to be non-negative) and are bounded above by `w_ee_max`.

Inhibitory weights are fixed; they do not learn in the current implementation.

---

# Training Paradigm

Each trial:

- A brief input (**CS**) drives **EA**
- After a fixed interval **Δ**, a second input (**US**) drives **EB**, and also drives **I** (feedforward inhibition)

Through plasticity, asymmetric **EA → EB** coupling emerges, so that after training the CS alone evokes a delayed response in **EB**. Testing uses CS-only trials (learning off).

---

# Emergent Timed Prediction Error

The prediction error arises from circuit interactions, not explicit computation:

- When the **US** occurs, it drives both **EB** and **I**; the recruited inhibition suppresses the internally generated predicted response
- When the **US** is omitted, this inhibition is absent, disinhibiting the predicted response

As a result, **omission responses exceed presence responses**, producing a timed prediction-error signal as an emergent circuit effect.

---

# Expected Outcomes

The included simulations test whether, after training:

- **CS alone** evokes a delayed response in **EB**, time-locked to the trained interval (`run_sim.py`)
- The **EB peak time tracks the trained interval** across different Δ (`interval_generalization.py`)
- **CS-only (omission)** trials show an enhanced late response relative to CS+US trials (omission > presence)

These behaviours correspond to the cortical timing phenomena in **Liu & Buonomano (2025)**.

---

# Implemented vs. Planned

**Implemented:**
- Rate-based E/I dynamics with threshold-linear nonlinearity
- Slow-current (`tau_s`) timing mechanism
- Eligibility-trace Hebbian learning on E → E synapses, with Dale's law and weight bounds
- Feedforward-inhibition prediction-error mechanism (US drives I)
- Emergent EA → EB directional coupling

**Planned / not yet implemented:**
- Inhibitory homeostatic plasticity (inhibitory weights currently fixed)
- Short-term synaptic plasticity (facilitation/depression) as an alternative timing source
- An anti-Hebbian decay term in the E → E rule

---

# Project Scope

This model prioritises interpretability over optimisation, minimal circuit complexity, and a direct correspondence between model components and biological mechanisms. 

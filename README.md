# Modeling the Gravitational Three-Body Problem with Deep Learning Architectures

## Abstract

The gravitational three-body problem exhibits deterministic chaos with positive maximal Lyapunov exponents, rendering long-term numerical trajectory predictions exceptionally sensitive to initial perturbations ($e^{\lambda t}$). While deep neural networks can approximate non-linear dynamical systems, unconstrained architectures typically suffer from rapid phase-space drift, secular energy non-conservation, and violation of fundamental physical symmetries.

This repository systematically benchmarks four neural modeling paradigms for the planar three-body problem, demonstrating how geometric inductive biases, equivariant message passing, Hamiltonian mechanics, and symplectic integrators mitigate error accumulation.

---

## Architecture Summary

| Model | Implementation | Mathematical Mapping | Integrator | Conservation Properties |
| :--- | :--- | :--- | :--- | :--- |
| **PINN** | `Three_Body_Problem_PINN_...ipynb` | $t \to [x, y]$ | Direct evaluation | Soft penalty via loss term; subject to drift |
| **Autoregressive MLP** | `autoregressive_3body.py` | $S_t \to S_{t+\Delta t}$ | Discrete stepping | Unconstrained; rapid secular drift |
| **Equivariant GNN + Neural ODE** | `gnn_ode_3body.py` | $dS/dt = \text{GNN}(S)$ | Runge-Kutta 4 (continuous) | Exact momentum ($\mathbf{P}, \mathbf{L}$) via central forces; bounded energy |
| **Hamiltonian GNN + Symplectic Verlet** | `hnn_symplectic_3body.py` | $\dot{q} = \partial \mathcal{H}/\partial p, \; \dot{p} = -\partial \mathcal{H}/\partial q$ | Störmer-Verlet (symplectic) | Strictly conservative force field; exact momentum ($\mathbf{P}, \mathbf{L}$); bounded energy |

---

## Methodological Formulations

### 1. Physics-Informed Neural Network (PINN)

**Implementation:** `Three_Body_Problem_PINN_(Physics_Informed_Neural_Network).ipynb`

The PINN parameterizes particle coordinates as a direct function of continuous time:

$$
f_\theta(t) \approx \mathbf{x}(t) = [x_1(t), y_1(t), x_2(t), y_2(t), x_3(t), y_3(t)]^T
$$

The network employs Fourier feature embeddings and sinusoidal activation functions. Optimization is performed against a composite objective:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}}
$$

where the physical residuals are evaluated at collocation points via automatic differentiation:

$$
\mathcal{L}_{\text{phys}} = \frac{1}{N} \sum_{i=1}^3 \left\| \frac{d^2 \mathbf{x}_i}{dt^2} - \frac{\mathbf{F}_i}{m_i} \right\|^2, \quad \mathcal{L}_{\text{energy}} = \frac{1}{N} \sum_{k=1}^N \left| \frac{dE}{dt}(t_k) \right|^2
$$

**Theoretical Limitations:** While effective for trajectory interpolation within a bounded observational window $[0, T]$, the model does not learn the autonomous dynamical generator $d\mathbf{S}/dt$. Consequently, it cannot extrapolate beyond the training horizon ($t > T$) or generalize across varied initial conditions without complete retraining.

---

### 2. Autoregressive State-Stepping MLP

**Implementation:** `autoregressive_3body.py`

The autoregressive baseline models the discrete transition operator over fixed time steps $\Delta t$:

$$
\mathbf{S}_{t+\Delta t} = \mathbf{S}_t + g_\theta(\mathbf{S}_t)
$$

where $\mathbf{S} \in \mathbb{R}^{12}$ denotes the complete phase-space coordinate vector $[x_1, y_1, x_2, y_2, x_3, y_3, v_{x1}, v_{y1}, v_{x2}, v_{y2}, v_{x3}, v_{y3}]^T$.

Training incorporates Gaussian perturbation noise and two-step temporal unrolling to improve operational stability:

$$
\mathcal{L} = \left\| g_\theta(\mathbf{S}_t + \epsilon) - \Delta \mathbf{S}_t \right\|^2 + \left\| g_\theta(\hat{\mathbf{S}}_{t+1}) - \Delta \mathbf{S}_{t+1} \right\|^2
$$

**Theoretical Limitations:** The model lacks relational inductive bias, permutation equivariance, and phase-space volume preservation (Liouville's theorem). Small errors compound exponentially over recursive evaluations, leading to secular drift and unphysical escape trajectories.

---

### 3. Equivariant Graph Neural Network with Neural ODE

**Implementation:** `gnn_ode_3body.py`

This architecture decouples interaction modeling from temporal integration. The three bodies are represented as nodes in a fully connected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, with pairwise directed edges $(i, j)$.

To enforce physical laws by construction, the edge model parameterizes scalar central forces:

$$
\mathbf{F}_{ij} = f_\theta\left( \left\| \mathbf{r}_{ij} \right\| \right) \frac{\mathbf{r}_j - \mathbf{r}_i}{\left\| \mathbf{r}_{ij} \right\| + \epsilon}
$$

where $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ is the relative displacement vector, and $f_\theta$ is a Multi-Layer Perceptron operating exclusively on scalar geometric invariants ($\left\| \mathbf{r}_{ij} \right\|$ and $\left\| \mathbf{r}_{ij} \right\|^{-2}$).

The net acceleration on body $i$ is obtained via message aggregation:

$$
\mathbf{a}_i = \frac{1}{m_i} \sum_{j \neq i} \mathbf{F}_{ij}
$$

The full state derivative is passed to an adaptive Runge-Kutta ODE solver:

$$
\frac{d\mathbf{S}}{dt} = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \\ \mathbf{v}_3 \\ \mathbf{a}_1 \\ \mathbf{a}_2 \\ \mathbf{a}_3 \end{bmatrix}, \quad \mathbf{S}(t) = \mathbf{S}(t_0) + \int_{t_0}^t \frac{d\mathbf{S}}{d\tau} d\tau
$$

**Symmetry Guarantees:**
- **Newton's Third Law:** Since $\mathbf{r}_{ji} = -\mathbf{r}_{ij}$ and $\left\| \mathbf{r}_{ji} \right\| = \left\| \mathbf{r}_{ij} \right\|$, the anti-symmetry $\mathbf{F}_{ji} = -\mathbf{F}_{ij}$ holds algebraically.
- **Linear Momentum Conservation:** Net internal forces vanish identically: $\sum_{i} \sum_{j \neq i} \mathbf{F}_{ij} = \mathbf{0}$.
- **Angular Momentum Conservation:** Central forces exert zero net internal torque: $\sum_i \mathbf{r}_i \times \sum_{j \neq i} \mathbf{F}_{ij} = \mathbf{0}$.
- **Rotational Equivariance:** The formulation is strictly equivariant under global rotations $SO(2)$.

---

### 4. Hamiltonian Graph Neural Network with Symplectic Integration

**Implementation:** `hnn_symplectic_3body.py`

This formulation incorporates Hamiltonian mechanics directly into the network architecture. Instead of parameterizing accelerations, the neural network learns a scalar pairwise potential $V_\theta(r)$:

$$
\mathcal{V}_\theta(\mathbf{q}) = \sum_{i < j} V_\theta\left( \left\| \mathbf{q}_j - \mathbf{q}_i \right\| \right)
$$

The total Hamiltonian is given by canonical kinetic and potential energy contributions:

$$
\mathcal{H}(\mathbf{q}, \mathbf{p}) = \sum_{i=1}^3 \frac{\left\| \mathbf{p}_i \right\|^2}{2m_i} + \mathcal{V}_\theta(\mathbf{q})
$$

Equations of motion are generated through Hamilton's canonical equations via automatic differentiation:

$$
\dot{\mathbf{q}}_i = \frac{\partial \mathcal{H}}{\partial \mathbf{p}_i} = \frac{\mathbf{p}_i}{m_i}, \quad \dot{\mathbf{p}}_i = -\frac{\partial \mathcal{H}}{\partial \mathbf{q}_i} = -\nabla_{\mathbf{q}_i} \mathcal{V}_\theta(\mathbf{q})
$$

Because forces are derived as gradients of a scalar potential, the vector field is guaranteed to be conservative:

$$
\nabla \times \mathbf{F} = \mathbf{0}
$$

**Symplectic Integration:**

Temporal evolution is computed using the Störmer-Verlet (Leapfrog) symplectic scheme:

$$
\begin{aligned}
\mathbf{p}\left(t + \frac{\Delta t}{2}\right) &= \mathbf{p}(t) + \frac{\Delta t}{2} \mathbf{F}(\mathbf{q}(t)) \\
\mathbf{q}(t + \Delta t) &= \mathbf{q}(t) + \Delta t \frac{\mathbf{p}\left(t + \frac{\Delta t}{2}\right)}{m} \\
\mathbf{p}(t + \Delta t) &= \mathbf{p}\left(t + \frac{\Delta t}{2}\right) + \frac{\Delta t}{2} \mathbf{F}(\mathbf{q}(t + \Delta t))
\end{aligned}
$$

This integration maps phase space while preserving the canonical differential 2-form $\omega = \sum_i d\mathbf{q}_i \wedge d\mathbf{p}_i$, preventing secular energy dissipation over long trajectories.

---

## Physical Conservation Diagnostics

Trajectories are evaluated against three first-principles physical invariants:

#### Relative Energy Drift

$$
\Delta E_{\text{rel}}(t) = \frac{\left| E(t) - E(0) \right|}{\left| E(0) \right|}
$$

#### Total Linear Momentum Invariant Error

$$
\Delta \mathbf{P}(t) = \left\| \mathbf{P}(t) - \mathbf{P}(0) \right\| = \left\| \sum_{i=1}^3 m_i \mathbf{v}_i(t) - \sum_{i=1}^3 m_i \mathbf{v}_i(0) \right\|
$$

#### Total Angular Momentum Invariant Error

$$
\Delta L(t) = \left| L(t) - L(0) \right| = \left| \sum_{i=1}^3 m_i \left( x_i(t) v_{yi}(t) - y_i(t) v_{xi}(t) \right) - L(0) \right|
$$

---

## Installation and Environment Setup

Dependencies can be installed using `uv` or standard Python `pip`:

```bash
# Using uv (recommended)
uv venv
uv pip install numpy scipy matplotlib torch torchdiffeq

# Using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib torch torchdiffeq
```

---

## Execution Guide

```bash
# 1. Hamiltonian GNN with Symplectic Velocity-Verlet Integration
python hnn_symplectic_3body.py

# 2. Equivariant GNN with Continuous Neural ODE
python gnn_ode_3body.py

# 3. Autoregressive State-Stepping MLP Baseline
python autoregressive_3body.py

# 4. Synchronize PINN Notebook
python update_pinn.py
```

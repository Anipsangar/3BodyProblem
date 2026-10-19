# Neural Networks and the 3-Body Problem

This repository explores different machine learning and neural network architectures to predict the highly chaotic dynamics of the gravitational 3-Body Problem.

Predicting chaotic systems is fundamentally difficult for standard neural networks because tiny perturbations grow exponentially over time ($\sim e^{\lambda t}$) due to positive Lyapunov exponents. Standard loss functions and unconstrained neural architectures quickly succumb to secular energy drift, momentum violations, and unphysical trajectories.

We implement, compare, and benchmark four distinct approaches:

---

## Architecture Overview

| Model | Script / Notebook | Mathematical Formulation | Symplectic? | Conserves Momentum & Energy? |
| :--- | :--- | :--- | :---: | :---: |
| **1. PINN** | [`Three_Body_Problem_PINN_...ipynb`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/Three_Body_Problem_PINN_(Physics_Informed_Neural_Network).ipynb) | $f(t) \to [x, y]$ | ❌ No | ❌ Soft penalty only (drifts) |
| **2. Autoregressive MLP** | [`autoregressive_3body.py`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/autoregressive_3body.py) | $f(S_t) \to S_{t+1}$ | ❌ No | ❌ None (compounding drift) |
| **3. Equivariant GNN + Neural ODE** | [`gnn_ode_3body.py`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/gnn_ode_3body.py) | $\frac{dS}{dt} = \text{GNN}(S)$ integrated via continuous ODE | ❌ No (RK4) | ✅ Momentum & Angular Momentum exact; Energy bounded |
| **4. Hamiltonian GNN + Symplectic Verlet** | [`hnn_symplectic_3body.py`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/hnn_symplectic_3body.py) | $\mathcal{H}(q, p) = \mathcal{T}(p) + \mathcal{V}_\theta(q)$ via Hamilton's eqns | ✅ **Yes (Verlet)** | ✅ **Exact linear & angular momentum; strictly conservative** |

---

## 1. Physics-Informed Neural Network (PINN)
**File:** [`Three_Body_Problem_PINN_(Physics_Informed_Neural_Network).ipynb`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/Three_Body_Problem_PINN_(Physics_Informed_Neural_Network).ipynb)

- **Method:** Uses a Multi-Layer Perceptron (MLP) with Sine activations and Fourier features mapping absolute time to particle positions: $f(t) \to [x, y]$. Includes differential loss penalties for equations of motion ($\ddot{x} - F/m = 0$) and energy conservation ($dE/dt \approx 0$).
- **Limitation:** Cannot extrapolate outside the training time window $[0, T]$. It is essentially a continuous trajectory regression model for a single initial condition; it does not learn the underlying dynamical vector field.

## 2. Autoregressive MLP (Discrete State-Stepping)
**File:** [`autoregressive_3body.py`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/autoregressive_3body.py)

- **Method:** An MLP takes the 12-dimensional state vector (positions and velocities) and predicts the residual change for the next step $S_{t+1} = S_t + \Delta S$.
- **Limitation:** Suffers from error accumulation (drift). Because standard MLPs have no geometric inductive bias or phase space conservation, errors compound iteratively and bodies rapidly spiral away or collapse. Includes built-in diagnostics for $\Delta E, \Delta P, \Delta L$.

## 3. Equivariant Graph Neural Network (GNN) + Neural ODE
**File:** [`gnn_ode_3body.py`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/gnn_ode_3body.py)

- **Method:**
  - **Equivariant Central Force GNN:** Treats bodies as nodes. Forces are parameterized as central interactions:
    $$\mathbf{F}_{ij} = f_\theta(\|\mathbf{r}_{ij}\|) \frac{\mathbf{r}_j - \mathbf{r}_i}{\|\mathbf{r}_{ij}\|}$$
    This **guarantees Newton's Third Law** ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$), exact linear momentum conservation, exact angular momentum conservation, and $SO(2)$ rotational equivariance by architectural construction.
  - **Neural ODE:** Continuous-time integration via Runge-Kutta (`torchdiffeq.odeint`). Eliminates discrete-step compounding drift.

## 4. Hamiltonian Graph Neural Network + Symplectic Integrator
**File:** [`hnn_symplectic_3body.py`](file:///c:/Users/farid/Downloads/New%20Dolanan/3BodyProblem/hnn_symplectic_3body.py)

- **Method:**
  - **Hamiltonian GNN (HNN):** Instead of directly predicting accelerations, a pairwise neural network learns the scalar gravitational interaction potential $V_\theta(r_{ij})$. The total potential is $\mathcal{V}(q) = \sum_{i < j} V_\theta(\|\mathbf{r}_i - \mathbf{r}_j\|)$.
  - **Conservative Force Derivation:** Forces are computed via automatic differentiation $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} \mathcal{V}(q)$. This guarantees that the force field is **strictly conservative** ($\nabla \times \mathbf{F} = 0$).
  - **Symplectic Velocity-Verlet Integration:** Preserves the symplectic 2-form $dq \wedge dp$. Unlike standard ODE solvers (such as RK4 or Adams-Bashforth), symplectic integrators ensure that energy errors remain bounded without secular drift over arbitrarily long time intervals.

---

## Quick Start & Installation

Create and activate a virtual environment, then install dependencies:

```bash
# Using uv (fastest)
uv venv
uv pip install numpy scipy matplotlib torch torchdiffeq

# Or standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib torch torchdiffeq
```

### Running the Models

```bash
# 1. Run Hamiltonian GNN + Symplectic Verlet (recommended state-of-the-art):
python hnn_symplectic_3body.py

# 2. Run Equivariant GNN + Neural ODE:
python gnn_ode_3body.py

# 3. Run Autoregressive MLP benchmark:
python autoregressive_3body.py

# 4. Update and sync PINN notebook:
python update_pinn.py
```

---

## Physical Conservation Metrics

Each script logs and visualizes key physical invariant diagnostics:
- **Relative Energy Drift:** $\frac{\|E(t) - E(0)\|}{\|E(0)\|}$
- **Linear Momentum Drift:** $\|\mathbf{P}(t) - \mathbf{P}(0)\|$
- **Angular Momentum Drift:** $\|L(t) - L(0)\|$

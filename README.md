# Neural Networks and the 3-Body Problem

This repository explores different machine learning and neural network architectures to predict the highly chaotic dynamics of the 3-Body Problem. 

Predicting chaotic systems is fundamentally difficult for standard neural networks because tiny errors accumulate rapidly over time (drift). We implement and compare three distinct approaches, ranging from simple function approximation to strictly constrained physics integration.

## 1. Physics-Informed Neural Network (PINN)
**File:** `Three_Body_Problem_PINN_(Physics_Informed_Neural_Network).ipynb`

This approach attempts to learn a direct mapping from absolute time to particle positions: $f(t) \to [x, y]$. 
- **Method:** Uses a Multi-Layer Perceptron (MLP) with Sine activations and Fourier features. It includes a custom physics loss function that penalizes deviations from energy conservation.
- **Limitation:** Fails to extrapolate far into the future. It essentially "memorizes" the trajectory within the training bounds but struggles to generalize chaotic dynamics for unseen time steps.

## 2. Autoregressive MLP (State-Stepping)
**File:** `autoregressive_3body.py`

Instead of predicting positions from absolute time, this model predicts the next state based on the current state: $f(S_t) \to S_{t+1}$.
- **Method:** An MLP takes the 12-dimensional state (positions and velocities) and predicts the residual change for the next step. Trained with input noise and 2-step unrolling to improve robustness.
- **Limitation:** Suffers from error accumulation (drift). During inference, tiny errors feed back into the input, and without exact physical constraints, the predicted trajectories eventually diverge wildly (fly away).

## 3. Graph Neural Network (GNN) + Neural ODE
**File:** `gnn_ode_3body.py`

This is the most advanced and robust approach, separating the learning of physical rules from the integration over time.
- **Method:** 
  - **GNN:** Treats the 3 bodies as nodes in a graph. An `EdgeModel` learns the gravitational forces ($1/r^2$) between bodies based on relative distances.
  - **Neural ODE:** Wraps the GNN in an ODE solver (`torchdiffeq.odeint`). Instead of taking discrete time steps, it predicts continuous time derivatives ($dS/dt$) and integrates them perfectly.
- **Advantage:** By learning the underlying physics rules (forces) rather than memorizing trajectories, and integrating over continuous time, this model eliminates the exponential drift seen in the simple MLP.

## Requirements
To run the scripts, you will need PyTorch and standard scientific libraries. For the GNN+ODE approach, you also need:
```bash
pip install torch_geometric torchdiffeq
```

## Conclusion

The 3-Body Problem serves as an excellent benchmark for evaluating the ability of neural networks to model chaotic physical systems. Our experiments reveal clear differences in architecture performance:

1. **PINNs** struggle to extrapolate beyond their training distribution due to their reliance on absolute time mapping, effectively memorizing trajectories without learning the underlying dynamics.
2. **Autoregressive MLPs** capture step-by-step state changes but are severely limited by error accumulation (drift), leading to rapid divergence in chaotic regimes.
3. **GNNs combined with Neural ODEs** provide the most robust solution. By explicitly modeling interactions (forces) and utilizing continuous-time integration, this approach mitigates drift and respects the physical rules governing the system.

**Note: This project is still a work in progress.** While the initial results for the GNN+ODE architecture are promising, the repository is not yet complete. There are numerous possibilities left to explore, including:
- Extensive hyperparameter tuning for the GNN models.
- Evaluating different Neural ODE solvers (e.g., Runge-Kutta variants) for improved speed and accuracy.
- Exploring more advanced Graph Neural Network architectures, such as Message Passing Neural Networks (MPNNs) with attention mechanisms.
- Adding strict energy and momentum conservation constraints directly into the GNN+ODE loss function.

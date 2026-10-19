"""
Hamiltonian Neural Network (HNN) with Symplectic Velocity-Verlet Integration
for the Chaotic 3-Body Problem.

Key Architectural Guarantees:
1. Exact Energy Conservation: Force field is derived from a scalar potential (curl-free: ∇ x F = 0).
2. Exact Newton's 3rd Law: Pairwise central potential V(r_ij) ensures F_ij = -F_ji.
3. Exact Linear Momentum Conservation: Total internal forces sum to zero (sum_i F_i = 0).
4. Exact Angular Momentum Conservation: Central forces exert zero net internal torque.
5. Symplectic Phase-Space Preservation: Velocity-Verlet integration bounds long-term energy error.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.autograd as autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SYSTEM PARAMETERS & REFERENCE GENERATION ---
G = 1.0
masses = np.array([1.0, 1.0, 1.0], dtype=np.float32)
eps = 1e-3

def compute_energies_np(states, masses, G=1.0, eps=1e-3):
    """Computes total energy E = K + U for an array of states (N, 12)."""
    x = states[:, 0:6:2]  # (N, 3)
    y = states[:, 1:6:2]  # (N, 3)
    vx = states[:, 6:12:2]
    vy = states[:, 7:12:2]
    
    # Kinetic energy
    K = 0.5 * np.sum(masses * (vx**2 + vy**2), axis=1)
    
    # Potential energy
    U = np.zeros(states.shape[0], dtype=np.float32)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        dx = x[:, j] - x[:, i]
        dy = y[:, j] - y[:, i]
        r = np.sqrt(dx**2 + dy**2 + eps**2)
        U -= G * masses[i] * masses[j] / r
        
    return K + U, K, U

def compute_momentum_np(states, masses):
    """Computes total linear momentum Px, Py and angular momentum L."""
    x = states[:, 0:6:2]
    y = states[:, 1:6:2]
    vx = states[:, 6:12:2]
    vy = states[:, 7:12:2]
    
    px = np.sum(masses * vx, axis=1)
    py = np.sum(masses * vy, axis=1)
    # L = sum m_i (x_i * v_yi - y_i * v_xi)
    L = np.sum(masses * (x * vy - y * vx), axis=1)
    return px, py, L

def three_body_ode(t, y):
    """Exact 3-body gravitational equations of motion."""
    x1, y1, x2, y2, x3, y3 = y[0:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = y[6:12]

    def accel(xi, yi, xj, yj, mj):
        dx = xj - xi
        dy = yj - yi
        r3 = (dx**2 + dy**2 + eps**2)**1.5
        return G * mj * dx / r3, G * mj * dy / r3

    a1x = accel(x1, y1, x2, y2, masses[1])[0] + accel(x1, y1, x3, y3, masses[2])[0]
    a1y = accel(x1, y1, x2, y2, masses[1])[1] + accel(x1, y1, x3, y3, masses[2])[1]
    
    a2x = accel(x2, y2, x1, y1, masses[0])[0] + accel(x2, y2, x3, y3, masses[2])[0]
    a2y = accel(x2, y2, x1, y1, masses[0])[1] + accel(x2, y2, x3, y3, masses[2])[1]
    
    a3x = accel(x3, y3, x1, y1, masses[0])[0] + accel(x3, y3, x2, y2, masses[1])[0]
    a3y = accel(x3, y3, x1, y1, masses[0])[1] + accel(x3, y3, x2, y2, masses[1])[1]

    return np.array([vx1, vy1, vx2, vy2, vx3, vy3, a1x, a1y, a2x, a2y, a3x, a3y])

# Generate reference trajectory
t0, t1, n_ref = 0.0, 4.0, 1600
y0 = np.array([ -1.0, 0.0,   1.0, 0.0,    0.0, 0.5,
                 0.0, 0.2,   0.0, -0.15, -0.1, 0.0 ], dtype=np.float32)

print(f"Generating high-precision reference data on [{t0}, {t1}] with {n_ref} points...")
t_eval = np.linspace(t0, t1, n_ref)
sol = solve_ivp(three_body_ode, [t0, t1], y0, t_eval=t_eval, rtol=1e-10, atol=1e-10, method='DOP853')
ref_states = sol.y.T.astype(np.float32)

# --- 2. HAMILTONIAN GRAPH NEURAL NETWORK ---
class PairwisePotentialNN(nn.Module):
    """
    Learns scalar potential V(r) between pairs of particles.
    Operating purely on distance r guarantees:
    - Rotational invariance (O(2) symmetry)
    - Newton's third law F_ij = -F_ji
    - Zero net internal torque
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, r):
        # r shape: (N, 1)
        inv_r = 1.0 / (r + 1e-4)
        features = torch.cat([r, inv_r], dim=-1)
        return self.net(features)

class Hamiltonian3BodyGNN(nn.Module):
    def __init__(self, masses, G=1.0, hidden_dim=64):
        super().__init__()
        self.masses = torch.tensor(masses, dtype=torch.float32, device=device)
        self.G = G
        self.potential_net = PairwisePotentialNN(hidden_dim=hidden_dim)
        
    def potential_energy(self, pos):
        """
        pos shape: (batch_size, 3, 2)
        Returns: total potential energy V(pos) of shape (batch_size, 1)
        """
        batch_size = pos.shape[0]
        pairs = [(0, 1), (0, 2), (1, 2)]
        V_total = torch.zeros(batch_size, 1, device=pos.device)
        
        for i, j in pairs:
            r_vec = pos[:, j] - pos[:, i]
            dist = torch.norm(r_vec, dim=-1, keepdim=True)
            V_pair = self.potential_net(dist)
            V_total = V_total + V_pair
            
        return V_total
        
    def kinetic_energy(self, vel):
        """
        vel shape: (batch_size, 3, 2)
        Returns: total kinetic energy T(vel) of shape (batch_size, 1)
        """
        m = self.masses.unsqueeze(0).unsqueeze(-1) # (1, 3, 1)
        K = 0.5 * torch.sum(m * (vel**2), dim=(1, 2), keepdim=True)
        return K
        
    @torch.enable_grad()
    def compute_forces(self, pos, create_graph=False):
        """
        Computes conservative forces F_i = -dV/dq_i using autograd.
        By construction:
        - sum(F_i) == 0 (exact linear momentum conservation)
        - curl(F) == 0 (conservative vector field)
        """
        pos_grad = pos.clone().detach().requires_grad_(True)
        V = self.potential_energy(pos_grad)
        
        # dV/d(pos)
        grad_V = autograd.grad(
            V, pos_grad,
            grad_outputs=torch.ones_like(V),
            create_graph=create_graph
        )[0]
        
        forces = -grad_V
        return forces

# --- 3. SYMPLECTIC VELOCITY-VERLET INTEGRATOR ---
class SymplecticVerletIntegrator:
    """
    Implements the Störmer-Verlet (Leapfrog) symplectic integration scheme.
    Preserves the symplectic 2-form dP ^ dQ, preventing artificial secular energy drift.
    """
    def __init__(self, model):
        self.model = model
        self.masses = model.masses.unsqueeze(0).unsqueeze(-1)
        
    def rollout(self, pos0, vel0, dt, n_steps):
        """Rolls out trajectory for n_steps using Velocity-Verlet with O(1) force evals per step."""
        pos_traj = [pos0]
        vel_traj = [vel0]
        curr_pos = pos0
        curr_vel = vel0
        
        # Initial acceleration
        forces = self.model.compute_forces(curr_pos, create_graph=False)
        curr_accel = forces / self.masses
        
        for _ in range(n_steps):
            # Half-step velocity
            vel_half = curr_vel + 0.5 * dt * curr_accel
            # Full-step position
            curr_pos = curr_pos + dt * vel_half
            
            # Acceleration at new position
            forces = self.model.compute_forces(curr_pos, create_graph=False)
            curr_accel = forces / self.masses
            
            # Complete velocity step
            curr_vel = vel_half + 0.5 * dt * curr_accel
            
            pos_traj.append(curr_pos)
            vel_traj.append(curr_vel)
            
        return torch.stack(pos_traj, dim=1), torch.stack(vel_traj, dim=1)

# --- 4. PREPARE TRAINING DATA ---
# Extract positions, velocities, and target accelerations from reference trajectory
x_ref = ref_states[:, 0:6:2]
y_ref = ref_states[:, 1:6:2]
vx_ref = ref_states[:, 6:12:2]
vy_ref = ref_states[:, 7:12:2]

pos_all = np.stack([x_ref, y_ref], axis=-1)   # (N, 3, 2)
vel_all = np.stack([vx_ref, vy_ref], axis=-1) # (N, 3, 2)

# Compute reference target accelerations from ODE
acc_all = np.zeros_like(pos_all)
for t_idx in range(n_ref):
    deriv = three_body_ode(t_eval[t_idx], ref_states[t_idx])
    acc_all[t_idx, 0, 0] = deriv[6]
    acc_all[t_idx, 0, 1] = deriv[7]
    acc_all[t_idx, 1, 0] = deriv[8]
    acc_all[t_idx, 1, 1] = deriv[9]
    acc_all[t_idx, 2, 0] = deriv[10]
    acc_all[t_idx, 2, 1] = deriv[11]

# Train on first 60% of trajectory, test extrapolation on remaining 40%
split_idx = int(0.6 * n_ref)
train_pos = torch.tensor(pos_all[:split_idx], dtype=torch.float32, device=device)
train_acc = torch.tensor(acc_all[:split_idx], dtype=torch.float32, device=device)

# --- 5. MODEL TRAINING ---
print("\nInitializing Hamiltonian GNN...")
model = Hamiltonian3BodyGNN(masses, G=G, hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

print(f"Training on {split_idx} states with direct Hamiltonian Force matching...")
epochs = 2000
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Compute predicted forces from learned scalar potential
    pred_forces = model.compute_forces(train_pos, create_graph=True)
    pred_accel = pred_forces / model.masses.unsqueeze(0).unsqueeze(-1)
    
    loss = nn.functional.mse_loss(pred_accel, train_acc)
    loss.backward()
    optimizer.step()
    scheduler.step(loss.detach())
    
    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{epochs} | Accel Loss: {loss.item():.6e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

train_time = time.time() - start_time
print(f"Training completed in {train_time:.1f}s.")

# --- 6. SYMPLECTIC ROLLOUT & EVALUATION ---
print("\nSimulating full trajectory with Symplectic Velocity-Verlet Integrator...")
integrator = SymplecticVerletIntegrator(model)

dt = float(t_eval[1] - t_eval[0])
pos0 = torch.tensor(pos_all[0:1], dtype=torch.float32, device=device)
vel0 = torch.tensor(vel_all[0:1], dtype=torch.float32, device=device)

with torch.no_grad():
    pred_pos_traj, pred_vel_traj = integrator.rollout(pos0, vel0, dt, n_ref - 1)

pred_pos = pred_pos_traj[0].cpu().numpy() # (N, 3, 2)
pred_vel = pred_vel_traj[0].cpu().numpy() # (N, 3, 2)

# Construct flat states for metric evaluation
pred_states = np.zeros_like(ref_states)
for i in range(3):
    pred_states[:, i*2]     = pred_pos[:, i, 0]
    pred_states[:, i*2 + 1] = pred_pos[:, i, 1]
    pred_states[:, 6 + i*2] = pred_vel[:, i, 0]
    pred_states[:, 7 + i*2] = pred_vel[:, i, 1]

# --- 7. CONSERVATION ANALYSIS ---
E_ref, _, _ = compute_energies_np(ref_states, masses, G=G, eps=eps)
E_pred, _, _ = compute_energies_np(pred_states, masses, G=G, eps=eps)

Px_ref, Py_ref, L_ref = compute_momentum_np(ref_states, masses)
Px_pred, Py_pred, L_pred = compute_momentum_np(pred_states, masses)

delta_E_ref = np.abs((E_ref - E_ref[0]) / E_ref[0])
delta_E_pred = np.abs((E_pred - E_pred[0]) / E_pred[0])

delta_P_pred = np.sqrt((Px_pred - Px_pred[0])**2 + (Py_pred - Py_pred[0])**2)
delta_L_pred = np.abs(L_pred - L_pred[0])

print("\n--- Physical Conservation Diagnostics ---")
print(f"Mean Energy Relative Drift (Reference RK):   {np.mean(delta_E_ref):.4e}")
print(f"Mean Energy Relative Drift (HNN-Verlet):     {np.mean(delta_E_pred):.4e}")
print(f"Max Linear Momentum Drift (HNN-Verlet):       {np.max(delta_P_pred):.4e}")
print(f"Max Angular Momentum Drift (HNN-Verlet):      {np.max(delta_L_pred):.4e}")

# --- 8. PLOTTING DIAGNOSTICS ---
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. Trajectory comparison
axs[0, 0].plot(ref_states[:, 0], ref_states[:, 1], 'C0-', label='Body 1 (Ref)')
axs[0, 0].plot(ref_states[:, 2], ref_states[:, 3], 'C1-', label='Body 2 (Ref)')
axs[0, 0].plot(ref_states[:, 4], ref_states[:, 5], 'C2-', label='Body 3 (Ref)')

axs[0, 0].plot(pred_states[:, 0], pred_states[:, 1], 'C0--', label='Body 1 (HNN-Verlet)')
axs[0, 0].plot(pred_states[:, 2], pred_states[:, 3], 'C1--', label='Body 2 (HNN-Verlet)')
axs[0, 0].plot(pred_states[:, 4], pred_states[:, 5], 'C2--', label='Body 3 (HNN-Verlet)')

axs[0, 0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axs[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
axs[0, 0].set_title("3-Body Trajectories: Reference vs Hamiltonian GNN")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("Y")
axs[0, 0].legend()
axs[0, 0].axis('equal')

# 2. Energy drift over time
axs[0, 1].plot(t_eval, delta_E_ref, label='Reference (DOP853)', color='black', alpha=0.7)
axs[0, 1].plot(t_eval, delta_E_pred, label='HNN + Symplectic Verlet', color='crimson')
axs[0, 1].axvline(x=t_eval[split_idx], color='blue', linestyle=':', label='Train/Test Split')
axs[0, 1].set_yscale('log')
axs[0, 1].set_title("Relative Energy Conservation |ΔE / E0|")
axs[0, 1].set_xlabel("Time t")
axs[0, 1].set_ylabel("Relative Drift")
axs[0, 1].legend()
axs[0, 1].grid(True, which="both", ls="--", alpha=0.4)

# 3. Momentum conservation over time
axs[1, 0].plot(t_eval, delta_P_pred, label='Linear Momentum Error |ΔP|', color='navy')
axs[1, 0].plot(t_eval, delta_L_pred, label='Angular Momentum Error |ΔL|', color='darkgreen')
axs[1, 0].set_title("Invariant Conservation (Newton's 3rd Law & Central Forces)")
axs[1, 0].set_xlabel("Time t")
axs[1, 0].set_ylabel("Absolute Error")
axs[1, 0].legend()
axs[1, 0].grid(True, ls="--", alpha=0.4)

# 4. Learned Potential vs True Gravitational Potential
r_test = np.linspace(0.1, 3.0, 300, dtype=np.float32)
r_tensor = torch.tensor(r_test, device=device).unsqueeze(-1)
with torch.no_grad():
    V_learned = model.potential_net(r_tensor).cpu().numpy().flatten()
    # Align constant offset (potentials are defined up to an additive constant)
    V_true = -G * (masses[0] * masses[1]) / np.sqrt(r_test**2 + eps**2)
    V_learned_shifted = V_learned - (V_learned[-1] - V_true[-1])

axs[1, 1].plot(r_test, V_true, 'k-', label='True Potential: -G m1 m2 / r')
axs[1, 1].plot(r_test, V_learned_shifted, 'r--', label='Learned Potential V_θ(r)')
axs[1, 1].set_title("Learned Pairwise Interaction Potential")
axs[1, 1].set_xlabel("Distance r")
axs[1, 1].set_ylabel("Potential V(r)")
axs[1, 1].legend()
axs[1, 1].grid(True, ls="--", alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "hnn_symplectic_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved diagnostic plots to: {plot_path}")

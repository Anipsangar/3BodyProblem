# To run in Colab, first install dependencies:
# !pip install torch_geometric torchdiffeq

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

try:
    from torchdiffeq import odeint
except ImportError:
    print("Please run: !pip install torchdiffeq")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. GENERATE REFERENCE DATA ---
G = 1.0
m1, m2, m3 = 1.0, 1.0, 1.0
eps = 1e-3

def three_body_ode(t, y):
    x1, y1, x2, y2, x3, y3 = y[0:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = y[6:12]

    def accel(xi, yi, xj, yj, mj):
        dx = xj - xi
        dy = yj - yi
        r3 = (dx**2 + dy**2 + eps**2)**1.5
        return G * mj * dx / r3, G * mj * dy / r3

    a1x = accel(x1, y1, x2, y2, m2)[0] + accel(x1, y1, x3, y3, m3)[0]
    a1y = accel(x1, y1, x2, y2, m2)[1] + accel(x1, y1, x3, y3, m3)[1]
    
    a2x = accel(x2, y2, x1, y1, m1)[0] + accel(x2, y2, x3, y3, m3)[0]
    a2y = accel(x2, y2, x1, y1, m1)[1] + accel(x2, y2, x3, y3, m3)[1]
    
    a3x = accel(x3, y3, x1, y1, m1)[0] + accel(x3, y3, x2, y2, m2)[0]
    a3y = accel(x3, y3, x1, y1, m1)[1] + accel(x3, y3, x2, y2, m2)[1]

    return np.array([vx1, vy1, vx2, vy2, vx3, vy3, a1x, a1y, a2x, a2y, a3x, a3y])

t0, t1, n_ref = 0.0, 2.0, 800
y0 = np.array([ -1.0, 0.0,   1.0, 0.0,    0.0, 0.5,
                 0.0, 0.2,   0.0, -0.15, -0.1, 0.0 ])

print("Generating reference data...")
t_eval = np.linspace(t0, t1, n_ref)
sol = solve_ivp(three_body_ode, [t0, t1], y0, t_eval=t_eval, rtol=1e-9, atol=1e-9, method='RK45')

states = sol.y.T.astype(np.float32) # (800, 12)
t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device)

# Reshape states to be graph-friendly: (T, 3_bodies, 4_features)
# Features: [x, y, vx, vy]
states_graph = np.zeros((n_ref, 3, 4), dtype=np.float32)
for i in range(3):
    states_graph[:, i, 0] = states[:, i*2]     # x
    states_graph[:, i, 1] = states[:, i*2+1]   # y
    states_graph[:, i, 2] = states[:, 6+i*2]   # vx
    states_graph[:, i, 3] = states[:, 6+i*2+1] # vy

Y_tensor = torch.tensor(states_graph, device=device) # (T, 3, 4)

# Fully connected graph for 3 bodies
edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                           [1, 2, 0, 2, 0, 1]], dtype=torch.long, device=device)


# --- 2. GNN & ODE MODELS ---
class EquivariantCentralEdgeModel(nn.Module):
    """
    Equivariant Central Force Model for Gravitational Interactions.
    Guarantees:
    - Newton's Third Law (F_ij = -F_ji) by construction
    - Exact conservation of total linear momentum (sum_i F_i = 0)
    - Exact conservation of total angular momentum (zero internal torque)
    - Rotational SO(2) equivariance
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Input: scalar distance and inverse square distance
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # scalar force magnitude f(r)
        )
        
    def forward(self, pos_i, pos_j):
        rel_pos = pos_j - pos_i  # Vector from i to j
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        inv_dist_sq = 1.0 / (dist**2 + 1e-5)
        scalar_features = torch.cat([dist, inv_dist_sq], dim=-1)
        force_magnitude = self.mlp(scalar_features)
        unit_vec = rel_pos / (dist + 1e-5)
        # Attractive central force directed towards body j
        return force_magnitude * unit_vec

class GNN_ODE_Func(nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.edge_model = EquivariantCentralEdgeModel()

    def forward(self, t, h):
        # h shape: (3, 4)  [x, y, vx, vy]
        pos = h[:, :2]
        vel = h[:, 2:]
        
        # Compute messages (forces)
        row, col = self.edge_index
        pos_i = pos[row]
        pos_j = pos[col]
        messages = self.edge_model(pos_i, pos_j)
        
        # Aggregate messages to compute acceleration
        accel = torch.zeros_like(pos)
        accel.scatter_add_(0, row.unsqueeze(1).expand(-1, 2), messages)
        
        # The derivative of state [pos, vel] is [vel, accel]
        dh_dt = torch.cat([vel, accel], dim=-1)
        return dh_dt

class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, h0, t):
        # We use rk4 or explicit_adams for speed, adjoint is optional but better for memory
        out = odeint(self.ode_func, h0, t, method='rk4')
        return out


# --- 3. TRAINING ---
ode_func = GNN_ODE_Func(edge_index).to(device)
model = NeuralODE(ode_func).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train on a smaller sequence initially to prevent gradient explosion
# 800 steps is too long for Neural ODE unrolling from scratch
seq_len = 50 
t_train = t_tensor[:seq_len]
Y_train = Y_tensor[:seq_len]
h0 = Y_train[0]

print("Training GNN-ODE (unrolled 50 steps)...")
epochs = 1500
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    pred = model(h0, t_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6e}")

# Expand to full sequence
print("Expanding to full 800 steps...")
epochs_full = 500
for epoch in range(1, epochs_full + 1):
    optimizer.zero_grad()
    pred = model(h0, t_tensor)
    loss = criterion(pred, Y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Full Epoch {epoch}/{epochs_full} | Loss: {loss.item():.6e}")

# --- 4. EVALUATION & CONSERVATION METRICS ---
print("Simulating full trajectory...")
model.eval()
with torch.no_grad():
    pred_states_raw = model(h0, t_tensor).cpu().numpy()

# Format predicted states to (n_ref, 12)
pred_states = np.zeros((n_ref, 12), dtype=np.float32)
for i in range(3):
    pred_states[:, i*2]     = pred_states_raw[:, i, 0]
    pred_states[:, i*2+1]   = pred_states_raw[:, i, 1]
    pred_states[:, 6+i*2]   = pred_states_raw[:, i, 2]
    pred_states[:, 6+i*2+1] = pred_states_raw[:, i, 3]

states = states.reshape(n_ref, 12)

def calc_conservation(s):
    x = s[:, 0:6:2]; y = s[:, 1:6:2]
    vx = s[:, 6:12:2]; vy = s[:, 7:12:2]
    K = 0.5 * np.sum(vx**2 + vy**2, axis=1)
    U = np.zeros(s.shape[0])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        r = np.sqrt((x[:, j] - x[:, i])**2 + (y[:, j] - y[:, i])**2 + eps**2)
        U -= G / r
    E = K + U
    Px = np.sum(vx, axis=1)
    Py = np.sum(vy, axis=1)
    P = np.sqrt(Px**2 + Py**2)
    L = np.sum(x * vy - y * vx, axis=1)
    return E, P, L

E_ref, P_ref, L_ref = calc_conservation(states)
E_pred, P_pred, L_pred = calc_conservation(pred_states)

delta_E = np.abs((E_pred - E_pred[0]) / E_pred[0])
delta_P = np.abs(P_pred - P_pred[0])
delta_L = np.abs(L_pred - L_pred[0])

print("\n--- Physical Conservation Diagnostics ---")
print(f"Mean Relative Energy Drift:      {np.mean(delta_E):.4e}")
print(f"Max Linear Momentum Drift:       {np.max(delta_P):.4e} (Newton's 3rd Law guaranteed)")
print(f"Max Angular Momentum Drift:      {np.max(delta_L):.4e} (Central force guaranteed)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Trajectory
ax1.plot(states[:,0], states[:,1], label='Body1 (Ref)')
ax1.plot(states[:,2], states[:,3], label='Body2 (Ref)')
ax1.plot(states[:,4], states[:,5], label='Body3 (Ref)')
ax1.plot(pred_states[:,0], pred_states[:,1], '--', label='Body1 (Equivariant GNN-ODE)')
ax1.plot(pred_states[:,2], pred_states[:,3], '--', label='Body2 (Equivariant GNN-ODE)')
ax1.plot(pred_states[:,4], pred_states[:,5], '--', label='Body3 (Equivariant GNN-ODE)')
ax1.legend()
ax1.set_title('Equivariant GNN + Neural ODE Trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')

# Conservation
ax2.plot(t_eval, delta_E, label='Relative Energy Drift |ΔE / E0|', color='crimson')
ax2.plot(t_eval, delta_P, label='Linear Momentum Drift |ΔP|', color='navy')
ax2.plot(t_eval, delta_L, label='Angular Momentum Drift |ΔL|', color='darkgreen')
ax2.set_title('Conservation Invariants over Time')
ax2.set_xlabel('Time t')
ax2.set_ylabel('Drift Magnitude')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig('gnn_ode_plot.png', dpi=150)
print("Saved plot to gnn_ode_plot.png")

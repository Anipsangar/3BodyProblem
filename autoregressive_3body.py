import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

states = sol.y.T.astype(np.float32)

X_data = states[:-1]
Y_data = states[1:]

X_tensor = torch.tensor(X_data, device=device)
Y_tensor = torch.tensor(Y_data, device=device)

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class AutoregressiveModel(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=128, n_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(Sine())
        for _ in range(n_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine())
        layers.append(nn.Linear(hidden_dim, in_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # We predict the delta (residual), which makes learning easier
        return x + self.net(x)

model = AutoregressiveModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training Autoregressive MLP...")
epochs = 10000
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # 1-step with noise
    noise = torch.randn_like(X_tensor) * 1e-3
    pred1 = model(X_tensor + noise)
    loss1 = criterion(pred1, Y_tensor)
    
    # 2-step unrolled
    pred2 = model(pred1[:-1])
    loss2 = criterion(pred2, Y_tensor[1:])
    
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6e}")

print("Simulating autoregressively...")
model.eval()
pred_states = [torch.tensor(y0, dtype=torch.float32, device=device)]
curr_state = pred_states[0].unsqueeze(0)

with torch.no_grad():
    for _ in range(n_ref - 1):
        curr_state = model(curr_state)
        pred_states.append(curr_state.squeeze(0))
        
pred_states = torch.stack(pred_states).cpu().numpy()

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
print(f"Mean Relative Energy Drift:  {np.mean(delta_E):.4e}")
print(f"Max Linear Momentum Drift:   {np.max(delta_P):.4e}")
print(f"Max Angular Momentum Drift:  {np.max(delta_L):.4e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.plot(states[:,0], states[:,1], label='Body1 (Ref)')
ax1.plot(states[:,2], states[:,3], label='Body2 (Ref)')
ax1.plot(states[:,4], states[:,5], label='Body3 (Ref)')

ax1.plot(pred_states[:,0], pred_states[:,1], '--', label='Body1 (Autoregressive)')
ax1.plot(pred_states[:,2], pred_states[:,3], '--', label='Body2 (Autoregressive)')
ax1.plot(pred_states[:,4], pred_states[:,5], '--', label='Body3 (Autoregressive)')

ax1.legend()
ax1.set_title('Autoregressive State-Stepping Prediction')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')

ax2.plot(t_eval, delta_E, label='Relative Energy Drift |ΔE / E0|', color='crimson')
ax2.plot(t_eval, delta_P, label='Linear Momentum Drift |ΔP|', color='navy')
ax2.plot(t_eval, delta_L, label='Angular Momentum Drift |ΔL|', color='darkgreen')
ax2.set_title('Conservation Invariants Drift (Drift Compounding)')
ax2.set_xlabel('Time t')
ax2.set_ylabel('Drift Magnitude')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig('autoregressive_plot.png', dpi=150)
print("Saved plot to autoregressive_plot.png")

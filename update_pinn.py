import json
import os

filepath = r"c:\Users\client-jempol-pc37\Downloads\Personal Project\3BodyProblem\Three_Body_Problem_PINN_(Physics_Informed_Neural_Network).ipynb"
with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

pinn_code = [
    "# ---------- PINN Model ----------\n",
    "class Sine(nn.Module):\n",
    "    def forward(self, x):\n",
    "        return torch.sin(x)\n",
    "\n",
    "class PINN(nn.Module):\n",
    "    def __init__(self, in_dim=1, out_dim=6, n_layers=4, n_hidden=128):\n",
    "        super().__init__()\n",
    "        self.B = nn.Parameter(torch.randn(in_dim, n_hidden) * 2.0, requires_grad=False)\n",
    "        layers = []\n",
    "        layers.append(nn.Linear(n_hidden, n_hidden))\n",
    "        layers.append(Sine())\n",
    "        for _ in range(n_layers-2):\n",
    "            layers.append(nn.Linear(n_hidden, n_hidden))\n",
    "            layers.append(Sine())\n",
    "        layers.append(nn.Linear(n_hidden, out_dim))\n",
    "        self.net = nn.Sequential(*layers)\n",
    "        for m in self.net:\n",
    "            if isinstance(m, nn.Linear):\n",
    "                nn.init.xavier_normal_(m.weight)\n",
    "                nn.init.zeros_(m.bias)\n",
    "\n",
    "    def forward(self, t):\n",
    "        x = torch.sin(t @ self.B)\n",
    "        return self.net(x)\n",
    "\n",
    "model = PINN(n_layers=n_layers, n_hidden=n_hidden).to(device)"
]

loss_code = [
    "# ---------- Loss functions ----------\n",
    "def compute_physics_residual(model, t_in):\n",
    "    t = t_in.clone().detach().requires_grad_(True).to(device)\n",
    "    pred = model(t)\n",
    "    x1 = pred[:,0:1]; y1 = pred[:,1:2]\n",
    "    x2 = pred[:,2:3]; y2 = pred[:,3:4]\n",
    "    x3 = pred[:,4:5]; y3 = pred[:,5:6]\n",
    "\n",
    "    v1x = autograd.grad(x1, t, torch.ones_like(x1), create_graph=True)[0]\n",
    "    v1y = autograd.grad(y1, t, torch.ones_like(y1), create_graph=True)[0]\n",
    "    v2x = autograd.grad(x2, t, torch.ones_like(x2), create_graph=True)[0]\n",
    "    v2y = autograd.grad(y2, t, torch.ones_like(y2), create_graph=True)[0]\n",
    "    v3x = autograd.grad(x3, t, torch.ones_like(x3), create_graph=True)[0]\n",
    "    v3y = autograd.grad(y3, t, torch.ones_like(y3), create_graph=True)[0]\n",
    "\n",
    "    a1x = autograd.grad(v1x, t, torch.ones_like(v1x), create_graph=True)[0]\n",
    "    a1y = autograd.grad(v1y, t, torch.ones_like(v1y), create_graph=True)[0]\n",
    "    a2x = autograd.grad(v2x, t, torch.ones_like(v2x), create_graph=True)[0]\n",
    "    a2y = autograd.grad(v2y, t, torch.ones_like(v2y), create_graph=True)[0]\n",
    "    a3x = autograd.grad(v3x, t, torch.ones_like(v3x), create_graph=True)[0]\n",
    "    a3y = autograd.grad(v3y, t, torch.ones_like(v3y), create_graph=True)[0]\n",
    "\n",
    "    def grav_acc(xi, yi, xj, yj, mj):\n",
    "        dx = xj - xi; dy = yj - yi\n",
    "        r2 = dx**2 + dy**2 + eps**2\n",
    "        r3 = r2 * torch.sqrt(r2)\n",
    "        return G * mj * dx / r3, G * mj * dy / r3\n",
    "\n",
    "    f1x_12, f1y_12 = grav_acc(x1, y1, x2, y2, m2)\n",
    "    f1x_13, f1y_13 = grav_acc(x1, y1, x3, y3, m3)\n",
    "    f1x = f1x_12 + f1x_13; f1y = f1y_12 + f1y_13\n",
    "\n",
    "    f2x_21, f2y_21 = grav_acc(x2, y2, x1, y1, m1)\n",
    "    f2x_23, f2y_23 = grav_acc(x2, y2, x3, y3, m3)\n",
    "    f2x = f2x_21 + f2x_23; f2y = f2y_21 + f2y_23\n",
    "\n",
    "    f3x_31, f3y_31 = grav_acc(x3, y3, x1, y1, m1)\n",
    "    f3x_32, f3y_32 = grav_acc(x3, y3, x2, y2, m2)\n",
    "    f3x = f3x_31 + f3x_32; f3y = f3y_31 + f3y_32\n",
    "\n",
    "    res = torch.cat([a1x - f1x, a1y - f1y, a2x - f2x, a2y - f2y, a3x - f3x, a3y - f3y], dim=1)\n",
    "    \n",
    "    v1_sq = v1x**2 + v1y**2\n",
    "    v2_sq = v2x**2 + v2y**2\n",
    "    v3_sq = v3x**2 + v3y**2\n",
    "    K = 0.5 * (m1*v1_sq + m2*v2_sq + m3*v3_sq)\n",
    "    \n",
    "    r12 = torch.sqrt((x2-x1)**2 + (y2-y1)**2 + eps**2)\n",
    "    r13 = torch.sqrt((x3-x1)**2 + (y3-y1)**2 + eps**2)\n",
    "    r23 = torch.sqrt((x3-x2)**2 + (y3-y2)**2 + eps**2)\n",
    "    U = -G * (m1*m2/r12 + m1*m3/r13 + m2*m3/r23)\n",
    "    \n",
    "    E = K + U\n",
    "    dE_dt = autograd.grad(E, t, torch.ones_like(E), create_graph=True)[0]\n",
    "    \n",
    "    return res, dE_dt\n",
    "\n",
    "def physics_loss(model, t_colloc):\n",
    "    res, dE_dt = compute_physics_residual(model, t_colloc)\n",
    "    return (res**2).mean() + (dE_dt**2).mean() * 0.1  # scale energy loss\n",
    "\n",
    "def data_loss(model, t_obs, pos_obs):\n",
    "    pred = model(t_obs)\n",
    "    return ((pred - pos_obs)**2).mean()"
]

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "# ---------- PINN Model ----------" in src:
            cell["source"] = pinn_code
        elif "# ---------- Loss functions ----------" in src:
            cell["source"] = loss_code

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Updated successfully")

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# -------------------------
# Config
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

PI = math.pi

# training
iters = 100000
lr = 1e-4
N_int = 2000
display_every = 1000

init_log_sigmas = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)

# -------------------------
# Exact IC / source term
# -------------------------
def u_ic(x):
    return (torch.sin(8 * x) / 8
            + torch.sin(x)
            + torch.sin(2 * x) / 2
            + torch.sin(3 * x) / 3
            + torch.sin(4 * x) / 4)

def r_term(x, t):
    return torch.exp(-t) * (
        3 * torch.sin(2 * x) / 2
        + 8 * torch.sin(3 * x) / 3
        + 15 * torch.sin(4 * x) / 4
        + 63 * torch.sin(8 * x) / 8
    )

def rx_term(x, t):
    return torch.exp(-t) * (
        63 * torch.cos(8 * x)
        + 15 * torch.cos(4 * x)
        + 8 * torch.cos(3 * x)
        + 3 * torch.cos(2 * x)
    )

def exact_solution(x, t):
    val = torch.sin(8 * x) / 8
    for i in range(1, 5):
        val = val + torch.sin(i * x) / i
    return torch.exp(-t) * val

# -------------------------
# Network + hard constraint
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=20, depth=3, out_dim=1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * depth + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

def output_transform(x, t, y_raw):
    ic = u_ic(x)
    return (x - PI) * (x + PI) * (1 - torch.exp(-t)) * y_raw + ic

class GPINN_Uncertainty(nn.Module):
    def __init__(self, base_net, init_log_sigmas):
        super().__init__()
        self.base_net = base_net
        self.log_sigmas = nn.Parameter(init_log_sigmas.clone().detach())

    def forward_u(self, x, t):
        z = torch.cat([x, t], dim=1)
        y_raw = self.base_net(z)
        u = output_transform(x, t, y_raw)
        return u

    def sigmas_weights(self):
        sigmas = torch.exp(self.log_sigmas)
        weights = 1.0 / (2.0 * sigmas * sigmas)
        return sigmas, weights

# -------------------------
# Derivatives + residuals
# -------------------------
def gpinn_residuals(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model.forward_u(x, t)

    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]
    
    u_tx = torch.autograd.grad(
        u_x, t, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]
    
    u_tt = torch.autograd.grad(
        u_t, t, grad_outputs=torch.ones_like(u_t),
        create_graph=True, retain_graph=True
    )[0]

    u_xxx = torch.autograd.grad(
        u_xx, x, grad_outputs=torch.ones_like(u_xx),
        create_graph=True, retain_graph=True
    )[0]
    
    u_xxt = torch.autograd.grad(
        u_xx, t, grad_outputs=torch.ones_like(u_xx),
        create_graph=True, retain_graph=True
    )[0]

    r = r_term(x, t)
    rx = rx_term(x, t)
    rt = -r

    f = u_t - u_xx - r
    fx = u_tx - u_xxx - rx
    ft = u_tt - u_xxt - rt

    return f, fx, ft

# -------------------------
# Sampling
# -------------------------
def sample_interior(n):
    x = (2.0 * torch.rand(n, 1, device=device) - 1.0) * PI
    t = torch.rand(n, 1, device=device)
    return x, t

# -------------------------
# Train
# -------------------------
def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    base = MLP(in_dim=2, hidden=20, depth=3, out_dim=1).to(device)
    model = GPINN_Uncertainty(base, init_log_sigmas=init_log_sigmas.to(device)).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("./results", exist_ok=True)

    # History tracking
    history = {
        "steps": [],
        "L_pde": [],
        "L_gx": [],
        "L_gt": [],
        "total_loss": [],
        "log_sigmas": [],
        "weights": [],
        "rel_l2_t0": [],
        "rel_l2_t05": [],
    }

    for step in range(1, iters + 1):
        x, t = sample_interior(N_int)

        f, fx, ft = gpinn_residuals(model, x, t)
        L_pde = (f * f).mean()
        L_gx = (fx * fx).mean()
        L_gt = (ft * ft).mean()

        sigmas, weights = model.sigmas_weights()

        total = (
            weights[0] * L_pde + torch.log(sigmas[0]) +
            weights[1] * L_gx  + torch.log(sigmas[1]) +
            weights[2] * L_gt  + torch.log(sigmas[2])
        )

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if step % display_every == 0 or step == 1:
            with torch.no_grad():
                x_te = torch.linspace(-PI, PI, 512, device=device).view(-1, 1)
                t0 = torch.zeros_like(x_te)
                tmid = 0.5 * torch.ones_like(x_te)

                u0_pred = model.forward_u(x_te, t0)
                u0_true = exact_solution(x_te, t0)
                rel0 = torch.norm(u0_pred - u0_true) / (torch.norm(u0_true) + 1e-12)

                um_pred = model.forward_u(x_te, tmid)
                um_true = exact_solution(x_te, tmid)
                relm = torch.norm(um_pred - um_true) / (torch.norm(um_true) + 1e-12)

            # Store history
            history["steps"].append(step)
            history["L_pde"].append(L_pde.item())
            history["L_gx"].append(L_gx.item())
            history["L_gt"].append(L_gt.item())
            history["total_loss"].append(total.item())
            history["log_sigmas"].append(model.log_sigmas.detach().cpu().numpy().copy())
            history["weights"].append(weights.detach().cpu().numpy().copy())
            history["rel_l2_t0"].append(rel0.item())
            history["rel_l2_t05"].append(relm.item())

            print(
                f"step {step:6d} | "
                f"L=[{L_pde.item():.3e}, {L_gx.item():.3e}, {L_gt.item():.3e}] | "
                f"total={total.item():.3e} | "
                f"log_sigmas={model.log_sigmas.detach().cpu().numpy()} | "
                f"weights={(weights.detach().cpu().numpy())} | "
                f"relL2(t=0)={rel0.item():.3e}, relL2(t=0.5)={relm.item():.3e}"
            )

    # Save final results
    sigmas, weights = model.sigmas_weights()
    print("\nFinal log_sigmas:", model.log_sigmas.detach().cpu().numpy())
    print("Final sigmas:", sigmas.detach().cpu().numpy())
    print("Final weights:", weights.detach().cpu().numpy())

    torch.save(
        {
            "model_state": model.state_dict(),
            "log_sigmas": model.log_sigmas.detach().cpu(),
            "history": history,
        },
        "./results/gpinn_uncertainty_model.pt",
    )

    with open("./results/gpinn_uncertainty_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print("\nSaved to ./results/gpinn_uncertainty_history.pkl")

if __name__ == "__main__":
    main()

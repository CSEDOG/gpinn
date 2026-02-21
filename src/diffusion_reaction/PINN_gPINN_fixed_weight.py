import deepxde as dde
import torch
import numpy as np
import pickle
import os

dde.config.set_default_float("float32")

print(f"DeepXDE version: {dde.__version__}")
print(f"Using backend: {dde.backend.backend_name}")

# Hard constraint output transform
def output_transform(x, y):
    x_in, t_in = x[:, 0:1], x[:, 1:2]
    ic = (
        torch.sin(8 * x_in) / 8
        + torch.sin(x_in)
        + torch.sin(2 * x_in) / 2
        + torch.sin(3 * x_in) / 3
        + torch.sin(4 * x_in) / 4
    )
    return (x_in - np.pi) * (x_in + np.pi) * (1 - torch.exp(-t_in)) * y + ic

def PINNpde(x, y):
    x_in, t_in = x[:, 0:1], x[:, 1:2]
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    r = torch.exp(-t_in) * (
        3 * torch.sin(2 * x_in) / 2
        + 8 * torch.sin(3 * x_in) / 3
        + 15 * torch.sin(4 * x_in) / 4
        + 63 * torch.sin(8 * x_in) / 8
    )
    return dy_t - dy_xx - r

def gPINNpde(x, y):
    x_in, t_in = x[:, 0:1], x[:, 1:2]
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    r = torch.exp(-t_in) * (
        3 * torch.sin(2 * x_in) / 2
        + 8 * torch.sin(3 * x_in) / 3
        + 15 * torch.sin(4 * x_in) / 4
        + 63 * torch.sin(8 * x_in) / 8
    )

    # x-gradient residual
    dy_tx = dde.grad.hessian(y, x, i=0, j=1)
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)
    dr_x = torch.exp(-t_in) * (
        63 * torch.cos(8 * x_in)
        + 15 * torch.cos(4 * x_in)
        + 8 * torch.cos(3 * x_in)
        + 3 * torch.cos(2 * x_in)
    )

    # t-gradient residual
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)
    dr_t = -r

    return [
        dy_t - dy_xx - r,
        dy_tx - dy_xxx - dr_x,
        dy_tt - dy_xxt - dr_t,
    ]

def solution(a):
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val

if __name__ == "__main__":
    geom = dde.geometry.Interval(-np.pi, np.pi)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    iters = 100000

    os.makedirs("./results", exist_ok=True)

    # ==================== PINN ====================
    print("\n" + "="*60)
    print("--- PINN ---")
    print("="*60)
    data = dde.data.TimePDE(
        geomtime, PINNpde, [], num_domain=50, solution=solution, num_test=1000
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
    net.apply_output_transform(output_transform)
    m_pinn = dde.Model(data, net)
    m_pinn.compile("adam", lr=1e-4, metrics=["l2 relative error"])
    losshistory_pinn, train_state_pinn = m_pinn.train(iterations=iters)
    dde.saveplot(losshistory_pinn, train_state_pinn, issave=True, isplot=True, output_dir="./PINN")
    
    # Save history
    with open("./results/pinn_history.pkl", "wb") as f:
        pickle.dump({
            "loss_train": losshistory_pinn.loss_train,
            "loss_test": losshistory_pinn.loss_test,
            "metrics_test": losshistory_pinn.metrics_test,
            "steps": losshistory_pinn.steps,
        }, f)

    # ==================== gPINN (fixed weights) ====================
    print("\n" + "="*60)
    print("--- gPINN (fixed weights [1, 0.1, 0.1]) ---")
    print("="*60)
    data_g = dde.data.TimePDE(
        geomtime, gPINNpde, [], num_domain=50, solution=solution, num_test=1000
    )
    net_g = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
    net_g.apply_output_transform(output_transform)
    m_gpinn = dde.Model(data_g, net_g)
    m_gpinn.compile("adam", lr=1e-4, loss_weights=[1, 0.1, 0.1], metrics=["l2 relative error"])
    losshistory_gpinn, train_state_gpinn = m_gpinn.train(iterations=iters)
    dde.saveplot(losshistory_gpinn, train_state_gpinn, issave=True, isplot=True, output_dir="./gPINN")
    
    # Save history
    with open("./results/gpinn_history.pkl", "wb") as f:
        pickle.dump({
            "loss_train": losshistory_gpinn.loss_train,
            "loss_test": losshistory_gpinn.loss_test,
            "metrics_test": losshistory_gpinn.metrics_test,
            "steps": losshistory_gpinn.steps,
        }, f)

    print("\n" + "="*60)
    print("DeepXDE baseline training complete!")
    print("="*60)

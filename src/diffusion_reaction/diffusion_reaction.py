import numpy as np
import matplotlib.pyplot as plt
import torch
import deepxde as dde

# (Include ScalarBRDR Class from above here)
class ScalarBRDR(dde.callbacks.Callback):
    def __init__(self, pde_index, grad_index, update_every=100, beta=0.99):
        super(ScalarBRDR, self).__init__()
        self.pde_idx = pde_index
        self.grad_idx = grad_index
        self.update_every = update_every
        self.beta = beta
        self.ema_sq_pde = 0.0
        self.ema_sq_grad = 0.0
        self.initialized = False

    def on_epoch_end(self):
        step = self.model.train_state.step
        losses = self.model.train_state.loss_train
        if losses is None or len(losses) <= max(self.pde_idx, self.grad_idx):
            return
        loss_pde = losses[self.pde_idx]
        loss_grad = losses[self.grad_idx]
        if not self.initialized:
            self.ema_sq_pde = loss_pde**2
            self.ema_sq_grad = loss_grad**2
            self.initialized = True
        self.ema_sq_pde = self.beta * self.ema_sq_pde + (1 - self.beta) * (loss_pde**2)
        self.ema_sq_grad = self.beta * self.ema_sq_grad + (1 - self.beta) * (loss_grad**2)
        if step % self.update_every == 0 and step > 0:
            eps = 1e-10
            irdr_pde = (loss_pde**2) / (np.sqrt(self.ema_sq_pde) + eps)
            irdr_grad = (loss_grad**2) / (np.sqrt(self.ema_sq_grad) + eps)
            irdr_mean = (irdr_pde + irdr_grad) / 2.0
            old_weight = self.model.loss_weights[self.grad_idx]
            target = irdr_grad / irdr_mean
            new_weight = old_weight + (1 - self.beta) * (target - old_weight)
            clipped_weight = max(1e-6, min(new_weight, 1.0))
            self.model.loss_weights[self.grad_idx] = clipped_weight
            print(f" [BRDR] Step {step}: P_decay={irdr_pde:.2f}, G_decay={irdr_grad:.2f}, W_grad={new_weight:.4e}")

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
    return [dy_t - dy_xx - r]

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
    
    dy_tx = dde.grad.hessian(y, x, i=0, j=1)
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)
    dr_x = torch.exp(-t_in) * (
        63 * torch.cos(8 * x_in) + 15 * torch.cos(4 * x_in)
        + 8 * torch.cos(3 * x_in) + 3 * torch.cos(2 * x_in)
    )

    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)
    dr_t = -r

    return [dy_t - dy_xx - r, dy_tx - dy_xxx - dr_x, dy_tt - dy_xxt - dr_t]

def solution(a):
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val

def output_transform(x, y):
    x_in, t_in = x[:, 0:1], x[:, 1:2]
    ic = (torch.sin(8 * x_in)/8 + torch.sin(x_in) + torch.sin(2*x_in)/2 + torch.sin(3*x_in)/3 + torch.sin(4*x_in)/4)
    return (x_in - np.pi) * (x_in + np.pi) * (1 - torch.exp(-t_in)) * y + ic

geom = dde.geometry.Interval(-np.pi, np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

iters = 100000

# PINN
print("--- PINN ---")
data = dde.data.TimePDE(geomtime, PINNpde, [], num_domain=50, solution=solution, num_test=1000)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)
m_pinn = dde.Model(data, net)
m_pinn.compile("adam", lr=1e-4, metrics=["l2 relative error"])
losshistory, train_state = m_pinn.train(iterations=iters)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir= "./PINN")
# gPINN
print("--- gPINN ---")
data_g = dde.data.TimePDE(geomtime, gPINNpde, [], num_domain=50, solution=solution, num_test=1000)
net_g = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
net_g.apply_output_transform(output_transform)
m_gpinn = dde.Model(data_g, net_g)
m_gpinn.compile("adam", lr=1e-4, loss_weights=[1, 0.1, 0.1],metrics=["l2 relative error"])
losshistory, train_state = m_gpinn.train(iterations=iters)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir="./gPINN" )
# gPINN BRDR
print("--- gPINN BRDR ---")
net_brdr = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
net_brdr.apply_output_transform(output_transform)
m_brdr = dde.Model(data_g, net_brdr)
m_brdr.compile("adam", lr=1e-4, loss_weights=[1.0, 0.1, 0.1],metrics=["l2 relative error"])
losshistory, train_state = m_brdr.train(iterations=iters, callbacks=[ScalarBRDR(0,1), ScalarBRDR(0,2)])
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir="./gPINN_BRDR" )

# Plot
#x_plot = geomtime.random_points(1000)
#plt.figure()
#plt.plot(x_plot[:,0], solution(x_plot), 'k.', label="Exact")
#plt.plot(x_plot[:,0], m_pinn.predict(x_plot), 'b.', label="PINN")
#plt.plot(x_plot[:,0], m_gpinn.predict(x_plot), 'g.', label="gPINN")
#plt.plot(x_plot[:,0], m_brdr.predict(x_plot), 'r.', label="gPINN (BRDR)")
#plt.legend()
#plt.title("Reaction Diffusion Random Sample")
#plt.savefig("DiffReact_Comparison.png")
#plt.show()
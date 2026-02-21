import numpy as np
import matplotlib.pyplot as plt
import torch
import deepxde as dde
def make_trainable_logsigmas(num_terms, init_values=None):
    if init_values is None:
        init_values = [0.0] * num_terms
    init_values = np.array(init_values, dtype=np.float32)
    # 用 torch.nn.Parameter 做 trainable 參數，DeepXDE 會透過 external_trainable_variables 更新它
    return torch.nn.Parameter(torch.from_numpy(init_values))


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
class UncertaintyWeightedNet(torch.nn.Module):
    def __init__(self, base_net, init_log_sigmas=None):
        super().__init__()
        self.base_net = base_net
        # log_sigma 參數（用 log 形式比較穩定）
        if init_log_sigmas is None:
            # 三個 loss：PDE, grad_x, grad_t
            init_log_sigmas = torch.zeros(3)
        self.log_sigmas = torch.nn.Parameter(init_log_sigmas.clone().detach())

    def forward(self, x):
        return self.base_net(x)

    def get_sigmas_and_weights(self):
        sigmas = torch.exp(self.log_sigmas)
        weights = 1.0 / (sigmas ** 2)
        return sigmas, weights

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
def uncertainty_loss(outputs, inputs, loss_fn, model):
    """
    outputs, inputs: DeepXDE 傳進來的標準參數（你可以不拿來用）
    loss_fn: 原本的 DeepXDE loss function，會回傳各個 loss term 的 list/tuple
    model: dde.Model 物件，我們需要從裡面的 net 拿 log_sigmas
    """
    # 先算原始的各 loss term
    losses = loss_fn(outputs, inputs, None)
    # losses 是 list: [L_pde, L_grad1, L_grad2]
    # 取出我們自定義的 net
    net = model.net
    if not isinstance(net, UncertaintyWeightedNet):
        # 如果不是用我們的 wrapper，就照原樣返回
        return losses

    sigmas, _ = net.get_sigmas_and_weights()
    # 確保長度一致
    num_terms = min(len(losses), len(sigmas))

    total = 0
    new_losses = []
    for i in range(num_terms):
        Li = losses[i]
        sigma_i = sigmas[i]
        # uncertainty-based weighting: 0.5 * Li / sigma^2 + log sigma
        weighted_Li = 0.5 * Li / (sigma_i ** 2) + torch.log(sigma_i)
        total = total + weighted_Li
        new_losses.append(weighted_Li)

    # 若 loss term 比 sigma 多，直接加上沒加權的（通常不會）
    for i in range(num_terms, len(losses)):
        total = total + losses[i]
        new_losses.append(losses[i])


    # DeepXDE 會用 list 各項來顯示，你也可以回傳 total
    return new_losses

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

# === Uncertainty-based weights for gPINN ===
# 我們對三個 loss: [PDE, grad_x, grad_t] 各有一個 log_sigma
log_sigmas_g = make_trainable_logsigmas(
    3,
    init_values=[0.0, 0.5, 0.5],  # PDE: logσ=0, grad: logσ=0.5 => 初始權重比較小
)

def get_uncertainty_weights():
    """回傳目前的 1/(2 σ_i^2)，用在 loss_weights 裡。"""
    # log_sigmas_g 是 torch.nn.Parameter
    log_s = log_sigmas_g.detach().cpu().numpy()
    sigmas = np.exp(log_s)
    weights = 1.0 / (2.0 * (sigmas ** 2))
    return weights


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
# gPINN with Uncertainty-based Adaptive Weights
print("--- gPINN Uncertainty ---")

net_unc = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
net_unc.apply_output_transform(output_transform)

m_unc = dde.Model(data_g, net_unc)

# 一開始先用初始的權重
init_weights = get_uncertainty_weights()
m_unc.compile(
    "adam",
    lr=1e-4,
    loss_weights=init_weights.tolist(),
    metrics=["l2 relative error"],
    external_trainable_variables=log_sigmas_g,
)


N = 1000  # 每 N 步刷新一次權重，可以視情況調

losshistory_list = []
train_state_last = None

for it in range(0, iters, N):
    cur_iters = min(N, iters - it)
    print(f" [Unc] Train steps {it} -> {it + cur_iters}")
    losshistory, train_state = m_unc.train(iterations=cur_iters, display_every=100)
    losshistory_list.append(losshistory)
    train_state_last = train_state

    new_weights = get_uncertainty_weights()
    m_unc.loss_weights = new_weights.tolist()
    print(" [Unc] log_sigmas:", log_sigmas_g.detach().cpu().numpy(),
        "weights:", new_weights)
# 存圖（簡單合併 history）
dde.saveplot(losshistory_list[-1], train_state_last, issave=True, isplot=True, output_dir="./gPINN_uncertainty")




x = geomtime.random_points(1000)
print("L2 relative error of u:")
print("\tPINN:", dde.metrics.l2_relative_error(solution(x), m_pinn.predict(x)))
print("\tgPINN:", dde.metrics.l2_relative_error(solution(x), m_gpinn.predict(x)))
print("\tgPINN_unc:", dde.metrics.l2_relative_error(solution(x), m_unc.predict(x)))

PINNresiduals = m_pinn.predict(x, operator=PINNpde)[0]
gPINNresiduals = m_gpinn.predict(x, operator=gPINNpde)[0]
gPINNresidualsunc = m_unc.predict(x, operator=gPINNpde)[0]
print("Mean absolute PDE residual:")
print("\tPINN:", np.mean(abs(PINNresiduals)))
print("\tgPINN:", np.mean(abs(gPINNresiduals)))
print("\tgPINNunc:", np.mean(abs(gPINNresidualsunc)))
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import torch
import deepxde as dde

# ==============================================================================
# 1. Standard PINN
# ==============================================================================
def PINNpde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    f = 8 * torch.sin(8 * x)
    for i in range(1, 5):
        f += i * torch.sin(i * x)
    return -dy_xx - f

def func(x):
    sol = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * np.sin(i * x)
    return sol

geom = dde.geometry.Interval(0, np.pi)
data = dde.data.PDE(geom, PINNpde, [], 15, 0, "uniform", solution=func, num_test=100)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

def output_transform(x, y):
    return x + torch.tanh(x) * torch.tanh(np.pi - x) * y

net.apply_output_transform(output_transform)

PINNmodel = dde.Model(data, net)
PINNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = PINNmodel.train(iterations=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)


# ==============================================================================
# 2. Self-Adaptive gPINN (Using Scalar BRDR)
# ==============================================================================

class ScalarBRDR(dde.callbacks.Callback):
    """
    Implements the Balanced Residual Decay Rate (BRDR) algorithm for Scalar Losses.
    Instead of balancing magnitudes, this balances the 'decay rate' (progress).
    """
    def __init__(self, update_every=100, beta=0.99):
        super(ScalarBRDR, self).__init__()
        self.update_every = update_every
        self.beta = beta
        
        # Exponential Moving Average (EMA) of squared losses
        self.ema_sq_pde = 0.0
        self.ema_sq_grad = 0.0
        self.initialized = False

    def on_epoch_end(self):
        step = self.model.train_state.step
        
        # Capture current losses
        losses = self.model.train_state.loss_train
        if losses is None or len(losses) < 2:
            return
        
        loss_pde = losses[0]
        loss_grad = losses[1]
        
        # Initialize EMAs on first run
        if not self.initialized:
            self.ema_sq_pde = loss_pde**2
            self.ema_sq_grad = loss_grad**2
            self.initialized = True
        
        # Update EMA (Tracks the "Typical History" of the loss)
        # E[t] = beta * E[t-1] + (1-beta) * L[t]^2
        self.ema_sq_pde = self.beta * self.ema_sq_pde + (1 - self.beta) * (loss_pde**2)
        self.ema_sq_grad = self.beta * self.ema_sq_grad + (1 - self.beta) * (loss_grad**2)
        
        # Perform Weight Update every N steps
        if step % self.update_every == 0 and step > 0:
            
            # 1. Compute IRDR (Inverse Residual Decay Ratio)
            # IRDR = Current_Loss^2 / sqrt(History_Variance)
            # High IRDR = Loss is stuck (stubborn) compared to history -> Needs higher weight
            # Low IRDR  = Loss is dropping fast (easy) -> Can reduce weight
            eps = 1e-10
            irdr_pde = (loss_pde**2) / (np.sqrt(self.ema_sq_pde) + eps)
            irdr_grad = (loss_grad**2) / (np.sqrt(self.ema_sq_grad) + eps)
            
            # 2. Compute Global Average IRDR
            irdr_mean = (irdr_pde + irdr_grad) / 2.0
            
            # 3. Update Weights based on stubbornness
            # Formula: W_new = W_old + alpha * (IRDR / IRDR_Mean - W_old)
            # We fix PDE weight to 1.0, so we only adapt Gradient weight relative to the mean.
            
            old_weight_grad = self.model.loss_weights[1]
            alpha = 1.0 - self.beta # Learning rate for the weight adaptation
            
            target_weight = irdr_grad / irdr_mean
            
            # Smooth update rule (Exponential Moving Average for weight)
            new_weight_grad = old_weight_grad + alpha * (target_weight - old_weight_grad)
            clipped_weight = max(1e-6, min(new_weight_grad, 1.0))

            self.model.loss_weights[1] = clipped_weight
            print(f" [BRDR] Step {step}: P_decay={irdr_pde:.2f}, G_decay={irdr_grad:.2f}, W_grad={new_weight_grad:.4e}")

def gPINNpde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    dy_xxx = dde.grad.jacobian(dy_xx, x)

    f = 8 * torch.sin(8 * x)
    for i in range(1, 5):
        f += i * torch.sin(i * x)
    
    df_x = (
        torch.cos(x)
        + 4 * torch.cos(2 * x)
        + 9 * torch.cos(3 * x)
        + 16 * torch.cos(4 * x)
        + 64 * torch.cos(8 * x)
    )

    return [-dy_xx - f, -dy_xxx - df_x]

geom = dde.geometry.Interval(0, np.pi)
data = dde.data.PDE(geom, gPINNpde, [], 15, 0, "uniform", solution=func, num_test=100)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

net.apply_output_transform(output_transform)

gPINNmodel = dde.Model(data, net)

# Initial weights: Start both at 1.0
gPINNmodel.compile(
    "adam", 
    lr=0.001, 
    metrics=["l2 relative error"], 
    loss_weights=[1.0, 0.01] 
)

# Use ScalarBRDR Callback
callbacks = [ScalarBRDR(update_every=100, beta=0.9)]

print("\n--- Training gPINN with BRDR Adaptive Weights ---")
losshistory, train_state = gPINNmodel.train(iterations=20000, callbacks=callbacks)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# ==============================================================================
# 3. Plots
# ==============================================================================

print("\n--- Generating Plots ---")

x = geom.uniform_points(1000)

plt.figure(figsize=(10, 6))
plt.plot(x, func(x), label="Exact", color="black", linewidth=2)
plt.plot(x, PINNmodel.predict(x), label="PINN", color="blue", linestyle="dashed")
plt.plot(x, gPINNmodel.predict(x), label="gPINN (BRDR)", color="red", linestyle="dashed")
plt.legend()
plt.title("Solution u(x)")
plt.savefig("Solution_Comparison.png")
plt.show()

def du_x(x):
    return 1 + np.cos(x) + np.cos(2 * x) + np.cos(3 * x) + np.cos(4 * x) + np.cos(8 * x)

plt.figure(figsize=(10, 6))
plt.plot(x, du_x(x), label="Exact", color="black", linewidth=2)
plt.plot(
    x,
    PINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="PINN",
    color="blue",
    linestyle="dashed",
)
plt.plot(
    x,
    gPINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="gPINN (BRDR)",
    color="red",
    linestyle="dashed",
)
plt.legend()
plt.title("Gradient u'(x)")
plt.savefig("Gradient_Comparison.png")
plt.show()
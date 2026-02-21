import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load histories
with open("./results/pinn_history.pkl", "rb") as f:
    pinn_hist = pickle.load(f)

with open("./results/gpinn_history.pkl", "rb") as f:
    gpinn_hist = pickle.load(f)

with open("./results/gpinn_uncertainty_history.pkl", "rb") as f:
    unc_hist = pickle.load(f)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Test L2 relative error
ax = axes[0, 0]
ax.semilogy(pinn_hist["steps"], pinn_hist["metrics_test"], 'b-', label='PINN', linewidth=2)
ax.semilogy(gpinn_hist["steps"], gpinn_hist["metrics_test"], 'g-', label='gPINN [1,0.1,0.1]', linewidth=2)
ax.semilogy(unc_hist["steps"], unc_hist["rel_l2_t05"], 'r-', label='gPINN Uncertainty', linewidth=2)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Relative L2 Error (t=0.5)', fontsize=12)
ax.set_title('Test Error Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Training loss
ax = axes[0, 1]
pinn_total = np.array(pinn_hist["loss_train"])
gpinn_total = np.array([sum(losses) for losses in gpinn_hist["loss_train"]])
ax.semilogy(pinn_hist["steps"], pinn_total, 'b-', label='PINN', linewidth=2)
ax.semilogy(gpinn_hist["steps"], gpinn_total, 'g-', label='gPINN [1,0.1,0.1]', linewidth=2)
ax.semilogy(unc_hist["steps"], [l[0]+l[1]+l[2] for l in zip(unc_hist["L_pde"], unc_hist["L_gx"], unc_hist["L_gt"])], 'r-', label='gPINN Uncertainty', linewidth=2)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Total Training Loss', fontsize=12)
ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Adaptive weights evolution
ax = axes[1, 0]
weights_arr = np.array(unc_hist["weights"])
ax.semilogy(unc_hist["steps"], weights_arr[:, 0], 'b-', label='PDE weight', linewidth=2)
ax.semilogy(unc_hist["steps"], weights_arr[:, 1], 'g-', label='Grad-x weight', linewidth=2)
ax.semilogy(unc_hist["steps"], weights_arr[:, 2], 'r-', label='Grad-t weight', linewidth=2)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Loss Weights', fontsize=12)
ax.set_title('Uncertainty-Based Weight Evolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Final error comparison (bar chart)
ax = axes[1, 1]
final_errors = [
    pinn_hist["metrics_test"][-1][0],
    gpinn_hist["metrics_test"][-1][0],
    unc_hist["rel_l2_t05"][-1]
]
methods = ['PINN', 'gPINN\n[1,0.1,0.1]', 'gPINN\nUncertainty']
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(methods, final_errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Final Relative L2 Error', fontsize=12)
ax.set_title('Final Test Error Comparison', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, err in zip(bars, final_errors):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{err:.2e}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./results/comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Final Results Summary")
print("="*60)
print(f"PINN final error:              {final_errors[0]:.6e}")
print(f"gPINN [1,0.1,0.1] final error: {final_errors[1]:.6e}")
print(f"gPINN Uncertainty final error: {final_errors[2]:.6e}")
print("="*60)
print(f"\nPlot saved to: ./results/comparison.png")

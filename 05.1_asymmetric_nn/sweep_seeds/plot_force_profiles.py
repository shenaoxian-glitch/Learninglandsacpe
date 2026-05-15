#!/usr/bin/env python
"""
plot_force_profiles.py — Force profiles F_x = -dPhi/dx for all seeds

Loads replot_data.npz (no re-training needed).
Plots overlaid force profiles at y = -0.5, 0.5, 1.5 for all 10 seeds + true.

Usage:
    python sweep_seeds/plot_force_profiles.py
"""
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt

SWEEP_DIR = os.path.dirname(__file__)
data = np.load(os.path.join(SWEEP_DIR, "replot_data.npz"))

x_line = data['x_line']
seeds = data['seeds_learned']
y_slices = [-0.5, 0.5, 1.5]

# Compute force from potential: F_x = -dPhi/dx (central differences)
dx = x_line[1] - x_line[0]

# ======================================================================
# Figure 1: Overlaid force profiles (all seeds + true)
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

for col, y_val in enumerate(y_slices):
    ax = axes[col]

    # True force
    phi_true = data[f'true_profile_y{y_val}']
    F_true = -np.gradient(phi_true, dx)
    ax.plot(x_line, F_true, 'k-', linewidth=3, label='True', zorder=10)

    # All learned seeds
    for i, s in enumerate(seeds):
        phi_l = data[f'profile_seed{s}_y{y_val}']
        F_l = -np.gradient(phi_l, dx)
        ax.plot(x_line, F_l, color=colors[i], linewidth=1, alpha=0.7,
                label=f's={s}')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$F_x = -\partial\Phi/\partial x$', fontsize=12)
    ax.set_title(f'y = {y_val}', fontsize=13)
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, alpha=0.3)
    if col == 2:
        ax.legend(fontsize=7, ncol=2, loc='upper right')

plt.suptitle(
    f'Force profiles across {len(seeds)} seeds '
    f'(seed_target={int(data["seed_target"])}, N=4000)',
    fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SWEEP_DIR, "summary_force_overlay.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary_force_overlay.png")

# ======================================================================
# Figure 2: Force envelope (mean ± std shading) + true
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for col, y_val in enumerate(y_slices):
    ax = axes[col]

    # True force
    phi_true = data[f'true_profile_y{y_val}']
    F_true = -np.gradient(phi_true, dx)

    # Stack all learned forces
    F_all = []
    for s in seeds:
        phi_l = data[f'profile_seed{s}_y{y_val}']
        F_all.append(-np.gradient(phi_l, dx))
    F_all = np.array(F_all)
    F_mean = F_all.mean(axis=0)
    F_std = F_all.std(axis=0)

    # Envelope
    ax.fill_between(x_line, F_mean - F_std, F_mean + F_std,
                    color='tab:red', alpha=0.2, label='Learned ±1σ')
    ax.fill_between(x_line, F_mean - 2*F_std, F_mean + 2*F_std,
                    color='tab:red', alpha=0.08, label='Learned ±2σ')
    ax.plot(x_line, F_mean, 'r-', linewidth=2, label='Learned mean')
    ax.plot(x_line, F_true, 'b-', linewidth=2.5, label='True')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$F_x = -\partial\Phi/\partial x$', fontsize=12)
    ax.set_title(f'y = {y_val}', fontsize=13)
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

plt.suptitle(
    f'Force envelope (mean ± std over {len(seeds)} seeds)',
    fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SWEEP_DIR, "summary_force_envelope.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary_force_envelope.png")

# ======================================================================
# Figure 3: Force error (learned - true) per seed
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for col, y_val in enumerate(y_slices):
    ax = axes[col]

    phi_true = data[f'true_profile_y{y_val}']
    F_true = -np.gradient(phi_true, dx)

    for i, s in enumerate(seeds):
        phi_l = data[f'profile_seed{s}_y{y_val}']
        F_l = -np.gradient(phi_l, dx)
        F_err = F_l - F_true
        ax.plot(x_line, F_err, color=colors[i], linewidth=1, alpha=0.7,
                label=f's={s}')

    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$\Delta F_x$ (learned $-$ true)', fontsize=12)
    ax.set_title(f'y = {y_val}', fontsize=13)
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, alpha=0.3)
    if col == 2:
        ax.legend(fontsize=7, ncol=2, loc='upper right')

plt.suptitle(
    f'Force error across {len(seeds)} seeds',
    fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SWEEP_DIR, "summary_force_error.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary_force_error.png")

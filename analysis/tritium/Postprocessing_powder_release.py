import matplotlib.pyplot as plt
import numpy as np
import json

# Import results from the JSON file
with open("LiOx_powder_results.json", "r") as f:
    data = json.load(f)

# rebuild arrays
time_points = np.array(data["time_points"])
data_intervals = np.array(data["data_intervals"])
dt_history = np.array(data["dt_history"])
rel_change_history = np.array(data["rel_change_history"])
total_inventory_history = np.array(data["total_inventory_history"])
bed_release_rate_history = np.array(data["bed_release_rate_history"])
grain_release_rate_history = np.array(data["grain_release_rate_history"])
C_sparge_outlet_history = np.array(data["C_sparge_outlet_history"])
Cm_history = np.array(data["Cm_history"])
C_sparge_profile_history = {k: np.array(v) for k, v in data["C_sparge_profile_history"].items()}
total_generated_history = np.array(data["total_generated_history"])
key_times = np.array(data["key_times"])
t_irr = data["t_irr"]
total_sim_time = data["total_sim_time"]
N = int(data["N"])
Nz = int(data["Nz"])
r = np.array(data["r"])
z = np.array(data["z"])
decay_constant = data["decay_constant"]
Q_sparge = data["Q_sparge"]
G_rate = data["G_rate"]
V_grain = data["V_grain"]

# --- Plot results ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Powder Bed Tritium Transport Simulation Results (Plug Flow Model)', fontsize=16)
axes = axes.flatten()
plot_time_days = np.array(time_points) / 86400

# a) Total Bed Inventory
axes[0].plot(plot_time_days, total_inventory_history, color='green', label='Simulated Inventory')
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Tritium Inventory (Bq)')
axes[0].set_title('Total Tritium Inventory in Packed Bed (Bq)')
axes[0].ticklabel_format(axis='y', style='plain')
axes[0].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[0].legend()
axes[0].grid(True)

# b) Bed Release Rate
axes[1].plot(plot_time_days, bed_release_rate_history, color='purple')
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Packed bed release rate (atoms/s)')
axes[1].set_title('Tritium Release Rate from Bed Outlet')
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[1].axvspan(0, t_irr/86400, color='red', alpha=0.3)
axes[1].grid(True)

# c) Sparge Gas Outlet Concentration
axes[2].plot(plot_time_days, C_sparge_outlet_history, label='Outlet Sparge Gas', color='orange')
axes[2].set_xlabel('Time (days)')
axes[2].set_ylabel('Concentration (T/cm³)')
axes[2].set_title('Sparge Gas Concentration at Bed Outlet')
axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[2].axvspan(0, t_irr/86400, color='red', alpha=0.3)
axes[2].legend(loc='upper left')
axes[2].grid(True)

# d) Sparge Gas Axial Profile
colors = plt.cm.viridis(np.linspace(0, 1, len(key_times) + 1))
i=0
for label, profile in C_sparge_profile_history.items():
    i += 1
    axes[3].plot(z, profile, color=colors[i], label=label)
axes[3].set_xlabel('Axial Position (z) in Packed Bed (cm)')
axes[3].set_ylabel('Sparge Gas Concentration (T/cm³)')
axes[3].set_title('Sparge Gas Concentration Axial Profile')
axes[3].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[3].legend()
axes[3].grid(True)

# e) Mobile Concentration Within Grain
Cm_history = np.array(Cm_history)
# Select 5 radial positions equally spaced from centre (index 0) to surface (index N)
selected_indices = np.linspace(0, N, 5).astype(int)
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))
for k, idx in enumerate(selected_indices):
    axes[4].plot(plot_time_days, Cm_history[:, idx], color=colors[k], label=f"r={r[idx]:.3f} cm (i={idx})")
axes[4].set_xlabel('Time (days)')
axes[4].set_ylabel('Mobile Concentration (T/cm³)')
axes[4].set_title('Mobile Tritium Concentration Profile in Grain Over Time, at Plug 0 (inlet)')
axes[4].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[4].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[4].legend(fontsize='small', ncol=2)
axes[4].grid(True)

# f) Mass Balance Check
cumulative_release = np.cumsum(np.array(bed_release_rate_history) * data_intervals) * decay_constant  # Cumulative release (Bq)
inventory_plus_release = np.array(total_inventory_history) + cumulative_release

axes[5].plot(plot_time_days, total_generated_history, 'k--', label='Total Generated')
axes[5].plot(plot_time_days, total_inventory_history, 'g-', label='Total Inventory')
axes[5].plot(plot_time_days, cumulative_release, 'b--', label='Cumulative Release')
axes[5].plot(plot_time_days, inventory_plus_release, 'r-', label='Inventory + Cumulative Release')
axes[5].set_xlabel('Time (days)')
axes[5].set_ylabel('Inventory (Bq)')
axes[5].set_title('Mass Balance Verification')
axes[5].ticklabel_format(axis='y', style='plain')
axes[5].axvspan(0, t_irr/86400, color='red', alpha=0.2)
axes[5].legend()


# --- PLOT TIMESTEP AND RELATIVE CHANGE HISTORY ---
fig2, ax2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig2.suptitle('Adaptive Timestep and Relative Change History', fontsize=15)

# Plot dt_history
ax2[0].plot(plot_time_days, dt_history, color='blue')
ax2[0].set_ylabel('Timestep (s)')
ax2[0].set_title('Adaptive Timestep (dt) vs Time')
ax2[0].axvspan(0, t_irr/86400, color='red', alpha=0.2)
ax2[0].grid(True)

# Plot rel_change_history
ax2[1].plot(plot_time_days, rel_change_history, color='red')
ax2[1].set_xlabel('Time (days)')
ax2[1].set_ylabel('Max Relative Change')
ax2[1].set_title('Max Relative Change vs Time')
ax2[1].axvspan(0, t_irr/86400, color='red', alpha=0.2)
ax2[1].grid(True)


# --- 7. PLOT GRAIN RELEASE RATE HISTORY ---
cumulative_grain_release_history = np.cumsum(np.array(grain_release_rate_history) * data_intervals) # Cumulative grain release (atoms)
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(plot_time_days, grain_release_rate_history, color='teal', label='Grain Release Rate')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Grain Release Rate at Inlet (atoms/s)')
ax3.set_title('Per Grain Release Rate at Bed Inlet Over Time')
ax3.axvspan(0, t_irr/86400, color='red', alpha=0.2, label='Irradiation Period')
ax3.grid(True)

# Add cumulative release on secondary y-axis
ax3b = ax3.twinx()
ax3b.plot(plot_time_days, cumulative_grain_release_history, color='orange', linestyle='--', label='Cumulative Grain Release')
ax3b.set_ylabel('Cumulative Grain Release (atoms)')
ax3b.set_ylim(bottom=0)

# Add horizontal dashed line for total tritium production
total_tritium_produced_atoms = G_rate * V_grain * t_irr
ax3b.axhline(total_tritium_produced_atoms, color='gray', linestyle='dashed', linewidth=2, label='Total Tritium Produced')

# Combine legends from both axes
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()

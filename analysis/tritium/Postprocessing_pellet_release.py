import matplotlib.pyplot as plt
import numpy as np
import json

# --- 1. LOAD DATA FROM JSON FILE ---
filename = 'pellet_release_results.json'
print(f"--- Loading data from {filename} ---")

with open(filename, 'r') as f:
    data = json.load(f)

# Extract parameters and results from the loaded data
params = data['simulation_parameters']
results = data['simulation_results']

# --- 2. RECONSTRUCT VARIABLES FOR PLOTTING ---
decay_constant = 1.785e-9 # Tritium Decay Constant (1/s)

# Simulation parameters
t_irr = params['irradiation_time_s']
r_g = params['grain_radius_cm']
z_bed = params['bed_length_cm']
N = params['radial_nodes_grain']
Nz = params['axial_nodes_bed']

# Time-series results
time_points = results['time_s']
total_inventory_history = results['total_inventory_Bq']
bed_release_rate_history = results['bed_release_rate_atoms_s']
C_sparge_outlet_history = results['outlet_sparge_concentration_T_cm3']
C_pore_outlet_history = results['outlet_pore_concentration_T_cm3']
total_generated_history = results['total_generated_tritium_Bq']
dt_history = results['adaptive_timestep_s']
rel_change_history = results['max_relative_change']

# Profile results
C_sparge_profile_history = results['sparge_gas_axial_profiles_T_cm3']
Cm_history = np.array(results['grain_mobile_concentration_profile_inlet_T_cm3'])
key_times = np.array(results['key_times'])

# Reconstruct spatial grids
# Radial grid for the grain model
r = np.linspace(0, r_g, N + 1)
# Axial grid for the packed bed model
dz = z_bed / Nz
z = np.linspace(dz/2, z_bed - dz/2, Nz)

# Reconstruct other necessary arrays
# The time interval 'dt' for each step, needed for cumulative sum
data_intervals = np.diff(time_points, prepend=0)

print("--- Data loaded and variables reconstructed successfully ---\n")

# --- 5. PLOT THE RESULTS ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Pellet Bed Tritium Transport Simulation Results (Plug Flow Model)', fontsize=16)
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
axes[2].plot(plot_time_days, C_pore_outlet_history, label='Outlet Pore Gas', color='blue', linestyle='--')
axes[2].set_xlabel('Time (days)')
axes[2].set_ylabel('Concentration (T/cm³)')
axes[2].set_title('Sparge Gas & Pore Gas Concentration at Bed Outlet')
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
    axes[4].plot(plot_time_days, Cm_history[:, idx], color=colors[k], label=f"r={r[idx]:.3f} cm")
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

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 6. PLOT TIMESTEP AND RELATIVE CHANGE HISTORY ---
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

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

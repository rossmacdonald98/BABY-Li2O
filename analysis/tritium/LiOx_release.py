import numpy as np
import matplotlib.pyplot as plt
import time 


# --- 1. PHYSICAL AND SIMULATION PARAMETERS ---

# --- Grain, Pellet & Packed Bed Properties ---
r_g = 0.01          # Average Grain Radius (cm)
r_p = 0.3           # Pellet Radius (cm)
fr = 0.8            # Surface Area Reduction Factor (accounts for necking)
porosity_pellet = 0.2  # Pellet porosity (void fraction, ε)
packing_density = 0.62 # Packing efficiency for random spheres (φ)
r_bed = 6.5          # Packed Bed Radius (cm)
z_bed = 8         # Packed Bed Length (cm)

# --- Diffusion, Trapping & Generation Properties ---
D = 1.0e-10          # Intrinsic Diffusion Coefficient (cm^2/s)
kt = 1.0e-24        # Trapping Coefficient (cm^3/(atom*s))
kd = 1.0e-3         # Detrapping Coefficient (1/s)
Nt = 1.0e20         # Trapping Site Density (sites/cm^3)
source_rate = 1e8   # Neutron Source Rate (n/s)
tbr = 5e-5          # Volumetric Tritium Breeding Ratio (T/n/cm^3)
G_rate = source_rate*tbr # Tritium Generation Rate (T/cm^3/s)

# --- Gas & System Properties ---
k_grain_ads = 5e-7  # Grain surface adsorption coeff
k_grain_des = 1e-7  # Grain surface desorption coeff
h_pellet = 1e-3   # Pore-Sparge Mass Transfer Coeff (cm/s)
Q_sparge = 8.33e-1 # Sparge Flow Rate (cm^3/s)
decay_constant = 1.785e-9 # Tritium Decay Constant (1/s)

# --- Simulation Parameters ---
t_irr = 7200 # Irradiation Time (s)
total_sim_time = 500000 # Total simulation time (s)
dt = 0.5 # User-defined time step

# --- Simulation Grid ---
# Radial grid for the grain model
N = 10              # Number of radial nodes
dr = r_g / N        # Radial step size (cm)
r = np.linspace(0, r_g, N + 1)  # Radial positions of each node

# Axial grid for the packed bed model
Nz = 15              # Number of axial nodes 
dz = z_bed / Nz      # Axial step size (cm)
z = np.linspace(dz/2, z_bed - dz/2, Nz)  # Axial positions of each node (center of each plug)

n_time_steps = int(total_sim_time / dt)

# --- Derived Parameters ---
V_pellet = (4/3) * np.pi * r_p**3  # Pellet Volume (cm^3)
V_grain = (4/3) * np.pi * r_g**3  # Grain Volume (cm^3)
A_external = 4 * np.pi * r_p**2 # External Surface Area per Pellet (cm^2)
V_pore = V_pellet * porosity_pellet # Pore Volume per Pellet (cm^3)
A_internal = fr * V_pellet * (1-porosity_pellet) * 3/r_g # Internal Surface Area per Pellet (cm^2)
N_grains_pellet = V_pellet * (1-porosity_pellet) / V_grain  # Number of grains per pellet

# Per-plug parameters
V_plug = np.pi * r_bed**2 * dz  # Volume of one axial 'plug' of the bed (cm^3)
V_sparge_plug = V_plug * (1 - packing_density)  # Sparge gas volume in one plug (cm^3)
N_pellets_plug = (V_plug * packing_density) / (V_pellet)  # Number of Pellets in one plug
A_pellets_plug = N_pellets_plug * A_external  # Total External Surface Area of Pellets in one plug (cm^2)

# Gas velocity
A_bed_cross_section = np.pi * r_bed**2  # Cross-sectional area of the packed bed (cm^2)
V_bed = A_bed_cross_section * z_bed  # Total volume of the packed bed (cm^3)
v_gas = Q_sparge / (A_bed_cross_section * (1 - packing_density))  # Correct interstitial gas velocity

# Tritium production
V_breeding_total = V_bed * packing_density * (1 - porosity_pellet)
T_est = G_rate * V_breeding_total * t_irr * decay_constant  # Estimated total tritium produced (Bq)


# --- 2. INITIALIZE CONCENTRATION ARRAYS ---
Cm = np.zeros((Nz, N + 1))
Ct = np.zeros((Nz, N + 1))
C_pore = np.zeros(Nz)
C_sparge = np.zeros(Nz)
J_grain = np.zeros(Nz)
J_pellet = np.zeros(Nz)

# --- Data Storage for Plotting ---
plot_interval = 100
time_points = []
total_inventory_history = []
bed_release_rate_history = []
C_sparge_outlet_history = []
Cm_history = []
C_sparge_profile_history = {}
total_generated_history = []
cumulative_release_history = []

print(f"--- Simulation Setup ---")
print(f"Time Step (dt): {dt:.3f} s, Total Steps: {n_time_steps}")
print(f'Estimated Total Tritium Produced: {T_est:.2e} Bq')
print(f"------------------------\n")

# --- 3. THE MAIN SIMULATION LOOP ---
start_time = time.time()
print("--- Starting Simulation ---")

for step in range(n_time_steps):
    Cm_old = Cm.copy()
    Ct_old = Ct.copy()
    C_pore_old = C_pore.copy()
    C_sparge_old = C_sparge.copy()

    G = G_rate if step * dt < t_irr else 0

    for j in range(Nz):
        # Center Node (i=0)
        dCm_dt_diffusion_center = 6 * D * (Cm_old[j, 1] - Cm_old[j, 0]) / dr**2
        dCm_dt_trapping_center = kt * Cm_old[j, 0] * (Nt - Ct_old[j, 0])
        dCm_dt_detrapping_center = kd * Ct_old[j, 0]
        Cm[j, 0] = Cm_old[j, 0] + (dCm_dt_diffusion_center + G - dCm_dt_trapping_center + dCm_dt_detrapping_center) * dt
        Ct[j, 0] = Ct_old[j, 0] + (dCm_dt_trapping_center - dCm_dt_detrapping_center) * dt

        # Interior Nodes (1 to N-1)
        for i in range(1, N):
            laplacian_term = (Cm_old[j, i+1] - 2*Cm_old[j, i] + Cm_old[j, i-1]) / dr**2
            geometric_term = (1/r[i]) * (Cm_old[j, i+1] - Cm_old[j, i-1]) / dr
            dCm_dt_diffusion = D * (laplacian_term + geometric_term)
            dCm_dt_trapping = kt * Cm_old[j, i] * (Nt - Ct_old[j, i])
            dCm_dt_detrapping = kd * Ct_old[j, i]
            Cm[j, i] = Cm_old[j, i] + (dCm_dt_diffusion + G - dCm_dt_trapping + dCm_dt_detrapping) * dt
            Ct[j, i] = Ct_old[j, i] + (dCm_dt_trapping - dCm_dt_detrapping) * dt

        # Surface Node (i=N)
        J_grain[j] = (Cm_old[j, N]**2 * k_grain_des - C_pore_old[j] * k_grain_ads)
        dCm_dt_diffusion_in = 2 * D * (Cm_old[j, N-1] - Cm_old[j, N]) / dr**2
        dCm_dt_surface_release_out = fr * 2 * J_grain[j] * (1/dr + 1/r[N])
        dCm_dt_trapping_surface = kt * Cm_old[j, N] * (Nt - Ct_old[j, N])
        dCm_dt_detrapping_surface = kd * Ct_old[j, N]
        Cm[j, N] = Cm_old[j, N] + (dCm_dt_diffusion_in - dCm_dt_surface_release_out + G - dCm_dt_trapping_surface + dCm_dt_detrapping_surface) * dt
        Ct[j, N] = Ct_old[j, N] + (dCm_dt_trapping_surface - dCm_dt_detrapping_surface) * dt

        # Pore Gas Concentration
        J_pellet[j] = h_pellet * (C_pore_old[j] - C_sparge_old[j])
        C_pore[j] = C_pore_old[j] + (dt * (J_grain[j] * A_internal - J_pellet[j] * A_external) / V_pore)

    # Update sparge gas concentrations
    source_term_0 = (J_pellet[0] * A_pellets_plug) / V_sparge_plug
    convection_term_0 = (-Q_sparge * C_sparge_old[0]) / V_sparge_plug
    C_sparge[0] = C_sparge_old[0] + dt * (source_term_0 + convection_term_0)
    for j in range(1, Nz):
        source_term = (J_pellet[j] * A_pellets_plug) / V_sparge_plug
        convection_term_in = (Q_sparge * C_sparge_old[j-1]) / V_sparge_plug
        convection_term_out = (Q_sparge * C_sparge_old[j]) / V_sparge_plug
        C_sparge[j] = C_sparge_old[j] + dt * (source_term + convection_term_in - convection_term_out)
    
    if np.isnan(Cm).any():
        print(f"\nERROR: NaN detected at time step {step} ({step*dt:.2f} s). Halting simulation.")
        break

    # --- 4. CALCULATE DERIVED QUANTITIES & STORE DATA ---
    if step % plot_interval == 0:
        current_time = step * dt
        time_points.append(current_time)

        # Total Inventory (Solid + Pore Gas + Sparge Gas)
        solid_inventory = 0
        pore_inventory = 0
        sparge_inventory = 0
        for j in range(Nz):
            total_conc_grain = Cm[j, :] + Ct[j, :]
            inventory_grain = np.sum(total_conc_grain * 4 * np.pi * r**2 * dr) # Total Tritium in grains (atoms)
            solid_inventory += inventory_grain * N_grains_pellet * N_pellets_plug * decay_constant # Solid phase inventory (Bq)
            pore_inventory += C_pore[j] * V_pore * N_pellets_plug * decay_constant # Pore gas inventory (Bq)
            sparge_inventory += C_sparge[j] * V_sparge_plug * decay_constant # Sparge gas inventory (Bq)    
        total_system_inventory = solid_inventory + pore_inventory + sparge_inventory # Total system inventory (Bq)
        total_inventory_history.append(total_system_inventory)

        # Total Generated History
        total_generated = G_rate * min(current_time, t_irr) * V_breeding_total * decay_constant  # Total generated tritium (Bq)
        total_generated_history.append(total_generated)

        # Release Rate
        release_rate = C_sparge[Nz-1] * Q_sparge # Release rate at the outlet (atoms/s)
        bed_release_rate_history.append(release_rate)
        
        # Other histories
        C_sparge_outlet_history.append(C_sparge[Nz-1])
        Cm_history.append(Cm[0, :].copy())
        if step == 0 or abs(current_time - t_irr) < dt or abs(current_time - total_sim_time/2) < dt or step == n_time_steps - 1:
            C_sparge_profile_history[f'{current_time/3600:.1f} hr'] = C_sparge.copy()

        # Print progress update to the console
        if n_time_steps > 20 and step % (n_time_steps / 20) < plot_interval: # Print ~20 updates
             print(f"  Progress: {step/n_time_steps*100:3.0f}%  |  Time: {current_time/3600:5.1f} hr / {total_sim_time/3600:5.1f} hr")


# --- 5. PLOT THE RESULTS ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Tritium Transport Simulation Results (Plug Flow Model)', fontsize=16)
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
axes[2].legend()
axes[2].grid(True)

# d) Sparge Gas Axial Profile
for label, profile in C_sparge_profile_history.items():
    axes[3].plot(z, profile, label=label)
axes[3].set_xlabel('Axial Position (z) in Packed Bed (cm)')
axes[3].set_ylabel('Sparge Gas Concentration (T/cm³)')
axes[3].set_title('Sparge Gas Concentration Axial Profile')
axes[3].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[3].legend()
axes[3].grid(True)

# e) Mobile Concentration Within Grain
colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
Cm_history = np.array(Cm_history)
for i in range(N + 1):
    axes[4].plot(plot_time_days, Cm_history[:, i], color=colors[i], label=f"r={r[i]:.3f} cm")
axes[4].set_xlabel('Time (days)')
axes[4].set_ylabel('Mobile Concentration (T/cm³)')
axes[4].set_title('Mobile Tritium Concentration Profile in Grain Over Time, at Plug 0')
axes[4].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[4].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[4].legend(fontsize='small', ncol=2)
axes[4].grid(True)

# f) Mass Balance Check
time_per_plot_interval = plot_interval * dt
cumulative_release = np.cumsum(np.array(bed_release_rate_history) * time_per_plot_interval) * decay_constant  # Cumulative release (Bq)
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


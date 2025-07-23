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
z_bed = 8         # Packed Bed Height (cm)

# --- Diffusion, Trapping & Generation Properties ---
D = 1.0e-10          # Intrinsic Diffusion Coefficient (cm^2/s)
kt = 1.0e-24        # Trapping Coefficient (cm^3/(atom*s))
kd = 1.0e-3         # Detrapping Coefficient (1/s)
Nt = 1.0e20         # Trapping Site Density (sites/cm^3)
source_rate = 1e8   # Neutron Source Rate (n/s)
tbr = 2e-5          # Volumetric Tritium Breeding Ratio (T/n/cm^3)
G = source_rate*tbr # Tritium Generation Rate (T/cm^3/s)

# --- Gas & System Properties ---
k_grain_ads = 5e-7  # Grain surface adsorption coeff
k_grain_des = 1e-7  # Grain surface desorption coeff
h_pellet = 1e-3   # Pore-Sparge Mass Transfer Coeff (cm/s)
Q_sparge = 8.33e-1 # Sparge Flow Rate (cm^3/s)
decay_constant = 1.785e-9 # Tritium Decay Constant (1/s)

# --- Simulation Parameters ---
t_irr = 7200 # Irradiation Time (s)
total_sim_time = 2000000 # Total simulation time (s)

# --- Simulation Grid ---
# Radial grid for the grain model
N = 10              # Number of radial nodes
dr = r_g / N        # Radial step size (cm)
r = np.linspace(0, r_g, N + 1)  # Radial positions of each node

# Axial grid for the packed bed model
Nz = 10              # Number of axial nodes (used when analysing packed bed)
dz = z_bed / Nz      # Axial step size (cm) (used when analysing packed bed)
z = np.linspace(dz/2, z_bed - dz/2, Nz)  # Axial positions of each node (center of each plug)

# Time grid
dt = 10              # Time step size (s) (Reduce if simulation is unstable)
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
v_gas = Q_sparge / A_bed_cross_section  # Average gas velocity in interstital space between pellets (cm/s)


# --- 2. INITIALIZE CONCENTRATION ARRAYS ---

# 2D arrays to hold concentrations for each axial plug (j) and radial grain node (i)
Cm = np.zeros((Nz, N + 1))  # Mobile concentration array
Ct = np.zeros((Nz, N + 1))  # Trapped concentration array

# 1D arrays for properties at each axial plug
C_pore = np.zeros(Nz)  # Pore gas concentration array
C_sparge = np.zeros(Nz)  # Sparge gas concentration array
J_pellet = np.zeros(Nz)  # Pellet to sparge release flux array

# --- Data Storage for Plotting ---
# store results at specific intervals to avoid saving massive arrays
plot_interval = 100 # Save data for plots every 100 steps
time_points = []
inventory_history = []
inventory_m_history = []
inventory_t_history = []
grain_flux_history = []
pellet_flux_history = []
C_pore_history = []
C_sparge_history = []
Cm_history = []
bed_release_rate_history = []
pore_profile_history = {}
sparge_profile_history = {}
C_sparge_outlet_history = []
inventory_plug_history = []
total_inventory_history = []


print(f"--- Simulation Setup ---")

print(f"Number of Radial Nodes: {N}")
print(f"Radial Step (dr): {dr:.2e} cm")
print(f"Number of Axial 'Plugs': {Nz-1}")
print(f"Axial Step (dz): {dz:.2e} cm")

print(f"Time Step (dt): {dt:.3f} s")
print(f"Total Simulation Time: {total_sim_time} s")
print(f"Total Number of Time Steps: {n_time_steps}")
print(f"------------------------\n")

print(f"--- Model Parameters ---")

print(f"Grain Radius: {r_g:.2e} cm")
print(f"Pellet Radius: {r_p:.2e} cm")

print(f"Packed Bed Radius: {r_bed:.2e} cm")
print(f"Packed Bed Height: {z_bed:.2e} cm")
print(f"Packed Bed Volume: {np.pi * r_bed**2 * z_bed:.2e} cm³")

print(f'Volume of one Pellet: {V_pellet:.2e} cm³')
print(f"Number of Grains per Pellet: {N_grains_pellet:.2f}")

print(f'Volume of one Axial Plug: {V_plug:.2e} cm³')
print(f'Number of Pellets in one Plug: {N_pellets_plug:.2f}')

print(f'Volumetric Tritium Breeding Ratio (TBR): {tbr:.2e} T/n/cm³')
print(f'Neutron Source Rate: {source_rate:.2e} n/s')
print(f'Tritium Generation Rate (G): {G:.2e} T/cm³/s')
print(f"------------------------\n")

# --- 3. THE MAIN SIMULATION LOOP ---
start_time = time.time()  # Start timer
for step in range(n_time_steps):
    # Store a copy of the old concentrations to use in calculations
    Cm_old = Cm.copy()
    Ct_old = Ct.copy()
    C_pore_old = C_pore.copy()
    C_sparge_old = C_sparge.copy()

    # --- Update generation term G  ---
    if step*dt < t_irr:
        # During irradiation, constant generation term (in T/cm^3/s)
        G = source_rate * tbr
    else:
        # After irradiation, the generation term is zero
        G = 0

    # --- Calculate new concentrations ---

    # === Loop through each axial plug ===

    for j in range(Nz):

        # --- a) Interior Nodes (1 to N-1) ---
        # Using the more accurate spherical coordinate formula for concentration curvature
        for i in range(1, N):
            # The standard part of the Laplacian
            laplacian_term = (Cm_old[j, i+1] - 2*Cm_old[j, i] + Cm_old[j, i-1]) / dr**2
            # The geometric part for spherical coordinates
            geometric_term = (1/r[i]) * (Cm_old[j, i+1] - Cm_old[j, i-1]) / dr
            
            # Combine terms
            dCm_dt_diffusion = D * (laplacian_term + geometric_term)
            
            # Trapping terms
            dCm_dt_trapping = kt * Cm_old[j, i] * (Nt - Ct_old[j, i])
            dCm_dt_detrapping = kd * Ct_old[j, i]

            # Update mobile concentration
            Cm[j, i] = Cm_old[j, i] + (dCm_dt_diffusion + G - dCm_dt_trapping + dCm_dt_detrapping) * dt
            
            # Update trapped concentration
            Ct[j, i] = Ct_old[j, i] + (dCm_dt_trapping - dCm_dt_detrapping) * dt

        # --- b) Center Node (i=0) ---
        # Uses special formula for r=0 in spherical coordinates
        dCm_dt_diffusion_center = 6 * D * (Cm_old[j, 1] - Cm_old[j, 0]) / dr**2
        dCm_dt_trapping_center = kt * Cm_old[j, 0] * (Nt - Ct_old[j, 0])
        dCm_dt_detrapping_center = kd * Ct_old[j, 0]
        
        Cm[j, 0] = Cm_old[j, 0] + (dCm_dt_diffusion_center + G - dCm_dt_trapping_center + dCm_dt_detrapping_center) * dt
        Ct[j, 0] = Ct_old[j, 0] + (dCm_dt_trapping_center - dCm_dt_detrapping_center) * dt

        # --- c) Surface Node (i=N) ---
        J_grain = (Cm_old[j, N] * k_grain_des - C_pore_old[j] * k_grain_ads)  # Tritium release rate from grain surface (T/cm²/s)

        # Uses the mass transfer boundary condition
        dCm_dt_diffusion_in = 2 * D * (Cm_old[j, N-1] - Cm_old[j, N]) / dr**2
        dCm_dt_surface_release_out = 2 * J_grain * (1/dr + 1/r[N])

        dCm_dt_trapping_surface = kt * Cm_old[j, N] * (Nt - Ct_old[j, N])
        dCm_dt_detrapping_surface = kd * Ct_old[j, N]
        
        Cm[j, N] = Cm_old[j, N] + (dCm_dt_diffusion_in - dCm_dt_surface_release_out + G - dCm_dt_trapping_surface + dCm_dt_detrapping_surface) * dt
        Ct[j, N] = Ct_old[j, N] + (dCm_dt_trapping_surface - dCm_dt_detrapping_surface) * dt

        # --- d) Pore Gas Concentration for plug j ---
        J_pellet[j] = h_pellet * (C_pore_old[j] - C_sparge_old[j])  # Tritium release from pellet boundary (T/cm²/s)
        C_pore[j] = C_pore_old[0] + (dt / V_pore) * (J_grain * A_internal - J_pellet[j] * A_external) 

    # === End of axial loop ===
    # --- Update sparge gas concentration for all plugs ---

    # Inlet plug (j=0)
    source_term_0 = J_pellet[0] * A_pellets_plug / V_sparge_plug  # Source term 
    convection_term_0 = -v_gas * C_sparge_old[0] / dz  # Convection term, assumes C_inlet = 0
    C_sparge[0] = C_sparge_old[0] + dt * (source_term_0 + convection_term_0)  # Update concentration at inlet

    # Downstream plugs (j > 0)
    for j in range(1, Nz):
        source_term = J_pellet[j] * A_pellets_plug / V_sparge_plug # Source term
        convection_term = -v_gas * (C_sparge_old[j] - C_sparge_old[j-1]) / dz  # Convection term
        C_sparge[j] = C_sparge_old[j] + dt * (source_term + convection_term) # Update concentration at each downstream plug
    
    # --- 4. CALCULATE DERIVED QUANTITIES & STORE DATA ---
    if step % plot_interval == 0:
        current_time = step * dt
        time_points.append(current_time)

        print(f"Processing plug {j+1}/{Nz} at time {current_time:.0f} s", end='\r')

        # Total inventory is the sum of inventories in all plugs
        total_inventory_bed = 0
        inventory_plug = np.zeros(Nz)

        for j in range(Nz):
            # Calculate the total mobile and trapped concentrations for grains in each plug
            total_conc_grain = Cm[j, :] + Ct[j, :]
            # Volume of grain shell i is approx 4*pi*r[i]^2*dr
            inventory_grain = np.sum(total_conc_grain * 4 * np.pi * r**2 * dr)  # Total inventory in each grain in this plug (T)
            inventory_plug[j] = inventory_grain * N_grains_pellet * N_pellets_plug * decay_constant # Total inventory in this plug (Bq)

        total_inventory_bed = np.sum(inventory_plug)  # Sum up all plugs [Bq]

        inventory_plug_history.append(inventory_plug) # Average inventory in each 'plug' of packed bed [Bq]
        total_inventory_history.append(total_inventory_bed) # Total inventory in the entire packed bed [Bq]

        C_sparge_outlet_history.append(C_sparge[Nz-1])  # Sparge gas concentration at outlet 

        # Bed release rate is the flux of tritium out of the bed outlet
        release_rate = C_sparge[Nz-1] * Q_sparge * decay_constant  # [Bq/s]
        bed_release_rate_history.append(release_rate)

        # Store sparge and pore gas concentrations at key times
        if step == 0 or abs(current_time - t_irr) < dt or abs(current_time - total_sim_time/2) < dt or step == n_time_steps - 1:
            pore_profile_history[f'{current_time/3600:.1f} hr'] = C_pore.copy()  # Store sparge gas profile for plotting
            sparge_profile_history[f'{current_time/3600:.1f} hr'] = C_sparge.copy()  # Store sparge gas profile for plotting

        # Store the mobile concentration profile at plug 1
        Cm_history.append(Cm[1, :].copy())

        # Print a progress update to the console
        if step % (n_time_steps / 20) < plot_interval: # Print ~10 updates
             print(f"Time: {current_time:.0f} s ({step/n_time_steps*100:.0f}%))")
             
# Calculate cumulative release per cm³ (time integral of bed release rate)
cumulative_release_history =  np.cumsum(bed_release_rate_history) * plot_interval * dt             

Cm_history = np.array(Cm_history)  # Convert to numpy array for easier indexing


# --- 5. PLOT THE RESULTS ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Tritium Transport Simulation Results (Plug Flow Model)', fontsize=16)
axes = axes.flatten()

# Convert time_points to days for plotting
plot_time_days = np.array(time_points) / 86400

# --- a) Total Bed Inventory Over Time ---
axes[0].plot(plot_time_days, total_inventory_history, color='green', label='Total Inventory')
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Tritium Inventory (Bq)')
axes[0].set_title('Total Tritium Inventory in Packed Bed (Bq)')
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[0].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[0].legend()
axes[0].grid(True)

# --- b) Bed Release Rate Over Time ---
axes[1].plot(plot_time_days, bed_release_rate_history, color='purple')
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Packed bed release rate (Bq/s)')
axes[1].set_title('Tritium Release Rate from Bed Outlet')
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[1].axvspan(0, t_irr/86400, color='red', alpha=0.3)
axes[1].grid(True)

# --- c) Sparge Gas Outlet Concentration Over Time ---
axes[2].plot(plot_time_days, C_sparge_outlet_history, label='Outlet Sparge Gas', color='orange')
axes[2].set_xlabel('Time (days)')
axes[2].set_ylabel('Concentration (T/cm³)')
axes[2].set_title('Sparge Gas Concentration at Bed Outlet')
axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[2].axvspan(0, t_irr/86400, color='red', alpha=0.3)
axes[2].legend()
axes[2].grid(True)

# --- d) Sparge Gas Axial Profile ---
for label, profile in sparge_profile_history.items():
    axes[3].plot(z, profile, label=label)
axes[3].set_xlabel('Axial Position (z) in Packed Bed (cm)')
axes[3].set_ylabel('Sparge Gas Concentration (T/cm³)')
axes[3].set_title('Sparge Gas Concentration Axial Profile')
axes[3].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[3].grid(True)


# --- e) Mobile Concentration Within Grain Over Time (Middle-Right) ---
colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
for i in range(N + 1):
    axes[4].plot(plot_time_days, Cm_history[:, i], color=colors[i], label=f"r={r[i]:.3f} cm")
axes[4].set_xlabel('Time (days)')
axes[4].set_ylabel('Mobile Concentration (T/cm³)')
axes[4].set_title('Mobile Tritium Concentration Profile in Grain Over Time, at Plug 1')
axes[4].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[4].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[4].legend(fontsize='small', ncol=2)
axes[4].grid(True)

# # --- e) Cumulative Release per cm³ Over Time (Bottom-Left) ---
# axes[4].plot(plot_time_days, cumulative_release_history, color='red')
# # Find the time where cumulative release reaches 99% of its final value
# final_cum_release = cumulative_release_history[-1]
# threshold = 0.99 * final_cum_release
# idx_99 = np.argmax(cumulative_release_history >= threshold)
# time_99 = time_points[idx_99]
# release_99 = cumulative_release_history[idx_99]
# axes[4].axvline(time_99/86400, color='black', linestyle='--', label='99% Release Time')
# axes[4].annotate(f"99% at {time_99/86400:.2f} d", xy=(time_99/86400, release_99), xytext=(time_99/86400, 0.7*final_cum_release),
#                  arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, color='black')
# axes[4].set_xlabel('Time (days)')
# axes[4].set_ylabel('Cumulative Release (Bq/cm³)')
# axes[4].set_title('Cumulative Tritium Release per cm³ Over Time')
# axes[4].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
# axes[4].legend(['Cumulative Release', '99% Release Time'])
# axes[4].grid(True)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

end_time = time.time()  # End timer
print(f"\nTotal script runtime: {end_time - start_time:.2f} seconds")





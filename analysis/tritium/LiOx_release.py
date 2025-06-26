import numpy as np
import matplotlib.pyplot as plt

# --- 1. PHYSICAL AND SIMULATION PARAMETERS ---

# --- Grain, Pellet & Packed Bed Properties ---
r_g = 0.01          # Average Grain Radius (cm)
r_p = 0.3           # Pellet Radius (cm)
fr = 0.8            # Surface Area Reduction Factor (accounts for necking)
porosity_pellet = 0.2  # Pellet porosity (void fraction, ε)
packing_density = 0.62 # Packing efficiency for random spheres (φ)
r_bed = 6.5          # Packed Bed Radius (cm)
z_bed = 8.0         # Packed Bed Height (cm)

# --- Diffusion, Trapping & Generation Properties ---
D = 2.0e-10          # Intrinsic Diffusion Coefficient (cm^2/s)
kt = 1.0e-24        # Trapping Coefficient (cm^3/(atom*s))
kd = 1.0e-3         # Detrapping Coefficient (1/s)
Nt = 1.0e20         # Trapping Site Density (sites/cm^3)
source_rate = 1e8   # Tritium Source Rate (T/s)
tbr = 2e-5          # Volumetric Tritium Breeding Ratio (T/n/cm^3)
G = source_rate*tbr # Tritium Generation Rate (T/cm^3/s)

# --- Gas & System Properties ---
h_grain = 1.0e-4    # Grain-Pore Mass Transfer Coeff (cm/s)
h_pellet = 1.0e-4   # Pore-Sparge Mass Transfer Coeff (cm/s)
Q_sparge = 8.33e-1 # Sparge Flow Rate (cm^3/s)

# --- Simulation Parameters ---
t_irr = 3600 # Irradiation Time (s)
total_sim_time = 1000000 # Total simulation time (s)
x = 0.1 # Position in Packed Bed (cm)

# --- Derived Parameters ---
V_bed = np.pi * r_bed**2 * z_bed  # Packed Bed Volume (cm^3)
A_internal = fr * (4 * np.pi * r_p**3)*(1-porosity_pellet)/r_g # Internal Surface Area per Pellet (cm^2)
A_external = 4 * np.pi * r_p**2 # External Surface Area per Pellet (cm^2)
V_pore = (4/3) * np.pi * r_p**3 * porosity_pellet # Pore Volume per Pellet (cm^3)
N_pellets = V_bed*packing_density / (4/3 * np.pi * r_p**3) # Number of Pellets in Packed Bed
V_gas = V_bed * (1 - packing_density) # Gas Volume in Packed Bed (cm^3)
Sv = fr * (3*(1-porosity_pellet)/r_g) # Pellet Volumetric Specific Surface Area (cm^-1)

# --- Simulation Grid ---
N = 10              # Number of radial nodes
dr = r_g / N          # Radial step size (m)

# To ensure stability, we must calculate the max possible time step
# For the interior nodes, the criterion is dt <= 0.5 * dr^2 / D
# For the surface node, it's dt <= 1 / (2*D/dr^2 + 2*hm/dr)
# We take the most restrictive (smallest) of these.
max_dt_interior = 0.5 * dr**2 / D
max_dt_boundary = 1 / (2 * D / dr**2 + 2 * h_grain / dr)
dt = min(max_dt_interior, max_dt_boundary) * 0.5 # Use 50% of max for safety

n_time_steps = int(total_sim_time / dt)



# --- 2. INITIALIZE CONCENTRATION ARRAYS ---
# Create arrays to hold the concentration at each node
# We use N+1 to have nodes from 0 to N inclusive
r = np.linspace(0, r_g, N + 1)  # Radial positions of each node

Cm = np.zeros(N + 1)  # Mobile concentration array, initialized to zero
Ct = np.zeros(N + 1)  # Trapped concentration array, initialized to zero

C_pore = np.zeros(1)  # Pore gas concentration array, initialized to zero
C_sparge = np.zeros(1)  # Sparge gas concentration array, initialized to zero

# --- Data Storage for Plotting ---
# store results at specific intervals to avoid saving massive arrays
plot_interval = 10 # Save data for plots every 10 steps
time_points = []
inventory_history = []
grain_flux_history = []
pellet_flux_history = []
C_pore_history = []
C_sparge_history = []
Cm_history = []
bed_release_rate_history = []


print(f"--- Simulation Setup ---")
print(f"Grain Radius: {r_g:.2e} cm")
print(f"Pellet Radius: {r_p:.2e} cm")
print(f"Pellet Internal Surface Area: {A_internal:.2e} cm2")
print(f"Pellet External Surface Area: {A_external:.2e} cm2")
print(f"Pellet Pore Volume: {V_pore:.2e} cm³")
print(f"Number of Pellets: {N_pellets:.2f}")
print(f"Sparge Gas Volume: {V_gas:.2e} cm³")
print(f"Number of Nodes: {N}")
print(f"Radial Step (dr): {dr:.2e} cm")
print(f"Calculated Stable Time Step (dt): {dt:.3f} s")
print(f"Total Simulation Time: {total_sim_time} s")
print(f"Total Number of Time Steps: {n_time_steps}")
print(f"------------------------\n")

# --- 3. THE MAIN SIMULATION LOOP ---
for step in range(n_time_steps):
    # Store a copy of the old concentrations to use in calculations
    Cm_old = Cm.copy()
    Ct_old = Ct.copy()
    C_pore_old = C_pore.copy()
    C_sparge_old = C_sparge.copy()

    # --- Update generation term G  ---
    if step*dt < t_irr:
        # During irradiation, constant generation term
        G = source_rate * tbr
    else:
        # After irradiation, the generation term is zero
        G = 0

    # --- Calculate new concentrations for each node ---

    # --- a) Interior Nodes (1 to N-1) ---
    # Using the more accurate spherical coordinate formula
    for i in range(1, N):
        # The standard part of the Laplacian
        laplacian_term = (Cm_old[i+1] - 2*Cm_old[i] + Cm_old[i-1]) / dr**2
        # The geometric part for spherical coordinates
        geometric_term = 0 #(Cm_old[i+1] - Cm_old[i-1]) / (2 * r[i] * dr)
        
        # Combine terms
        dCm_dt_diffusion = D * (laplacian_term + geometric_term)
        
        dCm_dt_trapping = kt * Cm_old[i] * (Nt - Ct_old[i])
        dCm_dt_detrapping = kd * Ct_old[i]

        # Update mobile concentration
        Cm[i] = Cm_old[i] + (dCm_dt_diffusion + G - dCm_dt_trapping + dCm_dt_detrapping) * dt
        
        # Update trapped concentration
        Ct[i] = Ct_old[i] + (dCm_dt_trapping - dCm_dt_detrapping) * dt

    # --- b) Center Node (i=0) ---
    # Uses the special formula for r=0 in spherical coordinates
    dCm_dt_diffusion_center = 6 * D * (Cm_old[1] - Cm_old[0]) / dr**2
    dCm_dt_trapping_center = kt * Cm_old[0] * (Nt - Ct_old[0])
    dCm_dt_detrapping_center = kd * Ct_old[0]
    
    Cm[0] = Cm_old[0] + (dCm_dt_diffusion_center + G - dCm_dt_trapping_center + dCm_dt_detrapping_center) * dt
    Ct[0] = Ct_old[0] + (dCm_dt_trapping_center - dCm_dt_detrapping_center) * dt


    # --- c) Surface Node (i=N) ---
    # Uses the mass transfer boundary condition
    diffusion_in = 2 * D * (Cm_old[N-1] - Cm_old[N]) / dr**2
    release_out =  h_grain * Sv * (Cm_old[N] - C_pore_old)
    
    dCm_dt_trapping_surface = kt * Cm_old[N] * (Nt - Ct_old[N])
    dCm_dt_detrapping_surface = kd * Ct_old[N]
    
    Cm[N] = Cm_old[N] + (diffusion_in - release_out + G - dCm_dt_trapping_surface + dCm_dt_detrapping_surface) * dt
    Ct[N] = Ct_old[N] + (dCm_dt_trapping_surface - dCm_dt_detrapping_surface) * dt

    # Surface release rates
    J_grain = h_grain * (Cm_old[N] - C_pore_old)  # Tritium release from grain surface (atoms/cm²/s)
    J_pellet = h_pellet * (C_pore_old - C_sparge_old)  # Tritium release from pellet surface (atoms/cm²/s)

    # --- d) Pore Gas Concentration ---
    C_pore[0] = C_pore_old[0] + (dt / V_pore) * (J_grain * A_internal - J_pellet * A_external) 

    # --- e) Sparge Gas Concentration ---
    C_sparge[0] = C_sparge_old[0] + dt * (J_pellet * A_external * N_pellets * (x / z_bed) - Q_sparge * C_sparge_old) / (V_gas * (x / z_bed))
    
    # --- 4. CALCULATE DERIVED QUANTITIES & STORE DATA ---
    if step % plot_interval == 0:
        current_time = step * dt
        time_points.append(current_time)

        # Calculate total inventory (volume-weighted average)
        # Each node i represents a shell of volume proportional to r[i]^2
        total_conc = Cm + Ct
        weights = r**2
        
        # For a proper average, we divide by the sum of weighted volumes.
        # But for total inventory, we sum the concentration in each shell's volume.
        # Volume of shell i is approx 4*pi*r[i]^2*dr
        inventory = np.sum(total_conc * 4 * np.pi * r**2 * dr)
        inventory_history.append(inventory)

        # Calculate release flux from the grain surface
        grain_flux = h_grain * (Cm[N] - C_pore)
        grain_flux_history.append(grain_flux)
        
        # Calculate release flux from the pellet surface
        pellet_flux = h_pellet * (C_pore - C_sparge)
        pellet_flux_history.append(grain_flux)

        # Calculate volumetric release rate from bed
        bed_release_rate = J_pellet * A_external * (packing_density / ((4/3) * np.pi * r_p**3))
        bed_release_rate_history.append(bed_release_rate)

        # Store the mobile concentration profile at this time step
        Cm_history.append(Cm.copy())

        # Print a progress update to the console
        if step % (n_time_steps / 20) < plot_interval: # Print ~10 updates
             print(f"Time: {current_time:.0f} s ({step/n_time_steps*100:.0f}%) | "
                   f"Surface Conc: {Cm[N]:.2e} atoms/cm^3 | "
                   f"Total Inventory: {inventory:.2e} atoms/grain")
Cm_history = np.array(Cm_history)  # Convert to numpy array for easier indexing


# --- 5. PLOT THE RESULTS ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 3, figsize=(14, 15))
fig.suptitle('Tritium Transport Simulation Results', fontsize=16)

# --- Flatten the axes array for easier indexing ---
axes = axes.flatten()

# --- b) Total Inventory Over Time (Top-Right) ---
axes[0].plot(time_points, inventory_history, color='green')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Total Tritium Inventory (atoms)')
axes[0].set_title('Total Inventory (atoms) Over Time')
axes[0].axvspan(0, t_irr, color='red', alpha=0.3, label='Irradiation Period')
axes[0].legend()
axes[0].grid(True)

# --- c) Release Rate Over Time (Bottom-Left) ---
axes[1].plot(time_points, bed_release_rate_history, color='purple')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Packed bed release rate (atoms/cm3/s)')
axes[1].set_title('Tritium Release Rate from Packed Bed')
axes[1].axvspan(0, t_irr, color='red', alpha=0.3)
axes[1].grid(True)

# --- e) Mobile Concentration at Nodes Over Time (Bottom-Right) ---
# Use a colormap to automatically assign different colors to each node's line
colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
for i in range(N + 1):
    # Transpose Cm_history so each row is a node's history
    axes[2].plot(time_points, Cm_history[:, i], color=colors[i], label=f'Node {i}')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Mobile Concentration (atoms/cm³)')
axes[2].set_title('Mobile Tritium Concentration at Nodes Over Time')
axes[2].axvspan(0, t_irr, color='red', alpha=0.3, label='Irradiation Period')
axes[2].legend(fontsize='small', ncol=2) # Add a legend
axes[2].grid(True)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()            

plt.style.use('seaborn-v0_8-darkgrid')

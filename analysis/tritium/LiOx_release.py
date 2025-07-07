import numpy as np
import matplotlib.pyplot as plt
import time 
from matplotlib.ticker import ScalarFormatter


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
source_rate = 1e8   # Tritium Source Rate (T/s)
tbr = 2e-5          # Volumetric Tritium Breeding Ratio (T/n/cm^3)
G = source_rate*tbr # Tritium Generation Rate (T/cm^3/s)

# --- Gas & System Properties ---
k_grain_ads = 1e-7  # Grain surface adsorption coeff
k_grain_des = 1e-6  # Grain surface desorption coeff
h_pellet = 2.0e-3   # Pore-Sparge Mass Transfer Coeff (cm/s)
Q_sparge = 8.33e-1 # Sparge Flow Rate (cm^3/s)
decay_constant = 1.785e-9 # Tritium Decay Constant (1/s)

# --- Simulation Parameters ---
t_irr = 3600 # Irradiation Time (s)
total_sim_time = 2000000 # Total simulation time (s)
x = 0.1 # Position in Packed Bed (cm) [Use 0.1 for inlet, 8 for outlet, or any value in between for intermediate positions]

simulate = "bed" # Set to "bed" to run analysis of the packed bed, or "grain" to run the grain model

# --- Derived Parameters ---
V_bed = np.pi * r_bed**2 * z_bed  # Packed Bed Volume (cm^3)
A_internal = fr * (4 * np.pi * r_p**3)*(1-porosity_pellet)/r_g # Internal Surface Area per Pellet (cm^2)
A_external = 4 * np.pi * r_p**2 # External Surface Area per Pellet (cm^2)
V_pore = (4/3) * np.pi * r_p**3 * porosity_pellet # Pore Volume per Pellet (cm^3)
N_pellets = (V_bed*packing_density) / (4/3 * np.pi * r_p**3) # Number of Pellets in Packed Bed
V_gas = V_bed * (1 - packing_density) # Gas Volume in Packed Bed (cm^3)
Sv = fr * (3*(1-porosity_pellet)/r_g) # Pellet Volumetric Specific Surface Area (cm^-1) (cm^2/cm^3)
grains_pellet = porosity_pellet*(((4/3) * np.pi * r_p**3)/((4/3) * np.pi * r_g**3))  # Grains per Pellet
grains_cm3 = (packing_density / ((4/3) * np.pi * r_p**3))*grains_pellet  # Grains per cm³

# --- Simulation Grid ---
N = 10              # Number of radial nodes
dr = r_g / N        # Radial step size (cm)
M = 10              # Number of axial nodes (used when analysing packed bed)
dz = z_bed / M      # Axial step size (cm) (used when analysing packed bed)

dt = 3              # Time step size (s) (Reduce if simulation is unstable)
n_time_steps = int(total_sim_time / dt)

# --- 2. INITIALIZE CONCENTRATION ARRAYS ---
# Create arrays to hold the concentration at each node
# Use N+1 to have nodes from 0 to N inclusive
r = np.linspace(0, r_g, N + 1)  # Radial positions of each node

Cm = np.zeros(N + 1)  # Mobile concentration array, initialized to zero
Ct = np.zeros(N + 1)  # Trapped concentration array, initialized to zero

C_pore = np.zeros(1)  # Pore gas concentration array, initialized to zero
C_sparge = np.zeros(1)  # Sparge gas concentration array, initialized to zero

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


print(f"--- Simulation Setup ---")
print(f"Grain Radius: {r_g:.2e} cm")
print(f"Pellet Radius: {r_p:.2e} cm")
print(f"Pellet Internal Surface Area: {A_internal:.2e} cm2")
print(f"Pellet External Surface Area: {A_external:.2e} cm2")
print(f"Pellet Pore Volume: {V_pore:.2e} cm³")
print(f"Packed Bed Volume: {V_bed:.2e} cm³")
print(f"Number of Pellets: {N_pellets:.2f}")
print(f"Sparge Gas Volume: {V_gas:.2e} cm³")

print(f"Number of Nodes: {N}")
print(f"Radial Step (dr): {dr:.2e} cm")

print(f"Calculated Stable Time Step (dt): {dt:.3f} s")
print(f"Total Simulation Time: {total_sim_time} s")
print(f"Total Number of Time Steps: {n_time_steps}")
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

    # --- Calculate new concentrations for each node ---
    # Concentrations are in T/cm^3

    # --- a) Interior Nodes (1 to N-1) ---
    # Using the more accurate spherical coordinate formula for concentration curvature
    for i in range(1, N):
        # The standard part of the Laplacian
        laplacian_term = (Cm_old[i+1] - 2*Cm_old[i] + Cm_old[i-1]) / dr**2
        # The geometric part for spherical coordinates
        geometric_term = (1/r[i]) * (Cm_old[i+1] - Cm_old[i-1]) / dr
        
        # Combine terms
        dCm_dt_diffusion = D * (laplacian_term + geometric_term)
        
        # Trapping terms
        dCm_dt_trapping = kt * Cm_old[i] * (Nt - Ct_old[i])
        dCm_dt_detrapping = kd * Ct_old[i]

        # Update mobile concentration
        Cm[i] = Cm_old[i] + (dCm_dt_diffusion + G - dCm_dt_trapping + dCm_dt_detrapping) * dt
        
        # Update trapped concentration
        Ct[i] = Ct_old[i] + (dCm_dt_trapping - dCm_dt_detrapping) * dt

    # --- b) Center Node (i=0) ---
    # Uses special formula for r=0 in spherical coordinates
    dCm_dt_diffusion_center = 6 * D * (Cm_old[1] - Cm_old[0]) / dr**2
    dCm_dt_trapping_center = kt * Cm_old[0] * (Nt - Ct_old[0])
    dCm_dt_detrapping_center = kd * Ct_old[0]
    
    Cm[0] = Cm_old[0] + (dCm_dt_diffusion_center + G - dCm_dt_trapping_center + dCm_dt_detrapping_center) * dt
    Ct[0] = Ct_old[0] + (dCm_dt_trapping_center - dCm_dt_detrapping_center) * dt


    # --- c) Surface Node (i=N) ---
    # Uses the mass transfer boundary condition
    diffusion_in = 2 * D * (Cm_old[N-1] - Cm_old[N]) / dr**2

    dCm_dt_trapping_surface = kt * Cm_old[N] * (Nt - Ct_old[N])
    dCm_dt_detrapping_surface = kd * Ct_old[N]

    # Surface release rates
    J_grain = (Cm_old[N] * k_grain_des - C_pore_old * k_grain_ads)  # Tritium release from grain surface (T/cm²/s)
    J_pellet = h_pellet * (C_pore_old - C_sparge_old)  # Tritium release from pellet boundary (T/cm²/s)

    release_out =  Sv * J_grain # Release rate per cm^3 of porous pellet volume.
    

    
    Cm[N] = Cm_old[N] + (diffusion_in - release_out + G - dCm_dt_trapping_surface + dCm_dt_detrapping_surface) * dt
    Ct[N] = Ct_old[N] + (dCm_dt_trapping_surface - dCm_dt_detrapping_surface) * dt



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
        inventory = grains_cm3 * np.sum(total_conc * 4 * np.pi * r**2 * dr)
        inventory_history.append(inventory)

        inventory_m = grains_cm3 * np.sum(Cm * 4 * np.pi * r**2 * dr)
        inventory_m_history.append(inventory_m)

        inventory_t = grains_cm3 * np.sum(Ct * 4 * np.pi * r**2 * dr)
        inventory_t_history.append(inventory_t)

        grain_flux_history.append(J_grain)
        
        # Calculate release flux from the pellet surface
        pellet_flux_history.append(J_pellet)

        # Calculate volumetric release rate from bed
        bed_release_rate = J_pellet * A_external * (packing_density / ((4/3) * np.pi * r_p**3)) * decay_constant
        bed_release_rate_history.append(bed_release_rate)

        # Store the mobile concentration profile at this time step
        Cm_history.append(Cm.copy())

        # Store pore and sparge gas concentrations for plotting
        C_pore_history.append(C_pore[0])
        C_sparge_history.append(C_sparge[0])

        # Print a progress update to the console
        if step % (n_time_steps / 20) < plot_interval: # Print ~10 updates
             print(f"Time: {current_time:.0f} s ({step/n_time_steps*100:.0f}%) | "
                   f"Surface Conc: {Cm[N]:.2e} T/cm^3 | "
                   f"Total Inventory: {inventory:.2e} T/cm3")
             
# Calculate cumulative release per cm³ (time integral of bed release rate)
cumulative_release_history =  np.cumsum(bed_release_rate_history) * plot_interval * dt             

Cm_history = np.array(Cm_history)  # Convert to numpy array for easier indexing


# --- 5. PLOT THE RESULTS ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(14, 12))
fig.suptitle('Tritium Transport Simulation Results', fontsize=16)

# --- Flatten the axes array for easier indexing ---
axes = axes.flatten()

# Convert time_points to days for plotting
plot_time_days = np.array(time_points) / 86400

# --- b) Total Inventory Over Time (Top-Left) ---
axes[0].plot(plot_time_days, inventory_history, color='green', label='Total Inventory')
axes[0].plot(plot_time_days, inventory_m_history, color='blue', linestyle='--', label='Mobile Inventory')
axes[0].plot(plot_time_days, inventory_t_history, color='orange', linestyle='--', label='Trapped Inventory')
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Tritium Inventory Concentration (T/cm³)')
axes[0].set_title('Total Inventory Within Grain (T/cm³)')
axes[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[0].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[0].legend()
axes[0].grid(True)

# --- c) Release Rate Over Time (Top-Right) ---
axes[1].plot(plot_time_days, bed_release_rate_history, color='purple')
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Packed bed release rate (Bq/cm3/s)')
axes[1].set_title('Tritium Release Rate from Packed Bed')
axes[1].axvspan(0, t_irr/86400, color='red', alpha=0.3)
axes[1].grid(True)

# --- d) Pore & Sparge Gas Concentrations Over Time (Middle-Left) ---
axes[2].plot(plot_time_days, C_pore_history, label='Pore Gas', color='blue')
axes[2].plot(plot_time_days, C_sparge_history, label='Sparge Gas', color='orange')
axes[2].set_xlabel('Time (days)')
axes[2].set_ylabel('Concentration (T/cm³)')
axes[2].set_title('Pore & Sparge Gas Concentrations Over Time')
axes[2].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[2].axvspan(0, t_irr/86400, color='red', alpha=0.3)
axes[2].legend()
axes[2].grid(True)

# --- e) Mobile Concentration Within Grain Over Time (Middle-Right) ---
colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
for i in range(N + 1):
    axes[3].plot(plot_time_days, Cm_history[:, i], color=colors[i], label=f"r={r[i]:.3f} cm")
axes[3].set_xlabel('Time (days)')
axes[3].set_ylabel('Mobile Concentration (T/cm³)')
axes[3].set_title('Mobile Tritium Concentration Profile in Grain Over Time')
axes[3].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[3].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[3].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[3].legend(fontsize='small', ncol=2)
axes[3].grid(True)

# --- e) Cumulative Release per cm³ Over Time (Bottom-Left) ---
axes[4].plot(plot_time_days, cumulative_release_history, color='red')
# Find the time where cumulative release reaches 99% of its final value
final_cum_release = cumulative_release_history[-1]
threshold = 0.99 * final_cum_release
idx_99 = np.argmax(cumulative_release_history >= threshold)
time_99 = time_points[idx_99]
release_99 = cumulative_release_history[idx_99]
axes[4].axvline(time_99/86400, color='black', linestyle='--', label='99% Release Time')
axes[4].annotate(f"99% at {time_99/86400:.2f} d", xy=(time_99/86400, release_99), xytext=(time_99/86400, 0.7*final_cum_release),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, color='black')
axes[4].set_xlabel('Time (days)')
axes[4].set_ylabel('Cumulative Release (Bq/cm³)')
axes[4].set_title('Cumulative Tritium Release per cm³ Over Time')
axes[4].axvspan(0, t_irr/86400, color='red', alpha=0.3, label='Irradiation Period')
axes[4].legend(['Cumulative Release', '99% Release Time'])
axes[4].grid(True)

end_time = time.time()  # End timer
print(f"\nTotal script runtime: {end_time - start_time:.2f} seconds")

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



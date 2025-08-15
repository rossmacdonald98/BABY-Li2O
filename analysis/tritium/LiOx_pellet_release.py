# -----------------------------------------------------------------------------
# Tritium Transport Simulation in a Packed Bed (Plug Flow Model)
#
# This script simulates the generation, transport, trapping, and release of
# tritium in a packed bed of a sintered ceramic pellet breeder material,
#  such as Li2O, under neutron irradiation. The model includes:
#   - Tritium generation within grains due to neutron irradiation.
#   - Diffusion and trapping/detrapping of tritium in grains.
#   - Mass transfer between solid, pore, and sparge gas phases.
#   - Axial transport of tritium in the sparge gas along the packed bed.
#   - Calculation of inventories, release rates, and concentration profiles.
#
# The simulation uses a finite difference approach with discretization in both
# radial (grain) and axial (bed) directions. Results are visualized in several
# plots, including total inventory, release rates, outlet concentrations, axial
# profiles, and mass balance verification.
#
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time 
import json

# --- 1. PHYSICAL AND SIMULATION PARAMETERS ---

# --- Grain, Pellet & Packed Bed Properties ---
r_g = 0.075          # Average Grain Radius (cm)
r_p = 0.3           # Pellet Radius (cm)
porosity_pellet = 0.2  # Pellet porosity (void fraction, ε)
fr = 7.716 * porosity_pellet**2  # Surface Area Reduction Factor (accounts for necking due to sintering between grains)
packing_density = 0.62 # Pellet bed packing efficiency (0.62 = Packing efficiency for random spheres) (φ)
r_bed = 6.5          # Packed Bed Radius (cm)
z_bed = 8         # Packed Bed Length (cm)

# --- Diffusion, Trapping & Generation Properties ---
D = 1.0e-9         # Diffusion Coefficient (cm^2/s)
kt = 1.0e-24        # Trapping Coefficient (cm^3/(atom*s))
kd = 1.0e-3         # Detrapping Coefficient (1/s)
Nt = 1.0e20         # Trapping Site Density (sites/cm^3)
source_rate = 8e8   # Neutron Source Rate (n/s)
tbr = 1e-5          # Volumetric Tritium Breeding Ratio (T/n/cm^3)
G_rate = source_rate*tbr # Tritium Generation Rate (T/cm^3/s)

# --- Gas & System Properties ---
k_grain_ads = 1e-9  # Grain surface adsorption coeff
k_grain_des = 1e-5  # Grain surface desorption coeff
h_pellet = 1e-3     # Pore-Sparge Mass Transfer Coeff (cm/s)
Q_sparge = 8.33e-1  # Sparge Flow Rate (cm^3/s)
decay_constant = 1.785e-9 # Tritium Decay Constant (1/s)

# --- Simulation Parameters ---
t_irr = 3600 * 2 # Irradiation Time (s)
total_sim_time = 3600 * 1000 # Total simulation time (s)
min_dt = 1e-5 # Min timestep for adaptive time-stepping (s)
dt = min_dt # Initial timestep (s)
allowed_change = 0.1 # Maximum relative change allowed in any variable per step

# --- Simulation Grid ---
# Radial grid for the grain model
N = 50              # Number of radial nodes
dr = r_g / N        # Radial step size (cm)
r = np.linspace(0, r_g, N + 1)  # Radial positions of each node

# Axial grid for the packed bed model
Nz = 15              # Number of axial nodes 
dz = z_bed / Nz      # Axial step size (cm)
z = np.linspace(dz/2, z_bed - dz/2, Nz)  # Axial positions of each node (center of each plug)

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
plot_interval = 50
time_points = []
data_intervals = []
dt_history = []
rel_change_history = []
total_inventory_history = []
bed_release_rate_history = []
C_pore_outlet_history = []
C_sparge_outlet_history = []
Cm_history = []
C_sparge_profile_history = {}
total_generated_history = []
cumulative_release_history = []

# --- Key times for sparge profile sampling ---
key_times = [
    0,
    t_irr / 4,
    t_irr / 2,
    3 * t_irr / 4,
    t_irr
]
# Five equally spaced times after irradiation
post_irr_times = np.linspace(t_irr, total_sim_time, 6)[1:]  # skip t_irr, already included
key_times += list(post_irr_times)
key_times = np.array(key_times)

print(f"--- Simulation Setup ---")
print(f'Estimated Total Tritium Produced: {T_est:.2e} Bq')
print(f"------------------------\n")

# --- 3. THE MAIN SIMULATION LOOP ---
start_time = time.time()
print("--- Starting Simulation ---")

step = 0
current_time = 0.0

while current_time < total_sim_time:

    # Update the current time in seconds
    current_time =  dt + current_time
    step += 1

    Cm_old = Cm.copy()
    Ct_old = Ct.copy()
    C_pore_old = C_pore.copy()
    C_sparge_old = C_sparge.copy()

    G = G_rate if current_time < t_irr else 0

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

    # Update sparge gas concentrations at bed inlet
    source_term_0 = (J_pellet[0] * A_pellets_plug) / V_sparge_plug
    convection_term_0 = (-Q_sparge * C_sparge_old[0]) / V_sparge_plug
    C_sparge[0] = C_sparge_old[0] + dt * (source_term_0 + convection_term_0)

    # Update sparge gas concentrations along the bed
    for j in range(1, Nz):
        source_term = (J_pellet[j] * A_pellets_plug) / V_sparge_plug
        convection_term_in = (Q_sparge * C_sparge_old[j-1]) / V_sparge_plug
        convection_term_out = (Q_sparge * C_sparge_old[j]) / V_sparge_plug
        C_sparge[j] = C_sparge_old[j] + dt * (source_term + convection_term_in - convection_term_out)
    
    if np.isnan(Cm).any():
        print(f"\nERROR: NaN detected at time step {step} ({step*dt:.2f} s). Halting simulation.")
        break

    # --- 4. CALCULATE DERIVED QUANTITIES & STORE DATA ---
    key_time_tol = (dt*plot_interval) / 2  # tolerance for time comparison

    if step % plot_interval == 0:
        time_points.append(current_time)
        if len(time_points) > 1:
            data_intervals.append(time_points[-1] - time_points[-2]) 
        else:
            data_intervals.append(dt)
        
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
        C_pore_outlet_history.append(C_pore[Nz-1])
        C_sparge_outlet_history.append(C_sparge[Nz-1])
        Cm_history.append(Cm[0, :].copy())
        # Only sample sparge profile at key times
        for key_time in key_times:
            if  abs(current_time - key_time) < key_time_tol:
                C_sparge_profile_history[f'{current_time/3600:.1f} hr'] = C_sparge.copy()

        # Store timestep and relative change history
        rel_change_history.append(max_rel_change)
        dt_history.append(dt)

    # Print single-line progress
    print(
        f"Step: {step:6d} | {current_time/total_sim_time*100:6.2f}% | Time: {current_time/3600:5.1f} hr / {total_sim_time/3600:5.1f} hr | dt: {dt:.2e} s",
        end='\r', flush=True
    )

    # --- Adaptive timestep logic ---
    # Compute max relative change for key variables
    max_rel_change = 0
    for arr, arr_old in [
        (Cm, Cm_old), (Ct, Ct_old),
        (C_pore, C_pore_old), (C_sparge, C_sparge_old)
    ]:
        rel_change = np.abs(arr - arr_old) / (np.abs(arr_old) + 1e-12)
        max_rel_change = max(max_rel_change, np.max(rel_change))

    new_dt = dt  # Default to current dt

    if max_rel_change > allowed_change and step > 1:
    # Reduce timestep quickly if change exceeds threshold:
        new_dt = 0.75 * dt
    elif max_rel_change < allowed_change:
    # Increase timestep slowly if change is below threshold
        new_dt = 1.00005 * dt
    
    if abs(t_irr - current_time) < 0.1:
        new_dt = min_dt # When near the end of irradiation, use minimum dt to ensure stability

    dt = max(new_dt, min_dt)



run_time = time.time() - start_time
print(f"\n--- Simulation Completed in {run_time:.2f} seconds ---")

# --- EXPORT RESULTS TO JSON ---
print("--- Exporting results to JSON file ---")

# First, calculate cumulative release as it's needed for the export
cumulative_release = np.cumsum(np.array(bed_release_rate_history) * data_intervals) * decay_constant

# Create a dictionary to hold all parameters
parameters = {
    "grain_radius_cm": r_g,
    "pellet_radius_cm": r_p,
    "pellet_porosity": porosity_pellet,
    "packing_density": packing_density,
    "bed_radius_cm": r_bed,
    "bed_length_cm": z_bed,
    "diffusion_coeff_cm2_s": D,
    "trapping_coeff_cm3_atom_s": kt,
    "detrapping_coeff_s": kd,
    "trap_density_sites_cm3": Nt,
    "tritium_generation_rate_T_cm3_s": G_rate,
    "sparge_flow_rate_cm3_s": Q_sparge,
    "decay_constant_s": decay_constant,
    "irradiation_time_s": t_irr,
    "total_sim_time_s": total_sim_time,
    "radial_nodes_grain": N,
    "axial_nodes_bed": Nz
}

# Create a dictionary for the results, converting numpy arrays to lists
results = {
    "time_s": time_points,
    "time_days": (np.array(time_points) / 86400).tolist(),
    "total_inventory_Bq": total_inventory_history,
    "bed_release_rate_atoms_s": bed_release_rate_history,
    "outlet_pore_concentration_T_cm3": C_pore_outlet_history,
    "outlet_sparge_concentration_T_cm3": C_sparge_outlet_history,
    "total_generated_tritium_Bq": total_generated_history,
    "cumulative_release_Bq": cumulative_release.tolist(),
    "grain_radial_positions_cm": r.tolist(),
    "grain_mobile_concentration_profile_inlet_T_cm3": np.array(Cm_history).tolist(),
    "sparge_gas_axial_profiles_T_cm3": {k: v.tolist() for k, v in C_sparge_profile_history.items()},
    "adaptive_timestep_s": dt_history,
    "max_relative_change": rel_change_history,
    "key_times": key_times.tolist()
}

# Combine into a single dictionary
export_data = {
    "simulation_parameters": parameters,
    "simulation_results": results
}

# Write to a JSON file
output_filename = 'pellet_release_results.json'
with open(output_filename, 'w') as f:
    json.dump(export_data, f, indent=4)

print(f"Results successfully exported to {output_filename}\n")






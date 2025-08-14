# -----------------------------------------------------------------------------
# Tritium Transport Simulation in a Powder Bed (Plug Flow Model)
#
# This script simulates the generation, transport, trapping, and release of
# tritium in a packed bed of a ceramic powder material,
#  such as Li2O, under neutron irradiation. The model includes:
#   - Tritium generation within powder grains due to neutron irradiation.
#   - Diffusion and trapping/detrapping of tritium in grains.
#   - Mass transfer between solid and sparge gas phases.
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
import time 
import json

# --- 1. PHYSICAL AND SIMULATION PARAMETERS ---

# --- Grain & Powder Bed Properties ---
r_g = 0.075          # Average Grain Radius (cm)
packing_density = 0.62 # Powder bed packing efficiency (0.62 = Packing efficiency for random spheres) (φ)
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
Nz = 10              # Number of axial nodes 
dz = z_bed / Nz      # Axial step size (cm)
z = np.linspace(dz/2, z_bed - dz/2, Nz)  # Axial positions of each node (center of each plug)

# --- Derived Parameters ---
V_grain = (4/3) * np.pi * r_g**3  # Grain Volume (cm^3)
A_external = 4 * np.pi * r_g**2 # External Surface Area per Grain (cm^2)

# Per-plug parameters
V_plug = np.pi * r_bed**2 * dz  # Volume of one axial 'plug' of the bed (cm^3)
V_sparge_plug = V_plug * (1 - packing_density)  # Sparge gas volume in one plug (cm^3)
N_grains_plug = (V_plug * packing_density) / (V_grain)  # Number of grains in one plug
A_grains_plug = N_grains_plug * A_external  # Total External Surface Area of Grains in one plug (cm^2)

# Gas velocity
A_bed_cross_section = np.pi * r_bed**2  # Cross-sectional area of the packed bed (cm^2)
V_bed = A_bed_cross_section * z_bed  # Total volume of the packed bed (cm^3)
v_gas = Q_sparge / (A_bed_cross_section * (1 - packing_density))  # Correct interstitial gas velocity

# Tritium production
V_breeding_total = V_bed * packing_density
T_est = G_rate * V_breeding_total * t_irr * decay_constant  # Estimated total tritium produced (Bq)


# --- 2. INITIALIZE CONCENTRATION ARRAYS ---
Cm = np.zeros((Nz, N + 1))
Ct = np.zeros((Nz, N + 1))
C_sparge = np.zeros(Nz)
J_grain = np.zeros(Nz)

# --- Data Storage for Plotting ---
plot_interval = 50
time_points = []
data_intervals = []
dt_history = []
rel_change_history = []
total_inventory_history = []
grain_release_rate_history = []
bed_release_rate_history = []
C_pore_outlet_history = []
C_sparge_outlet_history = []
Cm_history = []
C_sparge_profile_history = {}
total_generated_history = []
cumulative_release_history = []
cumulative_grain_release_history = []

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
        J_grain[j] = (Cm_old[j, N]**2 * k_grain_des - C_sparge_old[j] * k_grain_ads)
        dCm_dt_diffusion_in = 2 * D * (Cm_old[j, N-1] - Cm_old[j, N]) / dr**2
        dCm_dt_surface_release_out = 2 * J_grain[j] * (1/dr + 1/r[N])
        dCm_dt_trapping_surface = kt * Cm_old[j, N] * (Nt - Ct_old[j, N])
        dCm_dt_detrapping_surface = kd * Ct_old[j, N]
        Cm[j, N] = Cm_old[j, N] + (dCm_dt_diffusion_in - dCm_dt_surface_release_out + G - dCm_dt_trapping_surface + dCm_dt_detrapping_surface) * dt
        Ct[j, N] = Ct_old[j, N] + (dCm_dt_trapping_surface - dCm_dt_detrapping_surface) * dt

    # Update grain release rate (at inlet, j = 0))
    grain_release_rate = J_grain[0] * A_external # atoms/s
    
    # Update sparge gas concentrations at bed inlet
    source_term_0 = (J_grain[0] * A_grains_plug) / V_sparge_plug
    convection_term_0 = (-Q_sparge * C_sparge_old[0]) / V_sparge_plug
    C_sparge[0] = C_sparge_old[0] + dt * (source_term_0 + convection_term_0)

    # Update sparge gas concentrations along the bed
    for j in range(1, Nz):
        source_term = (J_grain[j] * A_grains_plug) / V_sparge_plug
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
            solid_inventory += inventory_grain * N_grains_plug * decay_constant # Solid phase inventory (Bq)
            sparge_inventory += C_sparge[j] * V_sparge_plug * decay_constant # Sparge gas inventory (Bq)    
        total_system_inventory = solid_inventory + sparge_inventory # Total system inventory (Bq)
        total_inventory_history.append(total_system_inventory)

        # Total Generated History
        total_generated = G_rate * min(current_time, t_irr) * V_breeding_total * decay_constant  # Total generated tritium (Bq)
        total_generated_history.append(total_generated)

        # Release Rate
        release_rate = C_sparge[Nz-1] * Q_sparge # Release rate at the outlet (atoms/s)
        bed_release_rate_history.append(release_rate)
        grain_release_rate_history.append(grain_release_rate)
        
        # Other histories
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
        (C_sparge, C_sparge_old)
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

## Save results to json file
def pyify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: pyify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pyify(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        # fallback for numpy scalar or unknown types
        try:
            return float(obj)
        except Exception:
            return str(obj)

# Prepare results dict (select the histories / variables you need)
results = {
    "time_points": np.array(time_points),
    "data_intervals": np.array(data_intervals),
    "dt_history": np.array(dt_history),
    "rel_change_history": np.array(rel_change_history),
    "total_inventory_history": np.array(total_inventory_history),
    "bed_release_rate_history": np.array(bed_release_rate_history),
    "grain_release_rate_history": np.array(grain_release_rate_history),
    "C_sparge_outlet_history": np.array(C_sparge_outlet_history),
    "Cm_history": np.array(Cm_history),
    "C_sparge_profile_history": C_sparge_profile_history,  # dict of numpy arrays
    "total_generated_history": np.array(total_generated_history),
    # metadata needed for plotting
    "key_times": np.array(key_times),
    "t_irr": t_irr,
    "total_sim_time": total_sim_time,
    "N": N,
    "Nz": Nz,
    "r": np.array(r),
    "z": np.array(z),
    "Q_sparge": Q_sparge,
    "decay_constant": decay_constant,
    "G_rate": G_rate,
    "V_grain": V_grain,
}

# Convert to JSON-serializable python types and write file
out_file = "LiOx_powder_results.json"
with open(out_file, "w") as f:
    json.dump(pyify(results), f, indent=2)

print(f"Results saved to {out_file}")




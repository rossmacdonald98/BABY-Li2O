# -----------------------------------------------------------------------------
# Diffusion-Desorption Tritium Release Model for Lithium Ceramic Spherical Grains.
# From the paper: # Johnson, J. R., et al. "Tritium transport and release from lithium ceramic breeder materials." Fusion Technology 26.1 (1994): 1-10.
#
# Adapted to apply a temporally varying generation rate to simulate neutron irradiation by superimposing a "ghost" generation rate after a specified time.
#
# The script is set up as follows:
# 1. Define functions.
# 2. Initialize physical parameters and simulation settings.
# 3. Pre-calculate the roots of the characteristic equation for the diffusion-desorption problem.
# 4. Calculate the tritium release rate, concentration profiles, and total inventory over time.
# 5. Plot the results, including concentration profiles, release rate, cumulative release, and inventory.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

# --- Core Model Functions based on Johnson et al. 1994 "TRITIUM TRANSPORT AND RELEASE FROM LITHIUM CERAMIC BREEDER MATERIALS" ---

def find_alpha_roots(n_roots, a, h):
    """
    Finds the first n_roots positive roots of the transcendental equation:
    a * alpha * cot(a * alpha) = 1 - a * h
    
    This equation determines the eigenvalues for the diffusion-desorption problem.
    We find the roots numerically by searching in intervals.
    
    Args:
        n_roots (int): The number of roots to find.
        a (float): Grain radius (m).
        h (float): Ratio of desorption to diffusion (K_d / D).
        
    Returns:
        numpy.ndarray: An array containing the first n_roots.
    """
    roots = []
    # The equation can be rewritten as: a*alpha*cos(a*alpha) - (1-a*h)*sin(a*alpha) = 0
    # Or tan(a*alpha) = a*alpha / (1 - a*h)
    # Roots are located near multiples of pi/a. We search between them.
    lower_bound = 1e-9 # Start just above zero
    for n in range(n_roots):
        # Define search interval for the nth root
        upper_bound = (n + 1) * np.pi / a
        
        # Define the function whose root we want to find
        f = lambda alpha: a * alpha * np.cos(a * alpha) - (1 - a * h) * np.sin(a * alpha)
        
        try:
            # Use Brent's method to find a root in the given interval
            root = brentq(f, lower_bound, upper_bound)
            roots.append(root)
        except ValueError:
            # This can happen if no root is in the interval, which is unlikely
            # for this problem but good practice to handle.
            print(f"Warning: Could not find a root in the interval ({lower_bound:.2e}, {upper_bound:.2e})")
            
        # The next search starts from the end of the current interval
        lower_bound = upper_bound
        
    return np.array(roots)

def calculate_release_rate(t, G, a, D, h, alpha_n):
    """
    Calculates the total tritium release rate from a single spherical grain at a given time t.
    This is based on Equation (4) from the paper, which is numerically stable.
    
    Args:
        t (float or numpy.ndarray): Time (s).
        G (float): Tritium generation rate (atoms/m^3/s).
        a (float): Grain radius (m).
        D (float): Tritium diffusivity in the grain (m^2/s).
        h (float): Ratio K_d/D (m^-1).
        alpha_n (numpy.ndarray): The pre-calculated roots of the characteristic equation.
        
    Returns:
        float or numpy.ndarray: The total tritium release rate (atoms/s).
    """
    # Steady-state part of the release flux (rate per unit area)
    steady_state_flux = (G * a) / 3.0
    
    # The transient part is a summation over the roots
    ah = a * h
    
    # Denominator of the summation term from Eq. (4)
    denominator = alpha_n**2 * (a**2 * alpha_n**2 + ah * (ah - 1))
    
    # Reshape t to a column vector (n_times, 1) and alpha_n to a row vector (1, n_roots)
    # to allow broadcasting for the exponential term calculation.
    # The result will be a (n_times, n_roots) array.
    exp_term = np.exp(-D * alpha_n**2 * t[:, np.newaxis])
    
    # Calculate the sum over all roots for each time point
    # The division is broadcast, and we sum along the roots axis (axis=1).
    sum_term = np.sum(exp_term / denominator, axis=1)
    
    prefactor = -2 * h**2 * a * G

    transient_flux = prefactor * sum_term
    
    # Total flux (rate per unit area)
    total_flux = steady_state_flux + transient_flux
    
    # Total release rate from the grain (flux * surface area)
    surface_area = 4 * np.pi * a**2
    total_rate = total_flux * surface_area
    
    return total_rate.squeeze() # Remove extra dimension if t was a single value

def calculate_concentration(r, t, G, a, D, h, alpha_n):
    """
    Calculates the tritium concentration inside the grain at a given SCALAR radius r and time t.
    This is based on Equation (3) from the paper, but reformulated to be numerically stable.
    
    Args:
        r (float): A single radial position inside the grain (m), where 0 <= r <= a.
        t (float): A single time point (s).
        G (float): Tritium generation rate (atoms/m^3/s).
        a (float): Grain radius (m).
        D (float): Tritium diffusivity in the grain (m^2/s).
        h (float): Ratio K_d/D (m^-1).
        alpha_n (numpy.ndarray): The pre-calculated roots of the characteristic equation.
        
    Returns:
        float: Tritium concentration (atoms/m^3).
    """
    # This function is designed for a scalar r, as used by scipy.integrate.quad
    if hasattr(r, "__len__"):
        raise TypeError(f"calculate_concentration is designed for a scalar r, but received array of shape {np.shape(r)}")

    # Steady-state part of the concentration
    steady_state_C = (G / (6 * h * D)) * (h * (a**2 - r**2) + 2 * a)
    
    # Transient part 
    ah = a * h
    exp_term = np.exp(-D * alpha_n**2 * t)
    
    # This is the denominator for the summation term in the release rate (Eq. 4)
    denominator = alpha_n**2 * (a**2 * alpha_n**2 + ah * (ah - 1)) * np.sin(a * (alpha_n))
    
    if r < 1e-12: # Use a small tolerance for floating point comparison to zero
        # Handle the r=0 case using L'Hopital's rule: lim_{r->0} sin(r*alpha)/r = alpha
        sum_numerator = alpha_n * exp_term
        sum_val = np.sum(sum_numerator / denominator)
        prefactor = -2 * h * a**2 * G / D
    else:
        sum_numerator = np.sin(r * alpha_n) * exp_term
        sum_val = np.sum(sum_numerator / denominator)
        prefactor = -2 * h * a**2 * G / (r * D)

    transient_C = prefactor * sum_val
    
    return steady_state_C + transient_C # Note: transient term is already negative

def calculate_inventory(t, G, a, D, h, alpha_n):
    """
    Calculates the total tritium inventory in a single spherical grain at time t.
    This is done by numerically integrating the concentration profile over the grain volume.
    
    Args:
        t (float or numpy.ndarray): Time point(s) (s).
        G (float): Tritium generation rate (atoms/m^3/s).
        a (float): Grain radius (m).
        D (float): Tritium diffusivity in the grain (m^2/s).
        h (float): Ratio K_d/D (m^-1).
        alpha_n (numpy.ndarray): The pre-calculated roots of the characteristic equation.
        
    Returns:
        float or numpy.ndarray: Total tritium inventory (atoms).
    """
    # The integrand is C(r, t) * 4 * pi * r^2
    integrand = lambda r, t_val: calculate_concentration(r, t_val, G, a, D, h, alpha_n) * 4 * np.pi * r**2
    
    # Integrate from r=0 to r=a for each time point
    if hasattr(t, "__len__"):
        inventory = [quad(integrand, 0, a, args=(t_val,))[0] for t_val in t]
        return np.array(inventory)
    else:
        inventory = quad(integrand, 0, a, args=(t,))[0]
        return inventory


# --- DEFINE PHYSICAL AND SIMULATION PARAMETERS ---
    
# Physical Parameters 
a = 0.0005 / 2    # Grain radius (m)
D = 1e-13      # Diffusivity (m^2/s)
K_d = 1e-3     # Desorption rate constant (m/s)
    
# Calculated parameter h
h = K_d / D    # Ratio K_d/D (m^-1)

# Tritium generation rate (G) parameters
source_rate = 8e8 # Neutron source rate (n/s)
volumetric_tbr = 2.2e1 # Volumetric tritium breeding ratio (T/m^3/n)
G1 = source_rate * volumetric_tbr # Initial tritium generation rate (atoms/m^3/s)
G2 = 0         # Secondary tritium generation rate (atoms/m^3/s)
    
# Simulation Parameters
t_start = 0
t_change = 3600 * 2 # Time of generation rate change in seconds
t_end = 3600 * 24  # Simulatione end time in seconds
n_steps = 500     # Number of time steps for the plot
n_roots = 100     # Number of roots to calculate for the series solution


# --- Main Script: Simulation and Plotting ---
if __name__ == '__main__':
    
    print("--- Model Parameters ---")
    print(f"Grain Radius (a): {a:.2e} m")
    print(f"Diffusivity (D): {D:.2e} m^2/s")
    print(f"Desorption Constant (K_d): {K_d:.2e} m/s")
    print(f"Generation Rate (G1): {G1:.2e} atoms/m^3/s")
    print(f"h (K_d/D): {h:.2e} m^-1")
    print("------------------------\n")
    
    # --- 2. PRE-CALCULATE ROOTS ---
    # This is the most computationally intensive part, so we do it once.
    print(f"Finding {n_roots} roots of the characteristic equation...")
    alpha_n = find_alpha_roots(n_roots, a, h)
    print("Roots found.\n")
    
    # --- 3. CALCULATE RESULTS OVER TIME ---
    print("Calculating release rate and inventory over time...")

    # Use logspace for time to better visualize the initial transient
    time_array = np.logspace(np.log10(0.1), np.log10(t_end), n_steps)

    # Create radial position array for calculating concentration profiles
    radial_pos = np.linspace(0, a, 20) # Radial positions from grain center to edge

    # Part 1: System evolves with G1 for full time
    # --- Release Rate Calculation ---
    release_rate_total = calculate_release_rate(time_array, G1, a, D, h, alpha_n)

    # --- Concentration Profiles Calculation ---
    concentration_profiles = np.array([[calculate_concentration(r, t, G1, a, D, h, alpha_n) for r in radial_pos] for t in time_array])

    # --- Inventory Calculation ---
    inventory_total = calculate_inventory(time_array, G1, a, D, h, alpha_n)

    # Part 2: Add the effect of the "ghost" generation rate (G2-G1) after t_change to simulate the change in generation rate.
    ghost_G = G2 - G1
    mask = time_array > t_change
    if ghost_G != 0:
        # Add ghost effect to the release rate
        ghost_times = time_array[mask] - t_change
        release_rate_total[mask] += calculate_release_rate(ghost_times, ghost_G, a, D, h, alpha_n)

        # Add ghost effect to concentration profiles
        ghost_profiles = np.array([[calculate_concentration(r, gt, ghost_G, a, D, h, alpha_n) for r in radial_pos] for gt in ghost_times])
        concentration_profiles[mask] += ghost_profiles

        # Add ghost effect to inventory
        ghost_inventory = calculate_inventory(ghost_times, ghost_G, a, D, h, alpha_n)
        inventory_total[mask] += ghost_inventory

    # Calculate cumulative release
    dt = np.diff(time_array, prepend=t_start)  # Time step for each interval
    cumulative_release = np.cumsum(release_rate_total * dt)

    print("Calculations complete.\n")

    # --- EXPORT RESULTS TO JSON ---
    print("Exporting results to JSON...")

    # Create a dictionary to hold all parameters
    parameters = {
        "grain_radius_m": a,
        "diffusivity_m2_per_s": D,
        "desorption_constant_m_per_s": K_d,
        "initial_generation_rate_atoms_per_m3_s": G1,
        "secondary_generation_rate_atoms_per_m3_s": G2,
        "t_change_s": t_change,
        "t_end_s": t_end,
        "n_roots": n_roots,
    }

    # Create a dictionary for the results, converting numpy arrays to lists
    results = {
        "time_s": time_array.tolist(),
        "time_hours": (time_array / 3600).tolist(),
        "release_rate_atoms_per_s": release_rate_total.tolist(),
        "cumulative_release_atoms": cumulative_release.tolist(),
        "inventory_atoms": inventory_total.tolist(),
        "radial_positions_m": radial_pos.tolist(),
        "concentration_profiles_atoms_per_m3": concentration_profiles.tolist(),
    }

    # Combine into a single dictionary
    export_data = {"simulation_parameters": parameters, "simulation_results": results}
    import json
    output_filename = "./results/LiOx_analytic_results.json"
    with open(output_filename, 'w') as f:
        json.dump(export_data, f, indent=4)
    print(f"Results exported to {output_filename}\n")
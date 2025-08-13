import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

# --- Core Model Functions based on Johnson et al. 1994 ---

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

# --- Main Script: Simulation and Plotting ---
if __name__ == '__main__':
    # --- 1. DEFINE PHYSICAL AND SIMULATION PARAMETERS ---
    
    # Physical Parameters (using illustrative values)
    # These should be changed to match the material and conditions of interest.
    a = 0.00025    # Grain radius (m), e.g., 20 micrometers
    D = 1e-14      # Diffusivity (m^2/s)
    K_d = 1e-8     # Desorption rate constant (m/s)
    G = 1e19       # Tritium generation rate (atoms/m^3/s)
    
    # Calculated parameter h
    h = K_d / D    # Ratio K_d/D (m^-1)
    
    # Simulation Parameters
    # User-defined timespan
    t_start = 0
    t_end = 3600 * 200  # End time in seconds (e.g., 5 hours)
    n_steps = 200     # Number of time steps for the plot
    n_roots = 100     # Number of roots to calculate for the series solution
    
    print("--- Model Parameters ---")
    print(f"Grain Radius (a): {a:.2e} m")
    print(f"Diffusivity (D): {D:.2e} m^2/s")
    print(f"Desorption Constant (K_d): {K_d:.2e} m/s")
    print(f"Generation Rate (G): {G:.2e} atoms/m^3/s")
    print(f"h (K_d/D): {h:.2e} m^-1")
    print("------------------------\n")
    
    # --- 2. PRE-CALCULATE ROOTS ---
    # This is the most computationally intensive part, so we do it once.
    print(f"Finding {n_roots} roots of the characteristic equation...")
    alpha_n = find_alpha_roots(n_roots, a, h)
    print("Roots found.\n")
    
    # --- 3. CALCULATE RESULTS OVER TIME ---
    print("Calculating release rate and inventory over time...")
    # Create a time array from t_start to t_end
    # We use a log space for time to better visualize the initial transient
    time_array = np.logspace(np.log10(t_end/10000), np.log10(t_end), n_steps)
    time_array = np.insert(time_array, 0, 0) # Add t=0

    # Create radial position array for calculating concentration profiles
    radial_pos = np.linspace(0, a, 10) # Radial positions from grain center to edge

    # Calculate concentration profiles at each time point
    concentration_profiles = np.array([[calculate_concentration(r, t, G, a, D, h, alpha_n) for r in radial_pos] for t in time_array])
    print("Calculations complete.\n")
    
    print(concentration_profiles.shape)
    print(concentration_profiles[0])  # Print the first time step for verification

    # Calculate the total release rate at each time point
    release_rates = calculate_release_rate(time_array, G, a, D, h, alpha_n)

    # --- 5. PLOT THE RESULTS ---
    print("Generating plots...")

        # Plot 1: Concentration Profiles at Different Times
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # Select a few time points to plot for clarity (e.g., 6 profiles)
    # We skip the t=0 profile as it's all zeros
    num_profiles_to_plot = 6
    plot_indices = np.linspace(1, len(time_array) - 1, num=num_profiles_to_plot, dtype=int)
    
    for i in plot_indices:
        time_val_hours = time_array[i] / 3600
        # Plot concentration vs. normalized radius (r/a)
        ax1.plot(radial_pos / a, concentration_profiles[i], marker='o', linestyle='-', label=f't = {time_val_hours:.2f} hours')

    ax1.set_title('Tritium Concentration Profiles in the Grain')
    ax1.set_xlabel('Normalized Radius (r/a)')
    ax1.set_ylabel('Concentration (atoms/m³)')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_ylim(bottom=0) # Concentration can't be negative

    plt.tight_layout()
    plt.show()

    # Plot 2: Release Rate vs. Time
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    # Plot release rate vs time in hours. We skip the first point (t=0) for better scaling if needed.
    ax2.plot(time_array[1:] / 3600, release_rates[1:], marker='.', linestyle='-')
    
    # At steady state, release rate equals generation rate (G * Volume)
    steady_state_rate = G * (4/3 * np.pi * a**3)
    ax2.axhline(y=steady_state_rate, color='r', linestyle='--', label=f'Steady-State Rate = {steady_state_rate:.2e} atoms/s')

    ax2.set_title('Tritium Release Rate Over Time')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Release Rate (atoms/s)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(bottom=0)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("Done.")


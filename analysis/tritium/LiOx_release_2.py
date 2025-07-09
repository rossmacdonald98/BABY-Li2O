import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy.optimize import fsolve
from matplotlib.ticker import ScalarFormatter


# --- 1. PHYSICAL AND SIMULATION PARAMETERS ---

# --- Grain Properties ---
a = 0.01          # Grain Radius (cm)

# --- Diffusion & Generation Properties ---
D = 1.0e-16          # Intrinsic Diffusion Coefficient (cm^2/s)
source_rate = 1e8    # Tritium Source Rate (T/s)
tbr = 2e-5           # Volumetric Tritium Breeding Ratio (T/n/cm^3)
G_active = source_rate*tbr # Tritium Generation Rate when source on (T/cm^3/s)

# --- Gas & System Properties ---
kd = 1.0e-5         # Desorption Coefficient 

decay_constant = 1.785e-9 # Tritium Decay Constant (1/s)

# --- Simulation Parameters ---
t_irradiate = 3600 * 0.1  # Duration of Phase 1 (generation is on) [s]
t_release = 3600 * 15 # Duration of Phase 2 (generation is off) [s]

data_points_irr = 100 # Number of data points for irradiation phase
data_points_release = 1000 # Number of data points for release phase

t_irr_phase = np.linspace(0, t_irradiate, data_points_irr)
t_release_phase = np.linspace(0, t_release, data_points_release)

roots_to_find = 25  # Number of roots to calculate for the sum approximation

# --- Derived parameters ---
h = kd / D   

# --- 2. Functions ---

def solve_transcendental_equation(roots_to_find, a, h):
    """
    Numerically finds the first n roots of the transcendental equation:
    a*alpha*cot(a*alpha) = 1 - a*h
    """
    # Define the function for the root-finding algorithm.
    # We want to find alpha where f(alpha) = 0.
    equation = lambda alpha: a * alpha / np.tan(a * alpha) - (1 - a * h)
    
    # The roots are roughly spaced by pi/a. We use this to provide
    # good initial guesses for the fsolve function.
    # The previous guess landed on poles of tan(), causing numerical instability.
    # This new guess is slightly offset to provide a stable starting point.
    initial_guesses = [(n + 0.01) * np.pi / a for n in range(1, roots_to_find + 1)]
    
    # Use SciPy's fsolve to find the roots from the initial guesses.
    roots = fsolve(equation, initial_guesses)
    
    return roots

def calculate_release_rate(t, G1, G2, t1, a, D, h, roots):
    """
    Calculates the tritium release rate R_total(t) using model for non-steady-state initial conditions.
    """
    # Unpack parameters for clarity
    ah = a * h
    
    # Calculate the summation term
    summation = 0
    for alpha_n in roots:
        # Denominator of the summation term
        denominator = (alpha_n**2) * (a**2 * alpha_n**2 + ah * (ah - 1))
        
        # Term inside the square brackets in the numerator
        bracket_term = (G2 - G1) + G1 * np.exp(-D * alpha_n**2 * t1)
        
        # Full numerator term
        numerator = bracket_term * np.exp(-D * alpha_n**2 * t)
        
        summation += numerator / denominator
        
    # Calculate the full release rate using the general model equation
    steady_state_term = (G2 * a) / 3
    transient_term = 2 * h**2 * a * summation
    
    R_total = steady_state_term - transient_term
    
    return R_total

# --- 3. Find the roots for the solution ---
start_time = time.time()

print("Finding roots of the transcendental equation...")
alpha_roots = solve_transcendental_equation(roots_to_find, a, h)
print(f"Found {len(alpha_roots)} roots.")
print(alpha_roots)

print("\n--- Verifying found roots ---")
equation_to_check = lambda alpha: a * alpha / np.tan(a * alpha) - (1 - a * h)
for i, root in enumerate(alpha_roots[:10]): # Check the first 10 roots
    residual = equation_to_check(root)
    print(f"Root {i+1}: {root:.4f},  Residual (should be ~0): {residual:.2e}")
print("---------------------------\n")

# --- 4. Run the Model for Both Phases ---   
# Phase 1: Irradiate from empty (G1=0, G2=G_active)
print("Simulating Phase 1: Irradiate...")
R_irradiate = [calculate_release_rate(t, 0, G_active, 0, a, D, h, alpha_roots) for t in t_irr_phase]
    
# Phase 2: Release from non-steady-state (G1=G_active, G2=0, t1=t_startup)
print("Simulating Phase 2: Release...")
R_release = [calculate_release_rate(t, G_active, 0, t_irradiate, a, D, h, alpha_roots) for t in t_release_phase]

# --- 5. Combine and Plot the Results ---
# Create a continuous time array for the plot
t_combined = np.concatenate((t_irr_phase, t_release_phase + t_irradiate))
R_combined = np.concatenate((R_irradiate, R_release))
    
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(t_combined / 3600, R_combined, color='mediumseagreen', linewidth=2.5)
    
# Add a vertical line to show where the phase change happens
ax.axvline(x=t_irradiate / 3600, color='k', linestyle='--', linewidth=1.5, label=f'Generation Off (at {t_irradiate/3600:.1f} hrs)')
    
# Formatting the plot
ax.set_title('Tritium Release for a Non-Steady-State Generation Pulse', fontsize=16, pad=20)
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Release Rate, R(t) [atoms/s]', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=10)
    
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

fig.tight_layout()
plt.show()


# Explanation of the Tritium Release Model

## 1. Introduction

This document provides a detailed explanation of the tritium transport and release model developed for the BABY-LiOx experiment. The primary objective of this model is to simulate the journey of tritium from its point of generation within a lithium ceramic breeder material (Li₂O) to its eventual release into a helium sparge gas stream.

The model is designed to handle two distinct physical forms of the breeder material, each with its own transport phenomena:

1.  **Li₂O Powder:** A packed bed of fine, spherical Li₂O grains.
2.  **Sintered Li₂O Pellets:** A packed bed of porous cylindrical pellets, which are themselves composed of sintered spherical grains.

The ultimate goal is to predict the tritium release rate from the packed bed over time, providing insights into the performance of the breeder material under experimental conditions.

## 2. Model Hierarchy

The model employs a multi-scale approach to accurately capture the complex physics involved in tritium transport. The problem is broken down into three distinct scales:

*   **Micro-scale (Intra-grain):** Focuses on the transport phenomena occurring *within* a single spherical Li₂O grain. This includes tritium generation, diffusion through the solid lattice, and interactions with trapping sites.
*   **Meso-scale (Inter-grain):** Describes the transport of tritium from the grain surface into the surrounding gas. The specifics of this scale differ significantly between the powder and pellet models.
*   **Macro-scale (Packed Bed):** Models the bulk flow and transport of tritium in the helium sparge gas as it moves axially through the entire packed bed column.

These scales are coupled, meaning the output from a smaller scale serves as the input for the next larger scale, creating a comprehensive, interconnected simulation.

## 3. Intra-Grain Transport Model (Micro-scale)

This model is the fundamental building block for both the powder and pellet simulations. It describes the behavior of tritium inside a single, spherical grain of radius $r_g$.

### Physical Processes

*   **Generation ($G$):** Tritium is produced by neutron-lithium reactions. This generation is assumed to be uniform throughout the volume of the grain and occurs at a constant rate during the irradiation period.
*   **Diffusion:** Mobile tritium atoms ($C_m$) are assumed to move through the solid Li₂O lattice via a diffusion mechanism governed by Fick's second law in spherical coordinates.
*   **Trapping & De-trapping:** Tritium atoms can be temporarily immobilized at defects or "trap sites" within the crystal lattice. The concentration of trapped tritium is denoted by $C_t$. This process is modeled as a reversible, first-order kinetic reaction where mobile atoms are trapped and trapped atoms are thermally excited and released (de-trapped).

### Governing Equations

The concentrations of mobile ($C_m$) and trapped ($C_t$) tritium as a function of radial position $r$ and time $t$ are described by a system of coupled partial differential equations:

**Mobile Tritium ($C_m$):**
$$
\frac{\partial C_m}{\partial t} = \underbrace{D \left( \frac{\partial^2 C_m}{\partial r^2} + \frac{2}{r}\frac{\partial C_m}{\partial r} \right)}_{\text{Diffusion}} + \underbrace{G}_{\text{Generation}} - \underbrace{k_t C_m (N_t - C_t)}_{\text{Trapping}} + \underbrace{k_d C_t}_{\text{De-trapping}}
$$

**Trapped Tritium ($C_t$):**
$$
\frac{\partial C_t}{\partial t} = \underbrace{k_t C_m (N_t - C_t)}_{\text{Trapping}} - \underbrace{k_d C_t}_{\text{De-trapping}}
$$

where:
*   $D$ is the diffusion coefficient.
*   $k_t$ and $k_d$ are the trapping and de-trapping rate coefficients, respectively.
*   $N_t$ is the density of available trap sites.

### Boundary Conditions

1.  **Grain Center ($r=0$):** Due to spherical symmetry, a zero-flux condition is applied.
    $$
    \frac{\partial C_m}{\partial r} \bigg|_{r=0} = 0
    $$

2.  **Grain Surface ($r=r_g$):** The diffusive flux of tritium reaching the grain surface from the bulk must equal the net flux of tritium leaving the surface.
    $$
    -D \frac{\partial C_m}{\partial r} \bigg|_{r=r_g} = J_{\text{grain}}
    $$
    The surface flux, $J_{\text{grain}}$, is determined by desorption from the surface and adsorption from the surrounding gas. This is a key coupling point to the meso-scale model.

## 4. Inter-Grain and Packed Bed Transport

This section describes how tritium moves from the grain surfaces into the bulk sparge gas. The mechanism depends on whether the breeder is a powder or a sintered pellet bed.

### 4.1. Powder Model

In the powder model, the grains are in direct contact with the helium sparge gas.

*   **Surface Flux:** The net flux from the grain surface ($J_{\text{grain}}$) is a balance between second-order desorption (recombination of two tritium atoms) and first-order adsorption from the sparge gas.
    $$
    J_{\text{grain}} = k_{\text{des}} C_{m, \text{surface}}^2 - k_{\text{ads}} C_{\text{sparge}}
    $$
*   **Packed Bed Transport:** The bed is modeled as a Plug Flow Reactor (PFR), which is solved numerically by discretizing it into a series of CSTRs (Continuous Stirred-Tank Reactors). For each axial plug $j$, the change in sparge gas concentration ($C_{\text{sparge},j}$) is given by a mass balance:
    $$
    \frac{dC_{\text{sparge}, j}}{dt} = \underbrace{\frac{J_{\text{grain}, j} \cdot A_{\text{grains, plug}}}{V_{\text{sparge, plug}}}}_{\text{Release from powder}} - \underbrace{\frac{Q_{\text{sparge}}}{V_{\text{sparge, plug}}}(C_{\text{sparge}, j-1} - C_{\text{sparge}, j})}_{\text{Removal via convection}} 
    $$
    where $Q_{\text{sparge}}$ is the sparge gas flow rate, $V_{\text{sparge, plug}}$ is the gas volume in the plug, and $A_{\text{grains, plug}}$ is the total surface area of all grains in that plug.

### 4.2. Sintered Pellet Model

For sintered pellets, an intermediate step is required to model the transport through the pellet's internal pore network.

*   **Step 1: Grain Surface to Pellet Pore:** Tritium desorbs from the grain surfaces into the gas-filled pores within the pellet. The surface flux equation is similar to the powder model, but depends on the pore gas concentration, $C_{\text{pore}}$.
    $$
    J_{\text{grain}} = k_{\text{des}} C_{m, \text{surface}}^2 - k_{\text{ads}} C_{\text{pore}}
    $$
*   **Step 2: Pellet Pore to Sparge Gas:** A mass balance on the tritium in the pore volume of a single pellet gives the rate of change of $C_{\text{pore}}$. This balances the total flux from all internal grains against the flux leaving the pellet's external surface.
    $$
    \frac{dC_{\text{pore}}}{dt} = \underbrace{\frac{J_{\text{grain}} \cdot A_{\text{internal}}}{V_{\text{pore}}}}_{\text{Release from grains}} - \underbrace{\frac{J_{\text{pellet}} \cdot A_{\text{external}}}{V_{\text{pore}}}}_{\text{Release from pellet}}
    $$
    The flux from the pellet to the sparge gas, $J_{\text{pellet}}$, is defined by a mass transfer correlation:
    $$
    J_{\text{pellet}} = h_{\text{pellet}} (C_{\text{pore}} - C_{\text{sparge}})
    $$
*   **Packed Bed Transport:** The macro-scale model is identical in form to the powder model, but the source term is now the flux from the pellets ($J_{\text{pellet}}$) rather than directly from the grains.
    $$
    \frac{dC_{\text{sparge}, j}}{dt} = \underbrace{\frac{J_{\text{pellet}, j} \cdot A_{\text{pellets, plug}}}{V_{\text{sparge, plug}}}}_{\text{Release from pellets}} - \underbrace{\frac{Q_{\text{sparge}}}{V_{\text{sparge, plug}}}(C_{\text{sparge}, j-1} - C_{\text{sparge}, j})}_{\text{Removal via convection}}
    $$

### Sintering Effect ($F_r$)

The sintering process used to create pellets causes "necking" between grains, which reduces the total exposed surface area available for release. This is accounted for by a **surface area reduction factor, $F_r$**, which scales the surface release term in the grain's surface boundary condition. The factor is derived from the change in specific surface area during sintering and is related to the pellet's final porosity ($\epsilon_p$) and the initial packing efficiency of the unsintered powder ($\phi_0$):

$$
F_r = \left( \frac{\epsilon_p}{1 - \phi_0} \right)^{2/3}
$$

For a random close packing of spheres, $\phi_0 \approx 0.64$, making the initial porosity $1 - \phi_0 = 0.36$.

## 5. Numerical Solution

The system of coupled PDEs and ODEs is solved numerically.

*   **Spatial Discretization:** A finite difference method is used to discretize the spherical grain into $N$ concentric radial shells. This converts the diffusion PDE into a system of $N+1$ coupled ODEs. Special formulations are used for the central node (to handle the $1/r$ singularity) and the surface node (to incorporate the surface flux boundary condition).

    *   **Center Node ($i=0$):**
        $$ \nabla^2 C_{m, i=0} = \frac{6(C_{m,1} - C_{m,0})}{\Delta r^2} $$

    *   **Interior Nodes ($0 < i < N$):**
        $$ \nabla^2 C_{m,i} = \frac{C_{m,i+1} - 2C_{m,i} + C_{m,i-1}}{\Delta r^2} + \frac{1}{r_i} \frac{C_{m,i+1} - C_{m,i-1}}{\Delta r} $$

    *   **Surface Node ($i=N$):** The diffusion term of the governing equation is discretized as:
        $$ \frac{\partial C_m}{\partial t} \bigg|_{\text{diffusion, surface}} = \frac{2D(C_{N-1}-C_N)}{\Delta r^2} - F_r \cdot 2 J_{\text{grain}} \left( \frac{1}{\Delta r} + \frac{1}{r_g} \right) $$
        Note the inclusion of the surface area reduction factor $F_r$ (where $F_r=1$ for the powder model).

*   **Time Integration:** The full system of ODEs (from the discretized grain model and the packed bed model) is solved over time. Because the characteristic times of the physical processes (e.g., diffusion vs. trapping) can vary by many orders of magnitude, the system is mathematically "stiff". To handle this, an **adaptive time-stepping** method is employed. The time step, $dt$, is dynamically adjusted at each iteration—decreased when concentrations are changing rapidly and increased when the system is approaching a steady state to balance the numerical stability and computational efficiency.

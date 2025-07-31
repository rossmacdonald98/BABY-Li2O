# Concept Li2O Pellet Release Model

This document outlines the formulation of a model to predict tritium production and release from a packed bed of sintered Lithium Oxide (Li2O) pellets. The model couples intra-grain diffusion and trapping with inter-grain pore diffusion and convection in a bulk sparge gas flow.

---

## 1. Model Formulation

### 1.1. Pellet Structure

The model considers a porous Li2O pellet composed of equally-sized, spherical grains. This structure arises from the sintering manufacturing process. Key assumptions include:

* **Open Porosity:** The pore network between grains provides a direct pathway for gas transport from the grain surfaces to the pellet's external boundary.
* **Surface Area Reduction:** The sintering process causes "necking" between adjacent grains, reducing the total exposed surface area available for mass transfer. This is accounted for by a surface area reduction factor, $F_r$.

Tritium is generated within the solid grains, diffuses to the grain surface, desorbs into the pore gas, and is subsequently transported into the main sparge gas flow.

---

## 2. Intra-Grain Tritium Transport

This section describes the transport of tritium within a single spherical grain.

### 2.1. Governing Equations

Tritium within the solid can exist in two states: mobile ($C_m$) and trapped ($C_t$).

**Mobile Tritium Concentration ($C_m$):**
The rate of change of mobile tritium is governed by diffusion, generation, trapping, and de-trapping. In spherical coordinates, the equation is:

$$
\frac{\partial C_m}{\partial t} = D \left( \frac{\partial^2 C_m}{\partial r^2} + \frac{2}{r}\frac{\partial C_m}{\partial r} \right) + G - k_t C_m (N_t - C_t) + k_d C_t
$$

* $D \nabla^2 C_m$: Diffusion of mobile tritium through the solid (Fick's second law).
* $G$: Generation rate from neutron reactions, assumed to be uniform.
* $k_t C_m (N_t - C_t)$: Trapping rate of mobile tritium at available trap sites.
* $k_d C_t$: De-trapping (release) rate of trapped tritium due to thermal excitation.

**Trapped Tritium Concentration ($C_t$):**
The rate of change of the trapped population is a balance between the trapping and de-trapping processes:

$$
\frac{\partial C_t}{\partial t} = k_t C_m (N_t - C_t) - k_d C_t
$$

### 2.2. Boundary Conditions

**At the Grain Center ($r=0$):**
Symmetry dictates a zero-flux condition:

$$
\frac{\partial C_m}{\partial r} \bigg|_{r=0} = 0
$$

**At the Grain Surface ($r=r_{grain}$):**
The diffusive flux from the bulk must equal the net flux from surface desorption/adsorption ($J_{grain}$).

$$
\frac{\partial C_m}{\partial r} \bigg|{r=r_{grain}} = -\frac{J_{grain}}{D}
$$

where the surface flux is defined as:

$$
J_{grain} = k_{des} C_{m, surface}^2 - k_{ads} C_{pore}
$$

This expression captures second-order desorption (recombination) and first-order adsorption from the pore gas.

---

## 3. Inter-Grain and Pellet Transport

This section describes the transport of tritium from the grain surfaces into the bulk sparge gas.

### 3.1. Pore Gas Concentration ($C_{pore}$)

Assuming the pore network within a pellet at a given axial location is well-mixed, the change in pore gas concentration is a mass balance between the release from all grains and the transfer to the sparge gas:

$$
\frac{dC_{pore}}{dt} = \frac{J_{grain} \cdot A_{internal}}{V_{pore}} - \frac{J_{pellet} \cdot A_{external}}{V_{pore}}
$$

* $J_{pellet}$: Flux of tritium from the pellet's external surface to the sparge gas, defined by a mass transfer correlation:

$$
J_{pellet} = h_{pellet} (C_{pore} - C_{sparge})
$$
  
* $A_{internal}$: Total exposed surface area of grains within a pellet.
* $A_{external}$: External surface area of a single pellet.
* $V_{pore}$: Total pore volume within a single pellet.

### 3.2. Sparge Gas Concentration ($C_{sparge}$)

The packed bed of pellets is modeled as a Plug Flow Reactor (PFR), solved numerically by discretizing the bed into a series of CSTRs (plugs).

* **For Plug ($j = 0$) (Inlet):**
    Assuming pure sparge gas inlet ($C_{inlet}=0$):
  
$$
\frac{dC_{sparge, j=0}}{dt} = \frac{J_{pellet, j=0} \cdot A_{pellets, plug}}{V_{sparge, plug}} - \frac{Q_{sparge} \cdot C_{sparge, j=0}}{V_{sparge, plug}}
$$

* **For Downstream Plugs ($j=n$):**
    The mass balance includes convective flow from the upstream plug:
  
$$
\frac{dC_{sparge, j=n}}{dt} = \frac{J_{pellet, j=n} \cdot A_{pellets, plug}}{V_{sparge, plug}} + \frac{Q_{sparge} \cdot C_{sparge, j=n-1}}{V_{sparge, plug}} - \frac{Q_{sparge} \cdot C_{sparge, j=n}}{V_{sparge, plug}}
$$

---

## 4. Numerical Implementation Notes

The system of coupled PDEs and ODEs is solved numerically.

### 4.1. Discretization of the Diffusion Equation

A finite difference method is used. The grain is discretized into $N$ radial nodes.

* **Interior Nodes ($0 < i < N$):** The Laplacian term $\nabla^2 C_{m,i}$ is discretized using a standard central difference scheme.

$$
\nabla^2 C_{m,i} = \frac{C_{m,i+1} - 2C_{m,i} + C_{m,i-1}}{\Delta r^2} + \frac{1}{r_i} \frac{C_{m,i+1} - C_{m,i-1}}{\Delta r}
$$

* **Center Node ($i=0$):** A special formulation is used to handle the singularity at $r=0$:

$$
\nabla^2 C_{m, i=0} = \frac{6(C_{m,1} - C_{m,0})}{\Delta r^2}
$$

* **Surface Node ($i=N$):** The boundary condition is enforced using a "ghost point" method. Substituting the discretized first and second derivatives into the diffusion part of the governing equation yields:

$$
\frac{\partial C_m}{\partial t} \bigg|{diffusion, surface} = \frac{2D(C_{N-1}-C_N)}{\Delta r^2} - 2 J_{grain} \left( \frac{1}{\Delta r} + \frac{1}{r_g} \right)
$$

* To account for sintering, the release term (related to $J_{grain}$) is scaled by the surface area reduction factor, $F_r$:
    
$$
\frac{\partial C_m}{\partial t} \bigg|{diffusion, surface} = \frac{2D(C_{N-1}-C_N)}{\Delta r^2} - F_r \cdot 2 J_{grain} \left( \frac{1}{\Delta r} + \frac{1}{r_g} \right)
$$

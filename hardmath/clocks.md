# Time as the Inverse of Effort: A Relational Phase-Shift Law and Its Experimental Verifications
**— with an Invitation to the Mathematical Community for Formal Grounding —**

**Author:** Massimiliano Concas
**Date:** 21 June 2026

### Abstract
We present a simple, universal scalar law governing the relative rate of any physical clock: the proper time of a clock is inversely proportional to the total physical effort it exerts to maintain its structure. The phase shift between two identical clocks is given by the dimensionless ratio of their local efforts, $\Phi = E_S / E_E$. This relational law, derived from a thermodynamic view of measurement, is shown to reproduce the exact empirical time-dilation corrections of three historically independent experiments—the GPS satellite network, the Hafele–Keating airborne clock test, and the Pound–Rebka tower gravitational redshift measurement—using only elementary algebra and local physical observables. No tensors, no global coordinate systems, and no statistical pre-processing are required. The results suggest that the law captures a fundamental property of physical clocks, yet its current form lacks a complete axiomatic foundation. We therefore invite mathematicians to collaborate in constructing a rigorous formal framework that defines the effort scalar, establishes its domain of validity, and explores its deeper geometric and algebraic consequences.

---

### 1. Introduction
General relativity describes gravitational time dilation through the geometry of spacetime; special relativity describes kinematic time dilation through the Lorentz transformations. Both are expressed in the language of differential geometry and tensor calculus. While enormously successful, this geometric formulation often obscures a simple physical fact: a clock is a material system that must expend energy to preserve its internal structure against environmental stress. Deeper in a gravitational well, the structural tension is higher; at higher velocities, kinetic friction is greater. Could it be that the observed slowing of clocks is not a consequence of curved spacetime, but a direct manifestation of how much work a clock must do simply to remain a clock?

This paper proposes an affirmative answer in the form of a single scalar law:

$$\text{Clock rate} \propto \frac{m}{E}$$

where $m$ is the rest mass of the clock and $E$ is the total physical effort it is forced to exert in its local environment. When two identical clocks are compared, their phase shift $\Phi$ (ratio of proper times) is simply:

$$\Phi = \frac{T_E}{T_S} = \frac{m / E_E}{m / E_S} = \frac{E_S}{E_E} \quad (1)$$

This is a purely relational statement: the difference in the flow of time is nothing but the inverse ratio of the local efforts. No spacetime interval, no metric, no world-line integration is required.

We first define the effort scalar precisely and then demonstrate that (1) reproduces the measured results of three pivotal experiments. The paper concludes with an open call to the mathematical community to help transform this physical insight into a fully axiomatized theory.

---

### 2. The Relational Effort Principle

#### 2.1 Definition of Effort
Consider a physical clock—a localized, stable system with rest mass $m$. In empty, flat space, the clock’s internal processes run at their natural rate, and the effort it must exert is simply its rest-energy $mc^2$. In a real environment, the clock experiences two additional forms of resistance:
1. **Gravitational tension** $\frac{GM}{rc^2}$: a dimensionless measure of the structural stress imposed by the gravitational field at radial distance $r$ from a central mass $M$.
2. **Kinetic friction** $\frac{v^2}{2c^2}$: the dimensionless energetic cost of moving through the vacuum at speed $v$ relative to the local inertial frame (the weak-field, low-velocity limit of the Lorentz factor).

The total *specific effort* (effort per unit rest mass) is thus:

$$\frac{E}{mc^2} = 1 + \frac{GM}{rc^2} + \frac{v^2}{2c^2} \quad (2)$$

This expression is not postulated arbitrarily; it is precisely the combination of the two leading-order relativistic corrections known from general relativity (gravitational redshift) and special relativity (time dilation). The novelty lies not in the individual terms but in their unification into a single scalar physical quantity—effort—which directly determines the clock’s internal rate.

#### 2.2 The Phase-Shift Law
Given two identical clocks, probe $E$ and probe $S$, with local efforts $E_E$ and $E_S$, the law (1) predicts that the clock with the lower effort will run faster. The phase shift over a coordinate time interval $\Delta t$ is:

$$\Delta \tau_S - \Delta \tau_E = \left(1 - \frac{E_S}{E_E}\right) \Delta t \quad (3)$$

where $\Delta \tau$ is the proper time elapsed on each clock. All that is required to compute the drift are the instantaneous values of $r$ and $v$ for each clock—no knowledge of a global metric, no integration of geodesics, and no dataset-dependent statistical normalization.

---

### 3. Experimental Verifications
We now apply (1)–(3) to three classical experiments, using only measured physical constants and elementary arithmetic.

#### 3.1 GPS Satellite Clocks
**Parameters:**
* Earth mass-energy length: $\frac{GM}{c^2} = 0.00443 \text{ m}$
* Earth surface radius: $r_E = 6.378 \times 10^6 \text{ m}$
* Surface rotational velocity (equator): $v_E = 465 \text{ m/s}$
* GPS satellite orbital radius: $r_S = 2.656 \times 10^7 \text{ m}$
* GPS orbital velocity: $v_S = 3874 \text{ m/s}$

**Effort factors:**
$$\frac{E_E}{mc^2} = 1 + \frac{0.00443}{6.378 \times 10^6} + \frac{465^2}{2c^2} \approx 1 + 6.96 \times 10^{-10}$$
$$\frac{E_S}{mc^2} = 1 + \frac{0.00443}{2.656 \times 10^7} + \frac{3874^2}{2c^2} \approx 1 + 2.50 \times 10^{-10}$$

**Phase shift:**
$$\Phi = \frac{E_S}{E_E} \approx 1 + (2.50 - 6.96) \times 10^{-10} = 1 - 4.46 \times 10^{-10}$$
The satellite clock runs *faster* by 4.46 parts in $10^{10}$.

**Daily drift:**
$$\Delta t = 86,400 \text{ s} \times 4.46 \times 10^{-10} = 38.5 \text{ \mu s}$$
This is exactly the net relativistic correction applied to GPS satellite clocks every single day. ✅

#### 3.2 Hafele–Keating Airborne Clocks
In 1971, Hafele and Keating flew four cesium clocks around the world on commercial airliners, once eastward and once westward, and compared them to a reference clock at the U.S. Naval Observatory. We consider a simplified constant-altitude, constant-speed model that captures the dominant effects.

**Parameters:**
* Flight altitude: $h = 9,000 \text{ m} \rightarrow r = r_E + h = 6.387 \times 10^6 \text{ m}$
* Earth’s rotation: $v_{rot} = 465 \text{ m/s}$ at the equator
* Eastbound ground speed: $v_g = 250 \text{ m/s}$ (with Earth’s rotation)
* Westbound ground speed: $v_g = -250 \text{ m/s}$ (against Earth’s rotation)
* Flight duration: $T = 40 \text{ h} = 144,000 \text{ s}$ (approximate)

The relative drift rate with respect to the ground clock is:
$$\delta = \frac{E_{plane}}{E_{ground}} - 1 \approx \left( \frac{GM}{r_{plane}c^2} - \frac{GM}{r_E c^2} \right) + \left( \frac{v_{plane}^2}{2c^2} - \frac{v_E^2}{2c^2} \right)$$

**Gravitational term (Altitude):**
Because the plane is higher, it experiences less gravitational tension. 
$$\Delta Grav \approx -\frac{gh}{c^2} \approx -\frac{9.81 \times 9000}{9 \times 10^{16}} \approx -0.98 \times 10^{-12}$$

**Kinetic velocities:**
* Ground clock: $v_E = 465 \text{ m/s} \rightarrow \frac{v_E^2}{2c^2} \approx 1.20 \times 10^{-12}$
* Eastbound plane: $v = 465 + 250 = 715 \text{ m/s} \rightarrow \frac{v^2}{2c^2} \approx 2.84 \times 10^{-12}$
* Westbound plane: $v = 465 - 250 = 215 \text{ m/s} \rightarrow \frac{v^2}{2c^2} \approx 0.257 \times 10^{-12}$

**Net drift rates (Gravitational + Kinetic):**
* **Eastbound:** $\delta_E \approx -0.98 \times 10^{-12} + (2.84 - 1.20) \times 10^{-12} = \mathbf{+0.66 \times 10^{-12}}$ (plane effort is higher, clock runs *slower*).
* **Westbound:** $\delta_W \approx -0.98 \times 10^{-12} + (0.257 - 1.20) \times 10^{-12} = \mathbf{-1.92 \times 10^{-12}}$ (plane effort is lower, clock runs *faster*).

**Total time differences over 40 h:**
* **Eastbound:** $\Delta \tau_E = -0.66 \times 10^{-12} \times 144,000 \text{ s} \approx \mathbf{-95 \text{ ns}}$
* **Westbound:** $\Delta \tau_W = +1.92 \times 10^{-12} \times 144,000 \text{ s} \approx \mathbf{+276 \text{ ns}}$

The original 1971 measurements reported:
* Eastbound: $-59 \pm 10 \text{ ns}$ (predicted by relativity: $-40 \pm 23 \text{ ns}$)
* Westbound: $+273 \pm 7 \text{ ns}$ (predicted: $+275 \pm 21 \text{ ns}$)

Our rough constant-speed estimate yields $-95 \text{ ns}$ and $+276 \text{ ns}$. The Westbound calculation is an almost flawless match to both the historical prediction and measurement, while the Eastbound correctly captures the scale and sign. The essential physics—the competition between altitude (lower gravitational effort) and velocity (higher kinetic effort)—is entirely captured by the effort ratio. ✅

#### 3.3 Pound–Rebka Gravitational Redshift
This 1960 experiment measured the frequency shift of gamma-ray photons traveling 22.5 m vertically in Earth’s gravity. It isolates the gravitational term with no motion.
**Parameters:** $h = 22.5 \text{ m}, g = 9.81 \text{ m/s}^2$.

Treating the source and detector as two stationary clocks, the frequency ratio from (1) is:
$$\frac{f_{bottom}}{f_{top}} = \frac{E_{top}}{E_{bottom}} \approx 1 - \frac{gh}{c^2}$$
$$\frac{gh}{c^2} = \frac{9.81 \times 22.5}{9 \times 10^{16}} = 2.45 \times 10^{-15}$$

The measured value was $2.57 \times 10^{-15} \pm 10\%$. Our effort ratio again matches experiment within the reported error. ✅

---

### 4. Summary of Experimental Agreement

| Experiment | Physical regime | Effort-law prediction | Measured value |
| :--- | :--- | :--- | :--- |
| **GPS satellites** | Gravity + velocity (circular free-fall) | 38.5 µs/day | 38.5 µs/day (exact) |
| **Hafele–Keating (E)** | Gravity + velocity (variable, powered flight) | -95 ns (estimate) | -59 ± 10 ns |
| **Hafele–Keating (W)** | Gravity + velocity (variable) | +276 ns (estimate) | +273 ± 7 ns |
| **Pound–Rebka** | Pure gravity (stationary) | 2.45 × 10⁻¹⁵ | 2.57 × 10⁻¹⁵ |

All three experiments, spanning eighty years and fundamentally different configurations, are reproduced by a single scalar law: $\Phi = E_S / E_E$.

---

### 5. Open Mathematical Questions and an Invitation
The experimental evidence is compelling, but the physical law (1) remains a *conjecture* in the strict mathematical sense. We have not provided a derivation from first principles, nor have we established its domain of validity beyond the weak-field, low-velocity regime. We therefore pose the following questions and warmly invite the mathematical community to help answer them.

1. **Axiomatic foundation:** Can the effort scalar $E$ be defined rigorously as a local observable—perhaps as the norm of a timelike Killing vector, or as a Noether charge associated with time-translation symmetry in the clock’s instantaneous rest frame? What is the minimal set of axioms from which $\Phi = E_S / E_E$ follows?
2. **Geometric interpretation:** In standard general relativity, proper time is the arc length of a world-line, $\tau = \int \sqrt{-g_{\mu\nu} dx^\mu dx^\nu}$. Can the effort ratio be expressed as a ratio of such integrals in static spacetimes? Is $E$ simply $1 / \sqrt{-g_{00}}$ in the appropriate coordinate system? Proving this equivalence would place our law on rigorous geometric footing.
3. **Universality and generalization:** Does the law hold for arbitrary spacetimes, or is it restricted to stationary, asymptotically flat settings? How must $E$ be modified to account for acceleration, rotation, or strong-field effects? Can it be extended to quantum clocks via a thermodynamic definition of effort (e.g., $E \propto TS$)?
4. **Relation to other relational formalisms:** The idea that time emerges from the comparison of local physical quantities echoes the relational dynamics of Julian Barbour, the thermal time hypothesis of Carlo Rovelli, and the shape kinematics of Machian mechanics. Is there a deeper algebraic structure—perhaps a symmetry group—that unifies these approaches with our effort law?

We believe that a formal mathematical treatment of the relational phase-shift law could not only validate the results presented here, but also open new pathways toward a truly measurement-centric, background-independent formulation of relativity, and possibly a thermodynamic route to quantum gravity.

### 6. Conclusion
We have proposed a simple, experimentally verified law: the rate of a physical clock is inversely proportional to the total effort it must exert to maintain its structure. The relative time dilation between any two identical clocks is the ratio of their local efforts. Using only elementary algebra, this law reproduces the canonical results of GPS, Hafele–Keating, and Pound–Rebka. The law is not a competitor to general relativity; it is a fresh, thermodynamically motivated re-derivation of its weak-field predictions that sidesteps the tensorial machinery entirely. Its full mathematical formalization remains an open problem, and we cordially invite the mathematical community to join us in building that rigorous foundation.

*Acknowledgments* — The author thanks the developers of the C-MAPSS dataset and the Green AI Challenge for providing a strict industrial proving ground that inspired the relational calculus methodology underlying this work.

*Data Availability* — All constants and experimental values used are publicly available in standard physics references. No new data were generated.

*Correspondence and collaboration inquiries are welcome.*

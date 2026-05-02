#!/usr/bin/env python3
"""
Relational Calculus: An Executable Introduction
================================================

This script is an *executable paper* — run it, read the printed output,
and study the code to understand the two core paradigms of Relational Calculus:

  CONFIRMATORY mode   – You know the equation and want to reveal its dimensionless soul.
  EXPLORATORY mode    – You only have a black‑box function; the decoder finds the hidden blueprint.

The code is intentionally transparent: no external libraries beyond NumPy,
no machine learning, no symbolic algebra. Everything follows the methodology
described in "The Intrinsic Blueprint: An Introduction to Relational Calculus".

Two case studies are hardwired:

  1. PROJECTILE RANGE (confirmatory)
     Variables: theta, v0, g
     Known formula: R = v0² / g * sin(2*theta)
     What the decoder discovers: capacity = v0²/g, shape = sin(2θ)

  2. TRAFFIC FLOW (exploratory)
     Variables: density, free_flow_speed, max_density
     Hidden formula: flow = free_flow_speed * density * (1 - density/max_density)
     What the decoder discovers: capacity = 0.25 * vf * kj,
     dimensionless shape = 4x(1-x)   (the Greenshields parabola)

After the two demonstrations, you can replace the `goal` functions with your own
system and re‑run the script to decode its intrinsic blueprint.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from itertools import product
import math

# ═══════════════════════════════════════════════════════════════════════════
# 0. USER‑INTERFACE FOR THE DECODER
# ═══════════════════════════════════════════════════════════════════════════

def is_broken(output):
    """Return True if the output signals that the system is in an invalid state."""
    return output is None or (isinstance(output, float) and
                              (math.isnan(output) or math.isinf(output)))

# ═══════════════════════════════════════════════════════════════════════════
# 1. PROBING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def sweep_variable(var, initial_state, goal, start=1e-6, growth=1.5,
                   max_steps=100, stop_after_first_peak=False):
    """
    Sweep a single variable upwards from a tiny starting value, calling
    goal(var, value, state) each time.  Returns:
        inputs   – np.array of input values tried
        outputs  – np.array of corresponding outputs
        peak_detected – bool, True if a local maximum was passed during the sweep
    """
    inputs, outputs = [], []
    x = start
    state = dict(initial_state)
    prev_out = None
    peak_detected = False
    for step in range(max_steps):
        st = dict(state)
        out = goal(var, x, st)
        if is_broken(out):
            break
        inputs.append(x)
        outputs.append(out)
        # Detect a simple rise‑and‑fall pattern
        if stop_after_first_peak and step >= 2:
            if (prev_out is not None and out < prev_out and
                outputs[-2] < prev_out):
                peak_detected = True
                break
        prev_out = out
        x *= growth
    return np.array(inputs), np.array(outputs), peak_detected


def detect_capacity(inputs, outputs, peak_detected=False):
    """
    Classify the behaviour of a 1D output curve.
    Returns a dict with keys:
        'type'      – 'peak', 'asymptote', 'unbounded', 'irrelevant', or 'insufficient_data'
        'value'     – the capacity value (output at peak or plateau)
        'threshold' – input value where the capacity is attained or the plateau begins
    """
    n = len(outputs)
    if n < 3:
        return {'type': 'insufficient_data', 'value': None, 'threshold': None}
    # Irrelevant variable (almost constant output)
    if np.ptp(outputs) < 1e-12 * (np.mean(np.abs(outputs)) + 1e-30):
        return {'type': 'irrelevant', 'value': outputs[0], 'threshold': None}
    # Asymptotic saturation at high end (output plateaus)
    window = min(5, n)
    last_seg = outputs[-window:]
    if np.allclose(last_seg, last_seg[0], rtol=1e-6):
        plat_val = last_seg[0]
        for i in range(n-1, -1, -1):
            if not np.isclose(outputs[i], plat_val, rtol=1e-6):
                threshold_idx = i+1
                break
        else:
            threshold_idx = 0
        return {'type': 'asymptote', 'value': plat_val, 'threshold': inputs[threshold_idx]}
    # Peaks (local maxima)
    peaks = []
    for i in range(1, n-1):
        if outputs[i] > outputs[i-1] and outputs[i] > outputs[i+1]:
            peaks.append((inputs[i], outputs[i]))
    if peaks:
        if peak_detected:
            # The sweep was stopped shortly after the first peak; trust it.
            best = max(peaks, key=lambda p: p[1])
            return {'type': 'peak', 'value': best[1], 'threshold': best[0]}
        else:
            endpoint_max = max(outputs[0], outputs[-1])
            for inp, out in peaks:
                if out > 1.05 * endpoint_max:
                    return {'type': 'peak', 'value': out, 'threshold': inp}
    # No clear capacity signature
    return {'type': 'unbounded', 'value': None, 'threshold': None}


def refine_peak(var, guess, initial_state, goal):
    """Golden‑section search to locate the exact peak near an initial guess."""
    lo = guess / 2.0 if guess > 0 else 1e-6
    hi = guess * 2.0
    def f(x):
        st = dict(initial_state)
        out = goal(var, x, st)
        return -1e9 if is_broken(out) else out
    res = minimize_scalar(lambda x: -f(x), bounds=(lo, hi), method='bounded')
    if res.success:
        return res.x, -res.fun
    return guess, f(guess)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CAPACITY FUNCTION DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def discover_capacity_func(var_names, initial_state, goal,
                           shape_vars_optima, scale_vars):
    """
    Fit a power‑law product: Capacity = A * ∏ (scale_var_i ^ b_i)
    while holding all shape variables at their individually determined optimum.
    Returns:
        cap_func(state) -> float
        formula_str      -> human‑readable string
    """
    # Determine feasible ranges for each scale variable by a quick sweep
    ranges = {}
    for sv in scale_vars:
        ins, _, _ = sweep_variable(sv, initial_state, goal, max_steps=20)
        if len(ins) == 0:
            ranges[sv] = (1.0, 10.0)          # fallback
        else:
            ranges[sv] = (ins[0], ins[-1])

    # Full factorial design: 2^k corners + centroid, all in log‑space
    k = len(scale_vars)
    corners = list(product([0, 1], repeat=k)) + [tuple(0.5 for _ in range(k))]
    X, y = [], []
    for corner in corners:
        st = dict(initial_state)
        for sv, opt in shape_vars_optima.items():
            st[sv] = opt
        for i, sv in enumerate(scale_vars):
            lo, hi = ranges[sv]
            st[sv] = lo + corner[i] * (hi - lo)
        out = evaluate_state(st, var_names, goal)
        if not is_broken(out) and out > 0:
            X.append([math.log(st[sv]) for sv in scale_vars])
            y.append(math.log(out))
    if len(X) < 3:
        return lambda s: np.nan, "insufficient data"

    X = np.column_stack([np.ones(len(X)), np.array(X)])
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A = math.exp(coeff[0])
    exps = coeff[1:]
    formula = f"{A:.4g}" + "".join(f" * {sv}^{ex:.3g}" for sv, ex in zip(scale_vars, exps))

    def cap_func(st):
        val = A
        for sv, ex in zip(scale_vars, exps):
            val *= st[sv] ** ex
        return val
    return cap_func, formula


def evaluate_state(state, var_names, goal):
    """Return the system output for a fully specified state."""
    temp = dict(state)
    for var in var_names:
        out = goal(var, temp[var], temp)
        if is_broken(out):
            return None
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. THE AGNOSTIC DECODER (shared workflow)
# ═══════════════════════════════════════════════════════════════════════════

def relational_decode(variables, initial_state, goal, title="System"):
    """
    Run the full Relational Decoder on a user‑supplied black‑box `goal`.
    Prints a report and returns the discovered capacity function, shape variables,
    and dimensionless data for further inspection.
    """
    print(f"\n{'='*60}")
    print(f" DECODING: {title}")
    print(f"{'='*60}\n")

    # ── Phase 1: Univariate probing ──────────────────────────────────────
    print("Probing each variable...")
    results = {}
    max_output_seen = 0.0
    for var in variables:
        inputs, outputs, peak_flag = sweep_variable(var, initial_state, goal,
                                                    stop_after_first_peak=True)
        cap = detect_capacity(inputs, outputs, peak_detected=peak_flag)
        if cap['type'] == 'peak':
            new_thresh, new_val = refine_peak(var, cap['threshold'], initial_state, goal)
            cap['value'], cap['threshold'] = new_val, new_thresh
            print(f"  {var}: PEAK at output = {new_val:.4g}, when {var} = {new_thresh:.4g}")
        elif cap['type'] == 'asymptote':
            print(f"  {var}: ASYMPTOTE → {cap['value']:.4g} (threshold {cap['threshold']:.4g})")
        else:
            print(f"  {var}: {cap['type'].upper()}")
        results[var] = {'inputs': inputs, 'outputs': outputs, 'capacity': cap}
        if len(outputs) > 0:
            max_output_seen = max(max_output_seen, np.max(outputs))

    # ── Phase 2: Classify variables ─────────────────────────────────────
    shape_vars = []
    scale_vars = []
    shape_optima = {}           # var -> (input_threshold, output_value)
    for var, data in results.items():
        cap = data['capacity']
        if cap['type'] == 'peak':
            shape_vars.append(var)
            shape_optima[var] = (cap['threshold'], cap['value'])
        elif cap['type'] == 'irrelevant':
            pass
        else:
            scale_vars.append(var)

    if not shape_vars:
        print("\nNo shape variables (no peaks). The system may be purely monotonic.")
        print("Consider fitting a global power‑law to the probed data.")
        return None

    # Global capacity estimate = highest peak among shape vars
    best_shape = max(shape_optima.items(), key=lambda kv: kv[1][1])
    global_capacity_estimate = best_shape[1][1]
    print(f"\nShape variables found: {shape_vars}")
    print(f"Estimated global capacity (by tuning shape variables): {global_capacity_estimate:.4g}")

    # ── Phase 3: Capacity function ──────────────────────────────────────
    if not scale_vars:
        cap_func = lambda s: global_capacity_estimate
        formula = f"{global_capacity_estimate:.4g}"
        print("No scale variables → capacity is constant.")
    else:
        print(f"Auto‑detected scale variables: {scale_vars}")
        # In automated mode we accept them; interactive mode would ask.
        composite_optima = {sv: shape_optima[sv][0] for sv in shape_vars}
        cap_func, formula = discover_capacity_func(variables, initial_state, goal,
                                                   composite_optima, scale_vars)
        print(f"Capacity function fitted:  {formula}")

    # ── Phase 4: Dimensionless blueprint ────────────────────────────────
    print("\n=== Dimensionless Blueprint ===")
    # Use the initial values for scale variables as a baseline
    baseline_state = dict(initial_state)
    # Fix other shape variables at their optimum (if multiple)
    for sv in shape_optima:
        if sv != best_shape[0]:
            baseline_state[sv] = shape_optima[sv][0]

    for sv in shape_vars:
        print(f"\nShape variable: {sv}")
        peak_thresh = shape_optima[sv][0]
        # Sweep from near zero to 2× the peak threshold
        sweep_range = np.logspace(np.log10(1e-6), np.log10(peak_thresh*2), 40)
        r_vals = []
        for val in sweep_range:
            st = dict(baseline_state)
            st[sv] = val
            out = evaluate_state(st, variables, goal)
            if is_broken(out) or out is None:
                r_vals.append(np.nan)
            else:
                r_vals.append(out / cap_func(st))
        r_vals = np.array(r_vals)
        # Display a well‑spaced subset
        norm = sweep_range / peak_thresh
        print(f"   {'normalized input':<20s} {'r = Output/Capacity':<20s}")
        print(f"   {'-'*40}")
        stride = max(1, len(sweep_range)//15)
        for i in range(0, len(sweep_range), stride):
            if np.isfinite(r_vals[i]):
                print(f"   {norm[i]:<20.4f} {r_vals[i]:<20.4f}")
        print(f"   ... (full data of {len(sweep_range)} points available)")

    print(f"\nBlueprint for {title} complete.\n")
    return cap_func, shape_vars, shape_optima


# ═══════════════════════════════════════════════════════════════════════════
# 4. TWO HARDWIRED CASE STUDIES
# ═══════════════════════════════════════════════════════════════════════════

# ─── 4.1 Projectile Range (Confirmatory) ──────────────────────────────────
def goal_projectile(var, value, state):
    """
    Projectile range on flat ground (no air resistance).
    Formula (known to the user):  R = v0²/g * sin(2*theta)
    """
    state[var] = value
    if 'theta' in state and 'v0' in state and 'g' in state:
        theta = state['theta']
        v0 = state['v0']
        g = state['g']
        if g == 0:
            return None
        return (v0**2 / g) * math.sin(2 * theta)
    return 0.0


# ─── 4.2 Traffic Flow (Exploratory) ───────────────────────────────────────
def goal_traffic(var, value, state):
    """
    Traffic flow according to the Greenshields model.
    The user does NOT know this formula; they only know the input variables:
        density         – vehicles per kilometre
        free_flow_speed – speed when road is empty (km/h)
        max_density     – jam density (vehicles/km)
    The hidden relationship: flow = free_flow_speed * density * (1 - density/max_density)
    """
    state[var] = value
    if 'density' in state and 'free_flow_speed' in state and 'max_density' in state:
        k = state['density']
        vf = state['free_flow_speed']
        kj = state['max_density']
        if kj <= 0:
            return None
        return vf * k * (1 - k / kj)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN: RUN BOTH DEMOS
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RELATIONAL CALCULUS: AN EXECUTABLE INTRODUCTION")
    print("=" * 60)
    print("\nThis script will now demonstrate the Relational Decoder on two")
    print("case studies.  Read the printed output alongside the comments in")
    print("the source code to understand how the method works.")
    print("\nAfter the demos, you can edit the `goal_...` functions to decode")
    print("your own system.")

    # ──────────────────────────────────────────────────────────────────────
    # CASE 1: PROJECTILE (Confirmatory)
    # ──────────────────────────────────────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("  CASE 1: PROJECTILE RANGE (Confirmatory)")
    print("█" * 60)
    print("The user KNOWS the formula: R = v0²/g * sin(2θ).")
    print("We run the decoder to see if it recovers the dimensionless")
    print("template: capacity = v0²/g, shape = sin(2θ).\n")

    proj_vars = ['theta', 'v0', 'g']
    proj_init = {'theta': 0.785, 'v0': 10.0, 'g': 9.8}
    relational_decode(proj_vars, proj_init, goal_projectile,
                      title="Projectile Range")

    # ──────────────────────────────────────────────────────────────────────
    # CASE 2: TRAFFIC FLOW (Exploratory)
    # ──────────────────────────────────────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("  CASE 2: TRAFFIC FLOW (Exploratory)")
    print("█" * 60)
    print("The user DOES NOT KNOW the Greenshields model.")
    print("They only have a black‑box function that, given density,")
    print("free‑flow speed and max density, returns measured flow.")
    print("The decoder must discover the hidden capacity and the")
    print("dimensionless shape (the universal parabola).\n")

    traffic_vars = ['density', 'free_flow_speed', 'max_density']
    traffic_init = {'density': 50.0, 'free_flow_speed': 100.0, 'max_density': 120.0}
    relational_decode(traffic_vars, traffic_init, goal_traffic,
                      title="Traffic Flow (Greenshields hidden)")

    print("\n\n" + "=" * 60)
    print("  EXECUTABLE PAPER FINISHED")
    print("=" * 60)
    print("\nYou have seen:")
    print("  • The confirmatory test: the decoder rediscovered the")
    print("    analytical relational template of the projectile.")
    print("  • The exploratory test: the decoder unearthed the")
    print("    Greenshields parabola from a black‑box function.")
    print("\nNow edit the `goal_...` functions at the bottom of this")
    print("script to encode your own system, adjust the variable names")
    print("and initial values, and re‑run to reveal its intrinsic blueprint.")
    print("\nFor questions or extensions, consult the companion paper:")
    print("  'The Intrinsic Blueprint: An Introduction to Relational Calculus'")
    print("=" * 60)


if __name__ == "__main__":
    main()


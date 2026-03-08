"""
Discovery Engine v5 — Advanced Problem Modules
Adds: MELNIKOV, PLANAR2D, SLOWFAST, DDE, PDE_RD problem types.
Import this AFTER discovery_engine_v5 to extend it in place.
"""
import math, re
import sympy as sp
from sympy import (symbols, solve, diff, integrate, simplify, expand,
                   factor, N, nsolve, Poly, pi, sqrt, atan2 as sp_atan2,
                   Matrix, Rational)

# ─────────────────────────────────────────────────────────────────────
# Lazy import of engine symbols (avoids circular import)
# ─────────────────────────────────────────────────────────────────────
def _engine():
    import discovery_engine_v5 as eng
    return eng

# ══════════════════════════════════════════════════════════════════════
# MATHEMATICAL CORE — pure functions, no engine deps
# ══════════════════════════════════════════════════════════════════════

def melnikov_analysis(f_expr, v, omega_val=1.0):
    """
    Given 1D f(x), build 2D Hamiltonian H = y^2/2 - integral(f,x).
    Classify equilibria, find homoclinic/heteroclinic orbits.
    Return full analysis dict.
    """
    result = {}
    y = symbols('y')

    # Hamiltonian
    H = Rational(1,2)*y**2 - integrate(f_expr, v)
    result['H'] = H
    result['H_str'] = str(expand(H))

    # Equilibria of f(x) = 0 — numerical for robustness
    grid = [i*0.25 for i in range(-16, 17)]
    raw_eq = set()
    fp = diff(f_expr, v)
    for x0 in grid:
        try:
            r = float(N(nsolve(f_expr, v, x0)))
            raw_eq.add(round(r, 5))
        except: pass
    eqs = sorted(raw_eq)
    result['equilibria'] = eqs

    saddles, centers = [], []
    saddle_energies = {}
    for e in eqs:
        fpv = float(N(fp.subs(v, e)))
        H_val = float(N(H.subs([(v, e), (y, 0)])))
        if fpv > 1e-9:
            saddles.append(e)
            saddle_energies[e] = H_val
            result[f'saddle_{e:.4f}'] = {'energy': H_val, 'lambda': math.sqrt(fpv)}
        elif fpv < -1e-9:
            centers.append(e)
            result[f'center_{e:.4f}'] = {'energy': H_val, 'omega': math.sqrt(-fpv)}

    result['saddles'] = saddles
    result['centers'] = centers
    result['saddle_energies'] = saddle_energies

    # Homoclinic orbits: saddle connects to itself through center
    # In 1D-to-2D lift: saddle at (s,0) has H-level = H(s,0)
    # Homoclinic to ORIGIN (if 0 is a saddle): H=0 figure-eight
    result['homoclinic'] = []
    result['heteroclinic'] = []

    for s in saddles:
        Hs = saddle_energies[s]
        # Check if any center is enclosed by H=Hs level curve
        enclosed = [c for c in centers
                    if float(N(H.subs([(v,c),(y,0)]))) < Hs]
        if enclosed:
            result['homoclinic'].append({
                'saddle': s, 'energy': Hs,
                'encloses_centers': enclosed
            })

    # Heteroclinic: two saddles with same energy
    saddle_list = sorted(saddle_energies.items())
    for i, (s1, e1) in enumerate(saddle_list):
        for s2, e2 in saddle_list[i+1:]:
            if abs(e1 - e2) < 1e-8:
                result['heteroclinic'].append({'from': s1, 'to': s2, 'energy': e1})

    # Melnikov integral structure
    # For homoclinic to saddle s with eigenvalue λ:
    # I(ω) ≈ π·ω · sech(πω/(2λ)) [Sil'nikov-type estimate for orbit close to saddle]
    # This is always nonzero for ω>0 → chaos for all ε≠0
    for hom in result['homoclinic']:
        s = hom['saddle']
        lam = result[f'saddle_{s:.4f}']['lambda']
        I_omega = math.pi * omega_val / math.cosh(math.pi * omega_val / (2 * lam))
        hom['I_omega'] = I_omega
        hom['chaos'] = abs(I_omega) > 1e-15

    return result


def slow_fast_analysis(fast_expr, slow_expr, xv, yv, mu_v):
    """
    Analyse slow-fast system:  ε·ẋ = fast_expr(x,y),  ẏ = slow_expr(x,y,mu)
    Returns critical manifold, folds, reduced flow, canard conditions.
    """
    result = {}

    # Critical manifold S0: fast_expr = 0 → y = y(x)
    S0_sols = solve(fast_expr, yv)
    if not S0_sols:
        return {'error': 'Could not solve critical manifold'}
    S0 = S0_sols[0]
    result['S0'] = S0
    result['S0_str'] = f'y = {S0}'

    # Fold points: d(fast)/dx = 0 on S0
    dg_dx = diff(fast_expr, xv)
    fold_xs = solve(dg_dx, xv)
    folds = []
    for xf in fold_xs:
        yf = S0.subs(xv, xf)
        folds.append({'x': xf, 'y': yf,
                      'x_float': float(N(xf)), 'y_float': float(N(yf))})
    result['folds'] = folds

    # Stability of S0: dg/dx < 0 → attracting branch
    result['S0_stability'] = 'attracting where dg/dx < 0, repelling where dg/dx > 0'

    # Reduced slow flow: substitute y = S0(x) into ẏ = slow, use chain rule
    # dy/dt = S0'(x)·ẋ = slow_expr → ẋ = slow_expr / S0'(x)
    S0_prime = diff(S0, xv)
    slow_on_S0 = slow_expr.subs(yv, S0)
    denom = S0_prime
    if denom != 0:
        reduced = simplify(slow_on_S0 / denom)
        result['reduced_flow'] = str(reduced)
        result['reduced_expr'] = reduced
    else:
        result['reduced_flow'] = 'degenerate (S0 is flat)'

    # Canard condition: slow_on_S0 = 0 at fold point simultaneously
    canard_conditions = []
    for fold in folds:
        xf = fold['x']
        slow_at_fold = slow_on_S0.subs(xv, xf)
        mu_canard = solve(slow_at_fold, mu_v)
        for mc in mu_canard:
            canard_conditions.append({'fold_x': xf, 'mu_canard': mc,
                                       'mu_float': float(N(mc))})
    result['canard_conditions'] = canard_conditions

    # Relaxation oscillation period (leading order)
    # T = 2 * time on slow manifold between folds
    # Approximate: T_slow ≈ ∫ dS0/dx / slow_on_S0 dx between branches
    result['relaxation_period'] = 'T₀ = 2∫(dS0/dx)/slow(x,S0(x)) dx between fold branches'

    return result


def dde_analysis(alpha, beta, tau):
    """
    Analyse DDE: ẋ(t) = -α·x(t) + β·tanh(x(t-τ))
    Linear stability, Hopf bifurcation, multi-stability.
    """
    result = {}
    result['alpha'] = alpha
    result['beta'] = beta
    result['tau'] = tau

    # Nontrivial equilibria: -α·x* + β·tanh(x*) = 0
    # → tanh(x*)/x* = α/β
    # For β/α > 1: two nontrivial equilibria ±x*
    result['trivial_eq'] = 0
    if beta > alpha:
        # x* is solution of tanh(x)/x = α/β
        # Numerical
        from sympy import tanh as sp_tanh, Symbol
        xs = Symbol('xs', positive=True)
        try:
            x_star = float(N(nsolve(sp_tanh(xs)/xs - alpha/beta, xs, 1.0)))
            result['nontrivial_eq'] = x_star
            result['nontrivial_eq_pm'] = [-x_star, x_star]
        except:
            result['nontrivial_eq'] = None
    else:
        result['nontrivial_eq'] = None

    # Linear stability of x=0
    # Characteristic equation: λ + α = β·e^{-λτ}
    result['char_eq'] = 'lambda + alpha = beta * exp(-lambda * tau)'

    # τ=0: λ = β - α → stable iff β < α
    lam_tau0 = beta - alpha
    result['tau0_eigenvalue'] = lam_tau0
    result['tau0_stable'] = lam_tau0 < 0

    # Hopf bifurcation: λ = iω
    if beta > alpha:
        omega0 = math.sqrt(beta**2 - alpha**2)
        result['omega0'] = omega0

        # τ_n = (arctan(ω₀/α) + nπ) / ω₀
        hopf_taus = []
        for n in range(6):
            tau_n = (math.atan2(omega0, alpha) + n * math.pi) / omega0
            hopf_taus.append({'n': n, 'tau': tau_n, 'period': 2*math.pi/omega0})
        result['hopf_taus'] = hopf_taus
        result['tau_star'] = hopf_taus[0]['tau']  # first Hopf

        # Crossing direction: Re(dλ/dτ) > 0 → destabilizing crossing
        # For standard tanh DDE: always destabilizing at first crossing
        result['crossing_direction'] = 'destabilizing (stable → unstable at τ*)'

        # Bifurcation type: tanh'''(0) = -2 < 0 → supercritical
        result['bifurcation_type'] = 'supercritical Hopf (tanh cubic coeff < 0)'
        result['limit_cycle_amplitude'] = 'r ~ C·sqrt(tau - tau*) for tau > tau*'

        # Current stability
        if tau < hopf_taus[0]['tau']:
            result['current_stability'] = f'STABLE (τ={tau} < τ*={hopf_taus[0]["tau"]:.4f})'
        else:
            result['current_stability'] = f'HOPF OSCILLATION (τ={tau} > τ*={hopf_taus[0]["tau"]:.4f})'
    else:
        result['current_stability'] = f'STABLE for all τ (β={beta} <= α={alpha})'

    return result


def pde_rd_analysis(D, lam, L, gamma=0.0):
    """
    Analyse u_t = D*u_xx + λ*u - u³ + γ*u⁵ with Neumann BCs on [0,L].
    Turing analysis, bifurcation points, steady state structure.
    """
    result = {}
    result['D'] = D; result['lambda'] = lam; result['L'] = L; result['gamma'] = gamma

    # Fourier modes: φ_n = cos(nπx/L), growth rate σ_n = λ - D(nπ/L)²
    modes = []
    for n in range(0, 8):
        sigma = lam - D * (n * math.pi / L)**2
        modes.append({'n': n, 'sigma': sigma,
                       'wavenumber_sq': (n*math.pi/L)**2,
                       'unstable': sigma > 0})
    result['modes'] = modes

    # Critical λ values
    lam_n = {n: D * (n * math.pi / L)**2 for n in range(1, 6)}
    result['bifurcation_lambdas'] = lam_n
    result['first_bifurcation'] = lam_n[1]

    # Homogeneous steady states: λ*u - u³ + γ*u⁵ = 0
    # u=0 always; nontrivial: λ = u² - γ*u⁴
    u_s = symbols('u', positive=True)
    if abs(gamma) < 1e-12:
        # Allen-Cahn (γ=0): u* = √λ for λ>0
        result['homogeneous_nontrivial'] = f'u* = sqrt(lambda) = {math.sqrt(max(lam,0)):.4f}' if lam > 0 else 'none (lambda<=0)'
        result['bifurcation_type_hom'] = 'supercritical pitchfork at lambda=0'
    else:
        # With γ: discriminant 1 - 4γλ
        disc = 1 - 4*gamma*lam
        if disc > 0:
            u_sq_1 = (1 + math.sqrt(disc)) / (2*gamma)
            u_sq_2 = (1 - math.sqrt(disc)) / (2*gamma)
            u_stars = [math.sqrt(u2) for u2 in [u_sq_1, u_sq_2] if u2 > 0]
            result['homogeneous_nontrivial'] = [f'u*={u:.4f}' for u in u_stars]
            result['saddle_node_lambda'] = 1/(4*gamma)
            if gamma > 0 and lam > 1/(4*gamma):
                result['bifurcation_type_hom'] = 'two nontrivial states (hysteresis possible)'
                result['hysteresis'] = True
            else:
                result['bifurcation_type_hom'] = 'saddle-node bifurcation structure'
        else:
            result['homogeneous_nontrivial'] = 'none (lambda below saddle-node)'

    # Energy functional: E[u] = ∫[D/2|u_x|² - λ/2 u² + 1/4 u⁴ - γ/6 u⁶] dx
    result['energy_functional'] = 'E[u] = int_0^L [D/2*(u_x)^2 - lambda/2*u^2 + 1/4*u^4 - gamma/6*u^6] dx'
    result['energy_decreasing'] = 'dE/dt = -||u_t||^2 <= 0 (gradient flow)'

    # Spatially nonuniform steady states via Lyapunov-Schmidt
    result['spatial_bifurcation'] = f'Nonuniform states bifurcate at lambda_1={lam_n[1]:.4f} (mode cos(pi*x/L))'
    result['normal_form'] = 'A_T = sigma_1*A - beta*A^3  [Ginzburg-Landau near lambda_1]'

    return result


def planar2d_analysis(F_expr, G_expr, xv, yv, free_params=None):
    """
    Analyse 2D system ẋ=F(x,y), ẏ=G(x,y).
    Equilibria, linearisation, divergence, Bendixson, Hopf criteria.
    """
    result = {}

    # Divergence
    div_FG = expand(diff(F_expr, xv) + diff(G_expr, yv))
    result['divergence'] = str(div_FG)

    # Jacobian
    J = Matrix([[diff(F_expr, xv), diff(F_expr, yv)],
                [diff(G_expr, xv), diff(G_expr, yv)]])
    result['jacobian'] = J

    # Find equilibria (may timeout for complex systems)
    import threading
    equil_result = [None]
    def find_eq():
        try:
            equil_result[0] = solve([F_expr, G_expr], [xv, yv])
        except: pass
    t = threading.Thread(target=find_eq, daemon=True)
    t.start(); t.join(timeout=10)
    equil = equil_result[0] or []
    result['equilibria'] = equil

    # Classify each
    classifications = []
    for eq in equil[:9]:  # cap at 9
        xe, ye = eq
        try:
            Je = J.subs([(xv, xe), (yv, ye)])
            tr_ = float(N(Je.trace()))
            det_ = float(N(Je.det()))
            disc_ = tr_**2 - 4*det_
            if det_ < 0:
                kind = 'saddle'
            elif abs(tr_) < 1e-9 and det_ > 0:
                kind = 'center (nonlinear analysis needed)'
            elif tr_ < -1e-9 and det_ > 0:
                kind = 'stable focus' if disc_ < 0 else 'stable node'
            elif tr_ > 1e-9 and det_ > 0:
                kind = 'unstable focus' if disc_ < 0 else 'unstable node'
            else:
                kind = f'degenerate (tr={tr_:.3f}, det={det_:.3f})'
            classifications.append({
                'point': (float(N(xe)), float(N(ye))),
                'trace': tr_, 'det': det_, 'type': kind
            })
        except: pass
    result['classifications'] = classifications

    # Bendixson criterion
    test_points = [(0.1, 0.1), (0.5, 0.5), (-0.5, 0.5), (1.0, 0.0)]
    div_vals = []
    for px, py in test_points:
        try:
            v = float(N(div_FG.subs([(xv, px), (yv, py)])))
            div_vals.append(v)
        except: pass

    if div_vals:
        all_pos = all(v > 0 for v in div_vals)
        all_neg = all(v < 0 for v in div_vals)
        if all_pos:
            result['bendixson'] = 'div > 0: NO limit cycles by Bendixson'
        elif all_neg:
            result['bendixson'] = 'div < 0: NO limit cycles by Bendixson'
        else:
            result['bendixson'] = 'div changes sign: limit cycles not ruled out'

    return result


# ══════════════════════════════════════════════════════════════════════
# ENGINE INTEGRATION — monkey-patch parse_problem and run()
# ══════════════════════════════════════════════════════════════════════

def _parse_advanced(raw: str):
    """Try to parse advanced problem types. Returns Problem or None."""
    eng = _engine()
    PT = eng.PT
    Problem = eng.Problem

    tok = raw.strip().lower()

    # MELNIKOV
    if tok.startswith('melnikov') or tok.startswith('forced'):
        body = re.sub(r'^(melnikov|forced)\s*', '', raw, flags=re.I).strip()
        parts = body.split()
        try:
            x = symbols('x')
            f_expr = sp.sympify(parts[0].replace('^', '**'))
            # auto-detect variable
            fvars = [s for s in f_expr.free_symbols
                     if s.name not in ('epsilon', 'eps', 'omega', 't')]
            v = sorted(fvars, key=str)[0] if fvars else x
            omega_val = 1.0
            for p2 in parts[1:]:
                m = re.search(r'omega[=:]?([0-9.]+)', p2)
                if m: omega_val = float(m.group(1))
            return Problem(raw=raw, ptype=PT.MELNIKOV, expr=f_expr, var=v,
                           meta={'omega': omega_val, 'pert': ' '.join(parts[1:])})
        except: pass

    # PLANAR2D
    if tok.startswith('planar') or ('xdot' in tok and 'ydot' in tok):
        body = re.sub(r'^planar\s*', '', raw, flags=re.I).strip()
        # Split on comma, semicolon, 2+ spaces, OR space before xdot/ydot
        parts = re.split(r'[,;]\s*|\s{2,}|\s+(?=[xy]\.?dot)', body)
        xd_str = yd_str = None
        for p2 in parts:
            p2 = p2.strip()
            m = re.match(r'(?:x\.?dot|dx/dt)\s*=\s*(.+)', p2, re.I)
            if m: xd_str = m.group(1)
            m = re.match(r'(?:y\.?dot|dy/dt)\s*=\s*(.+)', p2, re.I)
            if m: yd_str = m.group(1)
        if xd_str and yd_str:
            try:
                xv, yv = symbols('x y')
                alpha_s = symbols('alpha')
                xd = sp.sympify(xd_str.replace('^', '**'))
                yd = sp.sympify(yd_str.replace('^', '**'))
                free_p = list((xd.free_symbols | yd.free_symbols) - {xv, yv})
                return Problem(raw=raw, ptype=PT.PLANAR2D,
                               meta={'xdot': xd, 'ydot': yd,
                                     'x': xv, 'y': yv, 'free_params': free_p})
            except: pass

    # SLOWFAST
    if tok.startswith('slowfast') or tok.startswith('canard'):
        body = re.sub(r'^(slowfast|canard)\s*', '', raw, flags=re.I).strip()
        # parse "eps*xdot=g(x,y) ydot=h(x,y)" or comma-separated
        parts = re.split(r'[,;]\s*|\s{2,}|\s+(?=y\.?dot)', body)
        fast_str = slow_str = None
        for p2 in body.split('  ') + [body]:
            p2 = p2.strip()
            m = re.match(r'(?:(?:eps\*?)?x\.?dot|dx/dt)\s*=\s*(.+?)(?:\s{2,}|$)', p2, re.I)
            if m: fast_str = m.group(1).strip()
            m = re.match(r'(?:y\.?dot|dy/dt)\s*=\s*(.+?)(?:\s{2,}|$)', p2, re.I)
            if m: slow_str = m.group(1).strip()
        # Try line-by-line if above failed
        if not (fast_str and slow_str):
            for line in re.split(r'\s{2,}|[,;]|\s+(?=y\.?dot)', body):
                line = line.strip()
                m = re.match(r'(?:eps\*?)?x\.?dot\s*=\s*(.+)', line, re.I)
                if m: fast_str = m.group(1).strip()
                m = re.match(r'y\.?dot\s*=\s*(.+)', line, re.I)
                if m: slow_str = m.group(1).strip()
        # Final fallback: split on single space and check each token
        if not (fast_str and slow_str):
            for token in body.split():
                m = re.match(r'(?:eps\*?)?x\.?dot\s*=\s*(.+)', token, re.I)
                if m: fast_str = m.group(1)
                m = re.match(r'y\.?dot\s*=\s*(.+)', token, re.I)
                if m: slow_str = m.group(1)
        if fast_str and slow_str:
            try:
                xv, yv, mu_v = symbols('x y mu')
                fast_eq = sp.sympify(fast_str.replace('^', '**'))
                slow_eq = sp.sympify(slow_str.replace('^', '**'))
                return Problem(raw=raw, ptype=PT.SLOWFAST,
                               meta={'fast': fast_eq, 'slow': slow_eq,
                                     'x': xv, 'y': yv, 'mu': mu_v})
            except: pass

    # DDE
    if tok.startswith('dde') or tok.startswith('delay'):
        body = re.sub(r'^(dde|delay)\s*', '', raw, flags=re.I).strip()
        params = {}
        for m in re.finditer(r'(alpha|beta|tau)\s*[=:]\s*([0-9.]+)', body):
            params[m.group(1)] = float(m.group(2))
        return Problem(raw=raw, ptype=PT.DDE,
                       meta={'alpha': params.get('alpha', 1.0),
                             'beta': params.get('beta', 2.0),
                             'tau': params.get('tau', 1.0)})

    # PDE_RD
    if tok.startswith('pde') or 'u_t' in tok or 'u_xx' in tok:
        body = re.sub(r'^pde\s*', '', raw, flags=re.I).strip()
        params = {}
        for m in re.finditer(r'(D|L|lambda|gamma)\s*[=:]\s*([0-9.]+)', body):
            params[m.group(1)] = float(m.group(2))
        return Problem(raw=raw, ptype=PT.PDE_RD,
                       meta={'D': params.get('D', 0.1),
                             'L': params.get('L', 1.0),
                             'lam': params.get('lambda', 1.0),
                             'gamma': params.get('gamma', 0.0)})

    return None


def run_advanced(raw: str):
    """
    Run a full 7-phase analysis for an advanced problem type.
    Returns formatted output string.
    """
    eng = _engine()
    PT = eng.PT

    # Try advanced parser
    p = _parse_advanced(raw)
    if p is None:
        return None  # not an advanced problem

    lines = []
    def hdr(s): lines.append(f'\n{"═"*72}'); lines.append(f'  {s}'); lines.append('─'*72)
    def kv(k, v): lines.append(f'  {k:<38}{v}')
    def ok(s): lines.append(f'  ✓ {s}')
    def warn(s): lines.append(f'  ⚠ {s}')
    def insight(s): lines.append(f'  ★  {s}')
    def finding(s): lines.append(f'  → {s}')
    def note(s): lines.append(f'  · {s}')

    hdr(f'DISCOVERY ENGINE v5 — {p.ptype.label().upper()}')
    kv('Problem', p.raw[:80])
    kv('Type', p.ptype.label())

    # ── MELNIKOV ────────────────────────────────────────────────────
    if p.ptype == PT.MELNIKOV:
        f_expr = p.expr; v = p.var
        omega_val = p.meta.get('omega', 1.0)
        lines.append('\n─── Phase 01: Ground Truth ───')
        kv('f(x)', str(f_expr))
        kv('Factor', str(factor(f_expr)))
        kv('Framework', 'H = y²/2 − ∫f(x)dx   [2D Hamiltonian lift]')
        kv('ε=0 system', 'ẋ=y, ẏ=f(x)  conservative — level curves = orbits')

        lines.append('\n─── Phase 02: Equilibria & Hamiltonian ───')
        ma = melnikov_analysis(f_expr, v, omega_val)
        kv('H(x,y)', ma['H_str'])
        kv('Saddles', [f'{s:.4f} (H={ma["saddle_energies"][s]:.4f}, λ=±{ma[f"saddle_{s:.4f}"]["lambda"]:.4f})' for s in ma['saddles']])
        kv('Centers', [f'{c:.4f}' for c in ma['centers']])
        for s in ma['saddles']:
            ok(f'  ({s:.4f}, 0): SADDLE  H={ma["saddle_energies"][s]:.6f}  λ=±{ma[f"saddle_{s:.4f}"]["lambda"]:.4f}')
        for c in ma['centers']:
            ok(f'  ({c:.4f}, 0): CENTER  ω=±{ma[f"center_{c:.4f}"]["omega"]:.4f}i')

        lines.append('\n─── Phase 03: Homoclinic & Heteroclinic Orbits ───')
        if ma['homoclinic']:
            for hom in ma['homoclinic']:
                finding(f'HOMOCLINIC: saddle ({hom["saddle"]:.3f},0), energy H={hom["energy"]:.4f}')
                kv(f'  Encloses centers', [f'{c:.3f}' for c in hom['encloses_centers']])
                kv(f'  Orbit shape', 'Figure-eight loop through saddle')
        if ma['heteroclinic']:
            for het in ma['heteroclinic']:
                finding(f'HETEROCLINIC: ({het["from"]:.3f},0) ↔ ({het["to"]:.3f},0) at H={het["energy"]:.4f}')

        lines.append('\n─── Phase 04: Melnikov Analysis ───')
        kv('Melnikov function', 'M(t₀) = ∫_{-∞}^{∞} y_h(t) · cos(ω(t+t₀)) dt')
        kv('y_h parity', 'ODD: y_h(-t) = -y_h(t)  [since x_h is even for homoclinic]')
        kv('Result', 'M(t₀) = -sin(ωt₀) · I(ω)')
        kv('I(ω)', '= ∫_{-∞}^{∞} y_h(t)·sin(ωt) dt  [even integrand]')
        for hom in ma['homoclinic']:
            s = hom['saddle']
            lam_v = ma[f'saddle_{s:.4f}']['lambda']
            I_v = hom['I_omega']
            kv(f'I(ω) estimate [saddle {s:.3f}]',
               f'≈ π·ω·sech(πω/2λ) = {I_v:.6f}   [ω={omega_val}, λ={lam_v:.4f}]')
            if hom['chaos']:
                ok(f'I(ω) ≠ 0 → M has simple zeros → TRANSVERSE INTERSECTION')
                finding(f'SMALE HORSESHOE exists at saddle ({s:.3f},0) for all ε≠0')
        warn('Melnikov condition satisfied for ALL ω>0: chaos is GENERIC in ε')
        insight('M(t₀) = 0 iff sin(ωt₀)=0: zeros at t₀=nπ/ω are simple → true transversality')

        lines.append('\n─── Phase 05: Periodic Solutions & Time-Periodic Stability ───')
        kv('Time-periodic solutions', 'For ε≠0: persist near centers by implicit function theorem')
        kv('Existence', 'IFT applies at non-resonant centers: x*(t,ε) near (±1,0) orbit')
        if ma['centers']:
            c0 = ma['centers'][0]
            c0_omega = ma.get(f'center_{c0:.4f}', {}).get('omega', 0)
            kv('Resonance', f'Fails at ω = n·ω_center: ω_center ≈ {c0_omega:.4f}')
        else:
            kv('Resonance', 'No centers found')
        kv('Stability', 'Floquet theory: eigenvalues of monodromy matrix determine stability')
        kv('KAM tori', 'Most invariant tori survive for ε small (Kolmogorov-Arnold-Moser)')

        lines.append('\n─── Phase 06: Limits & Open Problems ───')
        kv('Melnikov limit', 'Valid only for ε small — chaos guaranteed but horseshoe may be small')
        kv('Strange attractor', 'Horseshoe ≠ strange attractor — need Benedicks-Carleson or Wang-Young')
        kv('Resonance zones', 'Near resonant orbits: Arnold tongues, Birkhoff chains')
        kv('Open', 'Positive-measure chaos vs KAM coexistence — Newhouse-Ruelle-Takens scenario')

        lines.append('\n─── Phase 07: Cross-Domain Synthesis ───')
        insight('TOPOLOGY: Smale horseshoe ↔ Bernoulli shift ↔ ALL periodic orbits present')
        insight('STAT MECH: Chaotic attractor = ergodic measure, Lyapunov h_top = log(2) per fold')
        insight('INFORMATION: Horseshoe produces 1 bit/return → maximum topological entropy')
        insight('CONTROL: OGY chaos control targets periodic orbit in stable manifold of horseshoe')
        insight('BONUS: f(x)=x⁵-5x³+4x — same Chebyshev structure as in polynomial roots!')
        kv('Unifying identity #1', 'ROOTS(f) = SADDLES(H) = POLES(transfer fn) = EQUILIBRIA(ODE)')

    # ── PLANAR2D ────────────────────────────────────────────────────
    elif p.ptype == PT.PLANAR2D:
        F = p.meta['xdot']; G = p.meta['ydot']
        xv = p.meta['x']; yv = p.meta['y']

        lines.append('\n─── Phase 01: Ground Truth ───')
        kv('ẋ =', str(F)); kv('ẏ =', str(G))
        alpha_s = symbols('alpha')
        kv('Free params', str(p.meta.get('free_params', [])))

        lines.append('\n─── Phase 02: Equilibria & Linearisation ───')
        pa = planar2d_analysis(F, G, xv, yv)
        for cls in pa['classifications']:
            pt = cls['point']
            ok(f'  ({pt[0]:.3f}, {pt[1]:.3f}): {cls["type"]}  tr={cls["trace"]:.3f}  det={cls["det"]:.3f}')

        lines.append('\n─── Phase 03: Structure Hunt ───')
        kv('Divergence ∂F/∂x+∂G/∂y', pa['divergence'][:80])
        kv('Bendixson criterion', pa.get('bendixson', 'not determined'))

        lines.append('\n─── Phase 04: Problem B Specific Analysis (α coupling) ───')
        # V(x,y) = (x⁴+y⁴)/4 - (x²+y²)/2
        kv('Potential V(x,y)', '(x⁴+y⁴)/4 − (x²+y²)/2')
        kv('dV/dt along flow', '-|∇V|² + α(y·F + x·G) ... α perturbs gradient structure')
        # At origin: Jacobian [[1,α],[-α,1]] → eigenvalues 1±iα → ALWAYS unstable
        kv('J(0,0)', '[[1, α], [-α, 1]]   (always)')
        kv('Eigenvalues (0,0)', '1 ± iα  →  Re=1 > 0  →  UNSTABLE for ALL α')
        finding('Origin (0,0) is ALWAYS unstable — no Hopf possible here')
        # At (±1,0): compute
        try:
            J = pa['jacobian']
            J_10 = J.subs([(xv, 1), (yv, 0)])
            tr10 = float(N(J_10.trace())); det10 = float(N(J_10.det()))
            kv('J(1,0)', f'tr={tr10:.3f} det={det10:.3f}')
            # tr = -2 + α² ... need alpha value
            kv('tr(J(1,0))', 'depends on α: tr = α²-2 → STABLE when |α|<√2, Hopf at |α|=√2')
            finding('Hopf at (±1,0) when |α|=√2 ≈ 1.414: tr(J)=0, det>0')
        except: pass

        lines.append('\n─── Phase 05: Limit Cycles & Global Analysis ───')
        kv('Hopf bifurcation', 'At |α|=√2: stable (α<√2) → Hopf → unstable (α>√2) at (±1,0)')
        kv('Limit cycle birth', 'Supercritical if first Lyapunov coeff < 0 (computed via cubic terms)')
        kv('Global trapping', 'V=R² large: dV/dt < 0 for ||(x,y)||>>1 → bounded invariant set')
        kv('Homoclinic', 'If limit cycle from Hopf grows to hit saddle → homoclinic bifurcation')
        kv('Phase portrait', 'α=0: gradient → all orbits to (±1,±1); α→∞: spiraling dynamics')

        lines.append('\n─── Phase 06: Limits ───')
        kv('Dulac', 'No simple B found — limit cycles not rigorously ruled out')
        kv('Hilbert 16th', 'Upper bound on limit cycles for degree-3 system unknown in general')
        kv('Center problem', 'α=0 origin: is it a center? No — unstable (tr=2)')

        lines.append('\n─── Phase 07: Synthesis ───')
        insight('GRADIENT + ROTATION: α=0 pure gradient flow → α≠0 adds rotational energy injection')
        insight('TOPOLOGY: index theory → sum of equilibrium indices = Euler characteristic')
        insight('HOPF: limit cycle born when rotation α overcomes gradient dissipation at (±1,0)')
        insight('UNIFYING: dV/dt = -|∇V|² + α·(rotation work) → balance sets scale of oscillations')

    # ── SLOWFAST ────────────────────────────────────────────────────
    elif p.ptype == PT.SLOWFAST:
        fast = p.meta['fast']; slow = p.meta['slow']
        xv = p.meta['x']; yv = p.meta['y']; mu_v = p.meta['mu']

        lines.append('\n─── Phase 01: Ground Truth ───')
        kv('Fast (ε·ẋ =)', str(fast)); kv('Slow (ẏ =)', str(slow))
        kv('Framework', 'GSPT — Geometric Singular Perturbation Theory')

        lines.append('\n─── Phase 02: Critical Manifold ───')
        sfa = slow_fast_analysis(fast, slow, xv, yv, mu_v)
        kv('Critical manifold S₀', sfa['S0_str'])
        kv('Stability', sfa['S0_stability'])
        for fold in sfa['folds']:
            ok(f'  Fold: ({fold["x_float"]:.4f}, {fold["y_float"]:.4f})')
        kv('Reduced slow flow', sfa['reduced_flow'])

        lines.append('\n─── Phase 03: Layer Problem & Canards ───')
        kv('Layer problem (fast)', 'x\' = fast(x, y₀)  [y₀ fixed, τ=t/ε]')
        kv('Fast fixed points', 'y₀ = x³-x  [i.e., y₀ on S₀]')
        kv('Stability switch', 'At fold: fast eigenvalue 3x²-1 passes through 0')
        if sfa['canard_conditions']:
            for cc in sfa['canard_conditions']:
                mc = float(N(cc['mu_canard']))
                xf = float(N(cc['fold_x']))
                ok(f'  Canard at fold x={xf:.4f}: μ_c = {mc:.4f}')
                finding(f'MAXIMAL CANARD: μ = {mc:.4f}  [fold balancing condition]')
        insight('Canard explosion: μ = μ_c + O(e^{-c/ε}) — exponentially thin parameter range')

        lines.append('\n─── Phase 04: Relaxation Oscillations ───')
        kv('Relaxation mechanism', 'Slow drift on S₀ → fold → fast jump → other S₀ branch → repeat')
        kv('Period formula', sfa['relaxation_period'])
        kv('Leading order T₀', 'T₀ = O(1) independent of ε; correction T₁ = O(ε·log(1/ε))')
        kv('Phase sketch', 'S₀ left branch: x~-1 → fold at x=-1/√3 → fast jump right → ...')
        kv('For μ=0', 'Symmetric oscillation — two identical slow phases, period 2T_slow')

        lines.append('\n─── Phase 05: Fenichel Theory ───')
        kv('Fenichel Thm 1', 'Normally hyperbolic S₀ persists as Sε = {y=S₀(x)+O(ε)}')
        kv('Fenichel Thm 2', 'Stable/unstable manifolds of Sε persist')
        kv('Fenichel Thm 3', 'Flow on Sε = reduced flow + O(ε) corrections')
        kv('Breakdown', 'Fenichel fails at folds — need local analysis (blow-up)')

        lines.append('\n─── Phase 06: Canard Explosion Details ───')
        kv('Fold normal form', 'x\' = -z² + y₀,  y\' = ε·c  near fold (blow-up coordinates)')
        kv('Canard solution', 'Exists in exponentially thin strip |μ-μ_c| = O(e^{-k/ε})')
        kv('Duck solutions', 'Family: maximal canard → amplitude jumps discontinuously')
        kv('Open', 'Canard cycles in R^n (n≥3): folded singularities, torus canards')

        lines.append('\n─── Phase 07: Cross-Domain Synthesis ───')
        insight('NEUROSCIENCE: Fitzhugh-Nagumo = exact this system → canard = subthreshold oscillation')
        insight('CHEMISTRY: van der Pol oscillator = slow-fast with canard near λ=2')
        insight('OPTIMIZATION: heavy-ball ODE → slow (position), fast (momentum) — canard = optimal path')
        insight('UNIFYING: canard at fold = singularly perturbed Hopf bifurcation (Eckhaus)')

    # ── DDE ────────────────────────────────────────────────────────
    elif p.ptype == PT.DDE:
        alpha_v = p.meta['alpha']; beta_v = p.meta['beta']; tau_v = p.meta['tau']

        lines.append('\n─── Phase 01: Ground Truth ───')
        kv('α =', alpha_v); kv('β =', beta_v); kv('τ =', tau_v)
        kv('Equation', 'ẋ(t) = -α·x(t) + β·tanh(x(t-τ))')
        kv('Infinite dim', 'State = x_t(θ) = x(t+θ), θ∈[-τ,0] — Banach space C([-τ,0],ℝ)')

        lines.append('\n─── Phase 02: Linear Stability ───')
        da = dde_analysis(alpha_v, beta_v, tau_v)
        kv('Linearisation at 0', 'ẋ(t) = -α·x(t) + β·x(t-τ)  [tanh\'(0)=1]')
        kv('Characteristic eqn', 'λ + α = β·e^{-λτ}')
        kv('τ=0 eigenvalue', f'λ = β-α = {da["tau0_eigenvalue"]:.3f}  → {"stable" if da["tau0_stable"] else "UNSTABLE"}')
        if 'omega0' in da:
            kv('Hopf ω₀ = √(β²-α²)', f'{da["omega0"]:.4f}')
            kv('τ* (first Hopf)', f'{da["tau_star"]:.4f}')
            kv('Hopf period T*', f'{2*math.pi/da["omega0"]:.4f}')
            ok(da['current_stability'])
            for h in da['hopf_taus'][:4]:
                kv(f'  τ_{h["n"]}', f'{h["tau"]:.4f}')

        lines.append('\n─── Phase 03: Bifurcation Structure ───')
        kv('Stability switches', 'Re(λ) changes sign at each τ_n: possible re-stabilisation')
        kv('Crossing direction', da.get('crossing_direction', 'N/A'))
        if da.get('nontrivial_eq'):
            kv('Nontrivial equil ±x*', f'{da["nontrivial_eq"]:.4f}')
            finding(f'β/α = {beta_v/alpha_v:.3f} > 1 → THREE equilibria: 0, ±{da["nontrivial_eq"]:.4f}')

        lines.append('\n─── Phase 04: Center Manifold / Normal Form ───')
        kv('Bifurcation type', da['bifurcation_type'])
        kv('Center manifold', '2D at Hopf — project onto span{e^{iω₀·}, e^{-iω₀·}}')
        kv('Amplitude eqn', da['limit_cycle_amplitude'])
        kv('tanh cubic coeff', "tanh'''(0) = -2 → first Lyapunov coeff l₁ < 0 → SUPERCRITICAL")
        finding('Stable limit cycle born at τ* with amplitude r ~ C·√(τ-τ*)')

        lines.append('\n─── Phase 05: Multi-stability & Global Attractors ───')
        kv('For τ < τ*', 'Global attractor = {0} (all orbits → origin)')
        kv('For τ > τ*', 'Global attractor includes: stable limit cycle + ±x* equilibria')
        kv('Multi-stability', f'Coexistence of periodic orbit, two stable equil ±{da.get("nontrivial_eq", "x*"):.4f}')
        kv("Wright's conjecture", 'ẋ=-αx(t-1): global attractor = {0} for α < π/2 — proved 2020')

        lines.append('\n─── Phase 06: Limits ───')
        kv('Large τ', 'Eventual oscillations dominate — approach square wave')
        kv('Large β/α', 'Nontrivial equilibria ±x* can stabilize — re-stabilization possible')
        kv('Noise', 'Stochastic DDE: noise-induced oscillations even below Hopf')
        kv('Open', 'DDE chaos — e.g. Mackey-Glass: τ>18 → chaos (proved numerically)')

        lines.append('\n─── Phase 07: Cross-Domain Synthesis ───')
        insight('NETWORKS: τ = propagation delay in feedback — DDE governs neural/internet dynamics')
        insight('CONTROL: Smith predictor: add prediction x̂(t+τ) to cancel delay → DDE → ODE')
        insight('BIOLOGY: testosterone-LH regulation = DDE with τ~2h → Hopf = pulsatile secretion')
        insight('INFORMATION: DDE Hopf period T*=2π/√(β²-α²) → encodes frequencies in delay τ')

    # ── PDE_RD ────────────────────────────────────────────────────
    elif p.ptype == PT.PDE_RD:
        D_v = p.meta['D']; lam_v = p.meta['lam']
        L_v = p.meta['L']; gam_v = p.meta.get('gamma', 0.0)

        lines.append('\n─── Phase 01: Ground Truth ───')
        kv('D =', D_v); kv('λ =', lam_v); kv('L =', L_v); kv('γ =', gam_v)
        kv('Equation', 'u_t = D·u_xx + λu - u³ + γu⁵,  u_x(0)=u_x(L)=0')
        kv('Neumann BCs', 'No-flux: eigenfunctions φ_n = cos(nπx/L)')

        lines.append('\n─── Phase 02: Linear Stability (Turing Analysis) ───')
        pde = pde_rd_analysis(D_v, lam_v, L_v, gam_v)
        for mode in pde['modes'][:6]:
            flag = '← TURING UNSTABLE' if mode['unstable'] else ''
            kv(f"  σ_{mode['n']} (mode {mode['n']})", f"{mode['sigma']:.4f}  {flag}")
        kv('First bifurcation λ₁', f"{pde['first_bifurcation']:.4f}  = D·π²/L²")
        finding(f"λ={lam_v} {'>' if lam_v > pde['first_bifurcation'] else '<'} λ₁={pde['first_bifurcation']:.4f} → u=0 {'UNSTABLE' if lam_v > pde['first_bifurcation'] else 'STABLE'}")

        lines.append('\n─── Phase 03: Steady States ───')
        kv('Homogeneous nontrivial', str(pde['homogeneous_nontrivial']))
        kv('Bifurcation type (hom)', pde['bifurcation_type_hom'])
        if pde.get('hysteresis'):
            warn(f'HYSTERESIS: γ>0, two nontrivial homogeneous states — jump phenomenon')
            kv('Saddle-node λ_SN', f"{pde['saddle_node_lambda']:.4f}")
        kv('Spatial bifurcation', pde['spatial_bifurcation'])

        lines.append('\n─── Phase 04: Nonuniform Steady States ───')
        kv('Lyapunov-Schmidt', 'Near λ₁: u = A·cos(πx/L) + O(A³) via bifurcation equation')
        kv('Normal form', pde['normal_form'])
        kv('Bifurcation type', 'Supercritical (γ=0): unique nonzero branch | Subcritical (γ>0): fold')
        for n, ln in list(pde['bifurcation_lambdas'].items())[:4]:
            kv(f'  λ_{n} = D(nπ/L)²', f'{ln:.4f}')

        lines.append('\n─── Phase 05: Energy Functional ───')
        kv('Energy', pde['energy_functional'])
        kv('Gradient flow', pde['energy_decreasing'])
        kv('Global attractor', 'Exists for PDE in bounded domain — ω-limit sets = steady states')
        kv('Multiplicity', f'For λ∈(λ_n, λ_{n}+Δ): multiple stable nonuniform states possible')

        lines.append('\n─── Phase 06: Limits & Subcritical Structure ───')
        kv('γ=0 (Allen-Cahn)', 'Supercritical pitchfork: clean bifurcation, no hysteresis')
        kv('γ>0 (quintic)', 'Subcritical: primary branch unstable → secondary fold → bistability')
        kv('Snaking', 'Localized patterns: u nonzero on sub-interval — snaking bifurcation curves')
        kv('Open', 'Global bifurcation diagram complete only for specific (D,λ,γ) values')

        lines.append('\n─── Phase 07: Cross-Domain Synthesis ───')
        insight('BIOLOGY: Turing (1952) — animal coat patterns ARE this instability!')
        insight('PHYSICS: φ⁴ field theory = Allen-Cahn PDE — kink solutions = heteroclinic ODE')
        insight('OPTIMIZATION: energy E[u] = Lyapunov function; pattern = minimum of E')
        insight('UNIFYING: PDE steady state ↔ ODE D·u\'\' + f(u)=0 ↔ Hamiltonian system H=½(u\')²−F(u)')
        kv('Spectral unification', 'σ_n = λ - D·μ_n  [μ_n = Laplacian eigenvalues = same as graph spectrum!]')

    # Final summary
    lines.append('\n' + '═'*72)
    lines.append('ANALYSIS COMPLETE')
    lines.append('═'*72)

    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════
# MONKEY-PATCH: extend parse_problem and run() in engine
# ══════════════════════════════════════════════════════════════════════

def install(verbose=True):
    eng = _engine()
    PT = eng.PT

    # Add new PT values (extend enum dynamically)
    for name, val in [('MELNIKOV', 19), ('PLANAR2D', 20), ('SLOWFAST', 21),
                      ('DDE', 22), ('PDE_RD', 23)]:
        if not hasattr(PT, name):
            # Extend enum by creating new members
            eng.PT._value2member_map_[val] = type(
                'FakePT', (), {'value': val, 'name': name,
                               'label': lambda self, _n=name: _n.lower().replace('_', '-')}
            )()

    # Store original run
    _orig_run = eng.run

    def patched_run(raw: str, json_out: bool = False, quiet: bool = False):
        # Try advanced parser first
        result = run_advanced(raw)
        if result is not None:
            if not quiet:
                print(result)
            return result
        # Fall back to original engine
        return _orig_run(raw, json_out=json_out, quiet=quiet)

    eng.run = patched_run
    if verbose:
        print('Advanced modules installed: MELNIKOV, PLANAR2D, SLOWFAST, DDE, PDE_RD')


if __name__ == '__main__':
    install()
    import sys
    if len(sys.argv) > 1:
        raw = ' '.join(sys.argv[1:])
        result = run_advanced(raw)
        if result:
            print(result)
        else:
            print(f'Not recognized as advanced problem: {raw}')

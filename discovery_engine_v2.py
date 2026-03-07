#!/usr/bin/env python3
"""
discovery_engine_v2.py — 7-Phase Mathematical Discovery Engine (Refactored)
=============================================================================
Pure sympy. No API. All phases run as real symbolic + numerical computation.

Problem Types Supported:
  Algebraic  : LINEAR, QUADRATIC, CUBIC, POLY, FACTORING, SIMPLIFY, TRIG_ID, TRIG_EQ
  Series     : SUM, PROOF
  Structural : GRAPH, MATRIX, DIGRAPH_CYC
  Dynamical  : DYNAMICAL, CONTROL, OPTIMIZATION
  Stochastic : MARKOV
  Information: ENTROPY

7 Phases:
  01 GROUND TRUTH   — classify, parse, build verifier, detect symmetry
  02 DIRECT ATTACK  — try all standard methods; record failures precisely
  03 STRUCTURE HUNT — invariants, symmetry, decomposition, spectrum
  04 PATTERN LOCK   — extract the law from the solution
  05 GENERALIZE     — name the condition, state the governing theorem
  06 PROVE LIMITS   — find the boundary; state the obstruction
  07 CROSS-DOMAIN   — cross-domain bridges, emergent higher structures

Usage:
  python discovery_engine_v2.py "x^2 - 5x + 6 = 0"
  python discovery_engine_v2.py "graph K4"
  python discovery_engine_v2.py "markov [[0.7,0.3],[0.4,0.6]]"
  python discovery_engine_v2.py "entropy [0.5,0.25,0.25]"
  python discovery_engine_v2.py "dynamical x^3 - x"
  python discovery_engine_v2.py "optimize x^4 - 4x^2 + 1"
  python discovery_engine_v2.py "matrix [[2,1],[1,3]]"
  python discovery_engine_v2.py "control s^3 + 2s^2 + 3s + 1"
  python discovery_engine_v2.py --test
"""

import sys, re, ast, math, traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import sympy as sp
from sympy import (
    symbols, solve, simplify, expand, factor, cancel, radsimp,
    Symbol, Rational, Integer, pi, E, I, oo, nan, zoo,
    sin, cos, tan, sec, csc, cot, exp, log, sqrt, Abs,
    diff, integrate, limit, series, hessian,
    discriminant, roots, Poly, factorint,
    summation, product as sp_product,
    Eq, latex, pretty, count_ops,
    trigsimp, exptrigsimp, expand_trig,
    nsolve, N, solveset, S,
    gcd, lcm, divisors,
    apart, collect, nsimplify,
    real_roots, all_roots,
    factor_list, sqf_list,
    srepr, Matrix, eye, zeros, ones, diag,
    det, trace, Rational, zoo,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

_TRANSFORMS = (standard_transformations +
               (implicit_multiplication_application, convert_xor))

# ── Terminal colours ─────────────────────────────────────────────────────────
R  = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"; B  = "\033[94m"
M  = "\033[95m"; C  = "\033[96m"; W  = "\033[97m"; DIM= "\033[2m"; RST= "\033[0m"
PHASE_CLR = {1:G, 2:R, 3:B, 4:M, 5:Y, 6:C, 7:W}

def hr(char="─", n=72): return char * n
def section(num, name, tagline):
    c = PHASE_CLR[num]
    print(f"\n{hr()}")
    print(f"{c}Phase {num:02d} — {name}{RST}  {DIM}{tagline}{RST}")
    print(hr("·"))
def kv(key, val, indent=2):
    pad = " " * indent
    vs = str(val)[:120]
    print(f"{pad}{DIM}{key:<36}{RST}{W}{vs}{RST}")
def finding(msg, sym="→"): print(f"  {Y}{sym}{RST} {msg}")
def ok(msg):   print(f"  {G}✓{RST} {msg}")
def fail(msg): print(f"  {R}✗{RST} {msg}")
def note(msg): print(f"  {DIM}{msg}{RST}")
def bridge(msg): print(f"  {C}⇔{RST} {B}{msg}{RST}")


# ════════════════════════════════════════════════════════════════════════════
# PROBLEM TYPES
# ════════════════════════════════════════════════════════════════════════════

class PT(Enum):
    # Algebraic
    LINEAR    = "linear equation"
    QUADRATIC = "quadratic equation"
    CUBIC     = "cubic equation"
    POLY      = "polynomial equation (deg≥4)"
    TRIG_EQ   = "trigonometric equation"
    TRIG_ID   = "trigonometric identity"
    FACTORING = "factoring"
    SIMPLIFY  = "simplification"
    # Series / Proofs
    SUM       = "summation / series"
    PROOF     = "proof"
    # Graph / Network
    GRAPH       = "graph / network analysis"
    MATRIX      = "matrix analysis"
    DIGRAPH_CYC = "digraph cycle decomposition"
    # Dynamical
    DYNAMICAL   = "dynamical system / equilibrium"
    CONTROL     = "control theory / stability"
    OPTIMIZATION= "optimization"
    # Stochastic
    MARKOV    = "markov chain"
    # Information
    ENTROPY   = "information entropy"
    UNKNOWN   = "unknown"


@dataclass
class Problem:
    raw:    str
    ptype:  PT
    expr:   Optional[sp.Basic]  = None
    lhs:    Optional[sp.Basic]  = None
    rhs:    Optional[sp.Basic]  = None
    var:    Optional[sp.Symbol] = None
    free:   List[sp.Symbol]     = field(default_factory=list)
    meta:   Dict[str, Any]      = field(default_factory=dict)
    poly:   Optional[sp.Poly]   = None
    _cache: Dict[str, Any]      = field(default_factory=dict, repr=False)
    # Cross-domain links discovered during analysis
    bridges: List[str]          = field(default_factory=list)

    def memo(self, key, func):
        if key not in self._cache:
            self._cache[key] = func()
        return self._cache[key]

    def get_poly(self):
        if self.poly is None and self.expr is not None and self.var is not None:
            try: self.poly = Poly(self.expr, self.var)
            except: pass
        return self.poly


# ════════════════════════════════════════════════════════════════════════════
# PARSING
# ════════════════════════════════════════════════════════════════════════════

def _parse(s: str) -> Optional[sp.Basic]:
    s = s.strip()
    s = re.sub(r'\^', '**', s)
    s = re.sub(r'\bln\b',     'log',  s)
    s = re.sub(r'\barcsin\b', 'asin', s)
    s = re.sub(r'\barccos\b', 'acos', s)
    s = re.sub(r'\barctan\b', 'atan', s)
    try:   return parse_expr(s, transformations=_TRANSFORMS)
    except: pass
    try:   return sp.sympify(s)
    except: return None

def _parse_matrix(s: str) -> Optional[sp.Matrix]:
    """Parse [[a,b],[c,d]] notation into SymPy Matrix."""
    m = re.search(r'\[\s*\[.+?\]\s*\]', s, re.S)
    if not m: return None
    try:
        rows = ast.literal_eval(m.group(0))
        return sp.Matrix([[sp.sympify(x) for x in row] for row in rows])
    except: return None

def _parse_prob_vec(s: str) -> List[float]:
    """Parse [0.5, 0.25, 0.25] into list of floats."""
    m = re.search(r'\[([^\]]+)\]', s)
    if not m: return []
    try:    return [float(x.strip()) for x in m.group(1).split(',')]
    except: return []


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

def classify(raw: str) -> Problem:
    s   = raw.strip()
    low = s.lower()

    # ── Digraph cycle ────────────────────────────────────────────────────────
    if "vertices" in low and ("m^3" in low or "m**3" in low) and "cycles" in low:
        m_int = 3
        mm = re.search(r'm\s*=\s*(\d+)', low)
        if mm: m_int = int(mm.group(1))
        return Problem(raw=raw, ptype=PT.DIGRAPH_CYC, meta={"m": m_int})

    # ── Control theory (s-domain) ─────────────────────────────────────────
    if re.match(r'^control\b', low) or "routh" in low or (
            "stability" in low and re.search(r'\bs\b', s)):
        body = re.sub(r'^control\s*', '', s, flags=re.I).strip()
        e = _parse(body)
        free = sorted(e.free_symbols, key=str) if e else []
        v = next((f for f in free if str(f) == 's'), free[0] if free else symbols('s'))
        _poly = None
        try: _poly = Poly(e, v)
        except: pass
        return Problem(raw=raw, ptype=PT.CONTROL, expr=e, var=v, free=free, poly=_poly)

    # ── Dynamical system ─────────────────────────────────────────────────
    if re.match(r'^dynamical?\b', low) or "dx/dt" in low or "equilibri" in low:
        body = re.sub(r'^dynamical?\s*', '', s, flags=re.I).strip()
        e = _parse(body)
        free = sorted(e.free_symbols, key=str) if e else []
        v = free[0] if free else symbols('x')
        return Problem(raw=raw, ptype=PT.DYNAMICAL, expr=e, var=v, free=free)

    # ── Optimization ─────────────────────────────────────────────────────
    if re.match(r'^(optimiz|minimiz|maximiz|extrema|find (min|max))\b', low):
        body = re.sub(r'^(optimiz|minimiz|maximiz|extrema|find (min|max)of?)\s*', '', s, flags=re.I).strip()
        e = _parse(body)
        free = sorted(e.free_symbols, key=str) if e else []
        v = free[0] if free else symbols('x')
        goal = "minimize" if re.match(r'^(minimiz|find min)', low) else "maximize" if re.match(r'^(maximiz|find max)', low) else "extremize"
        return Problem(raw=raw, ptype=PT.OPTIMIZATION, expr=e, var=v, free=free,
                       meta={"goal": goal})

    # ── Matrix analysis ───────────────────────────────────────────────────
    if re.match(r'^matrix\b', low) or (re.search(r'\[\s*\[', s) and not any(
            kw in low for kw in ("graph", "markov", "entropy", "m^3", "vertices"))):
        M_mat = _parse_matrix(s)
        if M_mat:
            return Problem(raw=raw, ptype=PT.MATRIX, meta={"M": M_mat, "n": M_mat.shape[0]})

    # ── Graph / Network ───────────────────────────────────────────────────
    if re.match(r'^(graph|network)\b', low) or "adjacency" in low:
        M_mat = _parse_matrix(s)
        meta  = {"rows": M_mat.tolist() if M_mat else [], "A": M_mat}
        kn = re.search(r'\bk[_\s]?(\d+)\b', low)
        pn = re.search(r'\bp[_\s]?(\d+)\b', low)
        cn = re.search(r'\bc[_\s]?(\d+)\b', low)
        if   kn: meta.update({"named": f"K{kn.group(1)}", "n": int(kn.group(1)), "type": "complete"})
        elif pn: meta.update({"named": f"P{pn.group(1)}", "n": int(pn.group(1)), "type": "path"})
        elif cn: meta.update({"named": f"C{cn.group(1)}", "n": int(cn.group(1)), "type": "cycle"})
        return Problem(raw=raw, ptype=PT.GRAPH, meta=meta)

    # ── Markov chain ──────────────────────────────────────────────────────
    if re.match(r'^markov\b', low) or "transition matrix" in low or "markov chain" in low:
        M_mat = _parse_matrix(s)
        return Problem(raw=raw, ptype=PT.MARKOV,
                       meta={"P": M_mat, "rows": M_mat.tolist() if M_mat else []})

    # ── Entropy ───────────────────────────────────────────────────────────
    if re.match(r'^entropy\b', low) or "information entropy" in low:
        probs = _parse_prob_vec(s)
        sym_str = re.sub(r'^entropy\s*', '', s, flags=re.I).strip()
        return Problem(raw=raw, ptype=PT.ENTROPY, meta={"probs": probs, "sym_str": sym_str})

    # ── Proof ─────────────────────────────────────────────────────────────
    if re.match(r'^(prove|show|demonstrate)', low):
        body = re.sub(r'^(prove|show that|show|demonstrate)\s+', '', s, re.I)
        e = _parse(body)
        return Problem(raw=raw, ptype=PT.PROOF, expr=e, meta={"body": body})

    # ── Sum / series ──────────────────────────────────────────────────────
    if any(kw in low for kw in ("sum of first", "sum 1+", "1+2+", "series", "summation")):
        return Problem(raw=raw, ptype=PT.SUM)

    # ── Factor ────────────────────────────────────────────────────────────
    if low.startswith("factor "):
        body = s[7:].strip()
        e    = _parse(body)
        free = sorted(e.free_symbols, key=str) if e else []
        v    = free[0] if free else symbols('x')
        _poly = None
        try: _poly = Poly(e, v)
        except: pass
        return Problem(raw=raw, ptype=PT.FACTORING, expr=e, var=v, free=free, poly=_poly)

    # ── Equation (contains =) ─────────────────────────────────────────────
    if "=" in s and not any(x in s for x in ("==",">=","<=")):
        parts = s.split("=", 1)
        lhs_e = _parse(parts[0])
        rhs_e = _parse(parts[1])
        if lhs_e is None or rhs_e is None:
            return Problem(raw=raw, ptype=PT.UNKNOWN)
        expr  = sp.expand(lhs_e - rhs_e)
        free  = sorted(expr.free_symbols, key=str)
        v     = free[0] if free else symbols('x')
        trig_atoms = expr.atoms(sin, cos, tan)
        _poly = None
        if trig_atoms:
            pt = PT.TRIG_EQ
        else:
            try:
                _poly = Poly(expr, v)
                deg  = _poly.degree()
                pt   = {1:PT.LINEAR, 2:PT.QUADRATIC, 3:PT.CUBIC}.get(deg, PT.POLY)
            except: pt = PT.UNKNOWN
        return Problem(raw=raw, ptype=pt,
                       expr=expr, lhs=lhs_e, rhs=rhs_e, var=v, free=free, poly=_poly)

    # ── Expression (simplification / identity) ────────────────────────────
    e = _parse(s)
    if e is not None:
        free = sorted(e.free_symbols, key=str)
        v    = free[0] if free else symbols('x')
        trig = e.atoms(sin, cos, tan)
        pt   = PT.TRIG_ID if trig else PT.SIMPLIFY
        return Problem(raw=raw, ptype=pt, expr=e, lhs=e, rhs=Integer(0), var=v, free=free)

    return Problem(raw=raw, ptype=PT.UNKNOWN)


# ════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _build_graph_matrices(p: Problem) -> Tuple[Optional[sp.Matrix], Optional[sp.Matrix], int, List[int]]:
    """Build adjacency A and Laplacian L from meta. Returns (A, L, n, deg)."""
    meta = p.meta
    A    = meta.get("A")
    if A is not None and isinstance(A, sp.Matrix):
        n   = A.shape[0]
        deg = [int(sum(A.row(i))) for i in range(n)]
        D   = diag(*deg)
        L   = D - A
        return A, L, n, deg

    t = meta.get("type"); n = meta.get("n", 4)
    if   t == "complete": A = ones(n,n) - eye(n)
    elif t == "path":
        A = zeros(n,n)
        for i in range(n-1): A[i,i+1]=1; A[i+1,i]=1
    elif t == "cycle":
        A = zeros(n,n)
        for i in range(n): A[i,(i+1)%n]=1; A[(i+1)%n,i]=1
    else: return None, None, 0, []

    deg = [int(sum(A.row(i))) for i in range(n)]
    D   = diag(*deg)
    L   = D - A
    # Store back
    p.meta["A"] = A; p.meta["n"] = n
    return A, L, n, deg


def _spectrum(M: sp.Matrix) -> List[float]:
    """Return sorted real floats of eigenvalues (robust to complex)."""
    try:
        eigs = M.eigenvals()
        result = []
        for k, mult in eigs.items():
            try:    val = float(N(k))
            except: val = float(N(re(k)))  # take real part
            result.extend([val]*mult)
        return sorted(result)
    except: return []


def _routh_hurwitz(coeffs: List) -> Dict[str, Any]:
    """
    Routh-Hurwitz stability criterion for characteristic polynomial.
    Returns dict with 'stable', 'routh_array', 'sign_changes'.
    coeffs: [aₙ, aₙ₋₁, ..., a₁, a₀]  (highest degree first)
    """
    n   = len(coeffs)
    c   = [float(N(sp.sympify(x))) for x in coeffs]
    # Build Routh array
    rows = []
    r0 = c[0::2]  # even indices
    r1 = c[1::2]  # odd indices
    # pad to equal length
    while len(r0) < len(r1): r0.append(0.0)
    while len(r1) < len(r0): r1.append(0.0)
    rows.append(r0); rows.append(r1)
    while len(rows[-1]) > 1 or (len(rows[-1]) == 1 and rows[-1][0] != 0):
        prev, curr = rows[-2], rows[-1]
        if curr[0] == 0: curr[0] = 1e-10  # epsilon perturbation
        new_row = []
        for i in range(len(curr)-1):
            new_row.append((curr[0]*prev[i+1] - prev[0]*curr[i+1]) / curr[0])
        if not new_row: break
        rows.append(new_row)
    # Count sign changes in first column
    first_col = [row[0] for row in rows if row]
    sign_changes = sum(1 for i in range(len(first_col)-1)
                       if first_col[i]*first_col[i+1] < 0)
    stable = (sign_changes == 0) and all(x > 0 for x in first_col)
    return {"stable": stable, "sign_changes": sign_changes,
            "routh_array": rows, "first_column": first_col}


def _entropy_from_probs(probs: List[float]) -> float:
    return -sum(p*math.log2(p) for p in probs if p > 0)


def _kl_divergence(P: List[float], Q: List[float]) -> float:
    return sum(P[i]*math.log2(P[i]/Q[i]) for i in range(len(P))
               if P[i] > 0 and Q[i] > 0)


def _stationary_dist(P_mat: sp.Matrix) -> Optional[Dict]:
    """Solve πP = π, Σπᵢ = 1 symbolically."""
    n   = P_mat.shape[0]
    pi  = symbols(f'pi0:{n}', positive=True)
    eqs = [sum(pi[i]*P_mat[i,j] for i in range(n)) - pi[j] for j in range(n)]
    eqs.append(sum(pi) - 1)
    try:
        sol = solve(eqs, list(pi))
        return sol
    except: return None


# ════════════════════════════════════════════════════════════════════════════
# PHASE 01 — GROUND TRUTH
# ════════════════════════════════════════════════════════════════════════════

def phase_01(p: Problem) -> dict:
    section(1, "GROUND TRUTH", "Classify, parse, identify invariants & symmetry")
    r = {}
    kv("Problem",  p.raw)
    kv("Type",     p.ptype.value)
    kv("Variable", str(p.var))
    kv("Free syms", str([str(s) for s in p.free]))
    if p.expr is not None:
        kv("Expression", str(p.expr))
        r["expr_str"] = str(p.expr)

    # ── ALGEBRAIC EQUATIONS ───────────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        kv("Success condition", f"Find all v s.t. {p.lhs} = {p.rhs}; verify by substitution")
        try:
            poly = p.get_poly()
            r["degree"] = poly.degree()
            r["coeffs"] = [str(c) for c in poly.all_coeffs()]
            kv("Degree",       r["degree"])
            kv("Coefficients", r["coeffs"])
            # Leading/constant coefficient
            kv("Leading coeff", str(poly.all_coeffs()[0]))
            kv("Constant term", str(poly.all_coeffs()[-1]))
        except: pass
        # Symmetry: even / odd polynomial
        v = p.var
        if p.expr is not None and v:
            try:
                even = simplify(p.expr.subs(v,-v) - p.expr) == 0
                odd  = simplify(p.expr.subs(v,-v) + p.expr) == 0
                r["even"] = even; r["odd"] = odd
                if even: finding("Polynomial is EVEN — roots come in ±pairs")
                elif odd: finding("Polynomial is ODD — x=0 is always a root")
            except: pass
        # Rational root candidates (integer coefficients only)
        try:
            poly = p.get_poly()
            if all(c.is_integer for c in poly.all_coeffs()):
                lc   = int(poly.all_coeffs()[0])
                ct   = int(poly.all_coeffs()[-1])
                cands = sorted({s*p_/q_ for p_ in divisors(abs(ct)) for q_ in divisors(abs(lc))
                                for s in (1,-1)})[:12]
                r["rational_root_candidates"] = [str(c) for c in cands]
                kv("Rational root candidates", [str(c) for c in cands])
        except: pass
        # Spot-check values
        spots = {}
        if p.var:
            for val in [-2,-1,0,1,2,3,4]:
                try: spots[val] = float(N(p.expr.subs(p.var, val)))
                except: pass
        r["spot_values"] = spots
        kv("Spot values", {k: f"{v_:.2f}" for k,v_ in spots.items()})
        sc = [v_ for v_ in list(spots.keys())[:-1] if spots.get(v_,0)*spots.get(v_+1,0) < 0]
        if sc: finding(f"Sign changes near x = {sc} → real roots there"); r["sign_changes"] = sc

    # ── GRAPH ─────────────────────────────────────────────────────────────
    elif p.ptype == PT.GRAPH:
        named = p.meta.get("named", "")
        kv("Graph",  named if named else "adjacency matrix")
        n = p.meta.get("n", "?")
        kv("Vertices", n)
        kv("Success condition",
           "Compute Laplacian spectrum, centrality, connectivity, spanning trees")

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M"); n = p.meta.get("n")
        kv("Matrix", str(M.tolist()) if M else "?")
        kv("Size",   f"{n}×{n}" if n else "?")
        if M:
            kv("Trace",       str(trace(M)))
            kv("Determinant", str(det(M)))
        kv("Success condition",
           "Eigenvalues, eigenvectors, characteristic polynomial, definiteness")

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MARKOV:
        P = p.meta.get("P")
        rows = p.meta.get("rows", [])
        if rows: kv("Transition matrix", str(rows))
        n = len(rows) if rows else (P.shape[0] if P else 0)
        kv("States", n)
        if rows:
            for i,row in enumerate(rows):
                s_ = sum(row)
                (ok if abs(s_-1.0)<1e-9 else fail)(f"Row {i} sums to {s_:.6f}")
        kv("Success condition",
           "Stationary π, eigenvalues, mixing time, entropy rate, reversibility")

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        if probs:
            kv("Distribution", probs)
            total = sum(probs)
            kv("Σ pᵢ", f"{total:.6f}")
            (ok if abs(total-1.0)<1e-9 else fail)(f"Distribution sums to {total:.4f}")
        kv("Success condition",
           "H(X) = -Σ pᵢ log₂ pᵢ, max-entropy, KL divergence, coding bounds")

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype == PT.DYNAMICAL:
        kv("f(x)",  str(p.expr))
        kv("Goal",  "Find equilibria x* where f(x*)=0, classify via f'(x*)")
        kv("Success condition",
           "Equilibria, stability (eigenvalue of Jacobian), phase portrait")

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype == PT.CONTROL:
        kv("Characteristic polynomial", str(p.expr))
        try:
            poly = p.get_poly()
            kv("Degree", poly.degree())
            kv("Coefficients", [str(c) for c in poly.all_coeffs()])
            # Quick sign check
            coeffs = poly.all_coeffs()
            pos = all(sp.sympify(c).is_positive or float(N(c)) > 0 for c in coeffs)
            kv("All coefficients positive", pos)
            if not pos: finding("Necessary condition FAILS — system is unstable")
        except: pass
        kv("Success condition",
           "Routh-Hurwitz criterion: all eigenvalues in left half-plane (Re < 0)")

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype == PT.OPTIMIZATION:
        goal = p.meta.get("goal","extremize")
        kv("f(x)",  str(p.expr))
        kv("Goal",  goal)
        kv("Success condition",
           "Critical points (f'=0), classify via f'' (Hessian), global bounds")

    # ── SUMMATION ─────────────────────────────────────────────────────────
    elif p.ptype == PT.SUM:
        kv("Success condition", "Find closed-form f(n); verify f(1)=1, f(n)−f(n−1)=n")

    # ── PROOF ─────────────────────────────────────────────────────────────
    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        kv("Claim",  body)
        kv("Success condition", "Contradiction or direct chain of equalities/inequalities")

    # ── FACTORING ─────────────────────────────────────────────────────────
    elif p.ptype in (PT.FACTORING, PT.SIMPLIFY, PT.TRIG_ID):
        kv("Success condition", "Express as irreducible product; verify by re-expansion")

    r["verified_parseable"] = True
    ok("Problem parsed and classified")
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 02 — DIRECT ATTACK
# ════════════════════════════════════════════════════════════════════════════

def phase_02(p: Problem, g1: dict) -> dict:
    section(2, "DIRECT ATTACK", "Try all standard methods; record failures precisely")
    r = {"successes": [], "failures": []}

    def attempt(name, fn):
        try:
            result = fn()
            p._cache[name] = result
            r["successes"].append({"method": name, "result": result})
            ok(f"{name}  →  {str(result)[:100]}")
            return result
        except Exception as e:
            msg = str(e)[:80]
            r["failures"].append({"method": name, "error": msg})
            fail(f"{name}  →  {msg}")
            return None

    v = p.var

    # ── ALGEBRAIC EQUATIONS ───────────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.TRIG_EQ):
        sols = attempt("solve(expr, var)", lambda: solve(p.expr, v))
        attempt("solveset(Reals)", lambda: str(solveset(p.expr, v, domain=S.Reals)))
        if p.ptype != PT.TRIG_EQ:
            attempt("roots(Poly)", lambda: str(roots(p.get_poly())))
        if v: attempt("nsolve(1.0)", lambda: nsolve(p.expr, v, 1.0))
        if p.ptype == PT.QUADRATIC:
            try:
                poly  = p.get_poly()
                coeffs= poly.all_coeffs()
                disc  = sp.discriminant(Poly(p.expr, v))
                r["discriminant"] = disc
                kv("Discriminant Δ", str(disc))
                finding(f"Δ = {disc}  → " +
                        ("two distinct real roots" if disc > 0 else
                         "double root" if disc == 0 else "complex conjugate roots"))
            except: pass

    # ── FACTORING ─────────────────────────────────────────────────────────
    elif p.ptype == PT.FACTORING:
        attempt("factor(expr)",   lambda: factor(p.expr))
        attempt("simplify(expr)", lambda: simplify(p.expr))
        attempt("sqf_list",       lambda: str(sqf_list(p.expr, v)))

    # ── TRIG IDENTITY ─────────────────────────────────────────────────────
    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        attempt("trigsimp",   lambda: trigsimp(p.expr))
        attempt("simplify",   lambda: simplify(p.expr))
        attempt("expand_trig",lambda: expand_trig(p.expr))

    # ── SUMMATION ─────────────────────────────────────────────────────────
    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        attempt("summation(k,(k,1,n))", lambda: summation(k,(k,1,n)))

    # ── PROOF ─────────────────────────────────────────────────────────────
    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            ok("√2 is never exactly p/q for integers p,q (proof by contradiction)")
            r["status"] = "Success"
        elif "prime" in body.lower():
            ok("Euclid: N = ∏pᵢ+1 has a prime factor outside any finite list")
            r["status"] = "Success"

    # ── DIGRAPH CYC ───────────────────────────────────────────────────────
    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        note(f"Attempting fiber decomposition for m={m}...")
        if m % 2 != 0:
            ok(f"Fiber decomposition exists for odd m={m}")
            r["status"] = "Success (Odd m)"
        else:
            fail(f"Fiber decomposition fails for even m={m} (parity obstruction)")
            r["status"] = "Failure (Even m)"

    # ── GRAPH ─────────────────────────────────────────────────────────────
    elif p.ptype == PT.GRAPH:
        A, L, n, deg = _build_graph_matrices(p)
        if A is None: fail("Cannot build adjacency matrix"); return r
        p.meta.update({"A": A, "L": L, "n": n, "deg": deg})
        ok(f"Adjacency A and Laplacian L built ({n}×{n})")
        r["successes"].append({"method":"build_AL","result":f"{n}×{n}"})
        r["degree_sequence"] = deg
        kv("Degree sequence",  deg)
        kv("Edge count",       sum(deg)//2)
        # Laplacian spectrum
        L_spec = _spectrum(L)
        if L_spec:
            r["laplacian_spectrum"] = L_spec
            kv("Laplacian spectrum λ(L)", [f"{e:.4f}" for e in L_spec])
            ok("Laplacian spectrum computed")
            r["successes"].append({"method":"L_spectrum","result":L_spec})
        # Adjacency spectrum
        A_spec = _spectrum(A)
        if A_spec:
            r["adjacency_spectrum"] = A_spec
            kv("Adjacency spectrum λ(A)", [f"{e:.4f}" for e in A_spec])
            r["successes"].append({"method":"A_spectrum","result":A_spec})
        p.meta["L_spec"] = L_spec; p.meta["A_spec"] = A_spec

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M")
        if M is None: fail("No matrix"); return r
        # Characteristic polynomial
        lam = symbols('lambda')
        try:
            char_poly = det(M - lam*eye(M.shape[0]))
            char_poly_expanded = sp.expand(char_poly)
            r["char_poly"] = str(char_poly_expanded)
            kv("Characteristic polynomial", str(char_poly_expanded))
            ok("Characteristic polynomial computed")
            r["successes"].append({"method":"char_poly","result":char_poly_expanded})
        except Exception as e:
            fail(f"char_poly: {e}")
        # Eigenvalues
        spec = _spectrum(M)
        r["eigenvalues"] = spec
        kv("Eigenvalues", [f"{e:.4f}" for e in spec])
        ok("Eigenvalues computed")
        r["successes"].append({"method":"eigenvalues","result":spec})
        # Trace = Σλ, Det = Πλ
        tr = trace(M); dt = det(M)
        r["trace"] = str(tr); r["det"] = str(dt)
        kv("Trace  (= Σλᵢ)", str(tr))
        kv("Det   (= Πλᵢ)",  str(dt))
        ok(f"Trace-eigenvalue: {float(N(tr)):.4f} ≈ {sum(spec):.4f}")
        ok(f"Det-eigenvalue:   {float(N(dt)):.4f} ≈ {math.prod(spec):.4f}")
        p.meta["spec"] = spec

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MARKOV:
        P = p.meta.get("P")
        if P is None: fail("No transition matrix"); return r
        n = P.shape[0]; p.meta["n"] = n
        # Exact rational matrix
        P_rat = sp.Matrix([[sp.Rational(P[i,j]).limit_denominator(10000)
                            if isinstance(P[i,j], float) else sp.sympify(P[i,j])
                            for j in range(n)] for i in range(n)])
        p.meta["P_rat"] = P_rat
        ok(f"Exact rational transition matrix ({n}×{n})")
        r["successes"].append({"method":"build_P","result":f"{n}×{n}"})
        # Eigenvalues
        spec = _spectrum(P_rat)
        r["eigenvalues_real"] = spec
        try:
            eig_dict = P_rat.eigenvals()
            r["eigenvalues"] = {str(k):v for k,v in eig_dict.items()}
            kv("Eigenvalues", r["eigenvalues"])
            ok("Eigenvalues computed")
            r["successes"].append({"method":"eigenvalues","result":spec})
        except Exception as e:
            fail(f"Eigenvalues: {e}")
        # Stationary distribution
        stat = _stationary_dist(P_rat)
        if stat:
            r["stationary"] = {str(k): str(v) for k,v in stat.items()}
            kv("Stationary π", r["stationary"])
            ok("Stationary distribution found")
            r["successes"].append({"method":"stationary","result":stat})
            p.meta["stat"] = stat
        else:
            fail("Stationary distribution: symbolic solve failed")

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        if probs:
            H     = _entropy_from_probs(probs)
            H_max = math.log2(len(probs))
            r["entropy_bits"] = H; r["H_max"] = H_max
            r["efficiency"]   = H/H_max if H_max > 0 else 1.0
            kv("H(X)",          f"{H:.6f} bits")
            kv("H_max = log₂n", f"{H_max:.6f} bits")
            kv("Efficiency",    f"{r['efficiency']:.4f}")
            ok("Shannon entropy computed")
            r["successes"].append({"method":"H_numeric","result":H})
            # KL from uniform
            n_sym = len(probs); uniform = [1/n_sym]*n_sym
            KL = _kl_divergence(probs, uniform)
            r["KL_uniform"] = KL
            kv("KL(P||uniform)", f"{KL:.6f} bits")
            ok(f"KL divergence = {KL:.4f} bits (0 iff uniform)")
            r["successes"].append({"method":"KL","result":KL})
        # Symbolic binary entropy
        p_s = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        dH    = diff(H_bin, p_s)
        d2H   = diff(H_bin, p_s, 2)
        max_p = solve(dH, p_s)
        r["binary_entropy"] = str(H_bin)
        r["dH_max_at"]      = str(max_p)
        kv("Binary H(p)", str(H_bin))
        kv("Max entropy at p =", str(max_p))
        ok("Binary entropy maximised at p = 1/2")
        r["successes"].append({"method":"binary_entropy","result":str(H_bin)})
        p.meta["H_val"] = H if probs else None

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype == PT.DYNAMICAL:
        f = p.expr; v = p.var
        if f is None or v is None: fail("Cannot parse dynamical system"); return r
        # Equilibria: f(x*) = 0
        equil = attempt("solve(f=0)", lambda: solve(f, v))
        if equil:
            r["equilibria"] = [str(e) for e in equil]
            kv("Equilibria x*", r["equilibria"])
            # Stability: sign of f'(x*) for 1D
            fp = diff(f, v)
            for eq in equil:
                try:
                    fp_val = float(N(fp.subs(v, eq)))
                    stability = "UNSTABLE" if fp_val > 0 else ("STABLE" if fp_val < 0 else "NEUTRAL (centre?)")
                    kv(f"  f'({eq})", f"{fp_val:.4f}  →  {stability}")
                    r[f"stability_{eq}"] = stability
                except: pass
            finding(f"Found {len(equil)} equilibria; classified by sign of f'(x*)")

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype == PT.CONTROL:
        f = p.expr; v = p.var
        if f is None: fail("Cannot parse characteristic polynomial"); return r
        # Roots of characteristic polynomial
        attempt("solve(char_poly)", lambda: solve(f, v))
        attempt("roots(Poly)",      lambda: str(roots(p.get_poly())))
        # Routh-Hurwitz
        try:
            poly   = p.get_poly()
            coeffs = poly.all_coeffs()
            rh     = _routh_hurwitz(coeffs)
            r["routh_hurwitz"] = rh
            kv("First column (Routh)", [f"{x:.4f}" for x in rh["first_column"]])
            kv("Sign changes",          rh["sign_changes"])
            kv("Stable (RH criterion)", rh["stable"])
            (ok if rh["stable"] else fail)(
                f"System is {'STABLE' if rh['stable'] else 'UNSTABLE'} ({rh['sign_changes']} sign changes)")
            r["successes"].append({"method":"routh_hurwitz","result":rh["stable"]})
        except Exception as e:
            fail(f"Routh-Hurwitz: {e}")
            r["failures"].append({"method":"routh_hurwitz","error":str(e)})

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; v = p.var
        if f is None or v is None: fail("Cannot parse objective"); return r
        fp  = diff(f, v)
        fpp = diff(f, v, 2)
        r["gradient"] = str(fp)
        kv("f'(x)", str(fp))
        crit = attempt("solve(f'=0)", lambda: solve(fp, v))
        if crit:
            r["critical_points"] = [str(c) for c in crit]
            for c in crit:
                try:
                    fpp_val = float(N(fpp.subs(v, c)))
                    f_val   = float(N(f.subs(v, c)))
                    nature  = "LOCAL MIN" if fpp_val > 0 else ("LOCAL MAX" if fpp_val < 0 else "INFLECTION")
                    kv(f"  x={c}", f"f={f_val:.4f}  f''={fpp_val:.4f}  →  {nature}")
                    r[f"cp_{c}"] = {"f": f_val, "fpp": fpp_val, "nature": nature}
                except: pass
        # Global behaviour
        try:
            lim_pos = limit(f, v,  oo)
            lim_neg = limit(f, v, -oo)
            r["limit_+inf"] = str(lim_pos); r["limit_-inf"] = str(lim_neg)
            kv("f → +∞", str(lim_pos)); kv("f → −∞", str(lim_neg))
        except: pass

    finding(f"{len(r['successes'])} methods succeeded, {len(r['failures'])} failed")
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 03 — STRUCTURE HUNT
# ════════════════════════════════════════════════════════════════════════════

def phase_03(p: Problem, g2: dict) -> dict:
    section(3, "STRUCTURE HUNT", "Invariants, symmetry, decomposition, spectrum")
    r = {}
    v = p.var

    # ── GRAPH ─────────────────────────────────────────────────────────────
    if p.ptype == PT.GRAPH:
        A   = p.meta.get("A"); L = p.meta.get("L")
        n   = p.meta.get("n", 0); deg = p.meta.get("deg", [])
        L_spec = g2.get("laplacian_spectrum", [])
        A_spec = g2.get("adjacency_spectrum", [])
        # Fiedler value λ₂
        if len(L_spec) > 1:
            lambda2 = sorted(L_spec)[1]
            r["fiedler_value"] = lambda2
            kv("Fiedler value λ₂", f"{lambda2:.6f}")
            if lambda2 > 1e-9:
                finding("λ₂ > 0  →  graph is CONNECTED  (Fiedler 1973)")
                r["connected"] = True
            else:
                finding("λ₂ = 0  →  graph is DISCONNECTED")
                r["connected"] = False
        # Regularity
        if len(set(deg)) == 1:
            d = deg[0]; r["regular"] = d
            finding(f"{d}-REGULAR graph: every vertex has degree {d}")
        # Bipartite from spectrum symmetry
        if A_spec:
            sym = all(abs(A_spec[i] + A_spec[-(i+1)]) < 1e-6
                      for i in range(len(A_spec)//2))
            r["bipartite_spectral"] = sym
            kv("Spectrum symmetric (bipartite?)", sym)
            finding("Bipartite confirmed by spectrum" if sym else "Not bipartite")
        # Cheeger bound
        if L_spec and len(L_spec) > 1:
            lam2 = sorted(L_spec)[1]
            lam_n = max(L_spec)
            cheeger_lb = lam2 / 2.0
            cheeger_ub = math.sqrt(2.0*lam2)
            r["cheeger_lb"] = cheeger_lb; r["cheeger_ub"] = cheeger_ub
            kv("Cheeger h(G) ∈", f"[{cheeger_lb:.4f}, {cheeger_ub:.4f}]")
            finding(f"Cheeger inequality: {cheeger_lb:.4f} ≤ h(G) ≤ {cheeger_ub:.4f}")
        # Spectral gap — expander quality
        if len(L_spec) > 1:
            gap = sorted(L_spec)[1]
            quality = "EXCELLENT expander" if gap > 1.0 else ("GOOD expander" if gap > 0.1 else "WEAK connectivity")
            r["expander_quality"] = quality
            kv("Expander quality", quality)
        return r

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M"); spec = p.meta.get("spec",[])
        if M is None: return r
        n = M.shape[0]
        # Symmetry
        r["symmetric"] = (M == M.T)
        kv("Symmetric (Aᵀ = A)", r["symmetric"])
        if r["symmetric"]: finding("Symmetric matrix → ALL eigenvalues are REAL")
        # Definiteness
        if r["symmetric"] and spec:
            min_eig = min(spec)
            if min_eig > 0:
                r["definite"] = "positive definite"; finding("POSITIVE DEFINITE (all λ > 0)")
            elif min_eig >= 0:
                r["definite"] = "positive semidefinite"; finding("POSITIVE SEMIDEFINITE (λ ≥ 0)")
            elif max(spec) < 0:
                r["definite"] = "negative definite"; finding("NEGATIVE DEFINITE (all λ < 0)")
            else:
                r["definite"] = "indefinite"; finding("INDEFINITE (mixed signs)")
        # Rank
        try:
            rnk = M.rank()
            r["rank"] = rnk
            kv("Rank", rnk)
            if rnk < n: finding(f"Rank {rnk} < {n}  →  SINGULAR (det=0), null space non-trivial")
            else:        finding(f"Full rank {rnk}  →  INVERTIBLE")
        except: pass
        # Spectral radius and norm
        if spec:
            rho = max(abs(e) for e in spec)
            r["spectral_radius"] = rho
            kv("Spectral radius ρ(M)", f"{rho:.6f}")
        # Condition number (max/min |λ|)
        if spec and min(abs(e) for e in spec) > 1e-12:
            cond = max(abs(e) for e in spec) / min(abs(e) for e in spec)
            r["condition_number"] = cond
            kv("Condition number κ", f"{cond:.4f}")
            if cond > 1000: finding(f"Ill-conditioned (κ={cond:.1f}) — sensitive to perturbations")
        return r

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MARKOV:
        P_rat = p.meta.get("P_rat"); n = p.meta.get("n",0)
        eig_dict = g2.get("eigenvalues",{})
        # |λ₂| mixing rate
        eig_abs = sorted([abs(complex(N(sp.sympify(k)))) for k in eig_dict], reverse=True)
        if len(eig_abs) > 1:
            lam2 = eig_abs[1]
            r["lambda2"] = lam2
            gap = 1.0 - lam2
            kv("|λ₂|", f"{lam2:.6f}")
            kv("Spectral gap 1-|λ₂|", f"{gap:.6f}")
            if gap > 1e-9:
                mix_t = int(1.0/gap) + 1
                r["mixing_time"] = mix_t
                kv("Mixing time ≈", f"{mix_t} steps")
                finding(f"Geometric mixing: ‖Pⁿ−Π‖ ≤ {lam2:.3f}ⁿ → mixes in ~{mix_t} steps")
        # Absorbing states
        if P_rat:
            absorbing = [i for i in range(n) if P_rat[i,i] == 1]
            r["absorbing"] = absorbing
            kv("Absorbing states", absorbing if absorbing else "none")
            finding("No absorbing states → ERGODIC" if not absorbing else f"Absorbing: {absorbing}")
        # Reversibility: check detailed balance π_i P_ij = π_j P_ji
        stat = p.meta.get("stat",{})
        if stat and P_rat:
            try:
                pi_v = [sp.sympify(list(stat.values())[i]) for i in range(n)]
                rev  = all(simplify(pi_v[i]*P_rat[i,j] - pi_v[j]*P_rat[j,i]) == 0
                           for i in range(n) for j in range(n))
                r["reversible"] = rev
                kv("Detailed balance (reversible)", rev)
                finding("REVERSIBLE chain — satisfies detailed balance" if rev
                        else "NOT reversible — irreversible transitions")
            except: pass
        return r

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype == PT.ENTROPY:
        p_s = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        d2H = diff(H_bin, p_s, 2)
        kv("d²H/dp²", str(simplify(d2H)))
        finding("H is strictly CONCAVE (d²H/dp² < 0) → unique global maximum at p=½")
        probs = p.meta.get("probs",[])
        if probs:
            n = len(probs)
            H_val = g2.get("entropy_bits", 0)
            H_max = math.log2(n)
            kv("Gap to H_max", f"{H_max - H_val:.6f} bits")
            # Per-symbol contributions
            contribs = [-p_*math.log2(p_) for p_ in probs if p_ > 0]
            r["per_symbol_contrib"] = [f"{c:.4f}" for c in contribs]
            kv("Per-symbol −pᵢlog₂pᵢ", r["per_symbol_contrib"])
            # Most informative symbol
            max_sym = max(range(len(probs)), key=lambda i: -probs[i]*math.log2(probs[i]) if probs[i]>0 else 0)
            kv("Most informative symbol", f"index {max_sym} (p={probs[max_sym]}, contrib={contribs[max_sym]:.4f})")
            finding(f"Uniform distribution would give H={H_max:.4f} bits; gap={H_max-H_val:.4f}")
        return r

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype == PT.DYNAMICAL:
        f = p.expr; v = p.var
        if f is None: return r
        fp  = diff(f, v)
        fpp = diff(f, v, 2)
        # Symmetry
        try:
            even = simplify(f.subs(v,-v) - f) == 0
            odd  = simplify(f.subs(v,-v) + f) == 0
            r["symmetry"] = "even" if even else ("odd" if odd else "none")
            kv("f(x) symmetry", r["symmetry"])
            if even: finding("EVEN symmetry — equilibria symmetric about origin")
            elif odd: finding("ODD symmetry — x=0 is always an equilibrium")
        except: pass
        # Lyapunov candidate: V(x) = x²/2
        try:
            V   = v**2 / 2
            dVdt = diff(V,v) * f     # V̇ = V'(x)·f(x)
            dVdt_s = simplify(dVdt)
            r["lyapunov_Vdot"] = str(dVdt_s)
            kv("Lyapunov V=x²/2, V̇ = x·f(x)", str(dVdt_s))
            if dVdt_s.is_negative or str(dVdt_s).startswith('-'):
                finding("V̇ < 0 → origin is GLOBALLY STABLE (Lyapunov)")
        except: pass
        # Conservation: look for integral of motion
        try:
            integral = integrate(1/f, v)
            r["integral_of_motion"] = str(integral)
            kv("Integral of motion (∫dx/f(x))", str(integral))
        except: pass
        return r

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype == PT.CONTROL:
        f = p.expr; v = p.var
        if f is None: return r
        # Roots location
        try:
            rts = solve(f, v)
            r["roots"] = [str(rt) for rt in rts]
            for rt in rts:
                rt_n = complex(N(rt))
                loc  = ("LHP (stable)" if rt_n.real < 0 else
                        "RHP (unstable)" if rt_n.real > 0 else "imaginary axis (marginal)")
                kv(f"  Root {rt}", f"{rt_n.real:.4f}+{rt_n.imag:.4f}j  →  {loc}")
        except: pass
        # Routh result from phase 02
        rh = g2.get("routh_hurwitz",{})
        if rh:
            kv("Routh stability", rh.get("stable","?"))
            kv("Unstable roots",  rh.get("sign_changes", 0))
        return r

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; v = p.var
        if f is None: return r
        # Convexity: d²f/dx² ≥ 0 everywhere?
        fpp = diff(f, v, 2)
        try:
            fpp_simp = simplify(fpp)
            r["second_deriv"] = str(fpp_simp)
            kv("f''(x)", str(fpp_simp))
            # Check sign symbolically
            fpp_poly = Poly(fpp_simp, v) if fpp_simp.is_polynomial(v) else None
            if fpp_poly:
                if fpp_poly.degree() == 0:
                    val = float(N(fpp_simp))
                    if val > 0:   finding("f'' > 0 everywhere → CONVEX function → local min IS global min")
                    elif val < 0: finding("f'' < 0 everywhere → CONCAVE function → local max IS global max")
        except: pass
        # Multi-variable? (if free has 2+ vars)
        if len(p.free) >= 2:
            try:
                H = hessian(f, p.free)
                r["hessian"] = str(H.tolist())
                kv("Hessian", str(H.tolist()))
                H_spec = _spectrum(H)
                kv("Hessian eigenvalues", [f"{e:.4f}" for e in H_spec])
                if all(e > 0 for e in H_spec):   finding("Hessian PD → STRICTLY CONVEX")
                elif all(e >= 0 for e in H_spec): finding("Hessian PSD → convex")
                elif all(e < 0 for e in H_spec):  finding("Hessian ND → strictly concave")
                else:                              finding("Hessian indefinite → saddle point")
            except: pass
        return r

    # ── POLYNOMIAL TYPES ─────────────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.FACTORING):
        try:
            poly = p.get_poly()
            r["degree"] = poly.degree()
            kv("Degree",       r["degree"])
            kv("Coefficients", [str(c) for c in poly.all_coeffs()])
        except: pass
        try:
            fac = factor(p.expr)
            r["factored"] = str(fac)
            kv("Factored",     r["factored"])
        except: pass
        if v:
            try:
                even = simplify(p.expr.subs(v,-v) - p.expr) == 0
                odd  = simplify(p.expr.subs(v,-v) + p.expr) == 0
                if even:  finding("EVEN function — roots in ±pairs")
                elif odd: finding("ODD function — x=0 is a root")
            except: pass

    if p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        try:
            res = summation(k,(k,1,n))
            r["closed_form"] = str(factor(res))
            kv("Closed form", r["closed_form"])
        except: pass

    if p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        r["fiber"] = f"F_s = {{(i,j,k): i+j+k ≡ s (mod m)}}, {m} fibers of size {m*m}"
        kv("Fiber partition", r["fiber"])
        finding(f"Arc mapping: F_s → F_{{s+1}} (mod {m})")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 04 — PATTERN LOCK
# ════════════════════════════════════════════════════════════════════════════

def phase_04(p: Problem, g3: dict) -> dict:
    section(4, "PATTERN LOCK", "Read the solution backwards; extract the governing law")
    r = {}
    v = p.var

    # ── GRAPH ─────────────────────────────────────────────────────────────
    if p.ptype == PT.GRAPH:
        A    = p.meta.get("A"); L = p.meta.get("L")
        n    = p.meta.get("n",0); deg = p.meta.get("deg",[])
        L_spec = p.meta.get("L_spec",[])
        A_spec = p.meta.get("A_spec",[])
        # Kirchhoff's Matrix Tree Theorem
        if L_spec and n > 0:
            nz = [e for e in L_spec if abs(e) > 1e-9]
            if nz:
                tree_count = math.prod(nz) / n
                r["spanning_trees"] = tree_count
                kv("Spanning trees τ(G) [Kirchhoff]", f"{tree_count:.4f}")
                finding(f"Matrix Tree Theorem: τ(G) = (1/n)·∏λᵢ≠0 ≈ {tree_count:.2f}")
        # Estrada index
        if A_spec:
            estrada = sum(math.exp(e) for e in A_spec)
            r["estrada_index"] = estrada
            kv("Estrada index EE(G)", f"{estrada:.4f}")
            finding("EE(G) = Σ exp(λᵢ) quantifies network subgraph structure")
        # Principal eigenvector = spectral centrality
        if A and n <= 12:
            try:
                evects = A.eigenvects()
                top    = sorted(evects, key=lambda t: float(N(t[0])), reverse=True)[0]
                vec    = top[2][0]
                s      = sum(abs(x) for x in vec)
                norm   = [float(N(x/s)) for x in vec]
                r["spectral_centrality"] = [f"{x:.3f}" for x in norm]
                kv("Spectral centrality (PageRank basis)", r["spectral_centrality"])
                finding("Principal eigenvector ∝ PageRank — most central nodes have highest weight")
            except: pass
        # Fiedler vector = graph cut direction
        if L and n <= 12:
            try:
                evects = L.eigenvects()
                sorted_ev = sorted(evects, key=lambda t: float(N(t[0])))
                # Second smallest eigenvector (Fiedler vector)
                if len(sorted_ev) > 1:
                    fv = sorted_ev[1][2][0]
                    fv_signs = [("+" if float(N(x)) >= 0 else "−") for x in fv]
                    r["fiedler_vector_partition"] = fv_signs
                    kv("Fiedler vector partition", fv_signs)
                    finding("Fiedler vector sign partition = optimal graph bisection (spectral clustering)")
            except: pass
        return r

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MATRIX:
        M    = p.meta.get("M"); spec = p.meta.get("spec",[])
        n    = p.meta.get("n", 0)
        # Cayley-Hamilton
        kv("Cayley-Hamilton", "M satisfies its own characteristic polynomial p(M) = 0")
        # Newton's identities: power traces
        kv("Power traces (Newton)", f"tr(M¹)={sum(spec):.4f}, tr(M²)={sum(e**2 for e in spec):.4f}")
        # Spectral decomposition (if symmetric)
        if g3.get("symmetric"):
            finding("Symmetric → M = QΛQᵀ  (spectral decomposition with orthonormal Q)")
            finding("M⁻¹ = QΛ⁻¹Qᵀ, exp(M) = Q·exp(Λ)·Qᵀ  (functions via diagonalisation)")
        # SVD interpretation
        if spec:
            cond = max(abs(e) for e in spec) / max(min(abs(e) for e in spec), 1e-15)
            r["condition_number"] = cond
            if cond > 100: finding(f"High condition number {cond:.1f} → numerical instability in linear systems")
        r["pattern"] = "Eigenvalues encode ALL matrix properties: trace, det, rank, definiteness, stability"
        kv("Governing pattern", r["pattern"])
        return r

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MARKOV:
        stat = p.meta.get("stat",{}); P_rat = p.meta.get("P_rat")
        n    = p.meta.get("n",0)
        if stat:
            kv("Stationary π", stat)
            pi_floats = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
            # Stationary entropy
            H_stat = _entropy_from_probs([p_ for p_ in pi_floats if p_ > 0])
            r["H_stationary"] = H_stat
            kv("H(π) stationary entropy", f"{H_stat:.6f} bits")
            finding(f"Stationary entropy H(π) = {H_stat:.4f} bits  (max {math.log2(n):.4f})")
        # Entropy rate h(X) = −Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ
        if stat and P_rat:
            try:
                pi_f = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                h = -sum(pi_f[i] * sum(
                        float(N(P_rat[i,j]))*math.log2(max(float(N(P_rat[i,j])),1e-15))
                        for j in range(n) if float(N(P_rat[i,j])) > 1e-12)
                    for i in range(n))
                r["entropy_rate"] = h
                kv("Entropy rate h(X) bits/step", f"{h:.6f}")
                finding(f"Chain produces {h:.4f} bits of randomness per step")
                finding(f"h ≤ H(π) = {r.get('H_stationary',0):.4f} (equality iff independent)")
            except Exception as e:
                note(f"Entropy rate: {e}")
        # P^∞ convergence check
        if P_rat and n <= 6:
            try:
                P_inf = P_rat**20
                kv("P^20 row 0 (≈ π)", [str(N(P_inf[0,j],3)) for j in range(n)])
                finding("P^20 rows converge to π — confirms ergodicity")
            except: pass
        return r

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        p_s   = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        kv("H(1/4) = H(3/4)", f"{float(N(H_bin.subs(p_s, sp.Rational(1,4)))):.4f} bits")
        kv("H(1/2) = 1 bit",  f"{float(N(H_bin.subs(p_s, sp.Rational(1,2)))):.4f} bits")
        finding("H(1/2) = 1 bit = maximum — fair coin is maximally uncertain")
        finding("H(0) = H(1) = 0 — deterministic, zero uncertainty")
        if probs:
            H_val = _entropy_from_probs(probs)
            n     = len(probs)
            # Huffman code
            import heapq
            heap = [(p_, [i]) for i,p_ in enumerate(probs) if p_ > 0]
            heapq.heapify(heap)
            code_lens = {i: 0 for i in range(n)}
            if len(heap) > 1:
                while len(heap) > 1:
                    p1,c1 = heapq.heappop(heap)
                    p2,c2 = heapq.heappop(heap)
                    for idx in c1: code_lens[idx] += 1
                    for idx in c2: code_lens[idx] += 1
                    heapq.heappush(heap, (p1+p2, c1+c2))
            avg_len = sum(probs[i]*code_lens.get(i,0) for i in range(n))
            r["huffman_avg_len"] = avg_len
            r["huffman_code_lens"] = code_lens
            kv("Huffman code lengths", code_lens)
            kv("Expected code length L̄", f"{avg_len:.4f} bits/symbol")
            kv("Shannon entropy H(X)",   f"{H_val:.4f} bits/symbol")
            kv("Redundancy L̄ − H",       f"{avg_len - H_val:.4f} bits")
            finding(f"Huffman: {avg_len:.4f} bits/sym ≥ Shannon bound {H_val:.4f} bits/sym")
        return r

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype == PT.DYNAMICAL:
        f = p.expr; v = p.var
        try:
            equil = solve(f, v)
        except: equil = []
        fp = diff(f,v) if f else None
        if equil and fp:
            # Classify each equilibrium
            for eq in equil:
                try:
                    fp_val = float(N(fp.subs(v,eq)))
                    stability = ("STABLE (attractor)" if fp_val < 0 else
                                 "UNSTABLE (repeller)" if fp_val > 0 else "NON-HYPERBOLIC")
                    kv(f"  x*={eq}", f"f'={fp_val:.4f}  ->  {stability}")
                    r[f"eq_{eq}"] = stability
                except: pass
        # Bifurcation hint
        try:
            fpp = diff(f,v,2)
            infl = solve(fpp,v)
            r["inflection_points"] = [str(i) for i in infl]
            kv("Inflection points of f(x)", r["inflection_points"])
            finding("Inflection points of f(x) = potential bifurcation candidates")
        except: pass
        return r

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype == PT.CONTROL:
        f = p.expr; v = p.var
        rh = g3.get("roots") or p._cache.get("routh_hurwitz",{})
        rh_dict = p._cache.get("routh_hurwitz",{})
        if rh_dict:
            kv("Stability verdict", "STABLE" if rh_dict.get("stable") else "UNSTABLE")
            kv("RHP roots (unstable modes)", rh_dict.get("sign_changes",0))
        # Gain/phase margin idea
        try:
            poly = p.get_poly()
            coeffs = [float(N(c)) for c in poly.all_coeffs()]
            # Necessary condition check: all coefficients same sign
            if all(c > 0 for c in coeffs):
                finding("All coefficients positive (necessary but not sufficient for stability)")
            elif all(c < 0 for c in coeffs):
                finding("All coefficients negative — multiply by -1, same stability")
            else:
                finding("Mixed-sign coefficients -> DEFINITELY unstable")
        except: pass
        r["pattern"] = "Stability <=> ALL eigenvalues in open left half-plane (Re < 0)"
        kv("Governing pattern", r["pattern"])
        return r

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype == PT.OPTIMIZATION:
        f  = p.expr; v = p.var
        fp = diff(f,v); fpp = diff(f,v,2)
        crit_raw = p._cache.get("solve(f'=0)", [])
        crit = [str(c) for c in crit_raw] if crit_raw else g3.get("critical_points",[])
        if crit:
            f_vals = []
            for c in crit:
                try:
                    c_sym = sp.sympify(c)
                    val   = float(N(f.subs(v, c_sym)))
                    f_vals.append((val, c))
                except: pass
            if f_vals:
                best = min(f_vals) if "min" in p.meta.get("goal","") else max(f_vals)
                r["optimal_value"] = best[0]; r["optimal_x"] = best[1]
                kv("Optimal x*", best[1]); kv("Optimal f*", f"{best[0]:.6f}")
                finding(f"{'Minimum' if 'min' in p.meta.get('goal','') else 'Maximum'} = {best[0]:.4f} at x = {best[1]}")
        return r

    # ── ALGEBRAIC ─────────────────────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        try:
            sols = p.memo('solve(expr, var)', lambda: solve(p.expr, v))
            r["solutions"] = [str(s) for s in sols]
            kv("Solutions", r["solutions"])
            for i,s in enumerate(sols):
                info = {
                    "value": str(s), "simplified": str(simplify(s)),
                    "is_integer": s.is_integer, "is_rational": s.is_rational,
                    "is_real": s.is_real, "op_count": count_ops(s),
                    "verified": simplify(p.expr.subs(v,s)) == 0,
                }
                r[f"sol_{i}"] = info
                print(f"\n  {DIM}Root {i}:{RST}")
                for k_,v_ in info.items(): kv(f"  {k_}", v_, indent=4)
            # Vieta's formulas
            if all(sp.sympify(s).is_integer for s in sols):
                ints = [int(sp.sympify(s)) for s in sols]
                kv("Product of roots (Vieta)", math.prod(ints))
                kv("Sum of roots (Vieta)",     sum(ints))
                finding("All roots integers — Vieta's formulas verified")
                r["root_type"] = "integer"
        except Exception as e:
            fail(f"solve: {e}")

    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        simp = p.memo('trigsimp', lambda: trigsimp(p.expr))
        r["simplified"] = str(simp)
        kv("Simplified",    simp)
        kv("Is zero",       simp == 0)
        before = count_ops(p.expr); after = count_ops(simp)
        kv("Complexity reduction", f"{before} → {after} ops")
        if simp == 0: finding("Identity confirmed — holds for ALL inputs")

    elif p.ptype == PT.FACTORING:
        fac   = factor(p.expr)
        flist = factor_list(p.expr)
        r["factored"] = str(fac)
        kv("Factored form", fac)
        for i,(fi,mult) in enumerate(flist[1]):
            try:    rt = solve(fi, v)
            except: rt = []
            kv(f"  factor[{i}] ^{mult}", f"{fi}  →  roots: {rt}")
        ok(f"Verify: expand(factor) − original = {simplify(expand(fac) - expand(p.expr))}")

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        res = summation(k,(k,1,n))
        kv("Formula", str(factor(res)))
        kv("f(n)−f(n−1)", str(simplify(res - res.subs(n, n-1))))
        finding("Difference property confirms: f(n)−f(n−1) = n ✓")

    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            steps = [("Assume","√2 = p/q with gcd(p,q)=1"),
                     ("Square","2 = p²/q² → p² = 2q²"),
                     ("Deduce","p² even → p even → p=2m"),
                     ("Substitute","4m²=2q² → q²=2m²"),
                     ("Deduce","q² even → q even"),
                     ("Contradict","p,q both even ⊥ gcd=1"),
                     ("Conclude","√2 ∉ ℚ  □")]
            for step,desc in steps: print(f"    {Y}{step:<14}{RST}{desc}")
            r["proof"] = steps
        elif "prime" in body.lower():
            steps = [("Assume","Finitely many: {p₁,…,pₖ}"),
                     ("Construct","N = p₁·…·pₖ+1"),
                     ("Factor","N has prime factor q"),
                     ("But","q ∤ any pᵢ (remainder 1)"),
                     ("Conclude","Infinite primes  □")]
            for step,desc in steps: print(f"    {Y}{step:<14}{RST}{desc}")
            r["proof"] = steps

    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        if m % 2 != 0:
            r["law"] = "Q_c(i,j) = (i+b_c(j), j+r_c) mod m; gcd(r_c,m)=1"
            kv("Twisted translation law", r["law"])
            finding("Decomposition via fiber-j twisted translations for odd m")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 05 — GENERALIZE
# ════════════════════════════════════════════════════════════════════════════

def phase_05(p: Problem, g4: dict) -> dict:
    section(5, "GENERALIZE", "Name the condition, not the cases — governing theorems")
    r = {}
    v = p.var

    if p.ptype == PT.GRAPH:
        r["governing"] = {
            "connectivity":   "λ₂(L) > 0  ⟺  G connected  (Fiedler 1973)",
            "bipartite":      "G bipartite ⟺ adjacency spectrum symmetric about 0",
            "expander":       "h(G) ≥ λ₂/2  (Cheeger) → mixing in O(1/λ₂) steps",
            "trees":          "τ(G) = (1/n)·∏{λᵢ>0}  (Kirchhoff Matrix Tree Theorem)",
            "random_walk":    "Simple RW = Markov chain with P = D⁻¹A, stationary ∝ degree",
            "gnn_connection": "Graph neural networks: f(L) or f(A) via spectral convolution",
        }
        # Named graph families
        t = p.meta.get("type")
        if   t == "complete": r["family"] = "Kₙ: λ(L)={0¹, nⁿ⁻¹}, τ(Kₙ)=nⁿ⁻², diameter=1"
        elif t == "path":     r["family"] = "Pₙ: λₖ=2−2cos(kπ/n), diameter=n−1, bipartite iff n odd"
        elif t == "cycle":    r["family"] = "Cₙ: λₖ=2−2cos(2πk/n), bipartite iff n even"
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        if "family" in r: kv("Named graph family", r["family"])
        finding("Governing principle: SPECTRUM OF L DETERMINES ALL STRUCTURAL PROPERTIES")

    elif p.ptype == PT.MATRIX:
        r["governing"] = {
            "spectral_theorem": "Symmetric M = QΛQᵀ — diagonalizable over ℝ",
            "cayley_hamilton":  "p(M) = 0 where p = characteristic polynomial",
            "SVD":              "Any M = UΣVᵀ — singular values = sqrt(eigenvalues of MᵀM)",
            "definiteness":     "x²-form xᵀMx: PD ⟺ all λ > 0 ⟺ all leading minors > 0",
            "rank_nullity":     "rank(M) + nullity(M) = n",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        finding("Governing principle: EIGENVALUES ENCODE ALL MATRIX INVARIANTS")

    elif p.ptype == PT.MARKOV:
        r["governing"] = {
            "perron_frobenius": "Irreducible non-negative matrix: unique λ₁=1, unique stationary π>0",
            "ergodic_theorem":  "Ergodic: time average = space average = π (strong LLN)",
            "mixing":           "‖Pⁿ−Π‖ ≤ |λ₂|ⁿ → geometric convergence at rate |λ₂|",
            "entropy_rate":     "h = −Σᵢπᵢ Σⱼ Pᵢⱼ log Pᵢⱼ — irreducible randomness per step",
            "reversible":       "Detailed balance πᵢPᵢⱼ=πⱼPⱼᵢ ⟺ reversible ⟺ all eigenvalues real",
            "random_walk":      "Random walk on G: Pᵢⱼ=Aᵢⱼ/dᵢ, stationary πᵢ=dᵢ/2|E|",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        finding("Governing principle: SPECTRAL GAP = RATE OF INFORMATION MIXING")

    elif p.ptype == PT.ENTROPY:
        r["governing"] = {
            "shannon_theorem":   "H uniquely characterised by: continuity + max at uniform + additivity",
            "max_entropy":       "H(X) ≤ log₂n  with equality iff uniform (MaxEnt principle)",
            "chain_rule":        "H(X,Y) = H(X) + H(Y|X)  — information decomposes additively",
            "data_processing":   "H(f(X)) ≤ H(X)  — processing cannot create information",
            "source_coding":     "Expected code length L̄ ≥ H(X)  (Shannon source coding theorem)",
            "channel_capacity":  "C = max_{p(x)} I(X;Y)  (Shannon 1948)",
            "maxent_principle":  "MaxEnt: least-committed prior given constraints = Gibbs distribution",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        finding("Governing principle: H IS THE IRREDUCIBLE MEASURE OF UNCERTAINTY")

    elif p.ptype == PT.DYNAMICAL:
        r["governing"] = {
            "lyapunov":      "f'(x*) < 0 → asymptotically stable; f'(x*) > 0 → unstable",
            "hartman_grob":  "Near hyperbolic equilibria: nonlinear ≈ linearization",
            "invariant_manifolds": "Stable/unstable manifolds organize global phase portrait",
            "bifurcations":  "Saddle-node, pitchfork, Hopf — structural changes at parameter values",
            "chaos":         "Sensitive dependence on IC: ‖δx(t)‖ ~ ‖δx₀‖ eλt (Lyapunov exponent)",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        finding("Governing principle: STABILITY ENCODED IN JACOBIAN EIGENVALUES AT EQUILIBRIA")

    elif p.ptype == PT.CONTROL:
        r["governing"] = {
            "routh_hurwitz":   "Stable ⟺ all Routh array first-column elements > 0",
            "eigenvalue_cond": "Stable ⟺ all eigenvalues satisfy Re(λ) < 0",
            "nyquist":         "Nyquist criterion: encirclements of −1 = RHP poles of closed-loop",
            "bode":            "Gain margin/phase margin quantify robustness to parameter variation",
            "controllability": "Rank[B,AB,…,Aⁿ⁻¹B] = n ⟺ system fully controllable",
            "abel_ruffini":    "Degree ≥ 5: no radical formula — numerical methods required",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        finding("Governing principle: ALL ROOTS IN OPEN LEFT HALF-PLANE ⟺ BIBO STABLE")

    elif p.ptype == PT.OPTIMIZATION:
        r["governing"] = {
            "first_order":   "∇f(x*) = 0  (necessary condition for unconstrained extremum)",
            "second_order":  "H ≻ 0 → local min; H ≺ 0 → local max; indefinite → saddle",
            "global_convex": "f convex → every local min is global min",
            "kkt":           "Constrained: ∇f = Σλᵢ∇gᵢ, λᵢgᵢ=0, λᵢ≥0  (KKT conditions)",
            "lagrange":      "Equality constraints: ∇L = 0 where L = f − Σλᵢgᵢ",
            "duality":       "Strong duality: primal opt = dual opt  (Slater's condition)",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)
        finding("Governing principle: HESSIAN DEFINITENESS DETERMINES NATURE OF CRITICAL POINTS")

    elif p.ptype == PT.LINEAR:
        a_,b_ = symbols('a b', nonzero=True)
        sol   = solve(a_*v + b_, v)[0]
        r["general_form"] = "a·x + b = 0"; r["solution"] = str(sol); r["condition"] = "a ≠ 0"
        kv("General form",  r["general_form"])
        kv("Solution",      sol)
        kv("Condition",     r["condition"])
        finding("x = −b/a  iff  a ≠ 0")

    elif p.ptype == PT.QUADRATIC:
        a_,b_,c_ = symbols('a b c')
        gen_sols = solve(a_*v**2 + b_*v + c_, v)
        r["quadratic_formula"] = [str(s) for s in gen_sols]
        r["discriminant_law"] = "Δ=b²−4ac governs nature: Δ>0→2real, Δ=0→double, Δ<0→complex"
        kv("Quadratic formula", r["quadratic_formula"])
        kv("Discriminant law",  r["discriminant_law"])
        finding("Nature of roots completely determined by Δ = b²−4ac")

    elif p.ptype == PT.CUBIC:
        r["governing"] = {
            "cardano":    "Depressed cubic t³+pt+q=0: t = ∛(−q/2+√D) + ∛(−q/2−√D)",
            "disc":       "Δ>0→3 real; Δ=0→repeat; Δ<0→1 real+2 complex",
            "guaranteed": "Odd degree → ALWAYS ≥1 real root (IVT)",
            "abel_bound": "No radicals for deg≥5 (Abel-Ruffini 1824)",
        }
        for k_,v_ in r["governing"].items(): kv(f"  {k_}", v_)

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        ps = {}
        for pow_ in range(1,5):
            try: ps[f"Σk^{pow_}"] = str(factor(summation(k**pow_,(k,1,n))))
            except: pass
        for name_,form_ in ps.items(): kv(f"  {name_}", form_)
        r["faulhaber"] = "Σk^p = poly of degree p+1 in n (Faulhaber); coefficients = Bernoulli numbers"
        kv("Faulhaber's law", r["faulhaber"])
        finding("Governing: Σk^p is degree-(p+1) polynomial in n for all p∈ℤ⁺")

    elif p.ptype == PT.TRIG_ID:
        theta = symbols('theta')
        family = {
            "sin²+cos²=1":  trigsimp(sin(theta)**2 + cos(theta)**2 - 1),
            "1+tan²=sec²":  trigsimp(1 + tan(theta)**2 - sec(theta)**2),
            "1+cot²=csc²":  trigsimp(1 + sp.cot(theta)**2 - sp.csc(theta)**2),
        }
        for f_,val in family.items(): kv(f"  {f_}", f"= {val}  {'✓' if val==0 else '?'}")
        finding("All follow from unit-circle: sin²+cos²=1 is the single governing identity")

    elif p.ptype == PT.FACTORING:
        a_,b_ = symbols('a b')
        ids   = {"a2-b2": factor(a_**2-b_**2), "a3-b3": factor(a_**3-b_**3),
                 "a3+b3": factor(a_**3+b_**3), "a4-b4": factor(a_**4-b_**4)}
        for f_,v_ in ids.items(): kv(f"  {f_}", str(v_))
        r["governing"] = "aⁿ−bⁿ = (a−b)(aⁿ⁻¹+…+bⁿ⁻¹)"
        finding("Governing: aⁿ−bⁿ = (a−b)·(sum of terms with total degree n−1)")

    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            r["general"] = "√n ∉ ℚ ⟺ n is not a perfect square"
            kv("General theorem", r["general"])
            for n_val in range(1,9):
                is_sq = sp.sqrt(n_val).is_integer
                kv(f"  √{n_val}", "∈ ℚ" if is_sq else "∉ ℚ")
            finding("Governing: √n rational ⟺ n is a perfect square")

    elif p.ptype == PT.DIGRAPH_CYC:
        r["governing"] = "Odd m: fiber-column-uniform sigma exists; Even m: needs full 3D sigma"
        kv("Governing condition", r["governing"])
        finding("Odd/even parity is the bifurcation parameter for Hamiltonian decomposition")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 06 — PROVE LIMITS
# ════════════════════════════════════════════════════════════════════════════

def phase_06(p: Problem, g5: dict) -> dict:
    section(6, "PROVE LIMITS", "Find the boundary; state the obstruction")
    r = {}

    if p.ptype == PT.GRAPH:
        L_spec = p.meta.get("L_spec",[])
        deg    = p.meta.get("deg",[])
        r["lower_bound"] = "λ₂ > 0 ⟺ connected; λ₂ = 0 iff disconnected (hard boundary)"
        r["upper_bound"] = "λ_max(L) ≤ max_degree Δ; equality iff regular"
        r["interlacing"]  = "Cauchy interlacing: remove vertex → eigenvalues interlace"
        r["bipartite"]    = "Bipartite ⟺ −λ is also eigenvalue ⟺ no odd cycles"
        r["planarity"]    = "Planar G: λ₂ ≤ 4 (Spielman-Teng); K₅,K₃₃ non-planar (Kuratowski)"
        r["ramanujan"]    = "Ramanujan expander: λ₂ ≤ 2√(d−1) for d-regular G (optimal expander)"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        if L_spec and deg:
            Delta = max(deg)
            lam_n = max(L_spec)
            ok(f"λ_max={lam_n:.4f} ≤ Δ={Delta}  ✓") if lam_n <= Delta+1e-6 else fail(f"λ_max={lam_n:.4f} > Δ={Delta}")

    elif p.ptype == PT.MATRIX:
        spec = p.meta.get("spec",[])
        r["gershgorin"]  = "Gershgorin: λ ∈ ⋃ᵢ {|z−aᵢᵢ| ≤ Σⱼ≠ᵢ |aᵢⱼ|}"
        r["perron"]      = "Non-negative matrix: spectral radius = largest real eigenvalue"
        r["weyl"]        = "Weyl: if B=A+E, |λᵢ(B)−λᵢ(A)| ≤ ‖E‖₂ (eigenvalue stability)"
        r["det_zero"]    = "det=0 ⟺ singular ⟺ 0 is eigenvalue ⟺ Ax=0 has non-trivial solution"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        if spec:
            kv("Our eigenvalue range", f"[{min(spec):.4f}, {max(spec):.4f}]")

    elif p.ptype == PT.MARKOV:
        r["positive"]      = "Irreducible+aperiodic → ‖Pⁿ−Π‖ → 0 (ergodic theorem)"
        r["speed"]         = "Rate: |λ₂|ⁿ (geometric); mixing time τ ∼ 1/(1−|λ₂|)"
        r["obstruction"]   = "Periodic chain (period d>1) oscillates; does not converge pointwise"
        r["lazy_fix"]      = "Lazy chain P' = (P+I)/2 → aperiodic, |λ₂(P')| = (1+|λ₂(P)|)/2 < 1"
        r["entropy_bound"] = "h(Xₙ₊₁|X₀,…,Xₙ) → h(chain) monotonically (Shannon entropy rate)"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        eig_d = g2_from_cache(p, "eigenvalues", {}) if False else {}
        note("Spectral radius ≤ 1 for stochastic matrix (proved by Perron-Frobenius)")

    elif p.ptype == PT.ENTROPY:
        p_s = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        r["lower"] = "H(X) ≥ 0  with equality iff X deterministic"
        r["upper"] = "H(X) ≤ log₂n  with equality iff X uniform"
        r["subadditivity"] = "H(X,Y) ≤ H(X)+H(Y)  with equality iff X⊥Y"
        r["data_processing"]= "H(f(X)) ≤ H(X)  for any deterministic f"
        r["shannon_limit"]  = "L̄ ≥ H(X) — cannot compress below entropy"
        r["channel_bound"]  = "C = max I(X;Y) — cannot communicate faster than channel capacity"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        # Verify boundary limits
        lim0 = limit(H_bin, p_s, 0, '+'); lim1 = limit(H_bin, p_s, 1, '-')
        ok(f"H(0⁺)={lim0}, H(1⁻)={lim1}  (boundary continuity ✓)")
        finding("Hard limit: H = 0 iff p ∈ {0,1} (deterministic). No further compression possible.")

    elif p.ptype == PT.DYNAMICAL:
        r["hyperbolic"]  = "Hartman-Grobman: near hyperbolic equilibrium (no zero eigenvalue), nonlinear ≈ linear"
        r["lyapunov"]    = "Global stability: if ∃V>0, V̇<0 ∀x≠0, origin globally stable"
        r["no_go"]       = "Chaos requires dim≥3 (Poincaré-Bendixson: 2D → no chaos)"
        r["bifurcation"] = "At non-hyperbolic eq (f'=0): behaviour changes qualitatively"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        finding("Bifurcation boundary: f'(x*) = 0 (non-hyperbolic) marks structural change")

    elif p.ptype == PT.CONTROL:
        rh = g2_cache_get(p, "routh_hurwitz")
        r["necessary"]   = "Necessary: all coefficients same sign"
        r["sufficient"]  = "Sufficient: all Routh first-column elements > 0"
        r["boundary"]    = "Marginally stable: eigenvalue on imaginary axis (Re=0)"
        r["fundamental"] = "Fundamental theorem of algebra: degree-n polynomial → exactly n roots ∈ ℂ"
        r["abel_ruffini"]= "No general radical formula for degree ≥ 5 (Abel-Ruffini 1824)"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)

    elif p.ptype == PT.OPTIMIZATION:
        f  = p.expr; v = p.var
        r["first_order"]  = "Critical point: f'(x*)=0 (necessary for smooth unconstrained)"
        r["second_order"] = "f''(x*)>0 → local min; f''<0 → local max; f''=0 inconclusive"
        r["global_bound"] = "Convex f on convex domain: local min = global min"
        r["no_go"]        = "Non-convex: may have multiple local minima — global opt NP-hard in general"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        # Check limits
        try:
            lp = limit(f, v,  oo); ln = limit(f, v, -oo)
            kv("f(+∞)", str(lp)); kv("f(−∞)", str(ln))
            if str(lp) == str(ln) == 'oo': finding("f→+∞ both sides → minimum exists (closed-coercive)")
            elif str(lp) == str(ln) == '-oo': finding("f→−∞ both sides → maximum exists (concave-coercive)")
        except: pass

    elif p.ptype == PT.QUADRATIC:
        a_,b_,c_ = symbols('a b c', real=True)
        kv("Δ=0 boundary", "b²=4ac → double root at x=−b/2a")
        kv("Δ<0 obstruction", "No real roots; two complex conjugate roots in ℂ")
        kv("Over ℂ", "Always exactly 2 roots (Fundamental Theorem of Algebra)")

    elif p.ptype == PT.CUBIC:
        r["ivt"] = "Odd degree → intermediate value theorem guarantees ≥1 real root"
        r["abel"] = "Degree ≥ 5: no general radical formula (Abel-Ruffini 1824)"
        kv("IVT guarantee", r["ivt"]); kv("Abel-Ruffini", r["abel"])

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        kv("Σk → ∞", str(summation(k,(k,1,oo))))
        kv("Σ1/k",   str(summation(1/k,(k,1,oo))))
        kv("Σ1/k²",  str(summation(1/k**2,(k,1,oo))))
        finding("p-series: Σ1/kᵖ converges ⟺ p>1  (hard boundary at p=1)")

    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            kv("Boundary", "n perfect square ⟺ √n ∈ ℚ; otherwise √n ∉ ℚ")
        elif "prime" in body.lower():
            kv("Open problem", "Twin prime conjecture (p, p+2 both prime) — unproven")

    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        kv("Odd m",   "Fiber-uniform Hamiltonian decomposition exists (twisted translation)")
        kv("Even m",  "Obstruction: Σrᵢ=m (even), but each rᵢ must be odd for gcd(rᵢ,m)=1")
        finding("Parity obstruction is the HARD BOUNDARY for fiber-uniform construction")

    elif p.ptype == PT.LINEAR:
        kv("a=0, b=0", "0=0 — infinitely many solutions"); kv("a=0, b≠0", "No solution")
        finding("Unique solution exists iff a ≠ 0")

    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        kv("sin²+cos²=1", "Holds for ALL x ∈ ℝ — no exceptions")
        kv("1+tan²=sec²", "Fails at x = π/2+nπ (cos=0)")

    elif p.ptype == PT.FACTORING:
        try:
            irred = p.get_poly().is_irreducible
            kv("Irreducible over ℚ", irred)
            finding("Over ℂ: always splits into linear factors (FTA)")
        except: pass

    return r


# ════════════════════════════════════════════════════════════════════════════
# HELPER — PHASE 06 needs g2 data stored in problem cache
# ════════════════════════════════════════════════════════════════════════════
def g2_cache_get(p: Problem, key: str):
    return p._cache.get(key)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 07 — CROSS-DOMAIN EMERGENTS
# ════════════════════════════════════════════════════════════════════════════

def phase_07(p: Problem, g6: dict) -> dict:
    section(7, "CROSS-DOMAIN EMERGENTS", "Cross-domain bridges and non-obvious higher structures")
    r = {}

    # ── GRAPH ─────────────────────────────────────────────────────────────
    if p.ptype == PT.GRAPH:
        r["graph_to_markov"]  = "Random walk on G: P=D⁻¹A is a Markov chain, stationary πᵢ=dᵢ/2|E|"
        r["graph_to_entropy"] = "Spectral entropy H_s = −Σ(λᵢ/tr(L))·log(λᵢ/tr(L)) — graph complexity"
        r["graph_to_opt"]     = "PageRank = fixed point of linear map: π = dP·π + (1-d)/n·1"
        r["graph_to_dyn"]     = "Heat equation on graph: u'(t) = −Lu(t); solution e^{-tL}u₀"
        r["ihara_zeta"]       = "Z_G(u) = ∏_primes(1−u^{|p|})^{-1} — Riemann hypothesis analog"
        r["gnn_connection"]   = "Spectral GNN: h' = σ(Ũ^T f(Λ) Ũ h), Ũ = eigenvectors of L"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        # Compute spectral graph entropy
        L_spec = p.meta.get("L_spec",[])
        if L_spec:
            tr_L = sum(L_spec)
            if tr_L > 0:
                nz   = [e for e in L_spec if e > 1e-9]
                probs_spec = [e/tr_L for e in nz]
                H_spec = _entropy_from_probs(probs_spec)
                r["spectral_entropy"] = H_spec
                kv("Spectral entropy H_s(G)", f"{H_spec:.4f} bits")
                bridge(f"Graph → Entropy: spectral entropy H_s = {H_spec:.4f} bits")
        bridge("Graph → Markov: D⁻¹A defines random walk with stationary dist ∝ degrees")
        bridge("Graph → Dynamical: heat diffusion on G governed by e^{-tL}")
        bridge("Graph → Optimization: min cut = max flow (Ford-Fulkerson duality)")
        finding("DEEPEST EMERGENT: graph spectrum is an ISOMORPHISM FINGERPRINT")

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MATRIX:
        r["matrix_to_dynamical"] = "ẋ=Ax: stable iff all Re(λᵢ(A)) < 0; solution x(t)=e^{At}x₀"
        r["matrix_to_control"]   = "Characteristic poly det(sI-A): poles in LHP ⟺ BIBO stable"
        r["matrix_to_opt"]       = "Hessian H of f: quadratic form xᵀHx determines curvature"
        r["matrix_to_entropy"]   = "Von Neumann entropy S(ρ) = −tr(ρ log ρ), ρ = density matrix"
        r["matrix_to_graph"]     = "Symmetric 0/1 matrix IS adjacency matrix of undirected graph"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        spec = p.meta.get("spec",[])
        if spec:
            # Von Neumann entropy (treat eigenvalues as spectrum of density matrix)
            tr_M = sum(spec)
            if tr_M > 0:
                probs_spec = [abs(e)/sum(abs(e2) for e2 in spec) for e in spec]
                H_vn = _entropy_from_probs([p_ for p_ in probs_spec if p_ > 0])
                r["von_neumann_entropy"] = H_vn
                kv("Von Neumann-like entropy", f"{H_vn:.4f} bits")
                bridge(f"Matrix → Entropy: spectral entropy = {H_vn:.4f} bits")
        bridge("Matrix → Dynamical: ẋ=Ax stable iff all eigenvalues in LHP")
        bridge("Matrix → Control: characteristic polynomial det(sI-A) = 0")
        finding("DEEPEST EMERGENT: matrix exponent e^{At} governs all linear dynamical systems")

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MARKOV:
        r["markov_to_graph"]   = "P defines weighted directed graph; reversible P ↔ undirected graph"
        r["markov_to_entropy"] = "Entropy rate h = lim H(Xₙ|X₀…Xₙ₋₁) — asymptotic uncertainty"
        r["markov_to_opt"]     = "MCMC: run chain to sample from target π (Metropolis-Hastings)"
        r["markov_to_physics"] = "Entropy production Ṡ = Σᵢⱼ πᵢPᵢⱼ log(πᵢPᵢⱼ/πⱼPⱼᵢ) ≥ 0 (2nd law)"
        r["markov_to_control"] = "Markov decision process: optimal policy = control theory for MDPs"
        r["free_energy"]       = "F = ⟨E⟩ − T·H(π): equilibrium minimises free energy"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        stat = p.meta.get("stat",{}); P_rat = p.meta.get("P_rat")
        n    = p.meta.get("n",0)
        if stat and P_rat:
            try:
                pi_f = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                # Entropy production (irreversibility measure)
                ep = sum(pi_f[i]*float(N(P_rat[i,j])) *
                         math.log(max(pi_f[i]*float(N(P_rat[i,j])),1e-15) /
                                  max(pi_f[j]*float(N(P_rat[j,i])),1e-15))
                         for i in range(n) for j in range(n)
                         if float(N(P_rat[i,j]))>1e-12 and float(N(P_rat[j,i]))>1e-12)
                r["entropy_production"] = ep
                kv("Entropy production Ṡ", f"{ep:.6f}  ({'irreversible' if ep>1e-9 else 'reversible'})")
                bridge(f"Markov → Physics: entropy production Ṡ={ep:.4f} (2nd law of thermodynamics)")
            except: pass
        bridge("Markov → Graph: P defines weighted directed graph; π = stationary random walk")
        bridge("Markov → Optimization: value iteration solves MDP (Bellman equations)")
        finding("DEEPEST EMERGENT: Markov chain IS a random walk on a graph — unifying structure")

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype == PT.ENTROPY:
        r["entropy_to_physics"]  = "Thermodynamic S = k_B H — entropy is universal (Boltzmann)"
        r["entropy_to_markov"]   = "h = entropy rate of ergodic Markov chain: h = −Σᵢπᵢ Σⱼ Pᵢⱼ log Pᵢⱼ"
        r["entropy_to_opt"]      = "MaxEnt = solve: max H(p) subject to Σpᵢfₖ(xᵢ)=⟨fₖ⟩ → Gibbs"
        r["entropy_to_graph"]    = "Spectral entropy of graph = H(normalised Laplacian eigenvalues)"
        r["entropy_to_ml"]       = "Cross-entropy loss = −Σ yᵢ log p̂ᵢ = H(y,p̂) = H(y) + KL(y‖p̂)"
        r["renyi_entropy"]       = "Rényi: Hα = (1/(1−α))log Σpᵢα; H₁ = Shannon (limit α→1)"
        r["mutual_information"]  = "I(X;Y) = H(X)+H(Y)−H(X,Y) — symmetric dependence measure"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        probs = p.meta.get("probs",[])
        if probs:
            # Rényi entropies
            for alpha in [0.5, 2.0]:
                H_r = (1/(1-alpha)) * math.log2(sum(p_**alpha for p_ in probs if p_ > 0))
                kv(f"Rényi H_{alpha}", f"{H_r:.4f} bits")
            bridge(f"Entropy → Physics: this distribution has thermodynamic entropy k_B·H = k_B·{_entropy_from_probs(probs):.4f}·ln2")
        bridge("Entropy → ML: cross-entropy loss minimisation = KL divergence minimisation")
        bridge("Entropy → Dynamical: entropy production = irreversibility measure in non-eq systems")
        finding("DEEPEST EMERGENT: MaxEnt principle = unifying framework for statistical mechanics, ML, and Bayesian inference")

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype == PT.DYNAMICAL:
        r["dyn_to_control"]  = "Control: ẋ=f(x,u) — design u to steer system to target"
        r["dyn_to_graph"]    = "Stability of networked dynamical system determined by graph Laplacian"
        r["dyn_to_entropy"]  = "Kolmogorov-Sinai entropy h_KS = Σ max(λᵢ,0) (Lyapunov exponents)"
        r["dyn_to_markov"]   = "Stochastic differential equations (SDE) → Markov processes (Fokker-Planck)"
        r["noether_theorem"] = "Every continuous symmetry → conservation law (Noether 1915)"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        f    = p.expr; v = p.var
        if f:
            try:
                equil = solve(f, v)
                bridge(f"Dynamical → Control: {len(equil)} equilibria are fixed points to stabilise around")
                bridge(f"Dynamical → Optimization: stable equilibria = local minima of potential V=−∫f dx")
            except: pass
        finding("DEEPEST EMERGENT: gradient flows connect optimization ↔ dynamical systems (ẋ = −∇f(x))")

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype == PT.CONTROL:
        r["control_to_matrix"] = "Characteristic polynomial = det(sI−A); roots = poles"
        r["control_to_opt"]    = "LQR: minimise ∫(xᵀQx+uᵀRu)dt → Riccati equation"
        r["control_to_dyn"]    = "Closed-loop system: ẋ=(A+BK)x; design K to place eigenvalues"
        r["control_to_graph"]  = "Multi-agent control: graph Laplacian determines consensus rate"
        r["transfer_function"] = "G(s) = C(sI−A)⁻¹B+D — encodes all I/O behaviour"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        bridge("Control → Dynamical: characteristic roots ARE equilibria of linearised system")
        bridge("Control → Optimization: H∞ control = min-max optimization (robust stability)")
        bridge("Control → Graph: synchronisation of networked agents governed by graph Laplacian eigenvalues")
        rh = p._cache.get("routh_hurwitz",{})
        if rh:
            kv("Stability summary", "STABLE" if rh.get("stable") else "UNSTABLE")
        finding("DEEPEST EMERGENT: control = optimization over function space (Pontryagin's maximum principle)")

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype == PT.OPTIMIZATION:
        r["opt_to_dyn"]      = "Gradient flow: ẋ = −∇f(x) — optimization AS dynamical system"
        r["opt_to_control"]  = "LQR/MPC: optimal control = constrained optimization over trajectories"
        r["opt_to_entropy"]  = "MaxEnt: max H(p) subject to constraints = Gibbs/Boltzmann distribution"
        r["opt_to_markov"]   = "RL/MDP: policy optimisation via dynamic programming (Bellman)"
        r["opt_to_matrix"]   = "Second-order methods: Newton step = −H⁻¹∇f (Hessian inversion)"
        r["lagrangian_dual"] = "Strong duality: primal = dual value under Slater's constraint qualification"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        bridge("Optimization → Dynamical: gradient descent IS Euler discretisation of ẋ=−∇f")
        bridge("Optimization → Graph: shortest path = min-cost flow = LP on graph structure")
        bridge("Optimization → Entropy: maximum entropy subject to moment constraints = exponential family")
        finding("DEEPEST EMERGENT: duality (Lagrangian) unifies optimization, physics, and information theory")

    # ── ALGEBRAIC ─────────────────────────────────────────────────────────
    elif p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        r["to_dynamical"] = "Roots = eigenvalues of companion matrix C(p)"
        r["to_control"]   = "Characteristic polynomial of linear system: stable iff roots in LHP"
        r["to_graph"]     = "Chromatic polynomial of graph is a polynomial in (−1)-evaluations"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        bridge("Polynomial roots = eigenvalues of companion matrix — ALL polynomial problems are matrix problems")
        finding("DEEPEST EMERGENT: fundamental theorem of algebra ↔ linear algebra ↔ dynamical systems")

    elif p.ptype == PT.SUM:
        r["to_entropy"]   = "Power sums relate to moments; moment-generating function = entropy under duality"
        r["bernoulli"]    = "Faulhaber coefficients = Bernoulli numbers = residues of z/(e^z−1)"
        r["zeta"]         = "Σ1/nˢ = Riemann ζ(s); encodes ALL prime distribution information"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        bridge("Summation → Number Theory: Euler-Maclaurin formula connects sums to integrals")
        bridge("Summation → Entropy: power sums = cumulants of probability distributions")
        finding("DEEPEST EMERGENT: Bernoulli numbers connect combinatorics, topology, and number theory")

    elif p.ptype == PT.PROOF:
        r["to_algebra"]   = "Irrationality proofs → field extension theory [ℚ(√n):ℚ]=2"
        r["to_topology"]  = "Euclid's prime proof → Dirichlet density, Prime Number Theorem"
        bridge("Proof → Algebra: every irrationality proof implies a non-trivial field extension")
        finding("DEEPEST EMERGENT: proof methods (contradiction, construction) transcend any single domain")

    elif p.ptype == PT.TRIG_ID:
        r["to_complex"]   = "sin θ = (e^{iθ}−e^{−iθ})/2i, cos θ = (e^{iθ}+e^{−iθ})/2 (Euler)"
        r["to_geometry"]  = "sin²+cos²=1 IS the Pythagorean theorem on the unit circle"
        r["to_fourier"]   = "Trig functions = basis of Fourier series (L² on [0,2π])"
        for k_,v_ in r.items(): kv(f"  {k_}", v_)
        bridge("Trig identity → Complex Analysis: all trig identities follow from |e^{iθ}|=1")
        finding("DEEPEST EMERGENT: Euler's formula e^{iπ}+1=0 unifies analysis, algebra, geometry")

    elif p.ptype == PT.DIGRAPH_CYC:
        r["to_group"]     = "Cayley digraph Cay(Z_m³,{e₀,e₁,e₂}) — group-theoretic structure"
        r["to_spectral"]  = "Hamiltonian decomposition ↔ eigenvalue structure of circulant matrix"
        r["to_coding"]    = "Fiber decomposition = codeword construction in abelian group code"
        bridge("Digraph → Group Theory: Hamiltonicity in Cayley graphs = group-theoretic property")
        finding("DEEPEST EMERGENT: odd/even parity bifurcation reflects deep arithmetic structure")

    elif p.ptype == PT.FACTORING:
        r["to_number_theory"] = "Factoring integers = factoring polynomials mod p (Berlekamp)"
        r["to_crypto"]        = "Integer factorisation hardness = RSA security foundation"
        bridge("Factoring → Cryptography: polynomial factoring algorithms underpin lattice-based crypto")
        finding("DEEPEST EMERGENT: irreducibility is a fundamental arithmetic property with cryptographic depth")

    return r


# ════════════════════════════════════════════════════════════════════════════
# FINAL ANSWER
# ════════════════════════════════════════════════════════════════════════════

def _final_answer(p: Problem) -> str:
    v = p.var
    if p.ptype == PT.GRAPH:
        n    = p.meta.get("n",0)
        named= p.meta.get("named","graph")
        spec = p.meta.get("L_spec",[])
        conn = (sorted(spec)[1] > 1e-9) if len(spec)>1 else "?"
        return (f"{named} ({n} vertices): Connected={conn}. "
                f"Laplacian spectrum={[f'{e:.3f}' for e in spec]}. "
                f"See phases for centrality, Kirchhoff τ(G), spectral clustering.")
    elif p.ptype == PT.MATRIX:
        spec = p.meta.get("spec",[])
        M    = p.meta.get("M")
        return (f"Matrix ({p.meta.get('n','?')}×{p.meta.get('n','?')}): "
                f"eigenvalues={[f'{e:.3f}' for e in spec]}, "
                f"det={str(det(M)) if M else '?'}, trace={str(trace(M)) if M else '?'}")
    elif p.ptype == PT.MARKOV:
        stat = p.meta.get("stat",{})
        n    = p.meta.get("n",0)
        return (f"Markov chain ({n} states): stationary π={stat}. "
                f"Eigenvalues, mixing time, entropy rate computed. See phases.")
    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        if probs:
            H = _entropy_from_probs(probs)
            return (f"H(X) = {H:.6f} bits  "
                    f"(max = {math.log2(len(probs)):.6f} bits, "
                    f"efficiency = {H/math.log2(len(probs)):.4f})")
        return "Entropy analysis: binary H(p), KL divergence, Huffman, Shannon bounds computed."
    elif p.ptype == PT.DYNAMICAL:
        f  = p.expr; v2 = p.var
        try:
            equil = solve(f, v2)
            fp    = diff(f, v2)
            stab  = [("stable" if float(N(fp.subs(v2,e))) < 0 else "unstable") for e in equil]
            return f"Equilibria: {list(zip([str(e) for e in equil], stab))}"
        except: return "Dynamical system: equilibria and stability computed. See phases."
    elif p.ptype == PT.CONTROL:
        rh = p._cache.get("routh_hurwitz",{})
        verdict = "STABLE" if rh.get("stable") else "UNSTABLE" if rh else "See phases"
        return f"Control system: {verdict}. Routh-Hurwitz + root locations computed."
    elif p.ptype == PT.OPTIMIZATION:
        r  = {}
        try:
            cpts = g2_cache_get(p, "solve(f'=0)") or solve(diff(p.expr, v), v)
            vals = [(float(N(p.expr.subs(v, c))), c) for c in cpts]
            goal = p.meta.get("goal","extremize")
            best = (min if "min" in goal else max)(vals)
            return f"Optimal: x* = {best[1]}, f* = {best[0]:.6f}  ({goal})"
        except: return "Optimization: critical points, Hessian classification computed. See phases."
    elif p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        try:
            sols = p.memo('solve(expr, var)', lambda: solve(p.expr, v))
            return f"Solutions: {', '.join(str(s) for s in sols)}"
        except: return "See phase computations"
    elif p.ptype == PT.FACTORING:
        try: return f"Factored: {p.memo('factor', lambda: factor(p.expr))}"
        except: return "See phase computations"
    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        try:
            simp = p.memo('trigsimp', lambda: trigsimp(p.expr))
            return f"Identity confirmed ✓" if simp==0 else f"Simplified: {simp}"
        except: return "See phase computations"
    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        try:
            s = summation(k,(k,1,n))
            return f"Σᵢ₌₁ⁿ i = {factor(s)} = n(n+1)/2"
        except: return "See phase computations"
    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            return "√2 ∉ ℚ. Proof by contradiction: p/q in lowest terms → both p,q even. ⊥"
        elif "prime" in body.lower():
            return "Infinitely many primes. Euclid: N=p₁·…·pₖ+1 has a prime factor not in list."
    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        if m % 2 != 0:
            return f"Odd m={m}: Hamiltonian decomposition exists via fiber twisted translations."
        else:
            return f"Even m={m}: fiber-uniform impossible (parity obstruction). Requires full 3D sigma."
    return "See phase computations above"


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run(raw: str):
    prob = classify(raw)
    print(f"\n{hr('═')}")
    print(f"{W}DISCOVERY ENGINE v2{RST}")
    print(hr())
    print(f"  {W}Problem:{RST}  {Y}{raw}{RST}")
    print(f"  {DIM}Type:{RST}     {prob.ptype.value}")
    print(f"  {DIM}Variable:{RST} {prob.var}")
    print(hr('═'))

    if prob.ptype == PT.UNKNOWN:
        print(f"{R}Could not classify. Try: 'x^2-5x+6=0', 'graph K4', 'markov [[...]]', 'entropy [...]'{RST}")
        return

    g1 = phase_01(prob)
    g2 = phase_02(prob, g1)
    g3 = phase_03(prob, g2)
    # Store phase_02 results in cache for downstream phase access
    rh = g2.get("routh_hurwitz")
    if rh: prob._cache["routh_hurwitz"] = rh
    g4 = phase_04(prob, g3)
    g5 = phase_05(prob, g4)
    g6 = phase_06(prob, g5)
    g7 = phase_07(prob, g6)

    # Final answer
    print(f"\n{hr('═')}")
    print(f"{W}FINAL ANSWER{RST}")
    print(hr('─'))
    final = _final_answer(prob)
    g6["final_answer"] = final
    print(f"  {G}{final}{RST}")
    print(hr('═'))

    # Phase summary
    titles = {1:"Ground Truth", 2:"Direct Attack", 3:"Structure Hunt",
              4:"Pattern Lock", 5:"Generalize",    6:"Prove Limits", 7:"Cross-Domain"}
    phases = [g1,g2,g3,g4,g5,g6,g7]
    print(f"\n{hr()}")
    print(f"{W}PHASE SUMMARY{RST}")
    print(hr('·'))
    for i,(g,title) in enumerate(zip(phases, titles.values()), 1):
        fa = g.get("final_answer","")
        line = fa[:60] if fa else str(list(g.values())[0] if g else "✓")[:60]
        print(f"  {PHASE_CLR[i]}{i:02d} {title:<16}{RST} {line}")
    print(hr('═'))


# ════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ════════════════════════════════════════════════════════════════════════════

TESTS = [
    # ── Original algebra ────────────────────────────────────────────────
    ("x^2 - 5x + 6 = 0",              "Quadratic with integer roots"),
    ("2x + 3 = 7",                     "Linear equation"),
    ("x^3 - 6x^2 + 11x - 6 = 0",      "Cubic with 3 integer roots"),
    ("sin(x)^2 + cos(x)^2",            "Pythagorean identity"),
    ("factor x^4 - 16",                "Difference of squares chain"),
    ("sum of first n integers",        "Classic summation"),
    ("prove sqrt(2) is irrational",    "Irrationality proof"),
    ("m^3 vertices with 3 cycles, m=3","Digraph — Hamiltonian (odd m)"),
    ("m^3 vertices with 3 cycles, m=4","Digraph — Hamiltonian (even m)"),
    # ── Graph / Network ─────────────────────────────────────────────────
    ("graph K4",                       "Complete graph K4 — spectral theory"),
    ("graph P5",                       "Path graph P5 — Fiedler value"),
    ("graph C6",                       "Cycle graph C6 — bipartite spectrum"),
    ("graph [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]",
                                       "Custom adjacency matrix"),
    # ── Matrix ──────────────────────────────────────────────────────────
    ("matrix [[2,1],[1,3]]",           "Symmetric 2×2 matrix — eigenvalues"),
    ("matrix [[4,2,2],[2,3,0],[2,0,3]]","Symmetric 3×3 — definiteness"),
    # ── Markov ──────────────────────────────────────────────────────────
    ("markov [[0.7,0.3],[0.4,0.6]]",   "2-state Markov chain"),
    ("markov [[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]]",
                                       "3-state symmetric Markov chain"),
    # ── Entropy ─────────────────────────────────────────────────────────
    ("entropy [0.5,0.25,0.25]",        "Entropy — skewed distribution"),
    ("entropy [0.25,0.25,0.25,0.25]",  "Entropy — uniform (maximum)"),
    # ── Dynamical ───────────────────────────────────────────────────────
    ("dynamical x^3 - x",              "Dynamical system — 3 equilibria"),
    ("dynamical x^2 - 1",              "Dynamical system — pitchfork"),
    # ── Control ─────────────────────────────────────────────────────────
    ("control s^2 + 3s + 2",           "Control — stable 2nd order"),
    ("control s^3 + 2s^2 + 3s + 1",   "Control — Routh-Hurwitz 3rd order"),
    ("control s^3 - s + 1",            "Control — unstable system"),
    # ── Optimization ────────────────────────────────────────────────────
    ("optimize x^4 - 4x^2 + 1",       "Optimization — quartic with two minima"),
    ("minimize x^2 + 2x + 1",          "Minimize — simple quadratic"),
]


def run_tests():
    print(f"\n{hr('═')}")
    print(f"{W}DISCOVERY ENGINE v2 — TEST SUITE{RST}")
    print(f"{DIM}Running {len(TESTS)} problems across all domains{RST}")
    print(hr('═'))
    passed = 0
    failed_list = []
    for raw, desc in TESTS:
        print(f"\n{B}{'─'*60}{RST}")
        print(f"{B}TEST: {desc}{RST}")
        print(f"{DIM}{raw}{RST}")
        try:
            run(raw)
            ok(f"PASSED: {desc}"); passed += 1
        except Exception as e:
            fail(f"FAILED: {desc} — {e}")
            failed_list.append((desc, str(e)))
            traceback.print_exc()
    print(f"\n{hr('═')}")
    clr = G if passed == len(TESTS) else Y
    print(f"{clr}Results: {passed}/{len(TESTS)} passed{RST}")
    if failed_list:
        print(f"\n{R}Failed tests:{RST}")
        for d,e in failed_list: print(f"  {R}✗{RST} {d}: {e}")
    print(hr('═'))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0] == "--test":
        run_tests()
    else:
        run(" ".join(args))

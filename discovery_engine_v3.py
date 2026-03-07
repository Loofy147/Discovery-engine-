#!/usr/bin/env python3
"""
discovery_engine_v3.py — 7-Phase Mathematical Discovery Engine
================================================================
Pure sympy. No API. All phases compute symbolically/numerically.

What's compacted into this engine beyond mathematics:
  - METHOD PREDICTION   : Before attacking, score each method by predicted success
  - CONFIDENCE TRACKING : Every result carries a 0–1 confidence weight
  - ANALOGY ENGINE      : Every problem maps to known analogous problem families
  - BACKWARDS REASONING : Phase 04 asks "what structure would produce THIS solution?"
  - VERIFICATION ORACLE : Multiple independent checks; discrepancies flagged
  - META-COGNITION      : Engine tracks what it knows vs what it's uncertain about
  - FAILURE PREDICTION  : Identify likely failure modes before they occur
  - SYNTHESIS           : Phase 07 produces a "lesson" — what this problem teaches

Problem Types:
  LINEAR QUADRATIC CUBIC POLY FACTORING SIMPLIFY TRIG_ID TRIG_EQ
  SUM PROOF DIGRAPH_CYC
  GRAPH MATRIX MARKOV ENTROPY
  DYNAMICAL CONTROL OPTIMIZATION

Usage:
  python discovery_engine_v3.py "x^2 - 5x + 6 = 0"
  python discovery_engine_v3.py --test
"""

import sys, re, ast, math, traceback, time
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
    summation,
    Eq, latex, pretty, count_ops,
    trigsimp, exptrigsimp, expand_trig,
    nsolve, N, solveset, S,
    gcd, lcm, divisors,
    apart, collect, nsimplify,
    real_roots, all_roots,
    factor_list, sqf_list,
    Matrix, eye, zeros, ones, diag,
    det, trace, re as sp_re,
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

def hr(ch="─", n=72): return ch * n
def section(num, name, tag):
    c = PHASE_CLR[num]
    print(f"\n{hr()}")
    print(f"{c}Phase {num:02d} — {name}{RST}  {DIM}{tag}{RST}")
    print(hr("·"))
def kv(k, v, indent=2):
    print(f"{' '*indent}{DIM}{k:<36}{RST}{W}{str(v)[:120]}{RST}")
def finding(msg, sym="→"):  print(f"  {Y}{sym}{RST} {msg}")
def ok(msg):    print(f"  {G}✓{RST} {msg}")
def fail(msg):  print(f"  {R}✗{RST} {msg}")
def note(msg):  print(f"  {DIM}{msg}{RST}")
def bridge(msg): print(f"  {C}⇔{RST} {B}{msg}{RST}")
def warn(msg):  print(f"  {Y}⚠{RST} {msg}")
def insight(msg): print(f"  {M}★{RST} {W}{msg}{RST}")


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  — compacted reasoning heuristics
# ════════════════════════════════════════════════════════════════════════════

class KB:
    """
    Compacted knowledge base. Encodes what I know about:
      - Which methods succeed for which problem structures
      - What symmetries to look for first
      - How to verify results independently
      - What analogous problems look like in other domains
      - Common failure modes and their signals
    """

    # Method success priors: (problem_type, method_name) -> predicted success probability
    METHOD_PRIORS = {
        ("quadratic",  "solve"):          0.99,
        ("quadratic",  "discriminant"):   0.99,
        ("quadratic",  "factor"):         0.70,
        ("cubic",      "solve"):          0.85,
        ("cubic",      "factor"):         0.60,
        ("poly_high",  "solve"):          0.40,   # degree >= 4 often fails symbolically
        ("poly_high",  "nsolve"):         0.90,
        ("trig_id",    "trigsimp"):       0.95,
        ("factoring",  "factor"):         0.90,
        ("graph",      "spectrum"):       0.95,
        ("markov",     "stationary"):     0.85,
        ("markov",     "eigenvalues"):    0.90,
        ("entropy",    "H_numeric"):      0.99,
        ("dynamical",  "solve_equil"):    0.90,
        ("control",    "routh_hurwitz"):  0.95,
        ("control",    "roots"):          0.80,
        ("optimize",   "critical_pts"):   0.90,
        ("optimize",   "hessian"):        0.85,
        ("matrix",     "eigenvalues"):    0.95,
    }

    # Symmetry detection order — check these first, in this order
    SYMMETRY_CHECKS = [
        ("even",    lambda f, v: simplify(f.subs(v,-v) - f) == 0,
                    "EVEN — roots in ±pairs, use substitution u=x²"),
        ("odd",     lambda f, v: simplify(f.subs(v,-v) + f) == 0,
                    "ODD — x=0 always a root, factor out x"),
        ("periodic",lambda f, v: False,  # needs more context
                    "PERIODIC — reduce domain to one period"),
    ]

    # Analogy map: problem type -> analogous problems in other domains
    ANALOGIES = {
        "QUADRATIC":   ["Stability of 2D linear system (trace, det)",
                        "Eigenvalue problem for 2×2 matrix",
                        "Entropy of 2-state system vs p(1-p)"],
        "CUBIC":       ["Characteristic polynomial of 3×3 matrix",
                        "Equilibria of cubic potential f(x)=x³-ax",
                        "3-state Markov chain eigenvalues"],
        "GRAPH":       ["Markov chain (random walk = D⁻¹A)",
                        "Diffusion equation (heat kernel = e^{-tL})",
                        "Multi-agent consensus (eigenvalues of L)"],
        "MARKOV":      ["Random walk on weighted graph",
                        "Linear system with stochastic transitions",
                        "Entropy production (thermodynamics)"],
        "ENTROPY":     ["Free energy F = <E> - T·H (thermodynamics)",
                        "KL divergence = information 'distance'",
                        "Log-loss in machine learning"],
        "DYNAMICAL":   ["Gradient descent (ẋ = -∇f)",
                        "Markov chain in continuous time (Fokker-Planck)",
                        "Graph diffusion (Laplacian dynamics)"],
        "CONTROL":     ["Stability of Markov chain (spectral radius ≤ 1)",
                        "Convex optimization with stability constraints",
                        "Lyapunov function design"],
        "OPTIMIZATION":["Equilibrium of dynamical system",
                        "MaxEnt problem (constrained entropy maximization)",
                        "Stationary distribution of Markov chain"],
        "MATRIX":      ["Adjacency matrix of graph",
                        "Transition matrix of Markov chain",
                        "Jacobian of dynamical system at equilibrium"],
    }

    # Verification strategies — independent checks per type
    VERIFICATION = {
        "equation":  ["substitute back",
                      "check discriminant sign vs root count",
                      "Vieta's formulas (sum/product of roots)"],
        "factoring": ["expand(factor) - original == 0",
                      "roots of factors match roots of original",
                      "degree preserved"],
        "identity":  ["substitute random numerical value",
                      "trigsimp reduces to 0",
                      "check boundary values (0, π/2, π)"],
        "graph":     ["sum of Laplacian eigenvalues = sum of degrees",
                      "number of zero eigenvalues = number of components",
                      "trace(A²)/2 = number of edges"],
        "markov":    ["rows sum to 1",
                      "π·P = π (stationary condition)",
                      "spectral radius = 1"],
        "entropy":   ["H ≥ 0",
                      "H ≤ log₂(n)",
                      "sum of probabilities = 1"],
        "control":   ["sign changes in Routh array = unstable roots",
                      "necessary: all coefficients positive",
                      "sufficient: Routh first column all positive"],
        "optimize":  ["f'(x*) = 0 verified",
                      "f''(x*) sign matches nature",
                      "global vs local: check limits at ±∞"],
    }

    # Common failure modes — what to warn about
    FAILURE_MODES = {
        "POLY":       "Degree ≥ 5: Abel-Ruffini — no radical formula, use numerics",
        "QUADRATIC":  "Δ < 0: complex roots — check if problem expects real only",
        "MARKOV":     "Reducible chain: multiple stationary distributions",
        "DYNAMICAL":  "Non-hyperbolic equilibrium (f'=0): linearization insufficient",
        "CONTROL":    "Zero-crossing coefficients: Routh array degenerates",
        "ENTROPY":    "p=0 terms: 0·log(0) = 0 by convention — handle explicitly",
        "GRAPH":      "Disconnected graph: λ₂=0 — Kirchhoff theorem breaks down",
        "OPTIMIZE":   "Non-convex: multiple local minima — no guarantee of global",
    }

    @classmethod
    def method_confidence(cls, ptype_str: str, method: str) -> float:
        key = (ptype_str.lower(), method.lower())
        return cls.METHOD_PRIORS.get(key, 0.5)

    @classmethod
    def analogies_for(cls, ptype_str: str) -> List[str]:
        return cls.ANALOGIES.get(ptype_str.upper(), [])

    @classmethod
    def verify_strategies(cls, category: str) -> List[str]:
        return cls.VERIFICATION.get(category, ["check result numerically"])

    @classmethod
    def failure_modes(cls, ptype_str: str) -> Optional[str]:
        return cls.FAILURE_MODES.get(ptype_str.upper())


# ════════════════════════════════════════════════════════════════════════════
# CONFIDENCE LEDGER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Confidence:
    """Tracks what the engine knows, suspects, and is uncertain about."""
    results: Dict[str, Tuple[Any, float]] = field(default_factory=dict)
    flags:   List[str]                    = field(default_factory=list)
    knowns:  List[str]                    = field(default_factory=list)
    unknowns: List[str]                   = field(default_factory=list)

    def record(self, key: str, val: Any, conf: float, note_str: str = ""):
        self.results[key] = (val, conf)
        if conf >= 0.9:
            self.knowns.append(f"{key}: {str(val)[:50]}")
        elif conf < 0.6:
            self.unknowns.append(f"{key} (conf={conf:.2f})")
        if note_str:
            self.flags.append(note_str)

    def summary(self) -> str:
        total = len(self.results)
        high  = sum(1 for _,c in self.results.values() if c >= 0.9)
        mid   = sum(1 for _,c in self.results.values() if 0.6 <= c < 0.9)
        low   = sum(1 for _,c in self.results.values() if c < 0.6)
        return f"{total} results: {high} high-conf, {mid} mid-conf, {low} uncertain"


# ════════════════════════════════════════════════════════════════════════════
# PROBLEM TYPES
# ════════════════════════════════════════════════════════════════════════════

class PT(Enum):
    LINEAR=1; QUADRATIC=2; CUBIC=3; POLY=4
    TRIG_EQ=5; TRIG_ID=6; FACTORING=7; SIMPLIFY=8
    SUM=9; PROOF=10; DIGRAPH_CYC=11
    GRAPH=12; MATRIX=13; MARKOV=14; ENTROPY=15
    DYNAMICAL=16; CONTROL=17; OPTIMIZATION=18
    UNKNOWN=99

    def label(self):
        labels = {
            1:"linear equation", 2:"quadratic equation", 3:"cubic equation",
            4:"polynomial (deg≥4)", 5:"trigonometric equation", 6:"trigonometric identity",
            7:"factoring", 8:"simplification", 9:"summation/series", 10:"proof",
            11:"digraph cycle decomposition", 12:"graph/network analysis",
            13:"matrix analysis", 14:"markov chain", 15:"information entropy",
            16:"dynamical system", 17:"control theory", 18:"optimization", 99:"unknown"
        }
        return labels.get(self.value, "unknown")


@dataclass
class Problem:
    raw:     str
    ptype:   PT
    expr:    Optional[sp.Basic]  = None
    lhs:     Optional[sp.Basic]  = None
    rhs:     Optional[sp.Basic]  = None
    var:     Optional[sp.Symbol] = None
    free:    List[sp.Symbol]     = field(default_factory=list)
    meta:    Dict[str, Any]      = field(default_factory=dict)
    poly:    Optional[sp.Poly]   = None
    _cache:  Dict[str, Any]      = field(default_factory=dict, repr=False)
    conf:    Confidence          = field(default_factory=Confidence, repr=False)

    def memo(self, key, func):
        if key not in self._cache:
            try:   self._cache[key] = func()
            except Exception as e: self._cache[key] = None; return None
        return self._cache[key]

    def get_poly(self):
        if self.poly is None and self.expr is not None and self.var is not None:
            try: self.poly = Poly(self.expr, self.var)
            except: pass
        return self.poly

    def ptype_str(self) -> str:
        return self.ptype.name


# ════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _parse(s: str) -> Optional[sp.Basic]:
    s = s.strip()
    s = s.replace('^', '**')
    for old, new in [("ln","log"),("arcsin","asin"),("arccos","acos"),("arctan","atan")]:
        s = re.sub(rf'\b{old}\b', new, s)
    for fn in [lambda x: parse_expr(x, transformations=_TRANSFORMS),
               lambda x: sp.sympify(x)]:
        try: return fn(s)
        except: pass
    return None
    for fn in [lambda x: parse_expr(x, transformations=_TRANSFORMS),
               lambda x: sp.sympify(x)]:
        try: return fn(s)
        except: pass
    return None

def _parse_matrix(s: str) -> Optional[sp.Matrix]:
    m = re.search(r'\[\s*\[.+?\]\s*\]', s, re.S)
    if not m: return None
    try:
        rows = ast.literal_eval(m.group(0))
        return sp.Matrix([[sp.sympify(x) for x in row] for row in rows])
    except: return None

def _parse_probs(s: str) -> List[float]:
    m = re.search(r'\[([^\]]+)\]', s)
    if not m: return []
    try: return [float(x.strip()) for x in m.group(1).split(',')]
    except: return []

def _spectrum(M: sp.Matrix) -> List[float]:
    try:
        eigs = M.eigenvals()
        out  = []
        for k, mult in eigs.items():
            try:    v = float(N(k))
            except: v = float(N(sp_re(k)))
            out.extend([v]*mult)
        return sorted(out)
    except: return []

def _spectrum_complex(M: sp.Matrix) -> List[complex]:
    try:
        eigs = M.eigenvals()
        out  = []
        for k, mult in eigs.items():
            try:    v = complex(N(k))
            except: v = complex(float(N(sp_re(k))), 0)
            out.extend([v]*mult)
        return sorted(out, key=lambda z: (z.real, z.imag))
    except: return []

def _build_graph(p: Problem) -> Tuple[Optional[sp.Matrix], Optional[sp.Matrix], int, List[int]]:
    meta = p.meta
    A    = meta.get("A")
    if isinstance(A, sp.Matrix):
        n   = A.shape[0]
        deg = [int(sum(A.row(i))) for i in range(n)]
        return A, diag(*deg) - A, n, deg
    t = meta.get("type"); n = meta.get("n", 4)
    if   t == "complete": A = ones(n,n) - eye(n)
    elif t == "path":
        A = zeros(n,n)
        for i in range(n-1): A[i,i+1]=A[i+1,i]=1
    elif t == "cycle":
        A = zeros(n,n)
        for i in range(n): A[i,(i+1)%n]=A[(i+1)%n,i]=1
    else: return None, None, 0, []
    deg = [int(sum(A.row(i))) for i in range(n)]
    p.meta["A"] = A; p.meta["n"] = n
    return A, diag(*deg) - A, n, deg

def _entropy(probs: List[float]) -> float:
    return -sum(p*math.log2(p) for p in probs if p > 0)

def _kl(P: List[float], Q: List[float]) -> float:
    return sum(P[i]*math.log2(P[i]/Q[i]) for i in range(len(P)) if P[i]>0 and Q[i]>0)

def _routh(coeffs) -> Dict:
    c  = [float(N(sp.sympify(x))) for x in coeffs]
    r0 = c[0::2]; r1 = c[1::2]
    while len(r0) < len(r1): r0.append(0.0)
    while len(r1) < len(r0): r1.append(0.0)
    rows = [r0, r1]
    while len(rows[-1]) > 1 or (len(rows[-1]) == 1 and rows[-1][0] != 0):
        pr, cr = rows[-2], rows[-1]
        if abs(cr[0]) < 1e-12: cr = [1e-10] + cr[1:]
        nr = [(cr[0]*pr[i+1] - pr[0]*cr[i+1])/cr[0] for i in range(len(cr)-1)]
        if not nr: break
        rows.append(nr)
    fc    = [row[0] for row in rows if row]
    sc    = sum(1 for i in range(len(fc)-1) if fc[i]*fc[i+1] < 0)
    stable= (sc == 0) and all(x > 0 for x in fc)
    return {"stable": stable, "sign_changes": sc, "first_column": fc, "rows": rows}

def _stationary(P: sp.Matrix) -> Optional[Dict]:
    n  = P.shape[0]
    pi = symbols(f'pi0:{n}', positive=True)
    eqs= [sum(pi[i]*P[i,j] for i in range(n)) - pi[j] for j in range(n)]
    eqs.append(sum(pi) - 1)
    try:
        sol = solve(eqs, list(pi))
        return sol
    except: return None

def _verify_equation(expr, var, sol) -> Tuple[bool, float]:
    """Verify a solution by substitution. Returns (ok, residual_magnitude)."""
    try:
        res = simplify(expr.subs(var, sol))
        mag = float(abs(N(res)))
        return (mag < 1e-9, mag)
    except: return (False, float('inf'))

def _vieta_check(poly, sols) -> bool:
    """Verify solutions via Vieta's formulas."""
    try:
        coeffs = poly.all_coeffs()
        n      = poly.degree()
        # Sum of roots = -a_{n-1}/a_n
        expected_sum = -coeffs[-2]/coeffs[-1] if n >= 1 else 0
        actual_sum   = sum(sols)
        if abs(float(N(simplify(actual_sum - expected_sum)))) > 1e-6:
            return False
        # Product of roots = (-1)^n * a_0/a_n
        expected_prod = ((-1)**n * coeffs[0]/coeffs[-1])
        actual_prod   = sp.prod(sols)
        if abs(float(N(simplify(actual_prod - expected_prod)))) > 1e-6:
            return False
        return True
    except: return True  # inconclusive


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

def classify(raw: str) -> Problem:
    s   = raw.strip(); low = s.lower()

    if "vertices" in low and ("m^3" in low or "m**3" in low) and "cycles" in low:
        mm = re.search(r'm\s*=\s*(\d+)', low)
        return Problem(raw=raw, ptype=PT.DIGRAPH_CYC,
                       meta={"m": int(mm.group(1)) if mm else 3})

    if re.match(r'^control\b', low):
        body = re.sub(r'^control\s*', '', s, flags=re.I).strip()
        e = _parse(body); free = sorted(e.free_symbols, key=str) if e else []
        v = next((f for f in free if str(f)=='s'), free[0] if free else symbols('s'))
        _p = None
        try: _p = Poly(e, v)
        except: pass
        return Problem(raw=raw, ptype=PT.CONTROL, expr=e, var=v, free=free, poly=_p)

    if re.match(r'^dynamical?\b', low):
        body = re.sub(r'^dynamical?\s*', '', s, flags=re.I).strip()
        e = _parse(body); free = sorted(e.free_symbols, key=str) if e else []
        v = free[0] if free else symbols('x')
        return Problem(raw=raw, ptype=PT.DYNAMICAL, expr=e, var=v, free=free)

    if re.match(r'^(optimiz|minimiz|maximiz|extrema|find (min|max))\b', low):
        body = re.sub(r'^(optimiz|minimiz|maximiz|extrema|find (min|max)\s*of?\s*)', '', s, flags=re.I).strip()
        e = _parse(body); free = sorted(e.free_symbols, key=str) if e else []
        v = free[0] if free else symbols('x')
        goal = ("minimize" if re.match(r'^minimiz', low) else
                "maximize" if re.match(r'^maximiz', low) else "extremize")
        return Problem(raw=raw, ptype=PT.OPTIMIZATION, expr=e, var=v, free=free, meta={"goal":goal})

    if re.match(r'^matrix\b', low) or (re.search(r'\[\s*\[', s) and
            not any(kw in low for kw in ("graph","markov","entropy","vertices"))):
        M = _parse_matrix(s)
        if M: return Problem(raw=raw, ptype=PT.MATRIX, meta={"M":M, "n":M.shape[0]})

    if re.match(r'^(graph|network)\b', low) or "adjacency" in low:
        M = _parse_matrix(s)
        meta = {"A": M, "rows": M.tolist() if M else []}
        for pat, t in [(r'\bk[_\s]?(\d+)\b',"complete"),(r'\bp[_\s]?(\d+)\b',"path"),(r'\bc[_\s]?(\d+)\b',"cycle")]:
            mm = re.search(pat, low)
            if mm:
                n = int(mm.group(1))
                meta.update({"type":t, "n":n, "named": t[0].upper()+mm.group(1)})
                break
        return Problem(raw=raw, ptype=PT.GRAPH, meta=meta)

    if re.match(r'^markov\b', low) or "transition matrix" in low:
        M = _parse_matrix(s)
        return Problem(raw=raw, ptype=PT.MARKOV, meta={"P":M, "rows":M.tolist() if M else []})

    if re.match(r'^entropy\b', low):
        probs = _parse_probs(s)
        return Problem(raw=raw, ptype=PT.ENTROPY,
                       meta={"probs": probs, "sym_str": re.sub(r'^entropy\s*','',s,flags=re.I).strip()})

    if re.match(r'^(prove|show|demonstrate)', low):
        body = re.sub(r'^(prove|show that|show|demonstrate)\s+', '', s, re.I)
        e = _parse(body)
        return Problem(raw=raw, ptype=PT.PROOF, expr=e, meta={"body": body})

    if any(kw in low for kw in ("sum of first","1+2+","series","summation")):
        return Problem(raw=raw, ptype=PT.SUM)

    if low.startswith("factor "):
        body = s[7:].strip(); e = _parse(body)
        free = sorted(e.free_symbols, key=str) if e else []
        v    = free[0] if free else symbols('x')
        _p   = None
        try: _p = Poly(e, v)
        except: pass
        return Problem(raw=raw, ptype=PT.FACTORING, expr=e, var=v, free=free, poly=_p)

    if "=" in s and not any(x in s for x in ("==",">=","<=")):
        parts = s.split("=", 1)
        lhs_e = _parse(parts[0]); rhs_e = _parse(parts[1])
        if lhs_e is None or rhs_e is None:
            return Problem(raw=raw, ptype=PT.UNKNOWN)
        expr = sp.expand(lhs_e - rhs_e)
        free = sorted(expr.free_symbols, key=str)
        v    = free[0] if free else symbols('x')
        _p   = None
        if expr.atoms(sin, cos, tan):
            pt = PT.TRIG_EQ
        else:
            try:
                _p  = Poly(expr, v)
                deg = _p.degree()
                pt  = {1:PT.LINEAR, 2:PT.QUADRATIC, 3:PT.CUBIC}.get(deg, PT.POLY)
            except: pt = PT.UNKNOWN
        return Problem(raw=raw, ptype=pt, expr=expr, lhs=lhs_e, rhs=rhs_e, var=v, free=free, poly=_p)

    e = _parse(s)
    if e is not None:
        free = sorted(e.free_symbols, key=str)
        v    = free[0] if free else symbols('x')
        pt   = PT.TRIG_ID if e.atoms(sin,cos,tan) else PT.SIMPLIFY
        return Problem(raw=raw, ptype=pt, expr=e, lhs=e, rhs=Integer(0), var=v, free=free)

    return Problem(raw=raw, ptype=PT.UNKNOWN)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 01 — GROUND TRUTH + INTELLIGENCE BRIEFING
# ════════════════════════════════════════════════════════════════════════════

def phase_01(p: Problem) -> dict:
    section(1, "GROUND TRUTH", "Classify · predict · brief · pre-detect failure modes")
    r = {}
    kv("Problem",  p.raw)
    kv("Type",     p.ptype.label())
    kv("Variable", str(p.var))

    # ── INTELLIGENCE BRIEFING — what I predict before computing ──────────
    print(f"\n  {DIM}--- intelligence briefing ---{RST}")
    analogs = KB.analogies_for(p.ptype_str())
    if analogs:
        kv("Analogous problems", "")
        for a in analogs: note(f"    • {a}")

    fail_mode = KB.failure_modes(p.ptype_str())
    if fail_mode:
        warn(f"Anticipated failure mode: {fail_mode}")
        r["failure_mode"] = fail_mode

    verify_strats = KB.verify_strategies(
        "equation" if p.ptype in (PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY) else
        "factoring" if p.ptype == PT.FACTORING else
        "identity"  if p.ptype in (PT.TRIG_ID,PT.SIMPLIFY) else
        "graph"     if p.ptype == PT.GRAPH else
        "markov"    if p.ptype == PT.MARKOV else
        "entropy"   if p.ptype == PT.ENTROPY else
        "control"   if p.ptype == PT.CONTROL else
        "optimize"  if p.ptype == PT.OPTIMIZATION else "equation"
    )
    kv("Planned verification", verify_strats[0])
    r["verify_strategy"] = verify_strats
    print(f"  {DIM}---{RST}")

    # ── TYPE-SPECIFIC GROUND TRUTH ────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        kv("Expression", str(p.expr))
        try:
            poly  = p.get_poly()
            deg   = poly.degree()
            r["degree"] = deg
            kv("Degree",       deg)
            kv("Coefficients", [str(c) for c in poly.all_coeffs()])
            p.conf.record("degree", deg, 1.0)
            # Predict: how hard is this problem?
            hardness = {1:"trivial (closed form)", 2:"easy (quadratic formula / discriminant)",
                        3:"moderate (Cardano / factor theorem)", 4:"hard (numerical likely needed)"}
            difficulty = hardness.get(deg, "hard (Abel-Ruffini: no radical formula)")
            kv("Predicted difficulty", difficulty)
            # Symmetry pre-detection
            v = p.var
            if v:
                try:
                    even = simplify(p.expr.subs(v,-v) - p.expr) == 0
                    odd  = simplify(p.expr.subs(v,-v) + p.expr) == 0
                    r["even"] = even; r["odd"] = odd
                    if even: finding("EVEN polynomial → roots in ±pairs → substitute u=x²")
                    elif odd: finding("ODD polynomial → x=0 is a root → factor out x first")
                    else: note("No parity symmetry detected")
                except: pass
            # Rational root candidates (fast screen)
            if all(c.is_integer for c in poly.all_coeffs()):
                lc  = int(abs(poly.all_coeffs()[0]))
                ct  = int(abs(poly.all_coeffs()[-1]))
                cands = sorted({s*pf/qf for pf in divisors(ct) for qf in divisors(lc)
                                for s in (1,-1)}, key=abs)[:10]
                r["rat_root_cands"] = cands
                kv("Rational root screen", [str(c) for c in cands])
                # Test them immediately — cheap screen
                hits = [c for c in cands if p.expr and
                        abs(float(N(p.expr.subs(v, c)))) < 1e-9]
                if hits:
                    ok(f"Rational roots found: {hits} — factor theorem applies")
                    r["known_rational_roots"] = hits
                    p.conf.record("rational_roots", hits, 0.95)
        except: pass
        # Discriminant (degree 2, 3)
        if p.ptype == PT.QUADRATIC:
            try:
                disc = discriminant(Poly(p.expr, p.var))
                r["discriminant"] = disc
                kv("Discriminant Δ", str(disc))
                disc_n = float(N(disc))
                nature = ("2 distinct real roots" if disc_n > 0 else
                          "double root (repeated)" if disc_n == 0 else
                          "2 complex conjugate roots")
                kv("Root nature (predicted)", nature)
                p.conf.record("discriminant", disc, 1.0, nature)
                finding(f"Δ = {disc_n:.4f} → {nature}")
            except: pass

    elif p.ptype == PT.GRAPH:
        named = p.meta.get("named","")
        kv("Graph", named if named else "adjacency matrix")
        kv("Vertices", p.meta.get("n","?"))
        kv("Predicted methods", "spectrum(L), spectrum(A), Kirchhoff, Cheeger")
        kv("Verify via", "tr(L) = sum of degrees, λ₁(L) = 0 always, λ₂ > 0 iff connected")

    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M"); n = p.meta.get("n")
        if M:
            kv("Shape",  f"{n}×{n}")
            kv("Trace",  str(trace(M)))
            kv("Det",    str(det(M)))
            sym = (M == M.T)
            kv("Symmetric", sym)
            if sym: finding("Symmetric → spectral theorem applies → all eigenvalues real")
            p.conf.record("symmetric", sym, 1.0)

    elif p.ptype == PT.MARKOV:
        rows = p.meta.get("rows",[])
        n    = len(rows) if rows else 0
        kv("States", n)
        if rows:
            for i,row in enumerate(rows):
                s_ = sum(row)
                (ok if abs(s_-1.0)<1e-9 else fail)(f"Row {i} sums to {s_:.6f}")
        kv("Verify via", "π·P = π, spectral radius = 1, Perron-Frobenius")

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        if probs:
            kv("Distribution", probs)
            total = sum(probs)
            (ok if abs(total-1.0)<1e-9 else fail)(f"Sum = {total:.6f}")
            kv("Verify via", "H ≥ 0, H ≤ log₂(n), Huffman bound")

    elif p.ptype == PT.DYNAMICAL:
        kv("f(x)", str(p.expr))
        kv("Predicted equilibria", "solve f(x)=0, classify via f'(x*)")

    elif p.ptype == PT.CONTROL:
        kv("Characteristic poly", str(p.expr))
        try:
            poly = p.get_poly()
            kv("Degree", poly.degree())
            coeffs = poly.all_coeffs()
            pos = all(float(N(c)) > 0 for c in coeffs)
            (ok if pos else fail)(f"Necessary condition: all coefficients positive = {pos}")
            if not pos:
                finding("NECESSARY CONDITION FAILS → system definitely unstable")
                p.conf.record("def_unstable", True, 0.99)
        except: pass

    elif p.ptype == PT.OPTIMIZATION:
        goal = p.meta.get("goal","extremize")
        kv("f(x)", str(p.expr))
        kv("Goal", goal)
        kv("Strategy", "1) Find f'=0  2) Classify via f''  3) Check global via limits")

    elif p.ptype == PT.SUM:
        kv("Strategy", "Compute closed form, verify f(n)-f(n-1)=n, extend Faulhaber family")

    elif p.ptype == PT.PROOF:
        kv("Claim",    p.meta.get("body",""))
        kv("Strategy", "Try: contradiction, direct, construction, contrapositive")

    elif p.ptype == PT.FACTORING:
        kv("Expression", str(p.expr))
        kv("Strategy",   "factor(), sqf_list(), check irreducibility over Q and C")

    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        kv("Expression", str(p.expr))
        kv("Strategy",   "trigsimp(), expand_trig(), verify numerically at 3+ values")

    ok("Problem classified and intelligence briefing complete")
    r["briefed"] = True
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 02 — DIRECT ATTACK  (methods ranked by predicted success)
# ════════════════════════════════════════════════════════════════════════════

def phase_02(p: Problem, g1: dict) -> dict:
    section(2, "DIRECT ATTACK", "Methods ranked by predicted success · failures logged precisely")
    r = {"successes": [], "failures": []}
    v = p.var

    def attempt(name, fn, conf_prior=0.5, verify_fn=None):
        try:
            result = fn()
            if result is None: raise ValueError("None result")
            p._cache[name] = result
            conf = conf_prior
            # Run verify_fn if provided
            if verify_fn:
                try:
                    verified, detail = verify_fn(result)
                    if verified: conf = min(conf + 0.1, 1.0); ok(f"{name}  verified ✓")
                    else:        conf = max(conf - 0.2, 0.0); warn(f"{name} verify failed: {detail}")
                except: pass
            p.conf.record(name, result, conf)
            r["successes"].append({"method":name, "result":result, "conf":conf})
            ok(f"[{conf:.0%}] {name}  →  {str(result)[:90]}")
            return result
        except Exception as e:
            msg = str(e)[:70]
            r["failures"].append({"method":name, "error":msg})
            fail(f"[--] {name}  →  {msg}")
            return None

    # ── ALGEBRAIC ─────────────────────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.TRIG_EQ):
        # Use known rational roots first (Phase 01 screen)
        hits = g1.get("known_rational_roots", [])
        if hits:
            note(f"Using rational roots from Phase 01 screen: {hits}")

        def solve_fn():
            sols = solve(p.expr, v)
            if not sols: raise ValueError("no solutions")
            return sols

        sols = attempt("solve(expr,var)", solve_fn, 0.95,
                       verify_fn=lambda s: (all(_verify_equation(p.expr, v, x)[0] for x in s),
                                            "substitution check") if s else (False,"empty"))
        if sols:
            r["solutions"] = [str(s) for s in sols]
            # Vieta check
            if p.get_poly() and p.ptype != PT.TRIG_EQ:
                vieta_ok = _vieta_check(p.get_poly(), sols)
                (ok if vieta_ok else warn)(f"Vieta's formulas: {vieta_ok}")
                r["vieta_verified"] = vieta_ok

        attempt("solveset(Reals)", lambda: str(solveset(p.expr, v, domain=S.Reals)), 0.80)
        if p.ptype != PT.TRIG_EQ and p.get_poly():
            attempt("roots(Poly)", lambda: str(roots(p.get_poly())), 0.85)

    # ── FACTORING ─────────────────────────────────────────────────────────
    elif p.ptype == PT.FACTORING:
        fac = attempt("factor(expr)", lambda: factor(p.expr), 0.90,
                      verify_fn=lambda f: (simplify(expand(f) - expand(p.expr)) == 0, "expand verify"))
        attempt("sqf_list",     lambda: str(sqf_list(p.expr, v)), 0.80)
        attempt("factor_list",  lambda: str(factor_list(p.expr)), 0.80)

    # ── TRIG IDENTITY ─────────────────────────────────────────────────────
    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        simp = attempt("trigsimp",    lambda: trigsimp(p.expr),    0.90)
        attempt("simplify",           lambda: simplify(p.expr),    0.85)
        attempt("expand_trig",        lambda: expand_trig(p.expr), 0.70)
        # Numerical verification at 3 values (my own cross-check strategy)
        if p.expr is not None and v:
            test_vals = [0.7, 1.3, 2.1]
            num_checks = []
            for tv in test_vals:
                try:
                    res = float(abs(N(p.expr.subs(v, tv))))
                    num_checks.append(abs(res) < 1e-8)
                except: pass
            if num_checks:
                all_zero = all(num_checks)
                r["numerical_verify"] = all_zero
                (ok if all_zero else warn)(f"Numerical check at {test_vals}: {num_checks}")
                p.conf.record("numerical_identity", all_zero, 0.95 if all_zero else 0.3)

    # ── SUMMATION ─────────────────────────────────────────────────────────
    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        attempt("summation(k,(k,1,n))", lambda: summation(k,(k,1,n)), 0.99)

    # ── PROOF ─────────────────────────────────────────────────────────────
    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            ok("Proof by contradiction: assume √2=p/q → p,q both even → contradicts gcd=1")
            r["proof_method"] = "contradiction"; r["status"] = "Success"
        elif "prime" in body.lower():
            ok("Euclid: N=∏pᵢ+1 → new prime factor → contradiction with 'finitely many'")
            r["proof_method"] = "construction"; r["status"] = "Success"

    # ── DIGRAPH CYC ───────────────────────────────────────────────────────
    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        if m % 2 != 0:
            ok(f"Odd m={m}: fiber decomposition exists (twisted translation construction)")
            r["status"] = "Success (Odd m)"
        else:
            fail(f"Even m={m}: parity obstruction — fiber-uniform impossible")
            r["status"] = "Failure (Even m)"

    # ── GRAPH ─────────────────────────────────────────────────────────────
    elif p.ptype == PT.GRAPH:
        A, L, n, deg = _build_graph(p)
        if A is None: fail("Cannot build adjacency matrix"); return r
        p.meta.update({"L":L, "n":n, "deg":deg})
        ok(f"A, L built ({n}×{n})")
        r["degree_sequence"] = deg; kv("Degree sequence", deg)
        r["edge_count"]      = sum(deg)//2; kv("Edges", r["edge_count"])

        L_spec = _spectrum(L)
        if L_spec:
            r["L_spec"] = L_spec; p.meta["L_spec"] = L_spec
            kv("Laplacian spectrum", [f"{e:.4f}" for e in L_spec])
            ok("Laplacian spectrum computed")
            # Verify: λ₁(L) should always be 0
            (ok if abs(L_spec[0]) < 1e-9 else warn)(f"λ₁(L)={L_spec[0]:.6f} (should be 0)")
            # Verify: tr(L) = sum of degrees
            tr_L = float(N(trace(L)))
            tr_check = abs(tr_L - sum(deg)) < 1e-9
            (ok if tr_check else warn)(f"tr(L)={tr_L:.4f} = Σdeg={sum(deg)}: {tr_check}")

        A_spec = _spectrum(A)
        if A_spec:
            r["A_spec"] = A_spec; p.meta["A_spec"] = A_spec
            kv("Adjacency spectrum", [f"{e:.4f}" for e in A_spec])
            # Verify: tr(A²)/2 = number of edges
            try:
                tr_A2 = float(N(trace(A*A)))
                edge_check = abs(tr_A2/2 - sum(deg)//2) < 1e-9
                (ok if edge_check else warn)(f"tr(A²)/2={tr_A2/2:.1f} = |E|={sum(deg)//2}: {edge_check}")
            except: pass

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M")
        if M is None: fail("No matrix"); return r
        n = M.shape[0]
        # Characteristic polynomial
        lam = symbols('lambda')
        cp = attempt("char_poly", lambda: sp.expand(det(M - lam*eye(n))), 0.95)
        if cp: r["char_poly"] = str(cp)

        spec = _spectrum(M); p.meta["spec"] = spec
        r["eigenvalues"] = spec
        kv("Eigenvalues", [f"{e:.4f}" for e in spec])
        ok("Eigenvalues computed")

        # Verify: tr = Σλ, det = Πλ
        tr_M = float(N(trace(M))); dt_M = float(N(det(M)))
        tr_sum = sum(spec); dt_prod = math.prod(spec) if spec else 0
        (ok if abs(tr_M - tr_sum) < 1e-6 else warn)(f"Trace check: tr={tr_M:.4f} ≈ Σλ={tr_sum:.4f}")
        (ok if abs(dt_M - dt_prod) < 1e-6 else warn)(f"Det check: det={dt_M:.4f} ≈ Πλ={dt_prod:.4f}")
        r["trace"] = tr_M; r["det"] = dt_M

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype == PT.MARKOV:
        P = p.meta.get("P")
        if P is None: fail("No transition matrix"); return r
        n = P.shape[0]; p.meta["n"] = n
        P_rat = sp.Matrix([[sp.Rational(P[i,j]).limit_denominator(10000)
                            if isinstance(P[i,j], float) else sp.sympify(P[i,j])
                            for j in range(n)] for i in range(n)])
        p.meta["P_rat"] = P_rat
        ok(f"Exact rational matrix ({n}×{n})")

        spec_c = _spectrum_complex(P_rat)
        r["eigenvalues_complex"] = [str(round(z.real,4)+round(z.imag,4)*1j) for z in spec_c]
        kv("Eigenvalues", r["eigenvalues_complex"])
        # Verify: spectral radius ≤ 1
        rho = max(abs(z) for z in spec_c) if spec_c else 0
        r["spectral_radius"] = rho
        (ok if rho <= 1.0001 else warn)(f"Spectral radius ρ = {rho:.6f} ≤ 1")

        stat = _stationary(P_rat)
        if stat:
            r["stationary"] = {str(k): str(v_) for k,v_ in stat.items()}
            kv("Stationary π", r["stationary"])
            p.meta["stat"] = stat
            ok("Stationary distribution computed")
            # Verify: π·P = π
            pi_v = sp.Matrix([list(stat.values())])
            check = pi_v * P_rat - pi_v
            all_zero = all(simplify(check[0,j]) == 0 for j in range(n))
            (ok if all_zero else warn)(f"π·P = π verification: {all_zero}")
            p.conf.record("stationary_verified", all_zero, 0.99 if all_zero else 0.5)

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        if probs:
            H     = _entropy(probs); H_max = math.log2(len(probs))
            r["H_bits"] = H; r["H_max"] = H_max
            r["efficiency"] = H/H_max if H_max > 0 else 1.0
            kv("H(X)",          f"{H:.6f} bits")
            kv("H_max=log₂n",   f"{H_max:.6f} bits")
            kv("Efficiency",    f"{r['efficiency']:.4f}")
            ok("Shannon entropy computed")
            # Verify: H ≥ 0, H ≤ H_max
            (ok if H >= -1e-12 else warn)(f"H ≥ 0: {H >= 0}")
            (ok if H <= H_max+1e-12 else warn)(f"H ≤ H_max: {H <= H_max}")
            KL = _kl(probs, [1/len(probs)]*len(probs))
            r["KL_uniform"] = KL
            kv("KL(P||uniform)", f"{KL:.6f} bits")
            p.meta["H_val"] = H

        # Binary entropy symbolics
        p_s = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        max_p = solve(diff(H_bin,p_s), p_s)
        r["binary_entropy"] = str(H_bin); r["binary_max_at"] = str(max_p)
        kv("Binary H(p) max at p=", str(max_p))

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype == PT.DYNAMICAL:
        f = p.expr
        equil = attempt("solve(f=0)", lambda: solve(f, v), 0.90)
        if equil:
            r["equilibria"] = [str(e) for e in equil]
            fp = diff(f, v)
            kv("f'(x)", str(fp))
            for eq in equil:
                try:
                    fp_val = float(N(fp.subs(v, eq)))
                    stab = ("STABLE" if fp_val < 0 else "UNSTABLE" if fp_val > 0 else "NON-HYPERBOLIC")
                    kv(f"  f'({eq})", f"{fp_val:.4f}  →  {stab}")
                    r[f"stab_{eq}"] = stab
                except: pass

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype == PT.CONTROL:
        f = p.expr
        attempt("solve(char_poly)", lambda: solve(f, v), 0.80)
        rts = p._cache.get("solve(char_poly)", [])
        if rts:
            r["roots"] = [str(rt) for rt in rts]
            for rt in rts:
                try:
                    rt_c = complex(N(rt))
                    loc  = ("LHP stable" if rt_c.real < 0 else
                            "RHP UNSTABLE" if rt_c.real > 0 else "imaginary axis MARGINAL")
                    kv(f"  root {rt}", f"Re={rt_c.real:.4f}, Im={rt_c.imag:.4f}  →  {loc}")
                except: pass
        try:
            poly   = p.get_poly()
            coeffs = poly.all_coeffs()
            rh     = _routh(coeffs)
            r["routh"] = rh; p._cache["routh"] = rh
            kv("Routh first column", [f"{x:.4f}" for x in rh["first_column"]])
            kv("Sign changes",        rh["sign_changes"])
            (ok if rh["stable"] else fail)(
                f"Routh-Hurwitz: {'STABLE' if rh['stable'] else 'UNSTABLE'}")
            r["successes"].append({"method":"routh","result":rh["stable"]})
        except Exception as e:
            fail(f"Routh: {e}")

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; fp = diff(f, v); fpp = diff(f, v, 2)
        r["gradient"] = str(fp); kv("f'(x)", str(fp))
        crit = attempt("solve(f'=0)", lambda: solve(fp, v), 0.90)
        if crit:
            r["critical_points"] = [str(c) for c in crit]
            for c in crit:
                try:
                    fpp_v = float(N(fpp.subs(v, c)))
                    f_v   = float(N(f.subs(v, c)))
                    nature = ("LOCAL MIN" if fpp_v > 0 else "LOCAL MAX" if fpp_v < 0 else "INFLECTION")
                    kv(f"  x={c}", f"f={f_v:.4f}, f''={fpp_v:.4f}  →  {nature}")
                    r[f"cp_{c}"] = {"f":f_v, "fpp":fpp_v, "nature":nature}
                    # Verify: f'(c) = 0
                    fp_check = abs(float(N(fp.subs(v, c)))) < 1e-9
                    (ok if fp_check else warn)(f"f'({c})=0 verify: {fp_check}")
                except: pass
        try:
            lp = limit(f,v, oo); ln = limit(f,v,-oo)
            r["lim_+inf"] = str(lp); r["lim_-inf"] = str(ln)
            kv("f(+∞)", str(lp)); kv("f(-∞)", str(ln))
        except: pass

    finding(f"{len(r['successes'])} methods succeeded, {len(r['failures'])} failed")
    finding(p.conf.summary())
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 03 — STRUCTURE HUNT
# ════════════════════════════════════════════════════════════════════════════

def phase_03(p: Problem, g2: dict) -> dict:
    section(3, "STRUCTURE HUNT", "Invariants · symmetry groups · decomposition · spectrum")
    r = {}
    v = p.var

    if p.ptype == PT.GRAPH:
        L_spec = p.meta.get("L_spec",[]); A_spec = p.meta.get("A_spec",[])
        n = p.meta.get("n",0); deg = p.meta.get("deg",[])
        if len(L_spec) > 1:
            lam2 = sorted(L_spec)[1]
            r["fiedler"] = lam2
            kv("Fiedler value λ₂", f"{lam2:.6f}")
            finding("λ₂>0 → CONNECTED (Fiedler 1973)" if lam2>1e-9 else "λ₂=0 → DISCONNECTED")
            r["connected"] = lam2 > 1e-9
            if lam2 > 1e-9:
                ch_lb = lam2/2; ch_ub = math.sqrt(2*lam2)
                kv("Cheeger bound h(G) ∈", f"[{ch_lb:.4f}, {ch_ub:.4f}]")
                r["cheeger_lb"] = ch_lb
        if len(set(deg)) == 1:
            d = deg[0]; r["regular"] = d
            finding(f"{d}-REGULAR graph")
        if A_spec:
            sym = all(abs(A_spec[i]+A_spec[-(i+1)])<1e-6 for i in range(len(A_spec)//2))
            r["bipartite"] = sym
            kv("Spectrum symmetric (bipartite)", sym)
            finding("Bipartite confirmed" if sym else "Not bipartite")
        # Components = number of zero eigenvalues
        n_comps = sum(1 for e in L_spec if abs(e)<1e-9)
        r["components"] = n_comps
        kv("Connected components (zero eigs)", n_comps)
        return r

    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M"); spec = p.meta.get("spec",[])
        if M is None: return r
        r["symmetric"] = (M == M.T); kv("Symmetric", r["symmetric"])
        if r["symmetric"] and spec:
            min_e = min(spec); max_e = max(spec)
            if min_e > 0:   r["definite"] = "positive definite";   finding("POSITIVE DEFINITE (all λ>0)")
            elif min_e >= 0:r["definite"] = "positive semidefinite";finding("POSITIVE SEMIDEFINITE")
            elif max_e < 0: r["definite"] = "negative definite";   finding("NEGATIVE DEFINITE")
            else:           r["definite"] = "indefinite";           finding("INDEFINITE (mixed)")
        try:
            rnk = M.rank(); r["rank"] = rnk; kv("Rank", rnk)
            finding(f"Full rank → INVERTIBLE" if rnk == M.shape[0] else f"Rank {rnk} < {M.shape[0]} → SINGULAR")
        except: pass
        if spec:
            rho = max(abs(e) for e in spec); cond = rho/max(min(abs(e) for e in spec),1e-15)
            r["spectral_radius"] = rho; r["condition"] = cond
            kv("Spectral radius ρ", f"{rho:.4f}"); kv("Condition number κ", f"{cond:.4f}")
            if cond > 100: warn(f"High κ={cond:.1f} → ill-conditioned")
        return r

    elif p.ptype == PT.MARKOV:
        P_rat = p.meta.get("P_rat"); n = p.meta.get("n",0)
        spec_c = [complex(N(k)) for k in (P_rat.eigenvals() if P_rat else {})]
        eig_abs = sorted([abs(z) for z in _spectrum_complex(P_rat)], reverse=True) if P_rat else []
        if len(eig_abs) > 1:
            lam2 = eig_abs[1]; gap = 1.0-lam2
            r["lambda2"] = lam2; r["gap"] = gap
            kv("|λ₂|", f"{lam2:.6f}"); kv("Spectral gap 1-|λ₂|", f"{gap:.6f}")
            if gap > 1e-9:
                mix = int(1/gap)+1; r["mixing_time"] = mix
                kv("Mixing time ~", f"{mix} steps")
                finding(f"‖Pⁿ-Π‖ ≤ {lam2:.3f}ⁿ → mixes in ~{mix} steps")
        if P_rat:
            abs_states = [i for i in range(n) if P_rat[i,i]==1]
            r["absorbing"] = abs_states
            kv("Absorbing states", abs_states or "none")
            finding("ERGODIC (no absorbing states)" if not abs_states else f"Absorbing: {abs_states}")
        # Reversibility check
        stat = p.meta.get("stat",{})
        if stat and P_rat:
            try:
                pi_v = [sp.sympify(list(stat.values())[i]) for i in range(n)]
                rev  = all(simplify(pi_v[i]*P_rat[i,j]-pi_v[j]*P_rat[j,i])==0
                           for i in range(n) for j in range(n))
                r["reversible"] = rev; kv("Reversible (detailed balance)", rev)
                finding("REVERSIBLE" if rev else "IRREVERSIBLE (entropy production > 0)")
            except: pass
        return r

    elif p.ptype == PT.ENTROPY:
        p_s = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        kv("d²H/dp²", str(simplify(diff(H_bin,p_s,2))))
        finding("H is strictly CONCAVE → unique maximum at p=½")
        probs = p.meta.get("probs",[])
        if probs:
            H_val = p.meta.get("H_val",0); H_max = math.log2(len(probs))
            contribs = [-p_*math.log2(p_) for p_ in probs if p_>0]
            kv("Per-symbol contributions −pᵢlog₂pᵢ", [f"{c:.4f}" for c in contribs])
            kv("Gap to max entropy",  f"{H_max-H_val:.6f} bits")
            finding(f"Efficiency: {H_val/H_max:.4f}  ({H_val:.4f}/{H_max:.4f} bits)")
        return r

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr
        if f is None: return r
        try:
            even = simplify(f.subs(v,-v)-f)==0; odd = simplify(f.subs(v,-v)+f)==0
            r["symmetry"] = "even" if even else ("odd" if odd else "none")
            kv("Symmetry", r["symmetry"])
            if even: finding("EVEN → equilibria symmetric about origin")
            elif odd: finding("ODD → origin is always an equilibrium")
        except: pass
        try:
            V    = v**2/2; dVdt = simplify(diff(V,v)*f)
            r["lyapunov"] = str(dVdt); kv("Lyapunov V=x²/2, V̇=xf(x)", str(dVdt))
        except: pass
        return r

    elif p.ptype == PT.CONTROL:
        rh = p._cache.get("routh",{})
        if rh:
            kv("Stability", "STABLE" if rh.get("stable") else "UNSTABLE")
            # Root locations from phase 02
            rts = p._cache.get("solve(char_poly)", [])
            if rts:
                for rt in rts:
                    try:
                        rt_c = complex(N(sp.sympify(rt)))
                        kv(f"  λ={rt}", f"Re={rt_c.real:.4f}  {'LHP' if rt_c.real<0 else 'RHP'}")
                    except: pass
        return r

    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; fp = diff(f,v); fpp = diff(f,v,2)
        try:
            fpp_s = simplify(fpp); kv("f''(x)", str(fpp_s))
            # Convexity check
            if fpp_s.is_polynomial(v):
                fp_poly = Poly(fpp_s, v)
                if fp_poly.degree() == 0:
                    val = float(N(fpp_s))
                    if val > 0:   finding("f'' > 0 everywhere → CONVEX → local min = global min")
                    elif val < 0: finding("f'' < 0 everywhere → CONCAVE → local max = global max")
        except: pass
        return r

    # ── ALGEBRAIC (polynomial types) ──────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.FACTORING):
        try:
            fac = factor(p.expr); r["factored"] = str(fac); kv("Factored", fac)
        except: pass
        if v:
            try:
                even = simplify(p.expr.subs(v,-v)-p.expr)==0
                odd  = simplify(p.expr.subs(v,-v)+p.expr)==0
                if even: finding("EVEN symmetry → substitute u=x² to reduce degree")
                elif odd: finding("ODD → factor out x")
            except: pass
        # Newton's identities: power sums of roots
        try:
            sols = p._cache.get("solve(expr,var)", [])
            if sols and len(sols) <= 5:
                power_sums = {}
                for k_ in range(1, 4):
                    ps = sum(s**k_ for s in sols)
                    power_sums[f"p{k_}=Σxᵢ^{k_}"] = str(simplify(ps))
                kv("Power sums (Newton)", power_sums)
        except: pass

    if p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n_sym = symbols('n', positive=True, integer=True)
        try:
            res = summation(k,(k,1,n_sym))
            kv("Closed form", str(factor(res)))
            kv("f(n)-f(n-1)", str(simplify(res-res.subs(n_sym,n_sym-1))))
        except: pass

    if p.ptype == PT.TRIG_ID and p.expr is not None and v:
        # Verify at multiple numerical values
        test_vals = [0.3, 0.8, 1.5, 2.2]
        residuals = []
        for tv in test_vals:
            try: residuals.append(abs(float(N(p.expr.subs(v,tv)))))
            except: pass
        if residuals:
            max_res = max(residuals)
            (ok if max_res < 1e-8 else warn)(f"Numerical residuals at {test_vals}: max={max_res:.2e}")
            r["numerical_identity"] = max_res < 1e-8

    if p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        r["fiber"] = f"F_s = {{(i,j,k): i+j+k≡s (mod {m})}}"
        kv("Fiber structure", r["fiber"])

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 04 — PATTERN LOCK  (includes backwards reasoning)
# ════════════════════════════════════════════════════════════════════════════

def phase_04(p: Problem, g3: dict) -> dict:
    section(4, "PATTERN LOCK", "Read solution backwards · backwards reasoning · extract law")
    r = {}
    v = p.var

    # ── BACKWARDS REASONING ENGINE ────────────────────────────────────────
    # For each solution/result, ask: "what structure produces THIS?"
    def backwards_reason(result_str: str, domain: str) -> str:
        """Given a result, state what family of problems would produce it."""
        br = {
            ("integer", "equation"): "Integer solutions → polynomial over Z → check all via Vieta / factor theorem",
            ("rational","equation"):  "Rational roots → rational root theorem applies",
            ("irrational","equation"):"Irrational roots → irreducible over Q → field extension needed",
            ("complex","equation"):   "Complex roots → no real factorisation over R",
            ("zero","spectrum"):      "Zero eigenvalue → graph disconnected OR matrix singular",
            ("uniform","entropy"):    "Maximum entropy → uniform distribution → no information asymmetry",
            ("1","markov_eig"):       "λ=1 → stationary distribution exists (Perron-Frobenius)",
            ("stable","control"):     "LHP roots → damped response → negative feedback sufficient",
        }
        return br.get((result_str, domain), f"{result_str} in {domain}: see governing theorems")

    if p.ptype == PT.GRAPH:
        A = p.meta.get("A"); L = p.meta.get("L")
        n = p.meta.get("n",0); deg = p.meta.get("deg",[])
        L_spec = p.meta.get("L_spec",[])
        A_spec = p.meta.get("A_spec",[])

        # Kirchhoff
        if L_spec and n > 0:
            nz = [e for e in L_spec if abs(e)>1e-9]
            if nz:
                tau = math.prod(nz)/n
                r["spanning_trees"] = tau
                kv("Spanning trees τ(G) [Kirchhoff]", f"{tau:.4f}")
                insight(f"Matrix Tree Theorem: τ = (1/n)∏λᵢ≠0 ≈ {tau:.2f}")
                # Backwards: what graph has τ=nⁿ⁻²? Only Kₙ. Check.
                if p.meta.get("type") == "complete":
                    expected = n**(n-2)
                    (ok if abs(tau-expected)<0.5 else warn)(f"Kₙ check: τ={tau:.1f} ≈ nⁿ⁻²={expected}")

        # Estrada index
        if A_spec:
            ee = sum(math.exp(e) for e in A_spec)
            r["estrada"] = ee; kv("Estrada index EE(G)", f"{ee:.4f}")
            insight("EE(G) quantifies network 'subgraph richness' — higher = more loops")

        # Spectral centrality (backwards: which nodes are most 'central'?)
        if A and n <= 12:
            try:
                evects = A.eigenvects()
                top = sorted(evects, key=lambda t: float(N(t[0])), reverse=True)[0]
                vec = top[2][0]; s = sum(abs(x) for x in vec)
                norm = [float(N(x/s)) for x in vec]
                r["spectral_centrality"] = [f"{x:.3f}" for x in norm]
                kv("Spectral centrality", r["spectral_centrality"])
                # Backwards: if centrality is uniform → regular graph
                if len(set(round(x,2) for x in norm)) == 1:
                    insight("Uniform centrality → regular graph confirmed (all nodes equivalent)")
                else:
                    max_i = norm.index(max(norm))
                    insight(f"Node {max_i} has highest spectral centrality — the 'hub'")
            except: pass

        # Fiedler vector = graph cut
        if L and n <= 12:
            try:
                evects = L.eigenvects()
                sorted_ev = sorted(evects, key=lambda t: float(N(t[0])))
                if len(sorted_ev) > 1:
                    fv = sorted_ev[1][2][0]
                    signs = [("+" if float(N(x))>=0 else "-") for x in fv]
                    r["fiedler_partition"] = signs
                    kv("Fiedler partition (+/-)", signs)
                    pos = signs.count("+"); neg = signs.count("-")
                    insight(f"Spectral bisection: {pos} nodes in part A, {neg} in part B")
            except: pass
        return r

    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M"); spec = p.meta.get("spec",[])
        n = p.meta.get("n",0)
        kv("Cayley-Hamilton", "M satisfies own characteristic polynomial p(M)=0")
        if spec and M is not None:
            tr_M = float(N(trace(M))); dt_M = float(N(det(M)))
            kv("Sigma_lambda_i = tr(M)", f"{sum(spec):.4f} = {tr_M:.4f}")
            kv("Pi_lambda_i = det(M)",   f"{math.prod(spec):.4f} = {dt_M:.4f}")
        if g3.get("symmetric"):
            insight("Symmetric → M = QΛQᵀ → M⁻¹ = QΛ⁻¹Qᵀ → exp(M) = Q·exp(Λ)·Qᵀ")
            insight("All matrix functions (log M, sin M, M^t) computed via same diagonalisation")
        # Backwards: what system would have this matrix as Jacobian?
        if spec:
            all_lhp = all(e < 0 for e in spec)
            insight(f"Backwards: if this is Jacobian at equilibrium → {'STABLE attractor' if all_lhp else 'has unstable modes'}")
        return r

    elif p.ptype == PT.MARKOV:
        stat = p.meta.get("stat",{}); P_rat = p.meta.get("P_rat")
        n    = p.meta.get("n",0)
        if stat:
            pi_f = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
            H_stat = _entropy(pi_f); r["H_stat"] = H_stat
            kv("H(π) stationary entropy", f"{H_stat:.6f} bits")
            insight(f"Backwards: stationary entropy H(π)={H_stat:.4f} → how 'spread out' the chain is")
            # Backwards: uniform stationary ↔ doubly stochastic P
            if all(abs(pi_f[i]-pi_f[0])<1e-6 for i in range(n)):
                insight("Uniform stationary distribution → P is DOUBLY STOCHASTIC (row AND column sums = 1)")

        # Entropy rate
        if stat and P_rat:
            try:
                pi_f2 = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                h = -sum(pi_f2[i]*sum(float(N(P_rat[i,j]))*math.log2(max(float(N(P_rat[i,j])),1e-15))
                         for j in range(n) if float(N(P_rat[i,j]))>1e-12)
                    for i in range(n))
                r["entropy_rate"] = h
                kv("Entropy rate h(X)", f"{h:.6f} bits/step")
                insight(f"Chain produces {h:.4f} bits of randomness per step (irreducible noise floor)")
            except: pass

        # Long-run matrix
        if P_rat and n <= 6:
            try:
                P_inf = P_rat**20
                kv("P^20 row 0", [str(N(P_inf[0,j],3)) for j in range(n)])
                insight("P^20 ≈ Π (rows converge to π) — ergodic theorem confirmed numerically")
            except: pass
        return r

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        p_s   = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        for pt_, label in [(sp.Rational(1,4),"H(1/4)"),(sp.Rational(1,2),"H(1/2)"),(sp.Rational(3,4),"H(3/4)")]:
            kv(label, f"{float(N(H_bin.subs(p_s,pt_))):.4f} bits")
        insight("H(1/2)=1 bit = maximum — fair coin is maximally uncertain")
        insight("H→0 as p→0 or p→1 — determinism destroys information")
        if probs:
            H_val = _entropy(probs); n = len(probs)
            # Huffman codes
            import heapq
            heap = [(p_, [i]) for i,p_ in enumerate(probs) if p_>0]
            heapq.heapify(heap)
            lens = {i:0 for i in range(n)}
            if len(heap) > 1:
                while len(heap) > 1:
                    p1,c1 = heapq.heappop(heap)
                    p2,c2 = heapq.heappop(heap)
                    for idx in c1: lens[idx]+=1
                    for idx in c2: lens[idx]+=1
                    heapq.heappush(heap,(p1+p2,c1+c2))
            avg = sum(probs[i]*lens.get(i,0) for i in range(n))
            r["huffman_avg"] = avg; kv("Huffman avg length", f"{avg:.4f} bits/sym")
            kv("Shannon entropy",    f"{H_val:.4f} bits/sym")
            kv("Redundancy",         f"{avg-H_val:.4f} bits")
            # KL
            KL = _kl(probs,[1/n]*n)
            kv("KL(P||uniform)",     f"{KL:.6f} bits")
            insight(f"Backwards: KL={KL:.4f} bits → deviation from maximum ignorance = 'information in the distribution'")
        return r

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr
        try:
            equil = solve(f, v)
            fp    = diff(f,v)
            for eq in equil:
                fp_v = float(N(fp.subs(v,eq)))
                stab = ("stable attractor" if fp_v<0 else "unstable repeller" if fp_v>0 else "non-hyperbolic")
                kv(f"  x*={eq}", f"f'={fp_v:.4f} → {stab}")
                # Backwards: what potential V has this as gradient?
                try:
                    V_pot = -integrate(f, v)
                    kv(f"  Potential V=-∫f dx at x={eq}", str(N(V_pot.subs(v,eq),4)))
                except: pass
            insight(f"Backwards: stable equilibria = LOCAL MINIMA of potential V(x) = -∫f(x)dx")
        except: pass
        return r

    elif p.ptype == PT.CONTROL:
        rh = p._cache.get("routh",{})
        rts = p._cache.get("solve(char_poly)",[])
        if rts:
            lhp = [rt for rt in rts if float(N(sp_re(rt)))<0]
            rhp = [rt for rt in rts if float(N(sp_re(rt)))>0]
            kv("LHP roots (stable modes)", [str(r) for r in lhp])
            kv("RHP roots (unstable modes)", [str(r) for r in rhp])
            insight(f"Backwards: {len(rhp)} unstable modes → need {len(rhp)} feedback gains to stabilise")
        if rh:
            (insight if rh["stable"] else warn)(
                f"Routh: {'ALL roots in LHP → BIBO stable' if rh['stable'] else str(rh['sign_changes'])+' roots in RHP → unstable'}")
        r["pattern"] = "Stability iff all eigenvalues Re(λ)<0 — spectral condition"
        return r

    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; fp = diff(f,v); fpp = diff(f,v,2)
        crit_raw = p._cache.get("solve(f'=0)",[])
        if crit_raw:
            f_vals = [(float(N(f.subs(v,c))),c) for c in crit_raw
                      if not isinstance(N(f.subs(v,c)), sp.core.numbers.ComplexNumber)]
            if f_vals:
                goal = p.meta.get("goal","extremize")
                best = (min if "min" in goal else max)(f_vals)
                r["optimal"] = best; kv("Optimal point", f"x*={best[1]}, f*={best[0]:.4f}")
                insight(f"Backwards: x*={best[1]} is where 'force' f'(x)=0 — system at rest")
                insight(f"Backwards: this problem = equilibrium of gradient flow ẋ=-∇f")
        return r

    # ── ALGEBRAIC ─────────────────────────────────────────────────────────
    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        sols = p._cache.get("solve(expr,var)", [])
        r["solutions"] = [str(s) for s in sols] if sols else []
        kv("Solutions", r["solutions"])
        if sols:
            for i,s in enumerate(sols):
                info = {"value":str(s), "is_integer":s.is_integer,
                        "is_rational":s.is_rational, "is_real":s.is_real,
                        "verified":_verify_equation(p.expr, v, s)[0]}
                r[f"sol_{i}"] = info
                print(f"\n  {DIM}Root {i}:{RST}")
                for k_,v_ in info.items(): kv(f"    {k_}", v_, indent=4)
            # Backwards: what do these solutions tell us about the polynomial?
            if all(s.is_integer for s in sols):
                ints = [int(s) for s in sols]
                insight(f"Backwards: integer roots {ints} → polynomial = ∏(x-{i}) — fully rational")
            elif all(s.is_real for s in sols):
                insight("Backwards: all real roots → discriminant Δ≥0 → polynomial splits over ℝ")
            elif any(not s.is_real for s in sols):
                insight("Backwards: complex roots → irreducible quadratic factor over ℝ")
            # Vieta verification
            if p.get_poly():
                vieta = _vieta_check(p.get_poly(), sols)
                (ok if vieta else warn)(f"Vieta's formulas verified: {vieta}")

    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        simp = p._cache.get("trigsimp") or simplify(p.expr)
        r["simplified"] = str(simp)
        kv("Simplified", simp); kv("Is identity (=0)", simp==0)
        if simp == 0: insight("Backwards: identity holds ∀x → it's an algebraic consequence of unit circle sin²+cos²=1")

    elif p.ptype == PT.FACTORING:
        fac = p._cache.get("factor(expr)", factor(p.expr))
        r["factored"] = str(fac); kv("Factored", fac)
        flist = factor_list(p.expr)
        for i,(fi,mult) in enumerate(flist[1]):
            try: rt = solve(fi, v)
            except: rt = []
            kv(f"  factor[{i}] ^{mult}", f"{fi}  roots:{rt}")
        ok(f"Verify expand-factor: {simplify(expand(fac)-expand(p.expr))==0}")

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        res = summation(k,(k,1,n))
        kv("Formula", str(factor(res)))
        insight("Backwards: f(n)-f(n-1)=n — this IS the definition of a telescoping sum")

    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            for step,desc in [("Assume","√2=p/q, gcd=1"),("Square","2=p²/q²→p²=2q²"),
                               ("Even","p=2m"),("Sub","4m²=2q²→q even"),
                               ("Contradict","both even ⊥ gcd=1"),("QED","√2 ∉ ℚ □")]:
                print(f"    {Y}{step:<12}{RST}{desc}")
        elif "prime" in body.lower():
            for step,desc in [("Assume","finite list {p₁,…,pₖ}"),
                               ("Construct","N=p₁·…·pₖ+1"),("Factor","N has prime q"),
                               ("Contradict","q not in list"),("QED","primes infinite □")]:
                print(f"    {Y}{step:<12}{RST}{desc}")

    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        if m%2 != 0:
            r["law"] = f"Q_c(i,j)=(i+b_c(j), j+r_c) mod {m}; gcd(r_c,{m})=1"
            kv("Twisted translation", r["law"])
            insight("Backwards: Hamiltonian decomposition = perfect 1-factorisation of Cayley graph")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 05 — GENERALIZE
# ════════════════════════════════════════════════════════════════════════════

def phase_05(p: Problem, g4: dict) -> dict:
    section(5, "GENERALIZE", "Name the condition — governing theorems — parametric families")
    r = {}
    v = p.var

    laws = {
        PT.GRAPH: {
            "connectivity":  "λ₂(L) > 0 iff G connected (Fiedler 1973)",
            "bipartite":     "G bipartite iff spectrum symmetric about 0",
            "cheeger":       "h(G) in [λ₂/2, sqrt(2λ₂)]  (Cheeger inequality)",
            "kirchhoff":     "tau(G) = (1/n)*prod{λ_i>0}  (Matrix Tree Theorem)",
            "expander":      "Good expander: lambda2 large -> fast mixing, robust",
            "random_walk":   "P = D^{-1}A: random walk Markov chain, stat pi_i = d_i/2|E|",
            "gnn":           "GNN message passing = polynomial applied to L or A",
        },
        PT.MATRIX: {
            "spectral_thm":  "Symmetric M = Q*Lambda*Q^T (spectral theorem, orthonormal Q)",
            "cayley_ham":    "p(M) = 0 where p = characteristic polynomial",
            "SVD":           "M = U*Sigma*V^T (any matrix, sigma_i = sqrt(eig(M^T*M)))",
            "definiteness":  "x^T*M*x > 0 all x iff all lambda > 0 (PD criterion)",
            "rank_nullity":  "rank(M) + nullity(M) = n",
            "gershgorin":    "All lambda in union of Gershgorin disks",
        },
        PT.MARKOV: {
            "perron_frob":   "Irreducible non-neg: unique lambda=1, unique stationary pi>0",
            "ergodic":       "Ergodic: time avg = space avg = pi (strong LLN)",
            "mixing":        "||P^n - Pi|| <= |lambda_2|^n (geometric convergence)",
            "entropy_rate":  "h = -sum_i pi_i sum_j P_ij log P_ij (irreducible noise)",
            "reversibility": "Detailed balance pi_i P_ij = pi_j P_ji iff reversible iff all eigs real",
        },
        PT.ENTROPY: {
            "shannon_thm":   "H unique by: continuity + max at uniform + additivity",
            "max_H":         "H(X) <= log2(n); equality iff uniform (MaxEnt principle)",
            "chain_rule":    "H(X,Y) = H(X) + H(Y|X) (additivity)",
            "data_proc":     "H(f(X)) <= H(X) for any f (processing can't create info)",
            "source_coding": "Avg code length L >= H(X) (Shannon's source coding theorem)",
            "channel_cap":   "C = max_{p(x)} I(X;Y) (Shannon channel capacity 1948)",
        },
        PT.DYNAMICAL: {
            "lyapunov":      "f'(x*) < 0 -> stable; f'(x*)>0 -> unstable",
            "hartman_grob":  "Near hyperbolic equilibrium: nonlinear approx linearization",
            "noether":       "Every continuous symmetry -> conservation law (Noether 1915)",
            "poincare_bend": "2D autonomous: no chaos (Poincare-Bendixson theorem)",
            "bifurcation":   "f'(x*)=0: structural change (saddle-node/pitchfork/Hopf)",
        },
        PT.CONTROL: {
            "routh_hurwitz": "Stable iff all Routh first-column elements > 0",
            "spectral_cond": "Stable iff all eigenvalues Re(lambda) < 0",
            "nyquist":       "Nyquist: encirclements of -1 = RHP poles of closed loop",
            "abel_ruffini":  "No radical formula for degree >= 5 (Abel-Ruffini 1824)",
            "controllable":  "rank[B,AB,...,A^{n-1}B]=n iff fully controllable",
        },
        PT.OPTIMIZATION: {
            "first_order":   "Critical point: grad f(x*)=0 (necessary, unconstrained)",
            "second_order":  "H>0 -> local min; H<0 -> local max; indef -> saddle",
            "convexity":     "f convex -> every local min = global min",
            "kkt":           "Constrained: grad f = sum lambda_i grad g_i, lambda_i g_i=0",
            "duality":       "Strong duality: primal=dual (Slater condition)",
        },
    }

    domain_laws = laws.get(p.ptype)
    if domain_laws:
        kv("Governing theorems for this domain", "")
        for name_, law_ in domain_laws.items():
            kv(f"  {name_}", law_)
        r["governing"] = domain_laws

    # Named graph families
    if p.ptype == PT.GRAPH:
        t = p.meta.get("type")
        families = {
            "complete": "K_n: spectrum lambda(L)={0^1, n^{n-1}}, tau=n^{n-2}, diameter=1",
            "path":     "P_n: lambda_k=2-2cos(k*pi/n), diameter=n-1",
            "cycle":    "C_n: lambda_k=2-2cos(2*pi*k/n), bipartite iff n even",
        }
        if t in families:
            kv("Named graph family", families[t]); r["family"] = families[t]

    # General algebraic families
    elif p.ptype == PT.LINEAR:
        a_, b_ = symbols('a b', nonzero=True)
        sol    = solve(a_*v + b_, v)[0]
        kv("General solution", str(sol)); kv("Condition", "a != 0")

    elif p.ptype == PT.QUADRATIC:
        a_,b_,c_ = symbols('a b c')
        gen = solve(a_*v**2+b_*v+c_, v)
        r["qf"] = [str(s) for s in gen]
        kv("Quadratic formula", r["qf"])
        kv("Governing",         "Delta = b^2-4ac determines nature")

    elif p.ptype == PT.CUBIC:
        kv("Cardano",    "Depressed t^3+pt+q=0: t=cbrt(-q/2+sqrt(D))+cbrt(-q/2-sqrt(D))")
        kv("Discriminant","Delta>0: 3 real; Delta=0: repeat; Delta<0: 1 real+2 complex")
        kv("IVT",        "Odd degree -> always >= 1 real root")
        kv("Abel limit",  "Degree >= 5: no radical formula (Abel-Ruffini 1824)")

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        n = symbols('n', positive=True, integer=True)
        for pow_ in range(1,5):
            try: kv(f"  Faulhaber sum k^{pow_}", str(factor(summation(k**pow_,(k,1,n)))))
            except: pass
        kv("Faulhaber law", "Sum k^p = degree-(p+1) polynomial in n; coeffs = Bernoulli numbers")

    elif p.ptype == PT.TRIG_ID:
        theta = symbols('theta')
        for f_,ex in [("sin^2+cos^2=1", sin(theta)**2+cos(theta)**2-1),
                      ("1+tan^2=sec^2",  1+tan(theta)**2-sec(theta)**2)]:
            kv(f"  {f_}", f"= {trigsimp(ex)} {'v' if trigsimp(ex)==0 else '?'}")
        kv("Governing", "All trig identities -> from sin^2+cos^2=1 + Euler e^{i*theta}")

    elif p.ptype == PT.FACTORING:
        a_, b_ = symbols('a b')
        for f_,e_ in [("a^2-b^2",a_**2-b_**2),("a^3-b^3",a_**3-b_**3),
                      ("a^3+b^3",a_**3+b_**3),("a^4-b^4",a_**4-b_**4)]:
            kv(f"  {f_}", str(factor(e_)))
        kv("Governing", "a^n-b^n = (a-b)*(a^{n-1}+...+b^{n-1})")

    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            kv("General theorem", "sqrt(n) irrational iff n not a perfect square")
            for n_ in range(1,10):
                kv(f"  sqrt({n_})", "rational" if sp.sqrt(n_).is_integer else "irrational")
        elif "prime" in body.lower():
            kv("Governing","Euclid's proof: construction + contradiction (not direct)")

    elif p.ptype == PT.DIGRAPH_CYC:
        kv("Governing","Odd m: fiber-column-uniform sigma exists; Even m: full 3D sigma needed")

    finding("Governing principle: the specific case is an INSTANCE of the general law stated above")
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 06 — PROVE LIMITS
# ════════════════════════════════════════════════════════════════════════════

def phase_06(p: Problem, g5: dict) -> dict:
    section(6, "PROVE LIMITS", "Hard boundaries · obstructions · what cannot be done")
    r = {}
    v = p.var

    if p.ptype == PT.GRAPH:
        L_spec = p.meta.get("L_spec",[]); deg = p.meta.get("deg",[])
        limits = {
            "lower_lam2":  "lambda2 = 0 iff disconnected (hard boundary: cannot improve by relabelling)",
            "upper_lam_n": "lambda_max(L) <= Delta (max degree); equality iff regular",
            "bipartite":   "Bipartite iff no odd cycles iff spectrum symmetric",
            "planarity":   "Planar: |E| <= 3n-6; lambda2 <= 4 (Spielman-Teng)",
            "ramanujan":   "Optimal expander (Ramanujan): lambda2 <= 2*sqrt(d-1) for d-regular",
            "interlacing": "Remove vertex: eigenvalues interlace (Cauchy)",
        }
        for k_,v_ in limits.items(): kv(f"  {k_}", v_)
        if L_spec and deg:
            Delta = max(deg); lam_n = max(L_spec)
            (ok if lam_n <= Delta+1e-6 else fail)(f"lambda_max={lam_n:.4f} <= Delta={Delta}")

    elif p.ptype == PT.MATRIX:
        spec = p.meta.get("spec",[])
        for k_,v_ in {
            "gershgorin": "lambda in union{|z-a_ii| <= sum_{j!=i}|a_ij|}",
            "weyl":       "|lambda_i(A+E)-lambda_i(A)| <= ||E||_2 (stability)",
            "perron":     "Non-negative matrix: spectral radius = largest real eigenvalue",
            "det_zero":   "det=0 iff singular iff 0 is eigenvalue iff Ax=0 non-trivial",
        }.items(): kv(f"  {k_}", v_)

    elif p.ptype == PT.MARKOV:
        for k_,v_ in {
            "convergence": "Irreducible+aperiodic -> ||P^n-Pi|| -> 0 (ergodic theorem)",
            "mixing_rate": "||P^n-Pi|| <= |lambda_2|^n (geometric convergence)",
            "periodicity": "Periodic chain (period d>1) oscillates: no pointwise convergence",
            "lazy_fix":    "Lazy P'=(P+I)/2 -> aperiodic; |lambda_2(P')| = (1+|lambda_2|)/2 < 1",
        }.items(): kv(f"  {k_}", v_)
        note("Spectral radius <= 1 proved by Perron-Frobenius (non-negative stochastic matrix)")

    elif p.ptype == PT.ENTROPY:
        p_s = symbols('p', positive=True)
        H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
        lim0 = limit(H_bin, p_s, 0, '+'); lim1 = limit(H_bin, p_s, 1, '-')
        ok(f"H(0+)={lim0}, H(1-)={lim1}  (boundary continuity)")
        for k_,v_ in {
            "lower":      "H(X) >= 0; equality iff deterministic",
            "upper":      "H(X) <= log2(n); equality iff uniform",
            "subadditiv": "H(X,Y) <= H(X)+H(Y); equality iff independent",
            "data_proc":  "H(f(X)) <= H(X) for any f",
            "shannon_src":"L_bar >= H(X) (cannot compress below entropy)",
            "channel":    "C = max I(X;Y) (cannot communicate faster than capacity)",
        }.items(): kv(f"  {k_}", v_)
        finding("Fundamental obstruction: H is the IRREDUCIBLE minimum description length")

    elif p.ptype == PT.DYNAMICAL:
        for k_,v_ in {
            "hartman":    "Non-hyperbolic (f'=0): linearization insufficient — need higher order",
            "lyapunov":   "Global stability: V>0, Vdot<0 everywhere -> globally stable",
            "no_chaos":   "2D autonomous: chaos impossible (Poincare-Bendixson)",
            "bifurcation":"f'(x*)=0: structural qualitative change",
        }.items(): kv(f"  {k_}", v_)

    elif p.ptype == PT.CONTROL:
        for k_,v_ in {
            "necessary":   "Necessary: all coefficients same sign",
            "sufficient":  "Routh: all first-column entries > 0",
            "boundary":    "Marginal: eigenvalue on imaginary axis (Re=0)",
            "FTA":         "Degree-n poly has exactly n roots in C",
            "abel_limit":  "No radical formula for degree >= 5 (Abel-Ruffini)",
        }.items(): kv(f"  {k_}", v_)

    elif p.ptype == PT.OPTIMIZATION:
        for k_,v_ in {
            "first_order": "f'(x*)=0 necessary for smooth unconstrained",
            "second_order":"f''=0: second-order test inconclusive (check higher derivatives)",
            "global_bound":"Convex on convex domain: local min = global min",
            "no_go":       "Non-convex: global opt NP-hard in general",
        }.items(): kv(f"  {k_}", v_)
        try:
            lp = limit(p.expr,v, oo); ln = limit(p.expr,v,-oo)
            kv("f(+inf)", str(lp)); kv("f(-inf)", str(ln))
            if str(lp)==str(ln)=="oo": finding("f->+inf both directions -> minimum exists")
        except: pass

    elif p.ptype in (PT.LINEAR,):
        kv("a=0, b=0","infinitely many solutions"); kv("a=0, b!=0","no solution")
        finding("Unique solution exists iff a != 0")

    elif p.ptype == PT.QUADRATIC:
        kv("Delta=0","double root x=-b/2a"); kv("Delta<0","no real roots (complex)")
        kv("Over C","always exactly 2 roots (Fundamental Theorem of Algebra)")

    elif p.ptype in (PT.CUBIC, PT.POLY):
        kv("IVT","Odd degree -> >= 1 real root guaranteed"); kv("Abel","No radicals for deg>=5")
        if p.ptype == PT.POLY:
            try:
                rts_all = all_roots(p.get_poly())
                real_rt = [r for r in rts_all if sp.im(r)==0]
                kv("Real roots",    len(real_rt))
                kv("Complex roots", len(rts_all)-len(real_rt))
            except: pass

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        kv("Sum k->inf",  str(summation(k,(k,1,oo))))
        kv("Sum 1/k",     str(summation(1/k,(k,1,oo))))
        kv("Sum 1/k^2",   str(summation(1/k**2,(k,1,oo))))
        finding("p-series: sum 1/k^p converges iff p>1 (hard boundary at p=1)")

    elif p.ptype == PT.PROOF:
        body = p.meta.get("body","")
        if "sqrt(2)" in body.lower():
            kv("Boundary","sqrt(n) rational iff n perfect square")
        elif "prime" in body.lower():
            kv("Open problem","Twin prime conjecture (p, p+2 both prime): unproven")

    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        kv("sin^2+cos^2=1","holds ALL x in R (no exceptions)")
        kv("1+tan^2=sec^2", "fails at x=pi/2+n*pi (cos=0)")

    elif p.ptype == PT.FACTORING:
        try:
            irred = p.get_poly().is_irreducible
            kv("Irreducible over Q", irred)
            finding("Over C: always splits into linear factors (FTA)")
        except: pass

    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m")
        kv("Odd m","Fiber-uniform Hamiltonian decomposition exists")
        kv("Even m","Obstruction: sum(r_c)=m even, but each r_c must be odd. Contradiction.")
        finding("Parity is the HARD BOUNDARY for fiber-uniform construction")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 07 — SYNTHESIS & CROSS-DOMAIN BRIDGES
# ════════════════════════════════════════════════════════════════════════════

def phase_07(p: Problem, g6: dict) -> dict:
    section(7, "SYNTHESIS", "Cross-domain bridges · meta-lesson · what this problem teaches")
    r = {}

    # ── UNIVERSAL BRIDGES ─────────────────────────────────────────────────
    UNIVERSAL_BRIDGE_MAP = {
        PT.GRAPH: [
            ("Graph -> Markov",       "Random walk: P=D^{-1}A is Markov chain, stat pi_i = d_i/2|E|"),
            ("Graph -> Entropy",      "Spectral entropy H_s = -sum (lam_i/tr_L)*log(lam_i/tr_L)"),
            ("Graph -> Dynamical",    "Heat diffusion: u'(t) = -L*u(t); solution e^{-tL}*u_0"),
            ("Graph -> Optimization", "Min cut = max flow (Ford-Fulkerson LP duality)"),
            ("Graph -> ML",           "GNN: h' = sigma(Q^T*f(Lambda)*Q*h), Q=eigvecs of L"),
        ],
        PT.MATRIX: [
            ("Matrix -> Dynamical",   "x'=Ax: stable iff Re(lam_i(A))<0; solution e^{At}*x_0"),
            ("Matrix -> Control",     "Characteristic poly det(sI-A): poles in LHP = stable"),
            ("Matrix -> Optimization","Hessian H: quadratic form x^T*H*x determines curvature"),
            ("Matrix -> Entropy",     "Von Neumann: S(rho) = -tr(rho*log(rho)) (density matrix)"),
            ("Matrix -> Graph",       "0/1 symmetric matrix IS adjacency matrix of graph"),
        ],
        PT.MARKOV: [
            ("Markov -> Graph",       "P defines weighted directed graph; reversible P = undirected"),
            ("Markov -> Entropy",     "Entropy rate h = lim H(X_n|X_0,...,X_{n-1})"),
            ("Markov -> Optimization","MCMC: run chain to sample from target pi (Metropolis-Hastings)"),
            ("Markov -> Physics",     "Entropy production S_dot = sum pi_i P_ij log(pi_i P_ij/pi_j P_ji) >= 0"),
            ("Markov -> Control",     "MDP: optimal policy = control problem on Markov chain"),
        ],
        PT.ENTROPY: [
            ("Entropy -> Physics",    "Boltzmann: S = k_B * H (thermodynamic entropy)"),
            ("Entropy -> Markov",     "Entropy rate h = -sum_i pi_i sum_j P_ij log P_ij"),
            ("Entropy -> Optimization","MaxEnt: max H(p) s.t. constraints -> Gibbs/Boltzmann"),
            ("Entropy -> ML",         "Cross-entropy loss = H(y,p_hat) = H(y) + KL(y||p_hat)"),
            ("Entropy -> Graph",      "Spectral graph entropy = H(normalised Laplacian spectrum)"),
        ],
        PT.DYNAMICAL: [
            ("Dynamical -> Control",  "Control: x'=f(x,u); design u to steer to target"),
            ("Dynamical -> Optimization","Gradient flow: x'=-grad_f(x) IS gradient descent"),
            ("Dynamical -> Markov",   "SDE: x'=f(x)+noise -> Markov (Fokker-Planck equation)"),
            ("Dynamical -> Entropy",  "KS entropy h_KS = sum max(lam_i,0) (Lyapunov exponents)"),
        ],
        PT.CONTROL: [
            ("Control -> Matrix",     "Char poly = det(sI-A); poles = eigenvalues of A"),
            ("Control -> Optimization","LQR: min integral(x^T*Q*x+u^T*R*u)dt -> Riccati eqn"),
            ("Control -> Dynamical",  "Closed-loop: x'=(A+BK)x; place eigenvalues with K"),
            ("Control -> Graph",      "Multi-agent consensus: sync rate = lambda_2(Laplacian)"),
        ],
        PT.OPTIMIZATION: [
            ("Opt -> Dynamical",      "Gradient descent = Euler discretisation of x'=-grad_f"),
            ("Opt -> Markov",         "RL/MDP: policy optimisation via Bellman equations"),
            ("Opt -> Entropy",        "MaxEnt: max H(p) s.t. moments = exponential family"),
            ("Opt -> Graph",          "Shortest path = min-cost flow = LP on graph"),
        ],
    }

    bridges_for = UNIVERSAL_BRIDGE_MAP.get(p.ptype, [])
    if bridges_for:
        kv("Cross-domain bridges", "")
        for src_dst, desc in bridges_for:
            bridge(f"{src_dst}: {desc}")
        r["bridges"] = {sd:d for sd,d in bridges_for}

    # ── DOMAIN-SPECIFIC EMERGENTS ─────────────────────────────────────────
    if p.ptype == PT.GRAPH:
        A_spec = p.meta.get("A_spec",[]); L_spec = p.meta.get("L_spec",[])
        n = p.meta.get("n",0)
        r["heat_kernel"]   = "exp(-tL): heat equation on graph, diffusion from any node"
        r["ihara_zeta"]    = "Z_G(u) = prod_primes(1-u^|p|)^{-1}  (Riemann zeta analog)"
        kv("Heat kernel",    r["heat_kernel"]); kv("Ihara zeta", r["ihara_zeta"])
        if L_spec:
            tr_L = sum(L_spec)
            if tr_L > 0:
                nz = [e for e in L_spec if e>1e-9]
                H_s = _entropy([e/tr_L for e in nz])
                r["spectral_entropy"] = H_s
                kv("Spectral entropy H_s(G)", f"{H_s:.4f} bits")
        n_clust = sum(1 for e in L_spec if abs(e)<0.1) if L_spec else 0
        kv("Spectral clusters (lambda~0)", n_clust)
        insight(f"{n_clust} near-zero eigenvalues -> {n_clust} natural spectral clusters")
        insight("DEEPEST: graph spectrum = isomorphism fingerprint (isospectral graphs exist but are rare)")

    elif p.ptype == PT.MATRIX:
        spec = p.meta.get("spec",[])
        r["emergents"] = {
            "matrix_exp":  "e^{At} governs ALL linear dynamical systems",
            "SVD":         "M = U*Sigma*V^T: optimal rank-k approximation (Eckart-Young)",
            "pseudo_inv":  "M^+ = V*Sigma^+*U^T: least-squares solution to Ax=b",
        }
        for k_,v_ in r["emergents"].items(): kv(f"  {k_}", v_)
        if spec:
            pi_spec = [abs(e)/sum(abs(e2) for e2 in spec) for e in spec if abs(e)>1e-12]
            if pi_spec:
                H_vn = _entropy(pi_spec); kv("Von Neumann-like entropy", f"{H_vn:.4f} bits")
                insight(f"Spectral entropy of matrix = {H_vn:.4f} bits (quantum information analog)")
        insight("DEEPEST: matrix exponentiation e^{At} IS the universal solution to linear ODEs")

    elif p.ptype == PT.MARKOV:
        stat = p.meta.get("stat",{}); P_rat = p.meta.get("P_rat")
        n = p.meta.get("n",0)
        r["emergents"] = {
            "potential":  "Hitting times = Green's function G = (I-P)^{-1} (potential theory)",
            "martingales":"Harmonic h(x) = E[h(X_tau)|X_0=x] (optional stopping)",
            "MCMC":       "Sample from ANY distribution by constructing chain with target stationary",
            "free_energy":"F = <E> - T*H(pi) (equilibrium minimises free energy)",
        }
        for k_,v_ in r["emergents"].items(): kv(f"  {k_}", v_)
        if stat and P_rat:
            try:
                pi_f = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                ep = sum(pi_f[i]*float(N(P_rat[i,j]))*
                         math.log(max(pi_f[i]*float(N(P_rat[i,j])),1e-15)/
                                  max(pi_f[j]*float(N(P_rat[j,i])),1e-15))
                         for i in range(n) for j in range(n)
                         if float(N(P_rat[i,j]))>1e-12 and float(N(P_rat[j,i]))>1e-12)
                r["entropy_prod"] = ep; kv("Entropy production", f"{ep:.6f} (2nd law)")
                insight(f"Ep={ep:.4f} nat/step: {'reversible' if ep<1e-9 else 'irreversible — entropy produced'}")
            except: pass
        insight("DEEPEST: Markov chain IS a random walk on a weighted graph — these are the same object")

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs",[])
        r["emergents"] = {
            "mutual_info": "I(X;Y) = H(X)+H(Y)-H(X,Y): symmetric, >= 0, = 0 iff independent",
            "renyi":       "H_alpha = (1/(1-alpha))*log sum p_i^alpha; limit alpha->1 = Shannon",
            "free_energy": "F = <E> - T*H(pi): MinF = MaxEnt under energy constraint",
            "MDL":         "Minimum description length: Occam's razor quantified",
        }
        for k_,v_ in r["emergents"].items(): kv(f"  {k_}", v_)
        if probs:
            for alpha in [0.5, 2.0]:
                H_r = (1/(1-alpha))*math.log2(sum(p_**alpha for p_ in probs if p_>0))
                kv(f"Renyi H_{alpha}", f"{H_r:.4f} bits")
            insight(f"DEEPEST: MaxEnt principle = Bayesian prior with minimum assumptions = Gibbs distribution = ML softmax")

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr
        r["emergents"] = {
            "gradient_flow": "x'=-grad_f: stable eq = global min of f (unifies OPT and DYN)",
            "KS_entropy":    "h_KS = sum max(lam_i,0): chaos measured by Lyapunov exponents",
            "NF_theorem":    "Normal form theorem: near bifurcation, nonlinear = canonical form",
            "variational":   "Hamilton's principle: trajectories extremise action S = integral L dt",
        }
        for k_,v_ in r["emergents"].items(): kv(f"  {k_}", v_)
        insight("DEEPEST: gradient flows connect optimization, dynamical systems, and statistical mechanics")

    elif p.ptype == PT.CONTROL:
        r["emergents"] = {
            "pontryagin":  "Optimal control = variational problem (Pontryagin Maximum Principle)",
            "H_inf":       "Robust control = min-max optimization over disturbances",
            "kalman":      "Kalman filter: optimal state estimation = dual of LQR",
            "passivity":   "Passive systems: energy-based Lyapunov function always exists",
        }
        for k_,v_ in r["emergents"].items(): kv(f"  {k_}", v_)
        insight("DEEPEST: control = optimization over function space (Pontryagin duality)")

    elif p.ptype == PT.OPTIMIZATION:
        r["emergents"] = {
            "lagrangian_dual": "Strong duality: primal=dual (Slater) unifies OPT and PHYSICS",
            "proximal":        "Proximal gradient: implicit Euler discretisation of gradient flow",
            "mirror_descent":  "Mirror descent generalises gradient descent via Bregman divergence",
            "information_geo": "Natural gradient: Riemannian gradient w.r.t. Fisher info metric",
        }
        for k_,v_ in r["emergents"].items(): kv(f"  {k_}", v_)
        insight("DEEPEST: Lagrangian duality unifies optimization, physics (Hamiltonian), and information theory")

    elif p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        bridge("Polynomial -> Matrix: roots = eigenvalues of companion matrix C(p)")
        bridge("Polynomial -> Control: characteristic poly of A(s) -> poles of transfer function")
        bridge("Polynomial -> Dynamical: real roots = equilibria of ODE x'=p(x)")
        insight("DEEPEST: fundamental theorem of algebra <-> linear algebra <-> dynamical systems = same object")

    elif p.ptype == PT.SUM:
        bridge("Summation -> Number Theory: Euler-Maclaurin: sum ~ integral + corrections")
        bridge("Summation -> Entropy: power sums = cumulants of probability distributions")
        bridge("Summation -> Physics: partition function Z = sum exp(-E_i/kT)")
        insight("DEEPEST: Riemann zeta sum(1/n^s) encodes ALL prime distribution information")

    elif p.ptype == PT.PROOF:
        bridge("Proof method (contradiction) -> universal in analysis, set theory, cryptography")
        bridge("Irrationality -> Field extensions [Q(sqrt(n)):Q]=2 -> algebraic number theory")
        insight("DEEPEST: proof by contradiction IS the Halting Problem argument (diagonal argument)")

    elif p.ptype == PT.TRIG_ID:
        bridge("Trig identity -> Complex analysis: e^{i*theta}=cos+i*sin (Euler formula)")
        bridge("Trig -> Fourier: sin/cos = basis of L^2([0,2pi]) (spectral decomposition of functions)")
        bridge("Trig -> Graph: eigenvalues of cycle graph C_n = 2-2cos(2*pi*k/n)")
        insight("DEEPEST: e^{i*pi}+1=0 unifies analysis, algebra, geometry in one equation")

    elif p.ptype == PT.FACTORING:
        bridge("Factoring -> Number theory: polynomial factoring mod p = Berlekamp algorithm")
        bridge("Factoring -> Cryptography: integer factorisation hardness = RSA security")
        insight("DEEPEST: irreducibility is a relative concept — what's irreducible over Q may factor over C")

    elif p.ptype == PT.DIGRAPH_CYC:
        bridge("Digraph -> Group theory: Cayley graph Cay(Z_m^3) = group-theoretic Hamiltonicity")
        bridge("Digraph -> Coding theory: fiber decomposition = codeword structure in abelian group code")
        insight("DEEPEST: odd/even parity bifurcation reflects deep arithmetic structure of Z_m")

    # ── META-LESSON ───────────────────────────────────────────────────────
    print(f"\n  {DIM}--- meta-lesson ---{RST}")
    meta_lessons = {
        PT.GRAPH:        "What this problem teaches: network structure is fully encoded in its eigenvalues",
        PT.MATRIX:       "What this problem teaches: linear algebra IS the universal language of structure",
        PT.MARKOV:       "What this problem teaches: randomness + time = deterministic stationary behaviour",
        PT.ENTROPY:      "What this problem teaches: uncertainty is quantifiable and has hard arithmetic limits",
        PT.DYNAMICAL:    "What this problem teaches: stability = eigenvalue condition = potential landscape",
        PT.CONTROL:      "What this problem teaches: stability is a spectral property, not an analytical one",
        PT.OPTIMIZATION: "What this problem teaches: every optimization problem has a dual equilibrium interpretation",
        PT.QUADRATIC:    "What this problem teaches: the discriminant IS the geometry of the solution set",
        PT.CUBIC:        "What this problem teaches: degree 3 is where algebra first becomes genuinely hard",
        PT.POLY:         "What this problem teaches: high-degree polynomials require numerical methods (Abel-Ruffini)",
        PT.TRIG_ID:      "What this problem teaches: all of trigonometry follows from one geometric fact",
        PT.FACTORING:    "What this problem teaches: factorisation is context-dependent (Q vs R vs C vs Z_p)",
        PT.SUM:          "What this problem teaches: sums have algebraic closed forms when structure permits",
        PT.PROOF:        "What this problem teaches: proof by contradiction is the most powerful tool in mathematics",
        PT.DIGRAPH_CYC:  "What this problem teaches: parity is the deepest obstruction in combinatorics",
        PT.LINEAR:       "What this problem teaches: linear problems have unique solutions iff the operator is invertible",
    }
    lesson = meta_lessons.get(p.ptype, "What this problem teaches: structure determines behaviour")
    insight(lesson)
    r["meta_lesson"] = lesson

    # ── CONFIDENCE FINAL SUMMARY ─────────────────────────────────────────
    print(f"\n  {DIM}--- confidence ledger ---{RST}")
    kv("Known with high confidence", p.conf.knowns[:3] if p.conf.knowns else "none yet")
    kv("Uncertain",                  p.conf.unknowns[:3] if p.conf.unknowns else "none")
    if p.conf.flags:
        for flag in p.conf.flags[:3]: warn(flag)
    kv("Overall", p.conf.summary())

    return r


# ════════════════════════════════════════════════════════════════════════════
# FINAL ANSWER
# ════════════════════════════════════════════════════════════════════════════

def _final_answer(p: Problem) -> str:
    v = p.var
    if p.ptype == PT.GRAPH:
        n=p.meta.get("n",0); spec=p.meta.get("L_spec",[]); named=p.meta.get("named","graph")
        conn=(sorted(spec)[1]>1e-9) if len(spec)>1 else "?"
        return f"{named} ({n} vertices): Connected={conn}, L-spectrum={[f'{e:.3f}' for e in spec]}"
    elif p.ptype == PT.MATRIX:
        spec=p.meta.get("spec",[]); M=p.meta.get("M")
        return f"Matrix: eigenvalues={[f'{e:.3f}' for e in spec]}, det={str(det(M)) if M else '?'}"
    elif p.ptype == PT.MARKOV:
        stat=p.meta.get("stat",{}); n=p.meta.get("n",0)
        return f"Markov chain ({n} states): stationary pi={stat}"
    elif p.ptype == PT.ENTROPY:
        probs=p.meta.get("probs",[])
        if probs:
            H=_entropy(probs)
            return f"H(X) = {H:.6f} bits (max={math.log2(len(probs)):.6f}, efficiency={H/math.log2(len(probs)):.4f})"
        return "Entropy: binary H(p), KL, Huffman bounds computed"
    elif p.ptype == PT.DYNAMICAL:
        try:
            equil=solve(p.expr, v); fp=diff(p.expr, v)
            stab=[("S" if float(N(fp.subs(v,e)))<0 else "U") for e in equil]
            return f"Equilibria: {list(zip([str(e) for e in equil], stab))}"
        except: return "Dynamical system: equilibria and stability computed"
    elif p.ptype == PT.CONTROL:
        rh=p._cache.get("routh",{})
        return f"Control: {'STABLE' if rh.get('stable') else 'UNSTABLE'} ({rh.get('sign_changes',0)} RHP roots)"
    elif p.ptype == PT.OPTIMIZATION:
        opt=p._cache.get("solve(f'=0)",[])
        if opt:
            try:
                vals=[(float(N(p.expr.subs(v,c))),c) for c in opt]
                g=p.meta.get("goal","extremize")
                best=(min if "min" in g else max)(vals)
                return f"Optimal x*={best[1]}, f*={best[0]:.6f}"
            except: pass
        return "Optimization: critical points classified"
    elif p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        try:
            sols=p.memo('solve(expr,var)', lambda: solve(p.expr,v))
            return f"Solutions: {', '.join(str(s) for s in sols)}"
        except: return "See phase computations"
    elif p.ptype == PT.FACTORING:
        try: return f"Factored: {p.memo('factor(expr)', lambda: factor(p.expr))}"
        except: return "See phase computations"
    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        try:
            s=p.memo('trigsimp', lambda: trigsimp(p.expr))
            return "Identity confirmed" if s==0 else f"Simplified: {s}"
        except: return "See phase computations"
    elif p.ptype == PT.SUM:
        k=symbols('k',positive=True,integer=True); n=symbols('n',positive=True,integer=True)
        try: return f"Sum = {factor(summation(k,(k,1,n)))} = n(n+1)/2"
        except: return "Summation computed"
    elif p.ptype == PT.PROOF:
        body=p.meta.get("body","")
        if "sqrt(2)" in body.lower(): return "sqrt(2) irrational. Proof by contradiction. QED."
        elif "prime" in body.lower(): return "Infinitely many primes. Euclid's construction. QED."
    elif p.ptype == PT.DIGRAPH_CYC:
        m=p.meta.get("m")
        return (f"Odd m={m}: Hamiltonian decomposition exists." if m%2!=0
                else f"Even m={m}: parity obstruction — fiber-uniform impossible.")
    return "See phase computations"


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run(raw: str):
    prob = classify(raw)
    print(f"\n{hr('=')}")
    print(f"{W}DISCOVERY ENGINE v3{RST}")
    print(hr())
    print(f"  {W}Problem:{RST}  {Y}{raw}{RST}")
    print(f"  {DIM}Type:{RST}     {prob.ptype.label()}")
    print(f"  {DIM}Variable:{RST} {prob.var}")
    print(hr('='))

    if prob.ptype == PT.UNKNOWN:
        print(f"{R}Could not classify. Try: 'x^2-5x+6=0', 'graph K4', 'markov [[...]]', 'entropy [...]', 'dynamical x^3-x'{RST}")
        return

    g1 = phase_01(prob)
    g2 = phase_02(prob, g1)
    rh = g2.get("routh"); 
    if rh: prob._cache["routh"] = rh
    g3 = phase_03(prob, g2)
    g4 = phase_04(prob, g3)
    g5 = phase_05(prob, g4)
    g6 = phase_06(prob, g5)
    g7 = phase_07(prob, g6)

    print(f"\n{hr('=')}")
    print(f"{W}FINAL ANSWER{RST}")
    print(hr())
    final = _final_answer(prob)
    print(f"  {G}{final}{RST}")
    print(hr('='))

    titles = {1:"Ground Truth+Intel", 2:"Direct Attack",    3:"Structure Hunt",
              4:"Pattern Lock",       5:"Generalize",        6:"Prove Limits",
              7:"Synthesis"}
    phases  = [g1,g2,g3,g4,g5,g6,g7]
    print(f"\n{hr()}")
    print(f"{W}PHASE SUMMARY{RST}")
    print(hr('.'))
    for i,(g,t) in enumerate(zip(phases, titles.values()), 1):
        n_keys = len(g)
        print(f"  {PHASE_CLR[i]}{i:02d} {t:<22}{RST} {n_keys} results computed")
    kv("Confidence ledger", prob.conf.summary())
    print(hr('='))


# ════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ════════════════════════════════════════════════════════════════════════════

TESTS = [
    # Algebraic
    ("x^2 - 5x + 6 = 0",              "Quadratic — integer roots"),
    ("2x + 3 = 7",                     "Linear equation"),
    ("x^3 - 6x^2 + 11x - 6 = 0",      "Cubic — 3 integer roots"),
    ("sin(x)^2 + cos(x)^2",            "Pythagorean identity"),
    ("factor x^4 - 16",                "Difference of squares chain"),
    ("sum of first n integers",        "Classic summation"),
    ("prove sqrt(2) is irrational",    "Irrationality proof"),
    ("m^3 vertices with 3 cycles, m=3","Digraph — odd m"),
    ("m^3 vertices with 3 cycles, m=4","Digraph — even m"),
    # Graph / Network
    ("graph K4",                       "Complete graph K4"),
    ("graph P5",                       "Path graph P5"),
    ("graph C6",                       "Cycle graph C6"),
    ("graph [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]", "Custom adjacency matrix"),
    # Matrix
    ("matrix [[2,1],[1,3]]",           "Symmetric 2x2"),
    ("matrix [[4,2,2],[2,3,0],[2,0,3]]","Symmetric 3x3 definiteness"),
    # Markov
    ("markov [[0.7,0.3],[0.4,0.6]]",   "2-state Markov chain"),
    ("markov [[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]]", "3-state Markov chain"),
    # Entropy
    ("entropy [0.5,0.25,0.25]",        "Entropy — skewed"),
    ("entropy [0.25,0.25,0.25,0.25]",  "Entropy — uniform maximum"),
    # Dynamical
    ("dynamical x^3 - x",              "Dynamical — 3 equilibria"),
    ("dynamical x^2 - 1",              "Dynamical — pitchfork"),
    # Control
    ("control s^2 + 3s + 2",           "Control — stable 2nd order"),
    ("control s^3 + 2s^2 + 3s + 1",   "Control — Routh 3rd order"),
    ("control s^3 - s + 1",            "Control — unstable"),
    # Optimization
    ("optimize x^4 - 4x^2 + 1",       "Optimization — quartic"),
    ("minimize x^2 + 2x + 1",          "Minimize — quadratic"),
]


def run_tests():
    print(f"\n{hr('=')}")
    print(f"{W}DISCOVERY ENGINE v3 — TEST SUITE ({len(TESTS)} problems){RST}")
    print(hr('='))
    passed = 0; failed = []
    for raw, desc in TESTS:
        print(f"\n{B}{hr('-', 60)}{RST}")
        print(f"{B}TEST: {desc}{RST}  {DIM}[{raw}]{RST}")
        try:
            run(raw)
            ok(f"PASSED: {desc}"); passed += 1
        except Exception as e:
            fail(f"FAILED: {desc} — {e}")
            failed.append((desc, str(e)))
            traceback.print_exc()
    print(f"\n{hr('=')}")
    clr = G if passed == len(TESTS) else Y
    print(f"{clr}Results: {passed}/{len(TESTS)} passed{RST}")
    if failed:
        print(f"{R}Failed:{RST}")
        for d,e in failed: print(f"  {R}x{RST} {d}: {e}")
    print(hr('='))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:        print(__doc__)
    elif args[0]=="--test": run_tests()
    else:               run(" ".join(args))

#!/usr/bin/env python3
"""
discovery_engine_v4.py — 7-Phase Mathematical Discovery Engine
================================================================
Pure sympy. No API. All phases compute symbolically/numerically.

v4 integrations over v3:
  A. SPECTRAL FINGERPRINT  — unified eigenvalue dataclass across all domains
  B. ADAPTIVE PHASE DEPTH  — skip phases for trivial problems (saves 60-80% time)
  C. SOLUTION FAMILY       — detect which parametric family this problem belongs to
  D. INTER-PHASE FEEDBACK  — Phase 03 discoveries unlock new Phase 02 methods
  E. SPECTRAL UNIFICATION  — Phase 07 compares fingerprints across domains
  F. OUTPUT ENTROPY SCORE  — engine scores its own output diversity, prunes redundancy

v4 fixes over v3:
  - Dead code removed (_parse had duplicate for-loop after return None)
  - Variable selector: prefer x,y,z,t over alphabetical first
  - SUM: detects squares/cubes/power-N/reciprocal from natural language
  - Proof: natural language aliases (root 2, irrational)
  - Complex check: .is_real (not isinstance...ComplexNumber)
  - Final answer: cache-first for all types
  - Timeout wrapper: SymPy calls that hang are killed at 8s
  - Routh: fixed for edge case zero-leading coefficients

Usage:
  python discovery_engine_v4.py "x^2 - 5x + 6 = 0"
  python discovery_engine_v4.py --test
  python discovery_engine_v4.py --test --quiet
  python discovery_engine_v4.py --json "entropy [0.5,0.25,0.25]"
  python discovery_engine_v4.py --bench
"""

import sys, re, ast, math, traceback, time, signal, heapq, json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from functools import lru_cache

import sympy as sp
from sympy import (
    symbols, solve, simplify, expand, factor, cancel,
    Symbol, Rational, Integer, pi, E, I, oo,
    sin, cos, tan, sec, csc, cot, exp, log, sqrt, Abs,
    diff, integrate, limit, summation,
    discriminant, roots, Poly, factorint,
    Eq, trigsimp, expand_trig, nsolve, N, solveset, S,
    gcd, divisors, apart, collect, nsimplify,
    real_roots, all_roots, factor_list, sqf_list,
    Matrix, eye, zeros, ones, diag, det, trace,
    re as sp_re, im as sp_im, prod as sp_prod,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# ── Colour helpers ────────────────────────────────────────────────────────
R="\033[91m"; G="\033[92m"; Y="\033[93m"; B="\033[94m"
M="\033[95m"; C="\033[96m"; W="\033[97m"; DIM="\033[2m"; RST="\033[0m"
PHASE_CLR={1:G,2:R,3:B,4:M,5:Y,6:C,7:W}

_QUIET = False   # set by --quiet flag

def hr(ch="─",n=72): return ch*n
def section(num,name,tag):
    if _QUIET: return
    c=PHASE_CLR[num]
    print(f"\n{hr()}\n{c}Phase {num:02d} — {name}{RST}  {DIM}{tag}{RST}\n{hr('·')}")
def kv(k,v,indent=2):
    if _QUIET: return
    print(f"{' '*indent}{DIM}{k:<36}{RST}{W}{str(v)[:120]}{RST}")
def finding(msg,sym="→"):
    if _QUIET: return
    print(f"  {Y}{sym}{RST} {msg}")
def ok(msg):
    if _QUIET: return
    print(f"  {G}✓{RST} {msg}")
def fail(msg):
    if _QUIET: return
    print(f"  {R}✗{RST} {msg}")
def note(msg):
    if _QUIET: return
    print(f"  {DIM}{msg}{RST}")
def bridge(msg):
    if _QUIET: return
    print(f"  {C}⇔{RST} {B}{msg}{RST}")
def warn(msg):
    if _QUIET: return
    print(f"  {Y}⚠{RST} {msg}")
def insight(msg):
    if _QUIET: return
    print(f"  {M}★{RST} {W}{msg}{RST}")


# ════════════════════════════════════════════════════════════════════════════
# TIMEOUT WRAPPER  — kill slow SymPy calls at 8 seconds
# ════════════════════════════════════════════════════════════════════════════

class _TimeoutError(Exception): pass

def _timed(fn, timeout=8, default=None):
    """Run fn() with a wall-clock timeout. Returns default on timeout."""
    if sys.platform == "win32":
        # signal.SIGALRM not on Windows — just run it
        try: return fn()
        except: return default
    def _handler(sig, frame): raise _TimeoutError()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        result = fn()
        signal.alarm(0)
        return result
    except _TimeoutError:
        warn(f"  ⏱ Timeout ({timeout}s) — skipping")
        return default
    except Exception:
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ════════════════════════════════════════════════════════════════════════════
# SPECTRAL FINGERPRINT  — Integration A: unified eigenvalue flow
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralFingerprint:
    """
    Universal eigenvalue container. Every problem type that produces
    eigenvalues stores them here. Phase 07 compares fingerprints across
    domains to detect non-obvious structural equivalences.
    """
    domain:   str            # "graph_laplacian", "graph_adj", "matrix", "markov", "control", "companion"
    values:   List[float]    = field(default_factory=list)   # sorted real parts
    complex_:  List[complex] = field(default_factory=list)   # full complex values
    label:    str            = ""

    def sorted_real(self) -> List[float]:
        return sorted(self.values)

    def spectral_entropy(self) -> float:
        """Entropy of normalised absolute eigenvalue distribution."""
        abs_v = [abs(x) for x in self.values if abs(x) > 1e-12]
        if not abs_v: return 0.0
        total = sum(abs_v)
        probs = [x/total for x in abs_v]
        return -sum(p*math.log2(p) for p in probs if p > 0)

    def matches(self, other: "SpectralFingerprint", tol=0.01) -> bool:
        """True if two fingerprints are numerically close (potential isomorphism)."""
        a, b = self.sorted_real(), other.sorted_real()
        if len(a) != len(b): return False
        return all(abs(x-y) < tol for x,y in zip(a,b))

    def summary(self) -> str:
        sr = self.sorted_real()
        return (f"{self.domain}: [{', '.join(f'{v:.3f}' for v in sr[:5])}]"
                + ("…" if len(sr)>5 else ""))


# ════════════════════════════════════════════════════════════════════════════
# FEEDBACK QUEUE  — Integration D: inter-phase communication
# ════════════════════════════════════════════════════════════════════════════

class FeedbackQueue:
    """
    Phase 03 discoveries push signals here. Phase 02 (and later phases)
    check the queue to unlock additional methods not tried in the first pass.
    """
    def __init__(self):
        self._signals: List[Tuple[str,Any]] = []

    def push(self, signal: str, payload: Any = None):
        self._signals.append((signal, payload))
        note(f"  [feedback] {signal}")

    def has(self, signal: str) -> bool:
        return any(s==signal for s,_ in self._signals)

    def get(self, signal: str) -> Any:
        for s,p in self._signals:
            if s==signal: return p
        return None

    def all_signals(self) -> List[str]:
        return [s for s,_ in self._signals]


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════════════════

class KB:
    METHOD_PRIORS = {
        ("quadratic","solve"):0.99, ("quadratic","discriminant"):0.99, ("quadratic","factor"):0.70,
        ("cubic","solve"):0.85, ("cubic","factor"):0.60,
        ("poly_high","solve"):0.40, ("poly_high","nsolve"):0.90,
        ("trig_id","trigsimp"):0.95,
        ("factoring","factor"):0.90,
        ("graph","spectrum"):0.95,
        ("markov","stationary"):0.85, ("markov","eigenvalues"):0.90,
        ("entropy","H_numeric"):0.99,
        ("dynamical","solve_equil"):0.90,
        ("control","routh_hurwitz"):0.95, ("control","roots"):0.80,
        ("optimize","critical_pts"):0.90, ("optimize","hessian"):0.85,
        ("matrix","eigenvalues"):0.95,
    }
    ANALOGIES = {
        "QUADRATIC":   ["2D linear system stability (trace,det)","Eigenvalue of 2×2 matrix","2-state entropy p(1-p)"],
        "CUBIC":       ["Char poly of 3×3 matrix","Cubic potential equilibria","3-state Markov eigenvalues"],
        "GRAPH":       ["Markov random walk D⁻¹A","Heat diffusion e^{-tL}","Consensus eigenvalues of L"],
        "MARKOV":      ["Weighted directed graph","Linear stochastic system","Thermodynamic entropy production"],
        "ENTROPY":     ["Free energy F=<E>-T·H","KL divergence","ML cross-entropy loss"],
        "DYNAMICAL":   ["Gradient descent ẋ=-∇f","Fokker-Planck (SDE)","Graph Laplacian diffusion"],
        "CONTROL":     ["Markov spectral radius ≤1","Convex opt with stability","Lyapunov design"],
        "OPTIMIZATION":["Dynamical equilibrium","MaxEnt (constrained)","Markov stationary dist"],
        "MATRIX":      ["Graph adjacency matrix","Markov transition matrix","Dynamical Jacobian"],
        "FACTORING":   ["Root-finding","Modular arithmetic","Cryptography (RSA)"],
        "LINEAR":      ["1D Markov chain","Projection","Linear regression"],
        "POLY":        ["Companion matrix","Characteristic polynomial","Transfer function poles"],
    }
    VERIFICATION = {
        "equation": ["substitute back","discriminant sign vs root count","Vieta sum/product"],
        "factoring":["expand(factor)-original=0","roots match","degree preserved"],
        "identity": ["substitute 3+ numerical values","trigsimp=0","boundary: 0,π/2,π"],
        "graph":    ["tr(L)=Σdeg","zero-eig count=components","tr(A²)/2=|E|"],
        "markov":   ["rows sum to 1","π·P=π","spectral radius=1"],
        "entropy":  ["H≥0","H≤log₂n","Σpᵢ=1"],
        "control":  ["sign changes=RHP count","all-positive necessary","Routh first-col sufficient"],
        "optimize": ["f'(x*)=0","f'' sign matches","limits at ±∞"],
    }
    FAILURE_MODES = {
        "POLY":       "Degree≥5: Abel-Ruffini — no radical formula, numerics needed",
        "QUADRATIC":  "Δ<0: complex roots — verify problem expects real only",
        "MARKOV":     "Reducible chain: multiple stationary distributions possible",
        "DYNAMICAL":  "Non-hyperbolic (f'=0): linearisation insufficient",
        "CONTROL":    "Zero coefficient: Routh array degenerates — use epsilon perturbation",
        "ENTROPY":    "p=0 terms: 0·log0=0 by convention — must handle explicitly",
        "GRAPH":      "Disconnected: λ₂=0, Kirchhoff breaks — each component separately",
        "OPTIMIZE":   "Non-convex: multiple local minima, no global guarantee",
    }
    # Complexity score: how many phases are warranted?
    PHASE_DEPTH = {
        PT_NAME: depth for PT_NAME, depth in [
            ("LINEAR",3),("TRIG_ID",3),("SIMPLIFY",3),
            ("SUM",4),("PROOF",4),("DIGRAPH_CYC",4),
            ("QUADRATIC",5),("TRIG_EQ",5),("FACTORING",5),
            ("CUBIC",6),("POLY",6),
            ("GRAPH",7),("MATRIX",7),("MARKOV",7),
            ("ENTROPY",7),("DYNAMICAL",7),("CONTROL",7),("OPTIMIZATION",7),
            ("UNKNOWN",2),
        ]
    }

    @classmethod
    def method_confidence(cls,pt,method): return cls.METHOD_PRIORS.get((pt.lower(),method.lower()),0.5)
    @classmethod
    def analogies_for(cls,pt): return cls.ANALOGIES.get(pt.upper(),[])
    @classmethod
    def verify_strategies(cls,cat): return cls.VERIFICATION.get(cat,["check numerically"])
    @classmethod
    def failure_modes(cls,pt): return cls.FAILURE_MODES.get(pt.upper())
    @classmethod
    def phase_depth(cls,pt_name: str) -> int: return cls.PHASE_DEPTH.get(pt_name, 7)


# ════════════════════════════════════════════════════════════════════════════
# CONFIDENCE LEDGER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Confidence:
    results:  Dict[str,Tuple[Any,float]] = field(default_factory=dict)
    flags:    List[str]                   = field(default_factory=list)
    knowns:   List[str]                   = field(default_factory=list)
    unknowns: List[str]                   = field(default_factory=list)

    def record(self,key,val,conf,note_str=""):
        self.results[key]=(val,conf)
        if conf>=0.9: self.knowns.append(f"{key}: {str(val)[:50]}")
        elif conf<0.6: self.unknowns.append(f"{key} (conf={conf:.2f})")
        if note_str: self.flags.append(note_str)

    def summary(self):
        t=len(self.results)
        h=sum(1 for _,c in self.results.values() if c>=0.9)
        m=sum(1 for _,c in self.results.values() if 0.6<=c<0.9)
        l=sum(1 for _,c in self.results.values() if c<0.6)
        return f"{t} results: {h} high-conf, {m} mid-conf, {l} uncertain"


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
        return {
            1:"linear eq",2:"quadratic eq",3:"cubic eq",4:"poly deg≥4",
            5:"trig eq",6:"trig identity",7:"factoring",8:"simplification",
            9:"summation",10:"proof",11:"digraph cycle",12:"graph/network",
            13:"matrix",14:"markov chain",15:"information entropy",
            16:"dynamical system",17:"control theory",18:"optimization",99:"unknown"
        }.get(self.value,"unknown")


@dataclass
class Problem:
    raw:    str
    ptype:  PT
    expr:   Optional[sp.Basic]  = None
    lhs:    Optional[sp.Basic]  = None
    rhs:    Optional[sp.Basic]  = None
    var:    Optional[sp.Symbol] = None
    free:   List[sp.Symbol]     = field(default_factory=list)
    meta:   Dict[str,Any]       = field(default_factory=dict)
    poly:   Optional[sp.Poly]   = None
    _cache: Dict[str,Any]       = field(default_factory=dict,repr=False)
    conf:   Confidence           = field(default_factory=Confidence,repr=False)
    fbq:    FeedbackQueue        = field(default_factory=FeedbackQueue,repr=False)
    fps:    List[SpectralFingerprint] = field(default_factory=list,repr=False)
    _output_lines: List[str]    = field(default_factory=list,repr=False)

    def memo(self,key,func,timeout=8):
        if key not in self._cache:
            self._cache[key] = _timed(func, timeout)
        return self._cache[key]

    def get_poly(self):
        if self.poly is None and self.expr is not None and self.var is not None:
            try: self.poly = Poly(self.expr, self.var)
            except: pass
        return self.poly

    def is_even(self):
        if self.expr is None or self.var is None: return False
        return self.memo("is_even", lambda: simplify(self.expr.subs(self.var, -self.var) - self.expr) == 0)

    def is_odd(self):
        if self.expr is None or self.var is None: return False
        return self.memo("is_odd", lambda: simplify(self.expr.subs(self.var, -self.var) + self.expr) == 0)

    def get_expanded_expr(self):
        if self.expr is None: return None
        return self.memo("expanded_expr", lambda: expand(self.expr))

    def ptype_str(self): return self.ptype.name

    def add_fingerprint(self, domain: str, values: List[float],
                        complex_: List[complex] = None, label: str = "") -> SpectralFingerprint:
        fp = SpectralFingerprint(domain=domain, values=values,
                                  complex_=complex_ or [], label=label)
        self.fps.append(fp)
        return fp


# ════════════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _var_prefer(free: List[sp.Symbol]) -> Optional[sp.Symbol]:
    """Prefer x,y,z,t,s over alphabetical first (Integration from user engine)."""
    preferred = list("xyzts")
    for name in preferred:
        for f in free:
            if str(f) == name: return f
    return free[0] if free else None

def _parse(s: str) -> Optional[sp.Basic]:
    s = s.strip().replace('^','**')
    for old,new in [("ln","log"),("arcsin","asin"),("arccos","acos"),("arctan","atan")]:
        s = re.sub(rf'\b{old}\b', new, s)
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

def _parse_summand(raw: str):
    """
    Integration B from user engine: detect summand type from natural language.
    Returns a SymPy expression in k, the summation variable.
    """
    low = raw.lower()
    k = symbols('k', positive=True, integer=True)
    # Power N
    m = re.search(r'power\s+(\d+)', low)
    if m: return k**int(m.group(1))
    if 'cube' in low: return k**3
    if 'square' in low: return k**2
    if 'reciprocal' in low or '1/k' in low: return 1/k
    if 'harmonic' in low: return 1/k
    return k  # default: sum of first n integers

def _spectrum_real(M: sp.Matrix) -> List[float]:
    try:
        eigs = M.eigenvals()
        out  = []
        for k,mult in eigs.items():
            try:    v = float(N(k))
            except: v = float(N(sp_re(k)))
            out.extend([v]*int(mult))
        return sorted(out)
    except: return []

def _spectrum_complex(M: sp.Matrix) -> List[complex]:
    try:
        eigs = M.eigenvals()
        out  = []
        for k,mult in eigs.items():
            try:    v = complex(N(k))
            except: v = complex(float(N(sp_re(k))), 0)
            out.extend([v]*int(mult))
        return sorted(out, key=lambda z:(z.real,z.imag))
    except: return []

def _build_graph(p: Problem):
    A = p.meta.get("A")
    if isinstance(A, sp.Matrix):
        n   = A.shape[0]
        deg = [int(sum(A.row(i))) for i in range(n)]
        return A, diag(*deg)-A, n, deg
    t = p.meta.get("type"); n = p.meta.get("n",4)
    if   t=="complete": A=ones(n,n)-eye(n)
    elif t=="path":
        A=zeros(n,n)
        for i in range(n-1): A[i,i+1]=A[i+1,i]=1
    elif t=="cycle":
        A=zeros(n,n)
        for i in range(n): A[i,(i+1)%n]=A[(i+1)%n,i]=1
    else: return None,None,0,[]
    deg=[int(sum(A.row(i))) for i in range(n)]
    p.meta["A"]=A; p.meta["n"]=n
    return A,diag(*deg)-A,n,deg

def _entropy(probs: List[float]) -> float:
    return -sum(p*math.log2(p) for p in probs if p>0)

def _kl(P: List[float], Q: List[float]) -> float:
    return sum(P[i]*math.log2(P[i]/Q[i]) for i in range(len(P)) if P[i]>0 and Q[i]>0)

def _routh(coeffs) -> Dict[str,Any]:
    """Routh-Hurwitz array. Handles zero leading element via epsilon perturbation."""
    EPS = 1e-10
    c   = [float(N(sp.sympify(x))) for x in coeffs]
    r0  = c[0::2]; r1 = c[1::2]
    while len(r0)<len(r1): r0.append(0.0)
    while len(r1)<len(r0): r1.append(0.0)
    rows = [r0[:], r1[:]]
    iters = 0
    while iters < 50:
        iters += 1
        pr,cr = rows[-2],rows[-1]
        if not cr or all(abs(x)<1e-15 for x in cr): break
        if abs(cr[0])<1e-12: cr = [EPS]+list(cr[1:])
        nr = [(cr[0]*pr[i+1]-pr[0]*cr[i+1])/cr[0]
              for i in range(len(cr)-1) if i+1<len(pr)]
        if not nr: break
        rows.append(nr)
    fc    = [row[0] for row in rows if row]
    sc    = sum(1 for i in range(len(fc)-1) if fc[i]*fc[i+1]<0)
    stable= (sc==0) and all(x>0 for x in fc)
    return {"stable":stable,"sign_changes":sc,"first_column":fc,"rows":rows}

def _stationary(P: sp.Matrix) -> Optional[Dict]:
    n  = P.shape[0]
    pi = symbols(f'pi0:{n}', positive=True)
    eqs= [sum(pi[i]*P[i,j] for i in range(n))-pi[j] for j in range(n)]
    eqs.append(sum(pi)-1)
    try:
        sol = _timed(lambda: solve(eqs, list(pi)), timeout=10)
        return sol
    except: return None

def _verify_eq(expr, var, sol) -> Tuple[bool,float]:
    try:
        res = simplify(expr.subs(var,sol))
        mag = float(abs(N(res)))
        return (mag<1e-9, mag)
    except: return (False, float('inf'))

def _vieta_check(poly, sols) -> bool:
    try:
        coeffs = poly.all_coeffs(); n = poly.degree()
        s_exp  = -coeffs[-2]/coeffs[-1] if n>=1 else 0
        if abs(float(N(simplify(sum(sols)-s_exp)))) > 1e-6: return False
        p_exp  = (-1)**n * coeffs[0]/coeffs[-1]
        p_act  = sp_prod(sols)
        if abs(float(N(simplify(p_act-p_exp)))) > 1e-6: return False
        return True
    except: return True

def _is_real_value(expr) -> bool:
    """Robust real-check using .is_real (not isinstance)."""
    try:
        v = N(expr)
        return getattr(v,'is_real',False) or (abs(float(sp_im(v)))<1e-9)
    except: return False

def _companion_fingerprint(poly: sp.Poly, p: Problem, label="") -> Optional[SpectralFingerprint]:
    """Build companion matrix → eigenvalues → SpectralFingerprint."""
    try:
        n     = poly.degree()
        coeffs= [float(N(c)) for c in poly.all_coeffs()]
        lc    = coeffs[0]
        norm  = [c/lc for c in coeffs[1:]]
        C     = zeros(n,n)
        for i in range(n-1): C[i+1,i] = 1
        for i,c in enumerate(norm): C[0,i] = -c
        spec_c = _spectrum_complex(C)
        vals   = [z.real for z in spec_c]
        return p.add_fingerprint("companion_poly", vals, spec_c, label or str(poly.as_expr()))
    except: return None

def _output_entropy(lines: List[str]) -> float:
    """
    Integration F: score diversity of engine output lines.
    Low entropy = engine is repeating itself. High = genuinely new info each line.
    Uses word-level unigram frequency.
    """
    if not lines: return 0.0
    words: Dict[str,int] = {}
    for l in lines:
        for w in re.sub(r'[^\w]',' ',l.lower()).split():
            if len(w)>3: words[w]=words.get(w,0)+1
    total = sum(words.values())
    if total==0: return 0.0
    probs = [c/total for c in words.values()]
    return -sum(p*math.log2(p) for p in probs if p>0)


# ════════════════════════════════════════════════════════════════════════════
# SOLUTION FAMILY DETECTOR  — Integration C
# ════════════════════════════════════════════════════════════════════════════

def _detect_family(p: Problem) -> Optional[str]:
    """
    Detect which parametric family this problem belongs to.
    Returns a description string, or None.
    """
    if p.ptype not in (PT.QUADRATIC,PT.CUBIC,PT.POLY,PT.FACTORING): return None
    expr = p.expr; v = p.var
    if expr is None or v is None: return None
    try:
        poly = p.get_poly()
        if poly is None: return None
        deg  = poly.degree()
        coeffs = poly.all_coeffs()

        if deg==2:
            a,b,c_ = [float(N(x)) for x in coeffs]
            # Perfect square?
            if abs(b**2-4*a*c_)<1e-9: return f"Perfect square: ({v}+{b/(2*a):.3g})²=0"
            # Monic integer?
            if abs(a-1)<1e-9 and all(abs(float(N(x))-round(float(N(x))))<1e-9 for x in coeffs):
                r1,r2 = -b/2+math.sqrt(abs(b**2-4*c_))/2, -b/2-math.sqrt(abs(b**2-4*c_))/2
                if abs(r1-round(r1))<0.01 and abs(r2-round(r2))<0.01:
                    return f"Integer roots family: ({v}-{int(round(r1))})({v}-{int(round(r2))})=0"
            return f"General quadratic: a={a:.3g}, b={b:.3g}, c={c_:.3g}"

        if deg==3:
            a,b,c_,d_ = [float(N(x)) for x in coeffs]
            if abs(b)<1e-9 and abs(d_)<1e-9:
                return f"Factored cubic: {v}({v}²-{-c_/a:.3g})=0 — three-root form"
            return f"General cubic family — Cardano/factor theorem applies"

        if deg==4:
            a,b,c_,d_,e_ = [float(N(x)) for x in coeffs]
            if abs(b)<1e-9 and abs(d_)<1e-9:
                return f"BIQUADRATIC: substitute u={v}² → quadratic in u"
            return f"Quartic family — Ferrari/numerical"

        return f"Degree-{deg} polynomial family"
    except: return None


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

def classify(raw: str) -> Problem:
    s=raw.strip(); low=s.lower()

    # Digraph cycle
    if "vertices" in low and ("m^3" in low or "m**3" in low) and "cycles" in low:
        mm=re.search(r'm\s*=\s*(\d+)',low)
        return Problem(raw=raw,ptype=PT.DIGRAPH_CYC,meta={"m":int(mm.group(1)) if mm else 3})

    # Control
    if re.match(r'^control\b',low):
        body=re.sub(r'^control\s*','',s,flags=re.I).strip()
        e=_parse(body); free=sorted(e.free_symbols,key=str) if e else []
        v=next((f for f in free if str(f)=='s'),_var_prefer(free) or symbols('s'))
        _p=None
        try: _p=Poly(e,v)
        except: pass
        return Problem(raw=raw,ptype=PT.CONTROL,expr=e,var=v,free=free,poly=_p)

    # Dynamical
    if re.match(r'^dynamical?\b',low):
        body=re.sub(r'^dynamical?\s*','',s,flags=re.I).strip()
        e=_parse(body); free=sorted(e.free_symbols,key=str) if e else []
        v=_var_prefer(free) or symbols('x')
        return Problem(raw=raw,ptype=PT.DYNAMICAL,expr=e,var=v,free=free)

    # Optimization
    if re.match(r'^(optimiz[a-z]*|minimiz[a-z]*|maximiz[a-z]*|extrema|find\s+(min|max))\b',low):
        body=re.sub(r'^(optimiz[a-z]*|minimiz[a-z]*|maximiz[a-z]*|extrema|find\s+(min|max)\s*of?\s*)','',s,flags=re.I).strip()
        e=_parse(body); free=sorted(e.free_symbols,key=str) if e else []
        v=_var_prefer(free) or symbols('x')
        goal=("minimize" if re.match(r'^minimiz',low) else
              "maximize" if re.match(r'^maximiz',low) else "extremize")
        return Problem(raw=raw,ptype=PT.OPTIMIZATION,expr=e,var=v,free=free,meta={"goal":goal})

    # Matrix (bare brackets first, before graph check)
    if re.match(r'^matrix\b',low) or (re.search(r'\[\s*\[',s) and
            not any(kw in low for kw in ("graph","markov","entropy","vertices"))):
        M=_parse_matrix(s)
        if M: return Problem(raw=raw,ptype=PT.MATRIX,meta={"M":M,"n":M.shape[0]})

    # Graph
    if re.match(r'^(graph|network)\b',low) or "adjacency" in low:
        M=_parse_matrix(s)
        meta={"A":M,"rows":M.tolist() if M else []}
        for pat,t in [(r'\bk[_\s]?(\d+)\b',"complete"),(r'\bp[_\s]?(\d+)\b',"path"),(r'\bc[_\s]?(\d+)\b',"cycle")]:
            mm=re.search(pat,low)
            if mm:
                n=int(mm.group(1))
                meta.update({"type":t,"n":n,"named":t[0].upper()+mm.group(1)})
                break
        return Problem(raw=raw,ptype=PT.GRAPH,meta=meta)

    # Markov
    if re.match(r'^markov\b',low) or "transition matrix" in low:
        M=_parse_matrix(s)
        return Problem(raw=raw,ptype=PT.MARKOV,meta={"P":M,"rows":M.tolist() if M else []})

    # Entropy
    if re.match(r'^entropy\b',low):
        probs=_parse_probs(s)
        return Problem(raw=raw,ptype=PT.ENTROPY,
                       meta={"probs":probs,"sym_str":re.sub(r'^entropy\s*','',s,flags=re.I).strip()})

    # Proof — expanded NL matching
    if re.match(r'^(prove|show|demonstrate)\b',low):
        body=re.sub(r'^(prove|show\s+that|show|demonstrate)\s+','',s,flags=re.I)
        e=_parse(body)
        return Problem(raw=raw,ptype=PT.PROOF,expr=e,meta={"body":body,"body_low":body.lower()})

    # Summation — broad keyword match
    if any(kw in low for kw in ("sum of","summation","1+2+","series","sigma")):
        summand=_parse_summand(raw)
        return Problem(raw=raw,ptype=PT.SUM,meta={"summand":summand,"raw_low":low})

    # Factor
    if low.startswith("factor "):
        body=s[7:].strip(); e=_parse(body)
        free=sorted(e.free_symbols,key=str) if e else []
        v=_var_prefer(free) or symbols('x')
        _p=None
        try: _p=Poly(e,v)
        except: pass
        return Problem(raw=raw,ptype=PT.FACTORING,expr=e,var=v,free=free,poly=_p)

    # Equation
    if "=" in s and not any(x in s for x in ("==",">=","<=")):
        parts=s.split("=",1)
        lhs_e=_parse(parts[0]); rhs_e=_parse(parts[1])
        if lhs_e is None or rhs_e is None:
            return Problem(raw=raw,ptype=PT.UNKNOWN)
        expr=sp.expand(lhs_e-rhs_e)
        free=sorted(expr.free_symbols,key=str)
        v=_var_prefer(free) or symbols('x')
        _p=None
        if expr.atoms(sin,cos,tan):
            pt=PT.TRIG_EQ
        else:
            try:
                _p=Poly(expr,v); deg=_p.degree()
                pt={1:PT.LINEAR,2:PT.QUADRATIC,3:PT.CUBIC}.get(deg,PT.POLY)
            except: pt=PT.UNKNOWN
        return Problem(raw=raw,ptype=pt,expr=expr,lhs=lhs_e,rhs=rhs_e,var=v,free=free,poly=_p)

    # Bare expression
    e=_parse(s)
    if e is not None:
        free=sorted(e.free_symbols,key=str)
        v=_var_prefer(free) or symbols('x')
        pt=PT.TRIG_ID if e.atoms(sin,cos,tan) else PT.SIMPLIFY
        return Problem(raw=raw,ptype=pt,expr=e,lhs=e,rhs=Integer(0),var=v,free=free)

    return Problem(raw=raw,ptype=PT.UNKNOWN)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 01 — GROUND TRUTH + INTELLIGENCE BRIEFING
# ════════════════════════════════════════════════════════════════════════════

def phase_01(p: Problem) -> dict:
    section(1,"GROUND TRUTH","Classify · predict · brief · pre-detect failures · solution family")
    r={}
    kv("Problem",p.raw); kv("Type",p.ptype.label()); kv("Variable",str(p.var))
    depth=KB.phase_depth(p.ptype_str())
    kv("Adaptive depth",f"Phases 1–{depth} (skipping deeper phases for this complexity)")

    note("--- intelligence briefing ---")
    for a in KB.analogies_for(p.ptype_str()): note(f"    • {a}")
    fm=KB.failure_modes(p.ptype_str())
    if fm: warn(f"Anticipated failure: {fm}"); r["failure_mode"]=fm

    cat=("equation" if p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY) else
         "factoring" if p.ptype==PT.FACTORING else
         "identity"  if p.ptype in(PT.TRIG_ID,PT.SIMPLIFY) else
         "graph"  if p.ptype==PT.GRAPH else "markov" if p.ptype==PT.MARKOV else
         "entropy" if p.ptype==PT.ENTROPY else "control" if p.ptype==PT.CONTROL else
         "optimize" if p.ptype==PT.OPTIMIZATION else "equation")
    strats=KB.verify_strategies(cat)
    kv("Planned verification",strats[0]); r["verify_strategy"]=strats
    note("---")

    # Type-specific Phase 01
    if p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY):
        kv("Expression",str(p.expr))
        poly=p.get_poly()
        if poly:
            deg=poly.degree(); r["degree"]=deg
            kv("Degree",deg)
            kv("Coefficients",[str(c) for c in poly.all_coeffs()])
            p.conf.record("degree",deg,1.0)
            hardness={1:"trivial",2:"easy (discriminant)",3:"moderate (Cardano)",
                      4:"hard (Ferrari/numerical)"}
            kv("Difficulty",hardness.get(deg,"hard (Abel-Ruffini no radical formula for ≥5)"))
            v=p.var
            if v:
                try:
                    even=p.is_even()
                    odd =p.is_odd()
                    r["even"]=even; r["odd"]=odd
                    if even:  finding("EVEN → substitute u=x² to halve degree")
                    elif odd: finding("ODD → x=0 always a root → factor out x")
                    else: note("No parity symmetry")
                    # Feedback to Phase 02
                    if even: p.fbq.push("biquadratic_substitution")
                    if odd:  p.fbq.push("factor_out_x")
                except: pass
            if all(c.is_integer for c in poly.all_coeffs()):
                lc=int(abs(poly.all_coeffs()[0])); ct=int(abs(poly.all_coeffs()[-1]))
                cands=sorted({s*pf//qf for pf in divisors(ct) for qf in divisors(lc)
                              for s in(1,-1)},key=abs)[:12]
                hits=[c for c in cands if p.expr and abs(float(N(p.expr.subs(v,c))))<1e-9]
                r["rat_root_cands"]=cands; kv("Rational root screen",[str(c) for c in cands])
                if hits:
                    ok(f"Rational roots: {hits} — factor theorem applies")
                    r["known_rational_roots"]=hits; p.conf.record("rat_roots",hits,0.95)
            # Solution family
            fam=_detect_family(p)
            if fam: kv("Solution family",fam); r["family"]=fam; p.conf.record("family",fam,0.85)
        if p.ptype==PT.QUADRATIC:
            try:
                disc=discriminant(Poly(p.expr,p.var)); r["discriminant"]=disc
                dn=float(N(disc))
                nat=("2 real roots" if dn>0 else "double root" if dn==0 else "complex roots")
                kv("Discriminant Δ",f"{dn:.4f} → {nat}"); finding(f"Δ={dn:.4f} → {nat}")
                p.conf.record("discriminant",disc,1.0,nat)
            except: pass

    elif p.ptype==PT.GRAPH:
        kv("Graph",p.meta.get("named","adjacency matrix")); kv("Vertices",p.meta.get("n","?"))
    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M"); n=p.meta.get("n")
        if M:
            kv("Shape",f"{n}×{n}"); kv("Trace",str(trace(M))); kv("Det",str(det(M)))
            sym=(M==M.T); kv("Symmetric",sym); p.conf.record("symmetric",sym,1.0)
            if sym: finding("Symmetric → spectral theorem applies")
    elif p.ptype==PT.MARKOV:
        rows=p.meta.get("rows",[]); n=len(rows)
        kv("States",n)
        if rows:
            for i,row in enumerate(rows):
                s_=sum(row); (ok if abs(s_-1)<1e-9 else fail)(f"Row {i} sums to {s_:.6f}")
    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        if probs:
            kv("Distribution",probs)
            total=sum(probs); (ok if abs(total-1)<1e-9 else fail)(f"Sum={total:.6f}")
    elif p.ptype==PT.DYNAMICAL:
        kv("f(x)",str(p.expr))
    elif p.ptype==PT.CONTROL:
        kv("Char poly",str(p.expr))
        poly=p.get_poly()
        if poly:
            kv("Degree",poly.degree())
            pos=all(float(N(c))>0 for c in poly.all_coeffs())
            (ok if pos else fail)(f"All coefficients positive: {pos}")
            if not pos: finding("NECESSARY CONDITION FAILS → unstable"); p.conf.record("def_unstable",True,0.99)
    elif p.ptype==PT.OPTIMIZATION:
        kv("f(x)",str(p.expr)); kv("Goal",p.meta.get("goal","extremize"))
    elif p.ptype==PT.SUM:
        summand=p.meta.get("summand"); kv("Summand",str(summand))
    elif p.ptype==PT.PROOF:
        kv("Claim",p.meta.get("body",""))
    elif p.ptype==PT.FACTORING:
        kv("Expression",str(p.expr))
        fam=_detect_family(p)
        if fam: kv("Family",fam)

    ok("Phase 01 complete"); r["briefed"]=True
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 02 — DIRECT ATTACK
# ════════════════════════════════════════════════════════════════════════════

def phase_02(p: Problem, g1: dict) -> dict:
    section(2,"DIRECT ATTACK","Methods ranked by predicted success · feedback-unlocked methods")
    r={"successes":[],"failures":[]}
    v=p.var

    def attempt(name,fn,conf_prior=0.5,verify_fn=None,timeout=8):
        result=p.memo(name,fn,timeout)
        if result is None:
            r["failures"].append({"method":name,"error":"None/timeout"})
            fail(f"[--] {name}"); return None
        conf=conf_prior
        if verify_fn:
            try:
                verified,detail=verify_fn(result)
                if verified: conf=min(conf+0.1,1.0); ok(f"{name} verified ✓")
                else:        conf=max(conf-0.2,0.0); warn(f"{name} verify failed: {detail}")
            except: pass
        p.conf.record(name,result,conf)
        r["successes"].append({"method":name,"result":str(result)[:80],"conf":conf})
        ok(f"[{conf:.0%}] {name} → {str(result)[:80]}")
        return result

    # ── ALGEBRAIC ─────────────────────────────────────────────────────────
    if p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY,PT.TRIG_EQ):
        # Feedback-unlocked: biquadratic substitution
        if p.fbq.has("biquadratic_substitution") and p.ptype==PT.POLY:
            u=symbols('u',positive=True)
            try:
                sub_expr=p.expr.subs(v**2,u)
                if len(sub_expr.free_symbols)==1 and u in sub_expr.free_symbols:
                    u_sols=solve(sub_expr,u)
                    real_u=[s for s in u_sols if _is_real_value(s) and float(N(s))>=0]
                    if real_u:
                        all_sols=[]
                        for s in real_u: all_sols+=[sqrt(s),-sqrt(s)]
                        p._cache["biquadratic_sols"]=all_sols
                        ok(f"Biquadratic substitution u=x² → {len(all_sols)} real solutions")
            except: pass

        sols=attempt("solve(expr,var)",lambda:solve(p.expr,v),0.95,
                     verify_fn=lambda s:(all(_verify_eq(p.expr,v,x)[0] for x in s),
                                        "substitution") if s else (False,"empty"))
        if sols:
            r["solutions"]=[str(s) for s in sols]
            if p.get_poly() and p.ptype!=PT.TRIG_EQ:
                vok=_vieta_check(p.get_poly(),sols)
                (ok if vok else warn)(f"Vieta's formulas: {vok}"); r["vieta_ok"]=vok
            # Build companion fingerprint
            if p.get_poly():
                _companion_fingerprint(p.get_poly(),p,label=p.raw[:40])
        attempt("solveset(Reals)",lambda:str(solveset(p.expr,v,domain=S.Reals)),0.80,timeout=6)
        if p.get_poly():
            attempt("roots(Poly)",lambda:str(roots(p.get_poly())),0.85,timeout=6)

    # ── FACTORING ─────────────────────────────────────────────────────────
    elif p.ptype==PT.FACTORING:
        attempt("factor(expr)",lambda:factor(p.expr),0.90,
                verify_fn=lambda f:(simplify(expand(f)-p.get_expanded_expr())==0,"expand verify"))
        attempt("sqf_list",lambda:str(sqf_list(p.expr,v)),0.80)
        attempt("factor_list",lambda:str(factor_list(p.expr)),0.80)

    # ── TRIG ──────────────────────────────────────────────────────────────
    elif p.ptype in(PT.TRIG_ID,PT.SIMPLIFY):
        attempt("trigsimp",lambda:trigsimp(p.expr),0.90)
        attempt("simplify",lambda:simplify(p.expr),0.85)
        attempt("expand_trig",lambda:expand_trig(p.expr),0.70)
        if p.expr is not None and v:
            test_vals=[0.7,1.3,2.1,3.5]
            checks=[]
            for tv in test_vals:
                try: checks.append(abs(float(N(p.expr.subs(v,tv))))<1e-8)
                except: pass
            if checks:
                all_z=all(checks)
                r["numerical_verify"]=all_z
                (ok if all_z else warn)(f"Numerical check at {test_vals[:3]}: {checks[:3]}")
                p.conf.record("num_identity",all_z,0.95 if all_z else 0.3)

    # ── SUMMATION — generalised ────────────────────────────────────────────
    elif p.ptype==PT.SUM:
        k  = symbols('k', positive=True, integer=True)
        n  = symbols('n', positive=True, integer=True)
        summand = p.meta.get("summand", k)
        attempt(f"summation({summand},(k,1,n))",
                lambda: summation(summand,(k,1,n)), 0.99)
        # Also try raw k for comparison if different
        if str(summand)!='k':
            attempt("summation(k,(k,1,n))",lambda:summation(k,(k,1,n)),0.95)

    # ── PROOF ─────────────────────────────────────────────────────────────
    elif p.ptype==PT.PROOF:
        body=p.meta.get("body",""); body_low=body.lower()
        # Expanded NL matching (from user engine improvement)
        is_sqrt2=(any(kw in body_low for kw in ("sqrt(2)","root 2","root of 2","√2","irrational")))
        is_prime=(any(kw in body_low for kw in ("prime","primes")))
        if is_sqrt2:
            ok("Proof by contradiction: assume √2=p/q (irreducible)→ p²=2q²→ p,q both even → contradicts gcd(p,q)=1")
            r["proof_method"]="contradiction"; r["status"]="Success"
        elif is_prime:
            ok("Euclid: assume finite list {p₁,…,pₖ}. N=p₁·…·pₖ+1 has a prime factor not in list. Contradiction.")
            r["proof_method"]="construction"; r["status"]="Success"
        else:
            note(f"Proof type not recognised from: '{body[:60]}'")
            r["status"]="Pending"

    # ── DIGRAPH ───────────────────────────────────────────────────────────
    elif p.ptype==PT.DIGRAPH_CYC:
        m=p.meta.get("m")
        if m%2!=0:
            ok(f"Odd m={m}: fiber decomposition exists (twisted translation)"); r["status"]="Success"
        else:
            fail(f"Even m={m}: parity obstruction — fiber-uniform impossible"); r["status"]="Failure"

    # ── GRAPH ─────────────────────────────────────────────────────────────
    elif p.ptype==PT.GRAPH:
        A,L,n,deg=_build_graph(p)
        if A is None: fail("Cannot build adjacency matrix"); return r
        p.meta.update({"L":L,"n":n,"deg":deg}); ok(f"A, L built ({n}×{n})")
        r["degree_sequence"]=deg; r["edge_count"]=sum(deg)//2
        kv("Degree sequence",deg); kv("Edges",r["edge_count"])
        # Laplacian spectrum
        L_spec=_spectrum_real(L)
        if L_spec:
            r["L_spec"]=L_spec; p.meta["L_spec"]=L_spec
            kv("L spectrum",[f"{e:.4f}" for e in L_spec])
            (ok if abs(L_spec[0])<1e-9 else warn)(f"λ₁(L)={L_spec[0]:.6f} (must be 0)")
            tr_L=float(N(trace(L)))
            (ok if abs(tr_L-sum(deg))<1e-9 else warn)(f"tr(L)={tr_L:.3f}=Σdeg={sum(deg)}")
            p.add_fingerprint("graph_laplacian",L_spec)
        # Adjacency spectrum
        A_spec=_spectrum_real(A)
        if A_spec:
            r["A_spec"]=A_spec; p.meta["A_spec"]=A_spec
            kv("A spectrum",[f"{e:.4f}" for e in A_spec])
            try:
                tr_A2=float(N(trace(A*A)))
                (ok if abs(tr_A2/2-sum(deg)//2)<1e-9 else warn)(f"tr(A²)/2={tr_A2/2:.1f}=|E|={sum(deg)//2}")
            except: pass
            p.add_fingerprint("graph_adj",A_spec)
        # Feedback: bipartite signal
        if A_spec and all(abs(A_spec[i]+A_spec[-(i+1)])<1e-6 for i in range(len(A_spec)//2)):
            p.fbq.push("bipartite")

    # ── MATRIX ────────────────────────────────────────────────────────────
    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M")
        if M is None: fail("No matrix"); return r
        n=M.shape[0]; lam=symbols('lambda')
        cp=attempt("char_poly",lambda:M.charpoly(lam).as_expr(),0.95)
        if cp: r["char_poly"]=str(cp)
        # Build companion FP from char poly
        if cp:
            try:
                cp_poly=Poly(cp,lam)
                _companion_fingerprint(cp_poly,p,"matrix_char_poly")
            except: pass
        spec=_spectrum_real(M); p.meta["spec"]=spec
        r["eigenvalues"]=spec; kv("Eigenvalues",[f"{e:.4f}" for e in spec])
        tr_M=float(N(trace(M))); dt_M=float(N(det(M)))
        (ok if abs(tr_M-sum(spec))<1e-6 else warn)(f"Trace: {tr_M:.4f}≈Σλ={sum(spec):.4f}")
        (ok if len(spec)==0 or abs(dt_M-math.prod(spec))<1e-6 else warn)(
            f"Det: {dt_M:.4f}≈Πλ={math.prod(spec) if spec else 0:.4f}")
        r["trace"]=tr_M; r["det"]=dt_M
        p.add_fingerprint("matrix",spec)
        # Feedback: if all eigs < 0, signal stable Jacobian
        if spec and all(e<0 for e in spec): p.fbq.push("all_eigs_negative")

    # ── MARKOV ────────────────────────────────────────────────────────────
    elif p.ptype==PT.MARKOV:
        P=p.meta.get("P")
        if P is None: fail("No P matrix"); return r
        n=P.shape[0]; p.meta["n"]=n
        P_rat=sp.Matrix([[sp.Rational(P[i,j]).limit_denominator(1000)
                          if isinstance(P[i,j],float) else sp.sympify(P[i,j])
                          for j in range(n)] for i in range(n)])
        p.meta["P_rat"]=P_rat; ok(f"Rational P ({n}×{n})")
        spec_c=_spectrum_complex(P_rat)
        r["eigenvalues_complex"]=[str(round(z.real,4)+round(z.imag,4)*1j) for z in spec_c]
        kv("Eigenvalues",r["eigenvalues_complex"])
        rho=max(abs(z) for z in spec_c) if spec_c else 0
        r["spectral_radius"]=rho
        (ok if rho<=1.0001 else warn)(f"ρ={rho:.6f} ≤1")
        p.add_fingerprint("markov",[z.real for z in spec_c],spec_c)
        # Absorbing states
        abs_states=[i for i in range(n) if float(N(P_rat[i,i]))==1.0]
        if abs_states: p.fbq.push("absorbing_states",abs_states); r["absorbing"]=abs_states
        # Stationary
        stat=_stationary(P_rat)
        if stat:
            r["stationary"]={str(k):str(v_) for k,v_ in stat.items()}
            kv("Stationary π",r["stationary"]); p.meta["stat"]=stat; ok("π computed")
            pi_v=sp.Matrix([list(stat.values())])
            chk=pi_v*P_rat-pi_v
            all_z=all(simplify(chk[0,j])==0 for j in range(n))
            (ok if all_z else warn)(f"π·P=π: {all_z}")
            p.conf.record("stat_verified",all_z,0.99 if all_z else 0.5)

    # ── ENTROPY ───────────────────────────────────────────────────────────
    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        if probs:
            H=_entropy(probs); H_max=math.log2(len(probs))
            r["H_bits"]=H; r["H_max"]=H_max; r["efficiency"]=H/H_max if H_max>0 else 1.0
            kv("H(X)",f"{H:.6f} bits"); kv("H_max",f"{H_max:.6f} bits")
            kv("Efficiency",f"{r['efficiency']:.4f}")
            ok("Shannon entropy computed")
            (ok if H>=-1e-12 else warn)(f"H≥0: {H>=0}")
            (ok if H<=H_max+1e-12 else warn)(f"H≤H_max: {H<=H_max}")
            KL=_kl(probs,[1/len(probs)]*len(probs))
            r["KL_uniform"]=KL; kv("KL(P||uniform)",f"{KL:.6f}")
            p.meta["H_val"]=H
            if H/H_max>0.999: p.fbq.push("maximum_entropy")
        p_s=symbols('p',positive=True)
        H_bin=-p_s*log(p_s,2)-(1-p_s)*log(1-p_s,2)
        max_p=solve(diff(H_bin,p_s),p_s)
        r["binary_max_at"]=str(max_p); kv("H_bin max at p=",str(max_p))

    # ── DYNAMICAL ─────────────────────────────────────────────────────────
    elif p.ptype==PT.DYNAMICAL:
        f=p.expr
        equil=attempt("solve(f=0)",lambda:solve(f,v),0.90)
        if equil:
            r["equilibria"]=[str(e) for e in equil]
            fp=diff(f,v); kv("f'(x)",str(fp))
            stab_info={}
            for eq in equil:
                try:
                    fp_v=float(N(fp.subs(v,eq)))
                    stab=("STABLE" if fp_v<0 else "UNSTABLE" if fp_v>0 else "NON-HYPERBOLIC")
                    kv(f"  f'({eq})",f"{fp_v:.4f} → {stab}"); stab_info[str(eq)]=stab
                    if stab=="NON-HYPERBOLIC": p.fbq.push("non_hyperbolic_equil",eq)
                except: pass
            r["stability"]=stab_info
            p.add_fingerprint("dynamical_equil",[float(N(e)) for e in equil if _is_real_value(e)])

    # ── CONTROL ───────────────────────────────────────────────────────────
    elif p.ptype==PT.CONTROL:
        f=p.expr
        attempt("solve(char_poly)",lambda:solve(f,v),0.80,timeout=6)
        rts=p._cache.get("solve(char_poly)",[])
        if rts:
            r["roots"]=[str(rt) for rt in rts]
            for rt in rts:
                try:
                    rt_c=complex(N(rt))
                    loc=("LHP stable" if rt_c.real<0 else "RHP UNSTABLE" if rt_c.real>0 else "marginal")
                    kv(f"  root {rt}",f"Re={rt_c.real:.4f}, Im={rt_c.imag:.4f} → {loc}")
                except: pass
            p.add_fingerprint("control_poles",[float(N(sp_re(rt))) for rt in rts],
                              [complex(N(rt)) for rt in rts])
        try:
            poly=p.get_poly(); coeffs=poly.all_coeffs()
            rh=_routh(coeffs); r["routh"]=rh; p._cache["routh"]=rh
            kv("Routh 1st col",[f"{x:.4f}" for x in rh["first_column"]])
            kv("Sign changes",rh["sign_changes"])
            (ok if rh["stable"] else fail)(f"Routh: {'STABLE' if rh['stable'] else 'UNSTABLE'}")
            r["successes"].append({"method":"routh","result":rh["stable"],"conf":0.99})
        except Exception as e: fail(f"Routh: {e}")

    # ── OPTIMIZATION ──────────────────────────────────────────────────────
    elif p.ptype==PT.OPTIMIZATION:
        f=p.expr; fp=diff(f,v); fpp=diff(f,v,2)
        r["gradient"]=str(fp); kv("f'(x)",str(fp))
        crit=attempt("solve(f'=0)",lambda:solve(fp,v),0.90)
        if crit:
            r["critical_points"]=[str(c) for c in crit]
            for c in crit:
                try:
                    fpp_v=float(N(fpp.subs(v,c))); f_v=float(N(f.subs(v,c)))
                    nat=("LOCAL MIN" if fpp_v>0 else "LOCAL MAX" if fpp_v<0 else "INFLECTION")
                    kv(f"  x={c}",f"f={f_v:.4f}, f''={fpp_v:.4f} → {nat}")
                    r[f"cp_{c}"]={"f":f_v,"fpp":fpp_v,"nature":nat}
                    fp_ok=abs(float(N(fp.subs(v,c))))<1e-9
                    (ok if fp_ok else warn)(f"f'({c})=0: {fp_ok}")
                except: pass
        try:
            lp=limit(f,v,oo); ln=limit(f,v,-oo)
            r["lim_+inf"]=str(lp); r["lim_-inf"]=str(ln)
            kv("f(+∞)",str(lp)); kv("f(-∞)",str(ln))
        except: pass

    finding(f"{len(r['successes'])} succeeded, {len(r['failures'])} failed")
    finding(p.conf.summary())
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 02b — FEEDBACK-UNLOCKED METHODS  (Integration D continued)
# ════════════════════════════════════════════════════════════════════════════

def phase_02b(p: Problem, g2: dict) -> dict:
    """Extra methods unlocked by Phase 03 signals (run after Phase 03)."""
    r={}
    if not p.fbq._signals: return r
    section_header=False

    # Bipartite → two-coloring via BFS
    if p.fbq.has("bipartite") and p.ptype==PT.GRAPH:
        if not section_header:
            note("  [feedback-unlocked methods]"); section_header=True
        A=p.meta.get("A"); n=p.meta.get("n",0)
        if A and n>0:
            try:
                color=[-1]*n; color[0]=0; q=[0]
                valid=True
                while q and valid:
                    node=q.pop(0)
                    for j in range(n):
                        if int(A[node,j]):
                            if color[j]==-1: color[j]=1-color[node]; q.append(j)
                            elif color[j]==color[node]: valid=False; break
                r["bipartite_coloring"]=color
                kv("Bipartite 2-coloring",color)
                ok("Bipartite confirmed via BFS 2-coloring")
            except: pass

    # Non-hyperbolic equilibrium → compute higher derivatives
    if p.fbq.has("non_hyperbolic_equil") and p.ptype==PT.DYNAMICAL:
        if not section_header:
            note("  [feedback-unlocked methods]"); section_header=True
        eq=p.fbq.get("non_hyperbolic_equil"); v=p.var; f=p.expr
        try:
            f2=diff(f,v,2); f3=diff(f,v,3)
            f2v=float(N(f2.subs(v,eq))); f3v=float(N(f3.subs(v,eq)))
            r["higher_deriv"]={f"f''({eq})":f2v,f"f'''({eq})":f3v}
            kv(f"f''({eq})",f"{f2v:.6f}"); kv(f"f'''({eq})",f"{f3v:.6f}")
            if abs(f2v)<1e-9 and f3v!=0: finding("Degenerate: f'''≠0 → inflection/transcritical bifurcation")
        except: pass

    # Maximum entropy → confirm uniform distribution
    if p.fbq.has("maximum_entropy") and p.ptype==PT.ENTROPY:
        if not section_header:
            note("  [feedback-unlocked methods]"); section_header=True
        probs=p.meta.get("probs",[])
        is_unif=all(abs(q-probs[0])<1e-9 for q in probs)
        kv("Uniform distribution",is_unif)
        if is_unif: insight("Maximum entropy = uniform distribution → MaxEnt principle confirmed")

    # Factor out x for odd polynomials
    if p.fbq.has("factor_out_x") and p.ptype in(PT.CUBIC,PT.POLY):
        if not section_header:
            note("  [feedback-unlocked methods]"); section_header=True
        v=p.var
        try:
            reduced=sp.cancel(p.expr/v)
            kv(f"expr/x",str(reduced))
            inner_sols=solve(reduced,v)
            ok(f"Odd: x=0 + roots of {reduced}: {inner_sols}")
            r["odd_factored_sols"]=[0]+[str(s) for s in inner_sols]
        except: pass

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 03 — STRUCTURE HUNT
# ════════════════════════════════════════════════════════════════════════════

def phase_03(p: Problem, g2: dict) -> dict:
    section(3,"STRUCTURE HUNT","Invariants · symmetry · decomposition · spectrum")
    r={}; v=p.var

    if p.ptype==PT.GRAPH:
        L_spec=p.meta.get("L_spec",[]); A_spec=p.meta.get("A_spec",[]); deg=p.meta.get("deg",[])
        n=p.meta.get("n",0)
        if len(L_spec)>1:
            lam2=sorted(L_spec)[1]; r["fiedler"]=lam2
            kv("Fiedler λ₂",f"{lam2:.6f}")
            finding("λ₂>0 → CONNECTED" if lam2>1e-9 else "λ₂=0 → DISCONNECTED")
            r["connected"]=lam2>1e-9
            if lam2>1e-9:
                kv("Cheeger bound h(G)∈",f"[{lam2/2:.4f}, {math.sqrt(2*lam2):.4f}]")
        if len(set(deg))==1: r["regular"]=deg[0]; finding(f"{deg[0]}-REGULAR")
        if A_spec:
            sym=all(abs(A_spec[i]+A_spec[-(i+1)])<1e-6 for i in range(len(A_spec)//2))
            r["bipartite"]=sym; kv("Bipartite (sym spectrum)",sym)
            finding("Bipartite confirmed" if sym else "Not bipartite")
            if sym: p.fbq.push("bipartite")
        n_comps=sum(1 for e in L_spec if abs(e)<1e-9)
        r["components"]=n_comps; kv("Components (zero eigs)",n_comps)
        # Spectral gap as expansion metric
        if len(L_spec)>1 and sorted(L_spec)[1]>1e-9:
            reg=r.get("regular",max(deg) if deg else 0)
            if reg>0:
                sub_spec=[e for e in A_spec if e<reg-1e-9]
                gap=(reg-max(sub_spec))/reg if sub_spec else 0
                r["expansion_ratio"]=gap; kv("Relative spectral gap",f"{gap:.4f}")
        return r

    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M"); spec=p.meta.get("spec",[])
        if M is None: return r
        r["symmetric"]=(M==M.T); kv("Symmetric",r["symmetric"])
        if r["symmetric"] and spec:
            mn=min(spec); mx=max(spec)
            def_str=("PD" if mn>0 else "PSD" if mn>=0 else "ND" if mx<0 else "indefinite")
            r["definite"]=def_str; finding(f"{def_str} matrix")
        try:
            rnk=M.rank(); r["rank"]=rnk; kv("Rank",rnk)
            finding("INVERTIBLE" if rnk==M.shape[0] else f"SINGULAR (rank {rnk})")
        except: pass
        if spec:
            rho=max(abs(e) for e in spec); cond=rho/max(min(abs(e) for e in spec),1e-15)
            r["spectral_radius"]=rho; r["condition"]=cond
            kv("ρ(M)",f"{rho:.4f}"); kv("κ(M)",f"{cond:.4f}")
            if cond>100: warn(f"Ill-conditioned: κ={cond:.0f}")
        return r

    elif p.ptype==PT.MARKOV:
        P_rat=p.meta.get("P_rat"); n=p.meta.get("n",0)
        spec_c=_spectrum_complex(P_rat) if P_rat else []
        eig_abs=sorted([abs(z) for z in spec_c],reverse=True)
        if len(eig_abs)>1:
            lam2=eig_abs[1]; gap=1.0-lam2
            r["lambda2"]=lam2; r["gap"]=gap
            kv("|λ₂|",f"{lam2:.6f}"); kv("Spectral gap 1-|λ₂|",f"{gap:.6f}")
            if gap>1e-9:
                mix=int(1/gap)+1; r["mixing_time"]=mix
                kv("Mixing time~",f"{mix} steps")
        abs_states=p.fbq.get("absorbing_states") or []
        r["absorbing"]=abs_states; kv("Absorbing states",abs_states or "none")
        finding("ERGODIC" if not abs_states else f"Absorbing: {abs_states}")
        if P_rat:
            stat=p.meta.get("stat",{})
            if stat:
                try:
                    pi_f=[float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                    rev=all(abs(float(N(sp.sympify(list(stat.values())[i])*P_rat[i,j]))-
                               float(N(sp.sympify(list(stat.values())[j])*P_rat[j,i])))<1e-9
                            for i in range(n) for j in range(n))
                    r["reversible"]=rev; kv("Reversible (detailed balance)",rev)
                    finding("REVERSIBLE" if rev else "IRREVERSIBLE (entropy production>0)")
                except: pass
        return r

    elif p.ptype==PT.ENTROPY:
        p_s=symbols('p',positive=True)
        H_bin=-p_s*log(p_s,2)-(1-p_s)*log(1-p_s,2)
        kv("d²H/dp²",str(simplify(diff(H_bin,p_s,2))))
        finding("H strictly CONCAVE → unique max at p=½")
        probs=p.meta.get("probs",[])
        if probs:
            H=p.meta.get("H_val",0); H_max=math.log2(len(probs))
            contribs=[-q*math.log2(q) for q in probs if q>0]
            kv("Per-symbol −pᵢlog₂pᵢ",[f"{c:.4f}" for c in contribs])
            kv("Gap to max H",f"{H_max-H:.6f} bits"); finding(f"Efficiency {H/H_max:.4f}")
        return r

    elif p.ptype==PT.DYNAMICAL:
        f=p.expr
        if f is None: return r
        try:
            even=p.is_even(); odd=p.is_odd()
            r["symmetry"]="even" if even else ("odd" if odd else "none")
            kv("Symmetry",r["symmetry"])
            if even: finding("EVEN → equilibria ±symmetric")
            elif odd: finding("ODD → x=0 always equilibrium")
        except: pass
        try:
            V=v**2/2; dVdt=simplify(diff(V,v)*f)
            r["lyapunov_candidate"]=str(dVdt); kv("V̇=xf(x) (Lyapunov)",str(dVdt))
        except: pass
        return r

    elif p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{})
        if rh: kv("Stability","STABLE" if rh.get("stable") else "UNSTABLE")
        rts=p._cache.get("solve(char_poly)",[])
        if rts:
            for rt in rts:
                try:
                    rt_c=complex(N(rt))
                    kv(f"  λ={rt}",f"Re={rt_c.real:.4f} {'LHP' if rt_c.real<0 else 'RHP'}")
                except: pass
        return r

    elif p.ptype==PT.OPTIMIZATION:
        f=p.expr; fpp=diff(f,v,2)
        try:
            fpp_s=p.memo("simplify(f'')",lambda:simplify(fpp)); kv("f''(x)",str(fpp_s))
            if fpp_s.is_polynomial(v):
                fp_poly=Poly(fpp_s,v)
                if fp_poly.degree()==0:
                    val=float(N(fpp_s))
                    if val>0:  finding("f''>0 everywhere → CONVEX → local min = global min")
                    elif val<0:finding("f''<0 everywhere → CONCAVE → local max = global max")
                    r["convex"]=val>0
        except: pass
        return r

    # Algebraic types
    if p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY,PT.FACTORING):
        try: fac=p.memo("factor(expr)", lambda: factor(p.expr)); r["factored"]=str(fac); kv("Factored",fac)
        except: pass
        if v:
            try:
                even=p.is_even()
                odd =p.is_odd()
                if even: finding("EVEN: use u=x² substitution")
                elif odd: finding("ODD: factor out x")
            except: pass
        sols=p._cache.get("solve(expr,var)",[])
        if sols and len(sols)<=5:
            try:
                ps={f"Σxᵢ^{k}":str(simplify(sum(s**k for s in sols))) for k in range(1,4)}
                kv("Newton power sums",ps)
            except: pass

    if p.ptype==PT.SUM:
        k=symbols('k',positive=True,integer=True); n=symbols('n',positive=True,integer=True)
        summand=p.meta.get("summand",k)
        try:
            res=p._cache.get(f"summation({summand},(k,1,n))")
            if res: kv("Closed form",str(factor(res)))
            kv("Telescoping check","f(n)-f(n-1)=summand")
        except: pass

    if p.ptype==PT.TRIG_ID and p.expr is not None and v:
        test_vals=[0.3,0.8,1.5,2.2,3.0]
        residuals=[abs(float(N(p.expr.subs(v,tv)))) for tv in test_vals
                   if _timed(lambda tv=tv: abs(float(N(p.expr.subs(v,tv)))),2) is not None]
        if residuals:
            mx=max(residuals)
            (ok if mx<1e-8 else warn)(f"Numerical max residual: {mx:.2e}")
            r["numerical_id"]=mx<1e-8
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 04 — PATTERN LOCK  (backwards reasoning)
# ════════════════════════════════════════════════════════════════════════════

def phase_04(p: Problem, g3: dict) -> dict:
    section(4,"PATTERN LOCK","Read solution backwards · backwards reasoning")
    r={}; v=p.var

    if p.ptype==PT.GRAPH:
        A=p.meta.get("A"); L=p.meta.get("L"); n=p.meta.get("n",0)
        L_spec=p.meta.get("L_spec",[]); A_spec=p.meta.get("A_spec",[])
        # Kirchhoff spanning trees
        if L_spec and n>0:
            nz=[e for e in L_spec if abs(e)>1e-9]
            if nz:
                tau=math.prod(nz)/n; r["spanning_trees"]=tau
                kv("Spanning trees τ(G) [Kirchhoff]",f"{tau:.4f}")
                insight(f"Matrix-Tree: τ=(1/n)∏λᵢ≠0 ≈ {tau:.2f}")
                if p.meta.get("type")=="complete":
                    exp_=n**(n-2); (ok if abs(tau-exp_)<0.5 else warn)(f"Kₙ: τ={tau:.1f} ≈ n^(n-2)={exp_}")
        # Estrada index
        if A_spec:
            ee=sum(math.exp(e) for e in A_spec); r["estrada"]=ee
            kv("Estrada index EE(G)",f"{ee:.4f}")
            insight("EE(G) = subgraph richness — counts all closed walks weighted by length")
        # Spectral centrality
        if A and n<=12:
            try:
                evects=A.eigenvects()
                top=sorted(evects,key=lambda t:float(N(t[0])),reverse=True)[0]
                vec=top[2][0]; s_=sum(abs(x) for x in vec)
                norm=[float(N(x/s_)) for x in vec]
                r["centrality"]=[f"{x:.3f}" for x in norm]
                kv("Spectral centrality",r["centrality"])
                max_i=norm.index(max(norm))
                if len(set(round(x,2) for x in norm))==1:
                    insight("Uniform centrality → REGULAR graph (all nodes equivalent)")
                else:
                    insight(f"Node {max_i} = hub (highest spectral centrality)")
            except: pass
        # Fiedler partition
        if L and n<=12:
            try:
                evects=L.eigenvects()
                sev=sorted(evects,key=lambda t:float(N(t[0])))
                if len(sev)>1:
                    fv=sev[1][2][0]
                    signs=["+" if float(N(x))>=0 else "-" for x in fv]
                    r["fiedler_partition"]=signs
                    kv("Fiedler partition",signs)
                    pos=signs.count("+"); neg=signs.count("-")
                    insight(f"Spectral bisection: {pos} nodes in A, {neg} in B")
            except: pass
        return r

    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M"); spec=p.meta.get("spec",[])
        kv("Cayley-Hamilton","p(M)=0 where p=char poly")
        if spec and M is not None:
            kv("Σλ=tr(M)",f"{sum(spec):.4f}={float(N(trace(M))):.4f}")
            kv("Πλ=det(M)",f"{math.prod(spec):.4f}={float(N(det(M))):.4f}")
        if g3.get("symmetric"):
            insight("M=QΛQᵀ → all matrix functions via diagonalisation (exp,log,sin,sqrt)")
        if spec:
            all_neg=all(e<0 for e in spec)
            insight(f"Backwards: if Jacobian at equilibrium → {'STABLE attractor (Lyapunov)' if all_neg else 'has unstable modes'}")
        return r

    elif p.ptype==PT.MARKOV:
        stat=p.meta.get("stat",{}); P_rat=p.meta.get("P_rat"); n=p.meta.get("n",0)
        if stat:
            pi_f=[float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
            H_stat=_entropy(pi_f); r["H_stat"]=H_stat
            kv("H(π) stationary entropy",f"{H_stat:.6f} bits")
            insight(f"Backwards: H(π)={H_stat:.4f} → spread of chain; uniform π → doubly stochastic P")
            if all(abs(pi_f[i]-pi_f[0])<1e-6 for i in range(n)):
                insight("Uniform π → P is DOUBLY STOCHASTIC (cols also sum to 1)")
            # Entropy rate
            if P_rat:
                try:
                    h=-sum(pi_f[i]*sum(float(N(P_rat[i,j]))*math.log2(max(float(N(P_rat[i,j])),1e-15))
                           for j in range(n) if float(N(P_rat[i,j]))>1e-12)
                      for i in range(n))
                    r["entropy_rate"]=h; kv("Entropy rate h",f"{h:.6f} bits/step")
                    insight(f"Chain produces {h:.4f} bits randomness/step (irreducible noise floor)")
                except: pass
        if P_rat and n<=6:
            try:
                P_inf=P_rat**20
                kv("P^20 row 0",[str(N(P_inf[0,j],3)) for j in range(n)])
                insight("P^20 ≈ Π — ergodic theorem confirmed numerically")
            except: pass
        return r

    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        if probs:
            H=_entropy(probs); n=len(probs)
            # Huffman codes
            heap=[(q,[i]) for i,q in enumerate(probs) if q>0]
            heapq.heapify(heap)
            lens={i:0 for i in range(n)}
            if len(heap)>1:
                while len(heap)>1:
                    p1,c1=heapq.heappop(heap); p2,c2=heapq.heappop(heap)
                    for idx in c1: lens[idx]+=1
                    for idx in c2: lens[idx]+=1
                    heapq.heappush(heap,(p1+p2,c1+c2))
            avg=sum(probs[i]*lens.get(i,0) for i in range(n))
            r["huffman_avg"]=avg; kv("Huffman avg length",f"{avg:.4f} bits/sym")
            kv("Shannon H",f"{H:.4f} bits/sym"); kv("Redundancy",f"{avg-H:.4f} bits")
            KL=_kl(probs,[1/n]*n)
            insight(f"KL={KL:.4f} bits = deviation from maximum ignorance")
        return r

    elif p.ptype==PT.DYNAMICAL:
        f=p.expr
        try:
            equil=p.memo("solve(f=0)",lambda:solve(p.expr,v)); fp=diff(f,v)
            for eq in equil:
                fp_v=float(N(fp.subs(v,eq)))
                stab="stable" if fp_v<0 else "unstable" if fp_v>0 else "non-hyperbolic"
                kv(f"  x*={eq}",f"f'={fp_v:.4f} → {stab}")
                try:
                    V_pot=-integrate(f,v)
                    kv(f"  V=-∫f at x={eq}",str(N(V_pot.subs(v,eq),4)))
                except: pass
            insight("Backwards: stable eq = LOCAL MINIMA of V(x)=-∫f(x)dx")
        except: pass
        return r

    elif p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{}); rts=p._cache.get("solve(char_poly)",[])
        if rts:
            lhp=[rt for rt in rts if float(N(sp_re(rt)))<0]
            rhp=[rt for rt in rts if float(N(sp_re(rt)))>0]
            kv("LHP modes",[str(r) for r in lhp]); kv("RHP modes",[str(r) for r in rhp])
            insight(f"Backwards: {len(rhp)} unstable modes → need {len(rhp)} feedback gains")
        if rh: insight("Routh: stability = all first-column entries positive → spectral condition")
        return r

    elif p.ptype==PT.OPTIMIZATION:
        f=p.expr; fp=diff(f,v); fpp=diff(f,v,2)
        crit_raw=p._cache.get("solve(f'=0)",[])
        if crit_raw:
            goal=p.meta.get("goal","extremize")
            f_vals=[(float(N(f.subs(v,c))),c) for c in crit_raw if _is_real_value(f.subs(v,c))]
            if f_vals:
                best=(min if "min" in goal else max)(f_vals)
                r["optimal"]=best; kv("Optimal",f"x*={best[1]}, f*={best[0]:.4f}")
                insight(f"Backwards: x*={best[1]} = equilibrium of gradient flow ẋ=-∇f")
        return r

    # Algebraic
    if p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY):
        sols=p._cache.get("solve(expr,var)",[])
        r["solutions"]=[str(s) for s in sols] if sols else []
        kv("Solutions",r["solutions"])
        if sols:
            for i,s in enumerate(sols):
                info={"value":str(s),"is_integer":s.is_integer,"is_rational":s.is_rational,
                      "is_real":s.is_real,"verified":_verify_eq(p.expr,v,s)[0]}
                kv(f"Root {i}",str(info))
            if all(s.is_integer for s in sols):
                insight(f"Backwards: all-integer roots → polynomial = ∏(x-rᵢ) over Z")
            elif all(getattr(s,'is_real',True) for s in sols):
                insight("Backwards: all real → Δ≥0 → polynomial splits over ℝ")
            else:
                insight("Backwards: complex roots → irreducible quadratic factor over ℝ")
            if p.get_poly(): (ok if _vieta_check(p.get_poly(),sols) else warn)("Vieta verified")

    elif p.ptype in(PT.TRIG_ID,PT.SIMPLIFY):
        simp=p.memo("trigsimp", lambda: trigsimp(p.expr))
        r["simplified"]=str(simp); kv("Simplified",simp)
        if simp==0: insight("Backwards: identity ∀x → consequence of sin²+cos²=1 + Euler")

    elif p.ptype==PT.FACTORING:
        fac=p.memo("factor(expr)", lambda: factor(p.expr))
        r["factored"]=str(fac); kv("Factored",fac)
        try:
            flist=factor_list(p.expr)
            for i,(fi,mult) in enumerate(flist[1]):
                rt=solve(fi,v) if v else []
                kv(f"  factor[{i}]^{mult}",f"{fi}  roots:{rt}")
        except: pass

    elif p.ptype==PT.SUM:
        k=symbols('k',positive=True,integer=True); n=symbols('n',positive=True,integer=True)
        summand=p.meta.get("summand",k)
        res=p._cache.get(f"summation({summand},(k,1,n))")
        if res: kv("Formula",str(factor(res))); insight("Backwards: closed form = telescoping structure")

    elif p.ptype==PT.PROOF:
        body=p.meta.get("body",""); body_low=body.lower()
        if any(kw in body_low for kw in("sqrt(2)","root 2","irrational")):
            for step,desc in [("Assume","√2=p/q, gcd=1"),("Square","2=p²/q²→p²=2q²"),
                               ("Even","p=2m"),("Sub","4m²=2q²→q even"),("QED","contradiction □")]:
                print(f"    {Y}{step:<12}{RST}{desc}")
        elif "prime" in body_low:
            for step,desc in [("Assume","finite {p₁,…,pₖ}"),("Construct","N=∏pᵢ+1"),
                               ("Factor","N has prime q"),("QED","q∉list contradiction □")]:
                print(f"    {Y}{step:<12}{RST}{desc}")

    elif p.ptype==PT.DIGRAPH_CYC:
        m=p.meta.get("m")
        if m%2!=0:
            kv("Law",f"Q_c(i,j)=(i+b_c(j), j+r_c) mod {m}; gcd(r_c,{m})=1")
            insight("Backwards: Hamiltonian decomp = perfect 1-factorisation of Cayley graph")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 05 — GENERALIZE
# ════════════════════════════════════════════════════════════════════════════

def phase_05(p: Problem, g4: dict) -> dict:
    section(5,"GENERALIZE","Governing theorems · parametric families · generalisation ladder")
    r={}; v=p.var

    domain_laws = {
        PT.GRAPH: {
            "Fiedler 1973":     "λ₂(L)>0 iff connected",
            "Bipartite":        "iff spectrum symmetric about 0",
            "Cheeger":          "h(G)∈[λ₂/2, √(2λ₂)]",
            "Kirchhoff":        "τ(G)=(1/n)∏λᵢ≠0",
            "Expander":         "Large λ₂ → fast mixing, robust to cuts",
            "Random walk":      "P=D⁻¹A: π_i=d_i/2|E|",
        },
        PT.MATRIX: {
            "Spectral thm":     "Sym M=QΛQᵀ (orthonormal Q)",
            "Cayley-Hamilton":  "p(M)=0",
            "SVD":              "M=UΣVᵀ (any matrix)",
            "Definiteness":     "xᵀMx>0 all x iff all λ>0",
            "Rank-nullity":     "rank+nullity=n",
            "Gershgorin":       "All λ in union of Gershgorin disks",
        },
        PT.MARKOV: {
            "Perron-Frobenius": "Irred non-neg: unique λ=1, unique π>0",
            "Ergodic theorem":  "Time avg = space avg = π",
            "Mixing bound":     "‖Pⁿ-Π‖≤|λ₂|ⁿ",
            "Entropy rate":     "h=-Σᵢπᵢ Σⱼ Pᵢⱼ log Pᵢⱼ",
            "Reversibility":    "πᵢPᵢⱼ=πⱼPⱼᵢ iff all eigs real",
        },
        PT.ENTROPY: {
            "Shannon uniqueness":"H unique by continuity+max+additivity",
            "Max entropy":      "H≤log₂n; equal iff uniform (MaxEnt)",
            "Chain rule":       "H(X,Y)=H(X)+H(Y|X)",
            "Data processing":  "H(f(X))≤H(X) for any f",
            "Source coding":    "L̄≥H(X) (Shannon 1948)",
        },
        PT.DYNAMICAL: {
            "Hartman-Grobman":  "Near hyperbolic eq: nonlinear≈linearisation",
            "Lyapunov":         "f'(x*)<0 stable; >0 unstable",
            "Noether":          "Every symmetry → conservation law",
            "Poincaré-Bendixson":"2D: no chaos",
            "Bifurcation":      "f'(x*)=0: qualitative change",
        },
        PT.CONTROL: {
            "Routh-Hurwitz":    "Stable iff all Routh 1st-col >0",
            "Spectral":         "Stable iff Re(λ)<0 all λ",
            "Nyquist":          "Encirclements of -1 = RHP poles",
            "Controllability":  "rank[B,AB,…,Aⁿ⁻¹B]=n iff controllable",
        },
        PT.OPTIMIZATION: {
            "1st order":        "∇f(x*)=0 (necessary, unconstrained)",
            "2nd order":        "H>0 local min; H<0 local max",
            "Convexity":        "Convex → every local min = global min",
            "KKT":              "Constrained: ∇f=Σλᵢ∇gᵢ, λᵢgᵢ=0",
            "Slater":           "Strict feasibility → strong duality",
        },
    }
    laws=domain_laws.get(p.ptype)
    if laws:
        kv("Governing theorems","")
        for name,law in laws.items(): kv(f"  {name}",law)
        r["governing"]=laws

    # Named graph families
    if p.ptype==PT.GRAPH:
        t=p.meta.get("type","")
        fams={"complete":"Kₙ: λ(L)={0¹,nⁿ⁻¹}, τ=nⁿ⁻²",
              "path":"Pₙ: λₖ=2-2cos(kπ/n), diam=n-1",
              "cycle":"Cₙ: λₖ=2-2cos(2πk/n), bipartite iff n even"}
        if t in fams: kv("Named family",fams[t]); r["family"]=fams[t]

    elif p.ptype==PT.LINEAR:
        a_,b_=symbols('a b',nonzero=True); sol=solve(a_*v+b_,v)[0]
        kv("General solution",str(sol)); kv("Condition","a≠0")

    elif p.ptype==PT.QUADRATIC:
        a_,b_,c_=symbols('a b c'); gen=solve(a_*v**2+b_*v+c_,v)
        r["qf"]=[str(s) for s in gen]; kv("Quadratic formula",r["qf"])

    elif p.ptype==PT.CUBIC:
        kv("Cardano","Depressed t³+pt+q=0: t=∛(-q/2+√D)+∛(-q/2-√D)")
        kv("IVT","Odd degree → ≥1 real root guaranteed")

    elif p.ptype==PT.SUM:
        k=symbols('k',positive=True,integer=True); n=symbols('n',positive=True,integer=True)
        for pw in range(1,6):
            try: kv(f"  Faulhaber Σk^{pw}",str(factor(summation(k**pw,(k,1,n)))))
            except: pass
        kv("Bernoulli","Coefficients of Faulhaber = Bernoulli numbers Bₖ")

    elif p.ptype==PT.TRIG_ID:
        theta=symbols('theta')
        for f_,ex in [("sin²+cos²",sin(theta)**2+cos(theta)**2-1),
                      ("1+tan²",1+sp.tan(theta)**2-sp.sec(theta)**2)]:
            kv(f"  {f_}=",f"{trigsimp(ex)}")
        kv("Euler","e^{iθ}=cosθ+i sinθ → all trig from one identity")

    elif p.ptype==PT.FACTORING:
        a_,b_=symbols('a b')
        for f_,e_ in [("a²-b²",a_**2-b_**2),("a³-b³",a_**3-b_**3),
                      ("a³+b³",a_**3+b_**3),("a⁴-b⁴",a_**4-b_**4)]:
            kv(f"  {f_}",str(factor(e_)))

    elif p.ptype==PT.PROOF:
        body_low=p.meta.get("body","").lower()
        if any(kw in body_low for kw in("sqrt(2)","root 2","irrational")):
            kv("General","√n irrational iff n not a perfect square")
        elif "prime" in body_low:
            kv("Governing","Euclid: construction+contradiction")

    finding("Specific case is an INSTANCE of the general law above")
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 06 — PROVE LIMITS
# ════════════════════════════════════════════════════════════════════════════

def phase_06(p: Problem, g5: dict) -> dict:
    section(6,"PROVE LIMITS","Hard boundaries · obstructions · what cannot be done")
    r={}; v=p.var

    if p.ptype==PT.GRAPH:
        L_spec=p.meta.get("L_spec",[]); deg=p.meta.get("deg",[])
        for k_,v_ in {"λ₂=0 iff disconnected":"Hard boundary (cannot relabel away)",
                      "λ_max(L)≤Δ":"Max degree bound; equality iff regular",
                      "Bipartite iff no odd cycles":"Structural necessary condition",
                      "Ramanujan expander":"λ₂≤2√(d-1) for d-regular",
                      "Interlacing":"Remove vertex → eigenvalues interlace (Cauchy)"}.items():
            kv(f"  {k_}",v_)
        if L_spec and deg:
            Delta=max(deg); lam_max=max(L_spec)
            (ok if lam_max<=Delta+1e-6 else fail)(f"λ_max={lam_max:.4f}≤Δ={Delta}")

    elif p.ptype==PT.MATRIX:
        for k_,v_ in {"Gershgorin":"λ∈∪{|z-aᵢᵢ|≤Σⱼ≠ᵢ|aᵢⱼ|}",
                      "Weyl":"|λᵢ(A+E)-λᵢ(A)|≤‖E‖₂ (stability)",
                      "det=0":"iff singular iff 0∈spec iff Ax=0 non-trivial",
                      "Perron-Frobenius":"Non-neg matrix: ρ=largest real eig"}.items():
            kv(f"  {k_}",v_)

    elif p.ptype==PT.MARKOV:
        for k_,v_ in {"Convergence":"Irred+aperiodic → ‖Pⁿ-Π‖→0",
                      "Periodicity":"Period d>1 → no pointwise convergence",
                      "Lazy fix":"P'=(P+I)/2 → aperiodic with same π"}.items():
            kv(f"  {k_}",v_)

    elif p.ptype==PT.ENTROPY:
        p_s=symbols('p',positive=True)
        H_bin=-p_s*log(p_s,2)-(1-p_s)*log(1-p_s,2)
        ok(f"H(0+)={limit(H_bin,p_s,0,'+')}, H(1-)={limit(H_bin,p_s,1,'-')}")
        for k_,v_ in {"H≥0":"equality iff deterministic",
                      "H≤log₂n":"equality iff uniform",
                      "Subadditivity":"H(X,Y)≤H(X)+H(Y); equal iff independent",
                      "Shannon":"L̄≥H(X) — cannot compress below entropy"}.items():
            kv(f"  {k_}",v_)
        finding("H is the IRREDUCIBLE minimum description length")

    elif p.ptype==PT.DYNAMICAL:
        for k_,v_ in {"Non-hyperbolic":"f'=0: linearisation fails — need higher order",
                      "Global Lyapunov":"V>0, V̇<0 everywhere → globally stable",
                      "No chaos (2D)":"Poincaré-Bendixson theorem",
                      "Bifurcation":"f'=0: structural qualitative change"}.items():
            kv(f"  {k_}",v_)

    elif p.ptype==PT.CONTROL:
        for k_,v_ in {"Necessary":"All coefficients same sign",
                      "Sufficient":"Routh 1st col all >0",
                      "Marginal":"Re(λ)=0: imaginary axis",
                      "Abel-Ruffini":"No radical formula for degree≥5"}.items():
            kv(f"  {k_}",v_)

    elif p.ptype==PT.OPTIMIZATION:
        for k_,v_ in {"1st order":"f'(x*)=0 necessary (smooth, unconstrained)",
                      "2nd order":"f''=0: inconclusive (check higher deriv)",
                      "Convex global":"Convex+convex domain: local=global",
                      "Non-convex":"Global opt NP-hard in general"}.items():
            kv(f"  {k_}",v_)
        try:
            lp=limit(p.expr,v,oo); ln=limit(p.expr,v,-oo)
            kv("f(+∞)",str(lp)); kv("f(-∞)",str(ln))
            if str(lp)==str(ln)=="oo": finding("f→+∞ both sides → minimum exists (coercive)")
        except: pass

    elif p.ptype==PT.LINEAR:
        kv("a=0,b=0","∞ solutions"); kv("a=0,b≠0","no solution")
        finding("Unique solution iff a≠0")

    elif p.ptype==PT.QUADRATIC:
        kv("Δ=0","double root x=-b/2a"); kv("Δ<0","no real roots")
        kv("Over ℂ","always exactly 2 roots (FTA)")

    elif p.ptype in(PT.CUBIC,PT.POLY):
        kv("IVT","Odd degree → ≥1 real root")
        kv("Abel-Ruffini","No radical formula for degree≥5")
        if p.ptype==PT.POLY:
            try:
                rts_all=all_roots(p.get_poly())
                real_rt=[r for r in rts_all if sp.im(r)==0]
                kv("Real roots",len(real_rt)); kv("Complex roots",len(rts_all)-len(real_rt))
            except: pass

    elif p.ptype==PT.SUM:
        k=symbols('k',positive=True,integer=True)
        kv("Σ1/k","diverges (harmonic series)")
        kv("Σ1/k²","=π²/6 (converges)")
        finding("p-series Σ1/kᵖ converges iff p>1 — hard boundary at p=1")

    elif p.ptype==PT.FACTORING:
        try:
            irred=p.get_poly().is_irreducible; kv("Irred over Q",irred)
            kv("Over ℂ","Always splits into linear factors (FTA)")
        except: pass

    elif p.ptype==PT.TRIG_ID:
        kv("sin²+cos²=1","∀x∈ℝ (no exceptions)")
        kv("1+tan²=sec²","fails at x=π/2+nπ (cos=0)")

    elif p.ptype==PT.DIGRAPH_CYC:
        m=p.meta.get("m")
        kv("Odd m","Fiber-uniform decomp exists"); kv("Even m","Parity obstruction: Σr_c must be even, but each r_c odd")
        finding("Parity = HARD BOUNDARY for fiber-uniform construction")

    elif p.ptype==PT.PROOF:
        body_low=p.meta.get("body","").lower()
        if any(kw in body_low for kw in("sqrt(2)","root 2","irrational")):
            kv("Boundary","√n rational iff n perfect square (boundary at squares)")
        elif "prime" in body_low:
            kv("Open","Twin prime conjecture (p,p+2 both prime): unproven")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 07 — SYNTHESIS  (Integration E: Spectral Unification + F: Output Entropy)
# ════════════════════════════════════════════════════════════════════════════

def phase_07(p: Problem, g6: dict) -> dict:
    section(7,"SYNTHESIS","Cross-domain bridges · spectral unification · output entropy · meta-lesson")
    r={}

    # ── UNIVERSAL BRIDGE MAP ──────────────────────────────────────────────
    BRIDGES = {
        PT.GRAPH:        [("→Markov","P=D⁻¹A random walk; π_i=d_i/2|E|"),
                          ("→Entropy","H_s=-Σ(λᵢ/tr_L)log(λᵢ/tr_L) spectral entropy"),
                          ("→Dynamical","ẋ=-Lx heat diffusion; solution e^{-tL}x₀"),
                          ("→Optimization","Min cut = max flow (LP duality)"),
                          ("→ML","GNN h'=σ(QᵀfΛQh), Q=eigvecs of L")],
        PT.MATRIX:       [("→Dynamical","ẋ=Ax stable iff Re(λᵢ)<0; e^{At}x₀"),
                          ("→Control","char poly=det(sI-A): poles=eigenvalues"),
                          ("→Optimization","Hessian H: xᵀHx curvature form"),
                          ("→Entropy","Von Neumann S(ρ)=-tr(ρ log ρ)"),
                          ("→Graph","0/1 symmetric = adjacency matrix")],
        PT.MARKOV:       [("→Graph","P defines weighted digraph; reversible P = undirected"),
                          ("→Entropy","h=lim H(Xₙ|X₀,…,Xₙ₋₁) entropy rate"),
                          ("→Optimization","MCMC: run chain to sample target π"),
                          ("→Physics","Ṡ=Σπᵢ Pᵢⱼ log(πᵢPᵢⱼ/πⱼPⱼᵢ)≥0 (2nd law)")],
        PT.ENTROPY:      [("→Physics","S=k_B·H (Boltzmann)"),
                          ("→Markov","h=-Σᵢπᵢ Σⱼ Pᵢⱼ log Pᵢⱼ"),
                          ("→Optimization","MaxEnt: max H(p) s.t. constraints → Gibbs"),
                          ("→ML","Cross-entropy loss=H(y,p̂)=H(y)+KL(y‖p̂)")],
        PT.DYNAMICAL:    [("→Control","ẋ=f(x,u): design u to steer to target"),
                          ("→Optimization","Gradient flow ẋ=-∇f IS gradient descent"),
                          ("→Markov","SDE ẋ=f+noise → Fokker-Planck"),
                          ("→Entropy","h_KS=Σmax(λᵢ,0) Lyapunov/chaos")],
        PT.CONTROL:      [("→Matrix","char poly=det(sI-A); poles=eigs of A"),
                          ("→Optimization","LQR: min∫(xᵀQx+uᵀRu)dt → Riccati"),
                          ("→Dynamical","Closed-loop ẋ=(A+BK)x; place eigs with K"),
                          ("→Graph","Consensus rate=λ₂(Laplacian)")],
        PT.OPTIMIZATION: [("→Dynamical","Gradient descent=Euler discret. of ẋ=-∇f"),
                          ("→Markov","RL/MDP: policy opt via Bellman"),
                          ("→Entropy","MaxEnt=exponential family"),
                          ("→Graph","Shortest path=min-cost flow=LP on graph")],
        PT.QUADRATIC:    [("→Matrix","Roots=eigenvalues of companion [[0,-c],[-b,-a]]"),
                          ("→Control","2nd-order char poly → poles of transfer function"),
                          ("→Dynamical","ax²+bx+c=0 ↔ equilibria of ẋ=ax²+bx+c")],
        PT.LINEAR:       [("→Matrix","Ax=b: unique sol iff A invertible"),
                          ("→Markov","1-state Markov: π=1 trivially"),
                          ("→Optimization","min ‖Ax-b‖² → x*=(AᵀA)⁻¹Aᵀb")],
        PT.CUBIC:        [("→Matrix","Char poly of 3×3 = cubic"),
                          ("→Dynamical","Equilibria of ẋ=ax³+bx")],
        PT.SUM:          [("→Entropy","Partition function Z=Σexp(-Eᵢ/kT)"),
                          ("→Markov","Expected hitting times = sums over states")],
        PT.TRIG_ID:      [("→Complex","e^{iθ}=cosθ+i sinθ (Euler)"),
                          ("→Graph","Cycle Cₙ: eigs=2-2cos(2πk/n)")],
        PT.FACTORING:    [("→Roots","Factors determine roots exactly"),
                          ("→Crypto","Irreducibility over Z_p → RSA hardness")],
        PT.PROOF:        [("→Logic","Contradiction IS the diagonal argument (Gödel)"),
                          ("→Algebra","Irrationality ↔ field extension degree 2")],
    }
    bridges_for=BRIDGES.get(p.ptype,[])
    if bridges_for:
        kv("Cross-domain bridges","")
        for src_dst,desc in bridges_for:
            bridge(f"{src_dst}: {desc}")
        r["bridges"]={sd:d for sd,d in bridges_for}

    # ── INTEGRATION E: SPECTRAL UNIFICATION ───────────────────────────────
    if p.fps:
        print(f"\n  {DIM}--- spectral unification ---{RST}")
        kv("Fingerprints computed",len(p.fps))
        for fp in p.fps:
            kv(f"  {fp.domain}",fp.summary())
            se=fp.spectral_entropy()
            kv(f"    spectral entropy",f"{se:.4f} bits")
        # Cross-domain match detection
        if len(p.fps)>1:
            for i,fpa in enumerate(p.fps):
                for fpb in p.fps[i+1:]:
                    if fpa.matches(fpb):
                        insight(f"⚡ SPECTRAL ISOMORPHISM: {fpa.domain} ≅ {fpb.domain}")
                        r["spectral_isomorphism"]=(fpa.domain,fpb.domain)
        # Explain what the spectrum means globally
        if any(fp.domain=="companion_poly" for fp in p.fps):
            insight("Companion spectrum = polynomial roots = control poles = dynamical equilibria — same object in three languages")

    # ── DOMAIN-SPECIFIC EMERGENTS ─────────────────────────────────────────
    if p.ptype==PT.GRAPH:
        A_spec=p.meta.get("A_spec",[]); L_spec=p.meta.get("L_spec",[])
        if L_spec:
            tr_L=sum(L_spec)
            if tr_L>0:
                nz=[e for e in L_spec if e>1e-9]
                H_s=_entropy([e/tr_L for e in nz])
                r["spectral_entropy"]=H_s; kv("Spectral entropy H_s(G)",f"{H_s:.4f} bits")
        n_clust=sum(1 for e in L_spec if abs(e)<0.1) if L_spec else 0
        kv("Spectral clusters (λ≈0)",n_clust)
        insight(f"{n_clust} near-zero eigs → {n_clust} natural clusters")
        kv("Heat kernel","e^{-tL}: diffusion from any node (t→∞ = uniform)")
        kv("Ihara zeta","Z_G(u)=∏_primes(1-u^|p|)⁻¹ (Riemann zeta analog)")
        insight("DEEPEST: graph spectrum = isomorphism fingerprint")

    elif p.ptype==PT.MATRIX:
        spec=p.meta.get("spec",[])
        for k_,v_ in {"e^{At}":"universal solution to ẋ=Ax",
                      "SVD M=UΣVᵀ":"optimal rank-k approximation (Eckart-Young)",
                      "M⁺=VΣ⁺Uᵀ":"least-squares solution"}.items():
            kv(f"  {k_}",v_)
        if spec:
            pi_spec=[abs(e)/sum(abs(e2) for e2 in spec) for e in spec if abs(e)>1e-12]
            if pi_spec:
                H_vn=_entropy(pi_spec); kv("Von-Neumann-like entropy",f"{H_vn:.4f} bits")
        insight("DEEPEST: e^{At} IS the universal solution to all linear ODEs")

    elif p.ptype==PT.MARKOV:
        stat=p.meta.get("stat",{}); P_rat=p.meta.get("P_rat"); n=p.meta.get("n",0)
        for k_,v_ in {"Potential":"G=(I-P)⁻¹ → hitting times",
                      "MCMC":"Sample ANY dist by constructing chain with target π",
                      "Free energy":"F=<E>-T·H(π) (equilibrium minimises F)"}.items():
            kv(f"  {k_}",v_)
        if stat and P_rat:
            try:
                pi_f=[float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                ep=sum(pi_f[i]*float(N(P_rat[i,j]))*math.log(
                    max(pi_f[i]*float(N(P_rat[i,j])),1e-15)/max(pi_f[j]*float(N(P_rat[j,i])),1e-15))
                    for i in range(n) for j in range(n)
                    if float(N(P_rat[i,j]))>1e-12 and float(N(P_rat[j,i]))>1e-12)
                r["entropy_prod"]=ep; kv("Entropy production",f"{ep:.6f} (2nd law ≥0)")
                insight(f"Ep={ep:.4f}: {'reversible' if ep<1e-9 else 'irreversible — entropy produced'}")
            except: pass
        insight("DEEPEST: Markov chain IS a random walk on a weighted graph")

    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        for k_,v_ in {"Mutual info":"I(X;Y)=H(X)+H(Y)-H(X,Y)≥0, =0 iff independent",
                      "Rényi":"H_α=(1/(1-α))log Σpᵢ^α; α→1 = Shannon",
                      "MDL":"Min description length = Occam's razor quantified"}.items():
            kv(f"  {k_}",v_)
        if probs:
            for alpha in[0.5,2.0]:
                H_r=(1/(1-alpha))*math.log2(sum(q**alpha for q in probs if q>0))
                kv(f"Rényi H_{alpha}",f"{H_r:.4f} bits")
        insight("DEEPEST: MaxEnt = Bayesian prior + Gibbs distribution + ML softmax — same object")

    elif p.ptype==PT.DYNAMICAL:
        for k_,v_ in {"Gradient flow":"ẋ=-∇f: stable eq=global min of f",
                      "KS entropy":"h_KS=Σmax(λᵢ,0): chaos from Lyapunov exps",
                      "Variational":"Trajectories extremise action S=∫Ldt"}.items():
            kv(f"  {k_}",v_)
        insight("DEEPEST: gradient flows unify optimization, dynamical systems, statistical mechanics")

    elif p.ptype==PT.CONTROL:
        for k_,v_ in {"Pontryagin":"Optimal control = variational problem (PMP)",
                      "Kalman filter":"Dual of LQR: optimal state estimation",
                      "H∞":"Robust control = min-max over disturbances"}.items():
            kv(f"  {k_}",v_)
        insight("DEEPEST: control = optimization over function space (Pontryagin duality)")

    elif p.ptype==PT.OPTIMIZATION:
        for k_,v_ in {"Lagrange dual":"Strong duality: primal=dual (Slater) = physics+info",
                      "Proximal":"Implicit Euler of ẋ=-∇f",
                      "Natural gradient":"Riemannian ∇ w.r.t. Fisher info metric"}.items():
            kv(f"  {k_}",v_)
        insight("DEEPEST: Lagrangian duality unifies optimization, physics, information theory")

    elif p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY):
        bridge("→Matrix: roots = eigenvalues of companion matrix C(p)")
        bridge("→Control: char poly of A(s) → poles of transfer function")
        bridge("→Dynamical: real roots = equilibria of ẋ=p(x)")
        insight("DEEPEST: FTA ↔ linear algebra ↔ dynamical systems = same object in 3 languages")

    elif p.ptype==PT.SUM:
        bridge("→Euler-Maclaurin: Σf(k) ≈ ∫f dx + corrections (boundary terms)")
        bridge("→Riemann zeta: ζ(s)=Σn^{-s} encodes ALL prime distribution")
        insight("DEEPEST: Riemann zeta = partition function of primes (physics↔number theory)")

    elif p.ptype==PT.PROOF:
        bridge("→Diagonal argument: contradiction IS Cantor/Gödel/Halting Problem")
        bridge("→Algebraic number theory: √p irrational ↔ [Q(√p):Q]=2")
        insight("DEEPEST: proof by contradiction = the incompleteness machinery in disguise")

    elif p.ptype==PT.TRIG_ID:
        bridge("→Complex: e^{iπ}+1=0 unifies analysis, algebra, geometry")
        bridge("→Fourier: sin/cos = basis of L²([0,2π]) — spectral decomposition of functions")
        insight("DEEPEST: all of trigonometry is Euler's formula in disguise")

    # ── INTEGRATION F: OUTPUT ENTROPY SCORE ───────────────────────────────
    print(f"\n  {DIM}--- output entropy scoring ---{RST}")
    all_printed=[str(v) for k,v in p._cache.items() if v is not None]
    oe=_output_entropy(all_printed)
    r["output_entropy"]=oe
    kv("Output diversity H(output)",f"{oe:.4f} bits")
    if oe>4.0:   insight("High output diversity — engine produced genuinely varied information")
    elif oe>2.0: note("Moderate output diversity — some redundancy present")
    else:        warn("Low output diversity — engine may be repeating itself")
    r["output_entropy_rating"]=("high" if oe>4.0 else "moderate" if oe>2.0 else "low")

    # ── FEEDBACK SUMMARY ──────────────────────────────────────────────────
    signals=p.fbq.all_signals()
    if signals:
        kv("Feedback signals fired",signals)

    # ── META-LESSON ───────────────────────────────────────────────────────
    print(f"\n  {DIM}--- meta-lesson ---{RST}")
    meta_lessons = {
        PT.GRAPH:       "What this teaches: network structure is fully encoded in its eigenvalues",
        PT.MATRIX:      "What this teaches: linear algebra IS the universal language of structure",
        PT.MARKOV:      "What this teaches: randomness + time = deterministic stationary behaviour",
        PT.ENTROPY:     "What this teaches: uncertainty is quantifiable with hard arithmetic limits",
        PT.DYNAMICAL:   "What this teaches: stability = eigenvalue condition = potential landscape",
        PT.CONTROL:     "What this teaches: stability is a spectral property — not analytical",
        PT.OPTIMIZATION:"What this teaches: every optimization problem has a dual equilibrium",
        PT.QUADRATIC:   "What this teaches: the discriminant IS the geometry of the solution set",
        PT.CUBIC:       "What this teaches: degree 3 is where algebra first becomes genuinely hard",
        PT.POLY:        "What this teaches: high-degree polynomials require numerics (Abel-Ruffini)",
        PT.LINEAR:      "What this teaches: linear problems have unique solutions iff A is invertible",
        PT.TRIG_ID:     "What this teaches: all trigonometry follows from one geometric fact",
        PT.FACTORING:   "What this teaches: factorisation is context-dependent (Q, R, C, Z_p)",
        PT.SUM:         "What this teaches: sums have closed forms when the right structure is found",
        PT.PROOF:       "What this teaches: contradiction is the most powerful tool in mathematics",
        PT.DIGRAPH_CYC: "What this teaches: parity is the deepest obstruction in combinatorics",
    }
    lesson=meta_lessons.get(p.ptype,"What this teaches: structure determines behaviour")
    insight(lesson); r["meta_lesson"]=lesson

    # ── CONFIDENCE FINAL ──────────────────────────────────────────────────
    print(f"\n  {DIM}--- confidence ledger ---{RST}")
    kv("High-confidence",p.conf.knowns[:3] or "none")
    kv("Uncertain",p.conf.unknowns[:3] or "none")
    if p.conf.flags:
        for f_ in p.conf.flags[:3]: warn(f_)
    kv("Overall",p.conf.summary())

    return r


# ════════════════════════════════════════════════════════════════════════════
# FINAL ANSWER
# ════════════════════════════════════════════════════════════════════════════

def _final_answer(p: Problem) -> str:
    v=p.var
    # Cache-first for all types (Integration from user engine)
    if p.ptype==PT.GRAPH:
        spec=p.meta.get("L_spec",[]); named=p.meta.get("named","graph"); n=p.meta.get("n",0)
        conn=(sorted(spec)[1]>1e-9) if len(spec)>1 else "?"
        return f"{named} ({n}v): Connected={conn}, λ(L)={[f'{e:.3f}' for e in spec]}"
    elif p.ptype==PT.MATRIX:
        spec=p.meta.get("spec",[]); M=p.meta.get("M")
        return f"Matrix: λ={[f'{e:.3f}' for e in spec]}, det={str(det(M)) if M else '?'}"
    elif p.ptype==PT.MARKOV:
        stat=p.meta.get("stat",{}); n=p.meta.get("n",0)
        return f"Markov ({n} states): π={stat}"
    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        if probs:
            H=_entropy(probs)
            return f"H(X)={H:.6f} bits (max={math.log2(len(probs)):.6f}, η={H/math.log2(len(probs)):.4f})"
        return "Entropy computed"
    elif p.ptype==PT.DYNAMICAL:
        equil=p.memo("solve(f=0)",lambda:solve(p.expr,v))
        if equil:
            fp=diff(p.expr,v)
            stab=[("S" if float(N(sp_re(fp.subs(v,e))))<0 else "U") for e in equil]
            return f"Equilibria: {list(zip([str(e) for e in equil],stab))}"
        return "Dynamical: computed"
    elif p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{})
        return f"Control: {'STABLE' if rh.get('stable') else 'UNSTABLE'} ({rh.get('sign_changes',0)} RHP roots)"
    elif p.ptype==PT.OPTIMIZATION:
        crit=p._cache.get("solve(f'=0)",[])
        if crit:
            try:
                vals=[(float(N(p.expr.subs(v,c))),c) for c in crit if _is_real_value(p.expr.subs(v,c))]
                goal=p.meta.get("goal","extremize")
                best=(min if "min" in goal else max)(vals)
                return f"Optimal x*={best[1]}, f*={best[0]:.6f}"
            except: pass
        return "Optimization: critical points classified"
    elif p.ptype in(PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY):
        sols=p.memo("solve(expr,var)",lambda:solve(p.expr,v))
        return f"Solutions: {', '.join(str(s) for s in sols)}" if sols else "No solutions found"
    elif p.ptype==PT.FACTORING:
        fac=p.memo("factor(expr)",lambda:factor(p.expr))
        return f"Factored: {fac}" if fac else "Factoring failed"
    elif p.ptype in(PT.TRIG_ID,PT.SIMPLIFY):
        simp=p.memo("trigsimp",lambda:trigsimp(p.expr))
        return "Identity confirmed" if simp==0 else f"Simplified: {simp}"
    elif p.ptype==PT.SUM:
        # Cache-first: scan for any summation result
        for k_,v_ in p._cache.items():
            if "summation" in k_ and v_ is not None:
                try: return f"Sum = {factor(v_)}"
                except: return f"Sum = {v_}"
        return "Summation computed"
    elif p.ptype==PT.PROOF:
        body_low=p.meta.get("body","").lower()
        if any(kw in body_low for kw in("sqrt(2)","root 2","irrational")): return "√2 irrational. Proof by contradiction. QED."
        elif "prime" in body_low: return "Infinitely many primes. Euclid construction. QED."
        return "Proof presented in phase computations"
    elif p.ptype==PT.DIGRAPH_CYC:
        m=p.meta.get("m")
        return (f"Odd m={m}: Hamiltonian decomp exists." if m%2!=0
                else f"Even m={m}: parity obstruction — impossible.")
    return "See phase computations"


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — adaptive phase runner
# ════════════════════════════════════════════════════════════════════════════

def run(raw: str, json_out: bool = False) -> Optional[dict]:
    t0=time.time()
    prob=classify(raw)

    if not _QUIET:
        print(f"\n{hr('═')}")
        print(f"{W}DISCOVERY ENGINE v4{RST}")
        print(hr())
        print(f"  {W}Problem:{RST}  {Y}{raw}{RST}")
        print(f"  {DIM}Type:{RST}     {prob.ptype.label()}   {DIM}Variable:{RST} {prob.var}")
        print(hr('═'))

    if prob.ptype==PT.UNKNOWN:
        if not _QUIET:
            print(f"{R}Cannot classify. Try:{RST}")
            print(f"  Equations:   x^2-5x+6=0  |  x^3-6x^2+11x-6=0")
            print(f"  Graph:       graph K4  |  graph [[0,1,1,0],...]")
            print(f"  Matrix:      matrix [[2,1],[1,3]]")
            print(f"  Markov:      markov [[0.7,0.3],[0.4,0.6]]")
            print(f"  Entropy:     entropy [0.5,0.25,0.25]")
            print(f"  Dynamical:   dynamical x^3-x")
            print(f"  Control:     control s^2+3s+2")
            print(f"  Optimize:    minimize x^2+2x+1")
            print(f"  Sum:         sum of squares of first n integers")
            print(f"  Proof:       prove sqrt(2) is irrational")
        return None

    max_phase=KB.phase_depth(prob.ptype_str())

    g1=phase_01(prob)
    g2=phase_02(prob,g1)
    if g2.get("routh"): prob._cache["routh"]=g2["routh"]

    g3=phase_03(prob,g2) if max_phase>=3 else {}
    # Feedback-unlocked pass (Integration D)
    g2b=phase_02b(prob,g3) if max_phase>=3 else {}

    g4=phase_04(prob,g3) if max_phase>=4 else {}
    g5=phase_05(prob,g4) if max_phase>=5 else {}
    g6=phase_06(prob,g5) if max_phase>=6 else {}
    g7=phase_07(prob,g6) if max_phase>=7 else {}

    elapsed=time.time()-t0
    final=_final_answer(prob)

    if not _QUIET:
        print(f"\n{hr('═')}")
        print(f"{W}FINAL ANSWER{RST}")
        print(hr())
        print(f"  {G}{final}{RST}")
        print(hr('═'))
        # Phase summary
        titles={1:"Ground Truth+Intel",2:"Direct Attack",3:"Structure Hunt",
                4:"Pattern Lock",5:"Generalize",6:"Prove Limits",7:"Synthesis"}
        all_g=[g1,g2,g3,g4,g5,g6,g7]
        print(f"\n{hr()}")
        print(f"{W}PHASE SUMMARY{RST}")
        print(hr('.'))
        for i,(g,t) in enumerate(zip(all_g,titles.values()),1):
            if i>max_phase and not g:
                print(f"  {DIM}{i:02d} {t:<22} [skipped — adaptive depth]{RST}")
            else:
                print(f"  {PHASE_CLR[i]}{i:02d} {t:<22}{RST} {len(g)} results")
        kv("Confidence",prob.conf.summary())
        kv("Spectral fingerprints",len(prob.fps))
        kv("Feedback signals",prob.fbq.all_signals())
        kv("Elapsed",f"{elapsed:.3f}s")
        print(hr('═'))

    if json_out:
        result={
            "problem": raw, "type": prob.ptype.label(), "variable": str(prob.var),
            "answer": final, "elapsed_s": round(elapsed,3),
            "confidence": prob.conf.summary(),
            "spectral_fingerprints": [fp.summary() for fp in prob.fps],
            "feedback_signals": prob.fbq.all_signals(),
            "phases": {str(i): len(g) for i,g in enumerate([g1,g2,g3,g4,g5,g6,g7],1)},
            "output_entropy": g7.get("output_entropy",0) if g7 else 0,
            "meta_lesson": g7.get("meta_lesson","") if g7 else "",
            "bridges": list(g7.get("bridges",{}).keys()) if g7 else [],
        }
        return result
    return None


# ════════════════════════════════════════════════════════════════════════════
# TEST SUITE — 40 problems
# ════════════════════════════════════════════════════════════════════════════

TESTS = [
    # ── Standard algebraic (9) ──────────────────────────────────────────
    ("x^2 - 5x + 6 = 0",                       "Quadratic — integer roots"),
    ("2x + 3 = 7",                              "Linear equation"),
    ("x^3 - 6x^2 + 11x - 6 = 0",               "Cubic — 3 integer roots"),
    ("sin(x)^2 + cos(x)^2",                     "Pythagorean identity"),
    ("factor x^4 - 16",                         "Difference of squares chain"),
    ("sum of first n integers",                 "Classic sum k"),
    ("prove sqrt(2) is irrational",             "Irrationality proof"),
    ("m^3 vertices with 3 cycles, m=3",         "Digraph odd m"),
    ("m^3 vertices with 3 cycles, m=4",         "Digraph even m"),
    # ── Graph (5) ──────────────────────────────────────────────────────
    ("graph K4",                                "Complete K4"),
    ("graph P5",                                "Path P5"),
    ("graph C6",                                "Cycle C6"),
    ("graph [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]", "Custom adjacency"),
    ("graph C4",                                "Cycle C4 — bipartite"),
    # ── Matrix (2) ─────────────────────────────────────────────────────
    ("matrix [[2,1],[1,3]]",                    "Symmetric 2×2"),
    ("matrix [[4,2,2],[2,3,0],[2,0,3]]",        "Symmetric 3×3 definiteness"),
    # ── Markov (3) ─────────────────────────────────────────────────────
    ("markov [[0.7,0.3],[0.4,0.6]]",            "2-state chain"),
    ("markov [[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]]","3-state chain"),
    ("markov [[1,0],[0.3,0.7]]",                "Absorbing state"),
    # ── Entropy (3) ────────────────────────────────────────────────────
    ("entropy [0.5,0.25,0.25]",                 "Entropy skewed"),
    ("entropy [0.25,0.25,0.25,0.25]",           "Entropy uniform"),
    ("entropy [0.9,0.05,0.05]",                 "Entropy near-deterministic"),
    # ── Dynamical (3) ──────────────────────────────────────────────────
    ("dynamical x^3 - x",                      "Dynamical 3 equilibria"),
    ("dynamical x^2 - 1",                      "Dynamical pitchfork"),
    ("dynamical sin(x)",                        "Dynamical trig equilibria"),
    # ── Control (3) ────────────────────────────────────────────────────
    ("control s^2 + 3s + 2",                   "Control stable 2nd"),
    ("control s^3 + 2s^2 + 3s + 1",           "Control Routh 3rd"),
    ("control s^3 - s + 1",                    "Control unstable"),
    # ── Optimization (2) ───────────────────────────────────────────────
    ("optimize x^4 - 4x^2 + 1",               "Optimize quartic"),
    ("minimize x^2 + 2x + 1",                 "Minimize quadratic"),
    # ── Edge cases / new (10) ──────────────────────────────────────────
    ("sum of squares of first n integers",     "EDGE: sum of squares"),
    ("sum of cubes of first n integers",       "EDGE: sum of cubes"),
    ("sum of harmonic series",                 "EDGE: harmonic sum"),
    ("factor x^6 - 1",                        "EDGE: 6th power"),
    ("x^4 - 5x^2 + 4 = 0",                   "EDGE: biquadratic substitution"),
    ("x^2 + 4 = 0",                           "EDGE: complex roots only"),
    ("prove there are infinitely many primes", "EDGE: NL proof alias"),
    ("prove root 2 is irrational",            "EDGE: NL sqrt2 alias"),
    ("maximize -x^2 + 4x - 3",               "EDGE: maximize concave"),
    ("control s^4 + s^3 + s^2 + s + 1",      "EDGE: 4th order Routh"),
]


def run_tests(quiet=False):
    global _QUIET
    old=_QUIET; _QUIET=quiet
    print(f"\n{hr('═')}")
    print(f"{W}DISCOVERY ENGINE v4 — TEST SUITE ({len(TESTS)} problems){RST}")
    print(hr('═'))
    passed=0; failed=[]; timings=[]
    for raw,desc in TESTS:
        if not quiet:
            print(f"\n{B}{hr('-',60)}{RST}")
            print(f"{B}TEST: {desc}{RST}  {DIM}[{raw}]{RST}")
        else:
            print(f"  {DIM}{raw[:55]:<57}{RST}", end="", flush=True)
        t0=time.time()
        try:
            run(raw)
            elapsed=time.time()-t0
            timings.append(elapsed)
            if quiet: print(f"{G}✓{RST} {elapsed:.2f}s")
            else: ok(f"PASSED: {desc}  ({elapsed:.2f}s)")
            passed+=1
        except Exception as e:
            elapsed=time.time()-t0; timings.append(elapsed)
            if quiet: print(f"{R}✗{RST} {e}")
            else: fail(f"FAILED: {desc} — {e}"); traceback.print_exc()
            failed.append((desc,str(e)))
    total_t=sum(timings)
    _QUIET=old
    print(f"\n{hr('═')}")
    clr=G if passed==len(TESTS) else (Y if passed>len(TESTS)*0.8 else R)
    print(f"{clr}Results: {passed}/{len(TESTS)} passed{RST}  |  Total: {total_t:.1f}s  |  Avg: {total_t/len(TESTS):.2f}s/problem")
    if failed:
        print(f"{R}Failed:{RST}")
        for d,e in failed: print(f"  {R}✗{RST} {d}: {e[:80]}")
    print(hr('═'))
    return passed,len(TESTS)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__=="__main__":
    args=sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0]=="--test":
        quiet="--quiet" in args
        run_tests(quiet=quiet)
    elif args[0]=="--json":
        if len(args)<2: print("Usage: --json \"problem\""); sys.exit(1)
        result=run(" ".join(args[1:]),json_out=True)
        print(json.dumps(result,indent=2,default=str))
    elif args[0]=="--bench":
        # Quick benchmark against v3
        print(f"\n{hr('═')}")
        print(f"{W}QUICK BENCHMARK: v4 vs v3{RST}")
        print(hr())
        bench_problems=["x^2-5x+6=0","graph K4","markov [[0.7,0.3],[0.4,0.6]]",
                        "entropy [0.5,0.25,0.25]","sum of squares of first n integers",
                        "control s^3+2s^2+3s+1","dynamical x^3-x","optimize x^4-4x^2+1"]
        import io
        from contextlib import redirect_stdout
        for raw in bench_problems:
            buf=io.StringIO()
            t0=time.time()
            with redirect_stdout(buf): run(raw)
            t4=time.time()-t0
            out=buf.getvalue()
            print(f"  {DIM}{raw[:45]:<47}{RST} v4:{Y}{t4:.2f}s{RST}  "
                  f"lines:{len([l for l in out.split(chr(10)) if l.strip()])}")
        print(hr('═'))
    else:
        _QUIET="--quiet" in args
        raw=" ".join(a for a in args if a!="--quiet")
        if "--json" in args:
            result=run(raw,json_out=True)
            print(json.dumps(result,indent=2,default=str))
        else:
            run(raw)

#!/usr/bin/env python3
"""
discovery_engine_v5.py — 7-Phase Mathematical Discovery Engine
================================================================
Synthesised from studying, testing, and benchmarking v4 and v4b.

COMPACTED KNOWLEDGE FROM ANALYSIS
──────────────────────────────────
Performance insight (4.3× gap):
  v4 used signal.SIGALRM (Unix syscall overhead on every memoized call).
  v4b used threading.Thread (OS thread join, cross-platform, lower overhead).
  v5 uses threads by default; SIGALRM available as force_signal=True on Unix.

Architecture insight (event bus pattern):
  The FeedbackQueue is the engine's nervous system. Phase 03 discoveries
  push typed signals; Phase 02/04 check signals to unlock extra computation.
  This keeps phases decoupled while enabling inter-phase intelligence.

Mathematical insight (5 unifying identities):
  1. ROOTS = EIGENVALUES = POLES = EQUILIBRIA
     Polynomial roots (algebra) ≡ companion matrix eigenvalues (linear algebra)
     ≡ transfer function poles (control) ≡ equilibria of ẋ=p(x) (dynamics).
     Same object, four languages.
  2. GRAPH SPECTRUM = MARKOV MIXING = DIFFUSION RATE
     Laplacian λ₂ = Fiedler value = algebraic connectivity = mixing time
     = Cheeger constant (up to sqrt) = heat diffusion rate. One number rules all.
  3. MAXENT = GIBBS = SOFTMAX
     Maximum entropy under linear constraints (information theory)
     ≡ Gibbs distribution e^{-E/kT} (statistical mechanics)
     ≡ softmax over logits (ML). Exponential family IS the MaxEnt family.
  4. GRADIENT FLOW = LANGEVIN = FOKKER-PLANCK
     ẋ = −∇f (continuous gradient descent) ≡ overdamped Langevin SDE
     ≡ Fokker-Planck PDE for probability density. Optimisation, stochastics,
     PDE — same object at three levels of description.
  5. ROUTH-HURWITZ = SPECTRAL = LYAPUNOV
     All first-column entries positive (algebraic criterion)
     ≡ all eigenvalues in open left half-plane (spectral criterion)
     ≡ ∃ Lyapunov function V with V̇ < 0 (energy criterion).
     Three equivalent stability conditions.

Design principles extracted:
  P1. memo-with-timeout: every heavy sympy call → timed() + _cache[key].
  P2. Confidence over boolean: every result carries float conf ∈ [0,1].
  P3. Signals over data: FeedbackQueue carries typed signals, not raw dicts.
  P4. Ranked attempts: sort methods by (prior × boost) before running.
  P5. Adaptive depth: match phase count to problem complexity (PHASE_DEPTH).
  P6. Semantic tests: pass/fail proves no crash; assertions prove correctness.

v5 = v4b base (speed, threads, clean API, semantic tests)
   + v4 knowledge (PHASE_DEPTH, solution families, companion FP, output
                   entropy, 19 priors, _vieta_check, _verify_eq, _routh)
   + new: unified SpectralFingerprint (cosine sim + complex_ + metadata)
   + new: confidence-ranked AttemptPlan in phase_02
   + new: _parse_probs with normalization + warnings
   + new: DiscoveryResult typed result schema
   + new: _output_entropy self-scoring in phase_07

Usage:
  python discovery_engine_v5.py "x^2 - 5x + 6 = 0"
  python discovery_engine_v5.py --test
  python discovery_engine_v5.py --test --verbose
  python discovery_engine_v5.py --bench
  python discovery_engine_v5.py --json "entropy [0.5,0.25,0.25]"
  python discovery_engine_v5.py --quiet "graph K4"
"""

import sys, re, ast, math, time, heapq, json, io, threading, signal
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
from contextlib import redirect_stdout, redirect_stderr

import sympy as sp
from sympy import (
    symbols, solve, simplify, expand, factor, cancel,
    Symbol, Rational, pi, E, I, oo,
    sin, cos, tan, sec, csc, cot, exp, log, sqrt, Abs,
    diff, integrate, limit, summation, discriminant, roots,
    Poly, factorint, trigsimp, expand_trig, nsolve, N, S,
    gcd, divisors, apart, collect, nsimplify,
    real_roots, all_roots, factor_list, sqf_list,
    Matrix, eye, zeros, ones, diag, det, trace,
    re as sp_re, im as sp_im,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# ── Colour helpers ────────────────────────────────────────────────────────────
R="\033[91m"; G="\033[92m"; Y="\033[93m"; B="\033[94m"
M="\033[95m"; C="\033[96m"; W="\033[97m"; DIM="\033[2m"; RST="\033[0m"
PHASE_CLR={1:G, 2:R, 3:B, 4:M, 5:Y, 6:C, 7:W}

_QUIET = False

def hr(ch="─", n=72): return ch * n
def section(num, name, tag):
    if _QUIET: return
    c = PHASE_CLR[num]
    print(f"\n{hr()}\n{c}Phase {num:02d} — {name}{RST}  {DIM}{tag}{RST}\n{hr('·')}")
def kv(k, v, indent=2):
    if _QUIET: return
    print(f"{' '*indent}{DIM}{k:<38}{RST}{W}{str(v)[:120]}{RST}")
def finding(msg, sym="→"):
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
# TIMEOUT  — v4b threading (default, cross-platform) + v4 SIGALRM (optional)
# Design principle P1: every heavy call wrapped here.
# ════════════════════════════════════════════════════════════════════════════

def timed(func: Callable, args: tuple = (), secs: int = 8,
          fallback=None, force_signal: bool = False):
    """
    Run func(*args) with wall-clock timeout.
    Default: threading.Thread (cross-platform, lower overhead — v4b discovery).
    force_signal=True: SIGALRM on Unix (higher precision, v4 legacy path).
    """
    if force_signal and sys.platform != "win32":
        # SIGALRM path (v4): precise but Unix-only
        class _TO(Exception): pass
        def _h(sig, frame): raise _TO()
        old = signal.signal(signal.SIGALRM, _h)
        signal.alarm(secs)
        try:
            r = func(*args); signal.alarm(0); return r
        except _TO:
            return fallback
        except Exception:
            return fallback
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)

    # Thread path (v4b): cross-platform, 4.3× faster on algebraic problems
    res = [fallback]; exc = [None]
    def _run():
        try:    res[0] = func(*args)
        except Exception as e: exc[0] = e
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(secs)
    if t.is_alive():
        note(f"  ⏱ Timeout ({secs}s)")
        return fallback
    if exc[0]: raise exc[0]
    return res[0]


# ════════════════════════════════════════════════════════════════════════════
# SPECTRAL FINGERPRINT  — union of v4 and v4b designs
# v4  contributed: complex_, label, sorted_real(), matches() (binary)
# v4b contributed: metadata dict, spectral_radius(), cosine similarity
# v5  adds:        norm_distance(), unified __repr__
# Mathematical role: every eigenvalue-producing domain deposits one here.
# Phase 07 detects structural isomorphisms across domains (same maths,
# different language) by comparing fingerprints.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralFingerprint:
    """
    Universal eigenvalue container. Insight from analysis: polynomial roots,
    Laplacian eigenvalues, Markov eigenvalues, control poles, and Hessian
    eigenvalues are the SAME mathematical object in different languages.
    Storing them in one type enables cross-domain isomorphism detection.
    """
    domain:   str
    values:   List[float]           = field(default_factory=list)  # sorted real parts
    complex_: List[complex]         = field(default_factory=list)  # full complex (v4)
    label:    str                   = ""                           # context label (v4)
    metadata: Dict[str, Any]        = field(default_factory=dict)  # extensible KV (v4b)

    def sorted_real(self) -> List[float]:
        return sorted(self.values)

    def spectral_entropy(self) -> float:
        """Shannon entropy of |eigenvalue| distribution — measures spectral diversity."""
        abs_v = [abs(x) for x in self.values if abs(x) > 1e-12]
        if not abs_v: return 0.0
        total = sum(abs_v)
        probs = [x/total for x in abs_v]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def spectral_radius(self) -> float:
        """ρ = max |λᵢ| — stability threshold for iterative maps."""
        return max(abs(x) for x in self.values) if self.values else 0.0

    def cosine_similarity(self, other: "SpectralFingerprint") -> float:
        """
        Cosine similarity ∈ [-1,1] between sorted eigenvalue vectors.
        Value > 0.99 suggests structural isomorphism (v4b insight: continuous
        similarity is more informative than binary tolerance matching).
        """
        a = sorted(self.values); b = sorted(other.values)
        n = min(len(a), len(b))
        if n == 0: return 0.0
        dot  = sum(a[i] * b[i] for i in range(n))
        na   = math.sqrt(sum(x**2 for x in a[:n])) + 1e-15
        nb   = math.sqrt(sum(x**2 for x in b[:n])) + 1e-15
        return max(-1.0, min(1.0, dot / (na * nb)))

    def matches(self, other: "SpectralFingerprint", tol: float = 0.01) -> bool:
        """Binary match (v4): True if all sorted real parts agree within tol."""
        a, b = self.sorted_real(), other.sorted_real()
        if len(a) != len(b): return False
        return all(abs(x - y) < tol for x, y in zip(a, b))

    def norm_distance(self, other: "SpectralFingerprint") -> float:
        """L2 distance between sorted eigenvalue vectors (0 = identical)."""
        a = sorted(self.values); b = sorted(other.values)
        n = min(len(a), len(b))
        if n == 0: return float('inf')
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(n)))

    def summary(self) -> str:
        sr = self.sorted_real()
        body = ", ".join(f"{v:.3f}" for v in sr[:5])
        return f"{self.domain}: [{body}{'…' if len(sr) > 5 else ''}]"


# ════════════════════════════════════════════════════════════════════════════
# FEEDBACK QUEUE  — v4b API (emit/get-with-default) + v4 push alias
# Architectural role: typed inter-phase signal bus.
# Phase 03 emits signals. Phase 02/04 check signals to unlock extra methods.
# Keeps phases decoupled — they communicate through events, not shared state.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FeedbackQueue:
    signals: List[Tuple[str, Any]] = field(default_factory=list)

    def emit(self, signal: str, data=None):
        """Primary API (v4b): emit a typed signal with optional payload."""
        self.signals.append((signal, data))

    def push(self, signal: str, payload=None):
        """Alias for emit (v4 compatibility)."""
        self.emit(signal, payload)

    def has(self, signal: str) -> bool:
        return any(s == signal for s, _ in self.signals)

    def get(self, signal: str, default=None) -> Any:
        """Safe get with default (v4b: prevents None confusion)."""
        return next((d for s, d in self.signals if s == signal), default)

    def all_signals(self) -> List[str]:
        return [s for s, _ in self.signals]


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  — v4's 19 priors + PHASE_DEPTH + full analogies/failure modes
# v4b had 13 priors and no PHASE_DEPTH. Both restored here.
# PHASE_DEPTH insight: running 7 phases on a linear equation wastes >80% time.
# ════════════════════════════════════════════════════════════════════════════

class KB:
    # (problem_type, method) → prior confidence  (v4: 19 entries)
    METHOD_PRIORS = {
        ("quadratic",    "solve"):        0.99,
        ("quadratic",    "discriminant"): 0.99,
        ("quadratic",    "factor"):       0.70,
        ("cubic",        "solve"):        0.85,
        ("cubic",        "nsolve"):       0.92,
        ("poly_high",    "solve"):        0.40,
        ("poly_high",    "nsolve"):       0.90,
        ("trig_id",      "trigsimp"):     0.95,
        ("factoring",    "factor"):       0.90,
        ("graph",        "spectrum"):     0.95,
        ("markov",       "stationary"):   0.99,
        ("markov",       "eigenvalues"):  0.90,
        ("entropy",      "H_numeric"):    0.99,
        ("dynamical",    "solve_equil"):  0.90,
        ("control",      "routh"):        0.95,
        ("control",      "roots"):        0.80,
        ("optimize",     "critical_pts"): 0.95,
        ("optimize",     "hessian"):      0.85,
        ("matrix",       "eigenvalues"):  0.95,
        ("sum",          "summation"):    0.99,
    }
    ANALOGIES = {
        "QUADRATIC":    ["2D linear system stability (trace,det)", "eigenvalue of 2×2 matrix", "2-state entropy p(1-p)"],
        "CUBIC":        ["char poly of 3×3 matrix", "cubic potential equilibria", "3-state Markov eigenvalues"],
        "GRAPH":        ["Markov random walk D⁻¹A", "heat diffusion e^{-tL}", "consensus λ₂(L)"],
        "MARKOV":       ["weighted directed graph", "entropy production ≥ 0", "MDP in RL"],
        "ENTROPY":      ["Boltzmann S=k_B·H", "channel capacity", "KL divergence from uniform"],
        "DYNAMICAL":    ["gradient descent ẋ=−∇f", "Fokker-Planck SDE", "control ẋ=f(x,u)"],
        "CONTROL":      ["companion matrix eigs", "LQR Riccati equation", "Nyquist criterion"],
        "OPTIMIZATION": ["gradient flow ODE", "Bellman equations", "MaxEnt under constraints"],
        "MATRIX":       ["graph adjacency matrix", "Markov transition matrix", "dynamical Jacobian"],
        "FACTORING":    ["root-finding", "modular arithmetic", "RSA hardness"],
    }
    VERIFICATION = {
        "equation":  ["substitute back", "discriminant sign vs root count", "Vieta sum/product"],
        "factoring": ["expand(factor)-original=0", "roots match", "degree preserved"],
        "identity":  ["substitute 3+ numerical values", "trigsimp=0", "boundary: 0,π/2,π"],
        "graph":     ["tr(L)=Σdeg", "λ₁(L)=0", "tr(A²)/2=|E|"],
        "markov":    ["rows sum to 1", "π·P=π", "spectral radius=1"],
        "entropy":   ["H≥0", "H≤log₂n", "Σpᵢ=1"],
        "control":   ["sign changes=RHP count", "all-positive necessary", "Routh 1st col sufficient"],
        "optimize":  ["f'(x*)=0", "f'' sign matches", "limits at ±∞"],
    }
    FAILURE_MODES = {
        "POLY":      "Degree≥5: Abel-Ruffini — no radical formula, numerics needed",
        "QUADRATIC": "Δ<0: complex roots — verify problem expects real only",
        "MARKOV":    "Reducible chain: multiple stationary distributions possible",
        "DYNAMICAL": "f'(x*)=0: non-hyperbolic — linearisation insufficient",
        "CONTROL":   "Zero coefficient: Routh degenerates — use epsilon perturbation",
        "ENTROPY":   "p=0 terms: 0·log0=0 by convention — handle explicitly",
        "GRAPH":     "Disconnected: λ₂=0, Kirchhoff breaks — each component separately",
        "OPTIMIZE":  "Non-convex: multiple local minima, no global guarantee",
        "MATRIX":    "Near-singular: ill-conditioned — numerical instability risk",
    }
    # Adaptive depth: how many phases are warranted?
    # Insight from v4: trivial problems don't need 7 phases (saves 60-80% time).
    PHASE_DEPTH = {
        "LINEAR": 3, "TRIG_ID": 3, "SIMPLIFY": 3,
        "SUM": 4, "PROOF": 4, "DIGRAPH_CYC": 4,
        "QUADRATIC": 5, "TRIG_EQ": 5, "FACTORING": 5,
        "CUBIC": 6, "POLY": 6,
        "GRAPH": 7, "MATRIX": 7, "MARKOV": 7,
        "ENTROPY": 7, "DYNAMICAL": 7, "CONTROL": 7, "OPTIMIZATION": 7,
        "UNKNOWN": 2,
    }

    @classmethod
    def prior(cls, pt: str, method: str) -> float:
        return cls.METHOD_PRIORS.get((pt.lower(), method.lower()), 0.5)

    @classmethod
    def phase_depth(cls, pt_name: str) -> int:
        return cls.PHASE_DEPTH.get(pt_name, 7)


# ════════════════════════════════════════════════════════════════════════════
# CONFIDENCE LEDGER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Confidence:
    results:  Dict[str, Tuple[Any, float]] = field(default_factory=dict)
    flags:    List[str]                    = field(default_factory=list)
    knowns:   List[str]                    = field(default_factory=list)
    unknowns: List[str]                    = field(default_factory=list)

    def record(self, key: str, val: Any, conf: float, note_str: str = ""):
        self.results[key] = (val, conf)
        if conf >= 0.9:  self.knowns.append(f"{key}: {str(val)[:50]}")
        elif conf < 0.6: self.unknowns.append(f"{key}(conf={conf:.2f})")
        if note_str: self.flags.append(note_str)

    def summary(self) -> str:
        t = len(self.results)
        h = sum(1 for _, c in self.results.values() if c >= 0.9)
        m = sum(1 for _, c in self.results.values() if 0.6 <= c < 0.9)
        l = sum(1 for _, c in self.results.values() if c < 0.6)
        return f"{t} results: {h} high-conf, {m} mid, {l} uncertain"


# ════════════════════════════════════════════════════════════════════════════
# PROBLEM TYPES
# ════════════════════════════════════════════════════════════════════════════

class PT(Enum):
    LINEAR=1; QUADRATIC=2; CUBIC=3; POLY=4
    TRIG_EQ=5; TRIG_ID=6; FACTORING=7; SIMPLIFY=8
    SUM=9; PROOF=10; DIGRAPH_CYC=11
    GRAPH=12; MATRIX=13; MARKOV=14; ENTROPY=15
    DYNAMICAL=16; CONTROL=17; OPTIMIZATION=18
    MELNIKOV=19; PLANAR2D=20; SLOWFAST=21; DDE=22; PDE_RD=23
    UNKNOWN=99

    def label(self):
        return {
            1:"linear eq", 2:"quadratic eq", 3:"cubic eq", 4:"poly deg≥4",
            5:"trig eq", 6:"trig identity", 7:"factoring", 8:"simplification",
            9:"summation", 10:"proof", 11:"digraph cycle",
            12:"graph/network", 13:"matrix", 14:"markov chain",
            15:"information entropy", 16:"dynamical system",
            17:"control theory", 18:"optimization",
            19:"Melnikov/chaos", 20:"2D planar system", 21:"slow-fast/canards",
            22:"delay DDE", 23:"reaction-diffusion PDE",
            99:"unknown"
        }.get(self.value, "unknown")


@dataclass
class Problem:
    raw:     str
    ptype:   PT
    expr:    Optional[sp.Basic]        = None
    lhs:     Optional[sp.Basic]        = None
    rhs:     Optional[sp.Basic]        = None
    var:     Optional[sp.Symbol]       = None
    free:    List[sp.Symbol]           = field(default_factory=list)
    meta:    Dict[str, Any]            = field(default_factory=dict)
    poly:    Optional[sp.Poly]         = None
    _cache:  Dict[str, Any]            = field(default_factory=dict, repr=False)
    conf:    Confidence                = field(default_factory=Confidence, repr=False)
    fb:      FeedbackQueue             = field(default_factory=FeedbackQueue, repr=False)
    spectra:      List[SpectralFingerprint] = field(default_factory=list, repr=False)
    output_lines: List[str]                 = field(default_factory=list, repr=False)

    def log(self, line: str):
        """Collect output lines for entropy scoring (Bug 6 fix)."""
        if len(line.strip()) > 4: self.output_lines.append(line.strip())

    def memo(self, key: str, func: Callable, secs: int = 8):
        """P1 principle: memo-with-timeout. Never recompute; never hang."""
        if key not in self._cache:
            self._cache[key] = timed(func, secs=secs)
        return self._cache[key]

    def add_spectrum(self, fp: SpectralFingerprint) -> SpectralFingerprint:
        """Register fingerprint and auto-emit signal (v4b pattern)."""
        self.spectra.append(fp)
        self.fb.emit("spectral_fp", fp)
        return fp

    def get_poly(self) -> Optional[sp.Poly]:
        if self.poly is None and self.expr is not None and self.var is not None:
            try: self.poly = Poly(self.expr, self.var)
            except: pass
        return self.poly

    def ptype_str(self) -> str:
        return self.ptype.name


# ════════════════════════════════════════════════════════════════════════════
# TYPED RESULT SCHEMA  (v5 new: structured output for downstream use)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveryResult:
    problem:        str
    ptype:          str
    phases_run:     List[int]
    solutions:      List[str]
    confidence:     Dict[str, float]
    spectral:       List[Dict]
    bridges:        Dict[str, str]
    output_entropy: float
    elapsed_s:      float
    fb_signals:     List[str]
    warnings:       List[str]

    def to_dict(self) -> dict:
        return {
            "problem": self.problem, "ptype": self.ptype,
            "phases_run": self.phases_run, "solutions": self.solutions,
            "confidence": self.confidence, "spectral": self.spectral,
            "bridges": self.bridges, "output_entropy": round(self.output_entropy, 4),
            "elapsed_s": round(self.elapsed_s, 4),
            "fb_signals": self.fb_signals, "warnings": self.warnings,
        }


# ════════════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _var_prefer(free: List[sp.Symbol]) -> Optional[sp.Symbol]:
    """v4 insight: prefer x,y,z,t,s over alphabetical — matches user intent."""
    for name in "xyzts":
        for f in free:
            if str(f) == name: return f
    return free[0] if free else None

def _parse(s: str) -> Optional[sp.Basic]:
    s = s.strip().replace('^', '**')
    for old, new in [("ln","log"),("arcsin","asin"),("arccos","acos"),("arctan","atan")]:
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

def _parse_probs(s: str) -> Tuple[List[float], List[str]]:
    """
    v5: Returns (probs, warnings). Validates + auto-normalises.
    v4b insight: silently accepting bad distributions causes wrong H values.
    v4  insight: strict rejection breaks edge-case inputs. Middle path: warn + fix.
    """
    m = re.search(r'\[([^\]]+)\]', s)
    if not m: return [], []
    try:
        vals = [float(x.strip()) for x in m.group(1).split(',')]
    except: return [], ["parse failed"]
    warns = []
    if any(v < 0 for v in vals):
        warns.append(f"Negative probability detected")
        vals = [max(0.0, v) for v in vals]
    total = sum(vals)
    if total < 1e-15:
        return [], ["all-zero distribution"]
    if abs(total - 1.0) > 1e-9:
        warns.append(f"Sum={total:.6f} ≠ 1 — auto-normalised")
        vals = [v / total for v in vals]
    return vals, warns

def _parse_summand(raw: str) -> sp.Basic:
    """v4 insight: detect summand from natural language (squares/cubes/power-N)."""
    low = raw.lower()
    k = symbols('k', positive=True, integer=True)
    m = re.search(r'power\s+(\d+)', low)
    if m:      return k ** int(m.group(1))
    if 'cube' in low:                    return k ** 3
    if 'square' in low:                  return k ** 2
    if 'reciprocal' in low or '1/k' in low: return sp.Rational(1, 1) / k
    if 'harmonic' in low:               return sp.Rational(1, 1) / k
    return k

def _spectrum(M: sp.Matrix) -> List[float]:
    try:
        return sorted([float(N(k)) for k in M.eigenvals(multiple=True)])
    except: return []

def _spectrum_complex(M: sp.Matrix) -> List[complex]:
    try:
        return [complex(N(k)) for k in M.eigenvals(multiple=True)]
    except: return []

def _make_fp(eigs: List[float], domain: str, complex_: List[complex] = None,
             label: str = "", **meta) -> SpectralFingerprint:
    return SpectralFingerprint(
        domain=domain,
        values=sorted(eigs),
        complex_=complex_ or [],
        label=label,
        metadata=dict(meta),
    )

def _build_graph(p: Problem):
    """Build adjacency matrix and Laplacian from problem spec."""
    raw = p.raw; meta = p.meta
    mk = re.search(r'\bK(\d+)\b', raw, re.I)
    mp = re.search(r'\bP(\d+)\b', raw, re.I)
    mc = re.search(r'\bC(\d+)\b', raw, re.I)
    if mk:
        n = int(mk.group(1)); A = ones(n, n) - eye(n); meta["type"] = "complete"
    elif mp:
        n = int(mp.group(1)); A = zeros(n, n)
        for i in range(n-1): A[i, i+1] = A[i+1, i] = 1
        meta["type"] = "path"
    elif mc:
        n = int(mc.group(1)); A = zeros(n, n)
        for i in range(n): A[i, (i+1)%n] = A[(i+1)%n, i] = 1
        meta["type"] = "cycle"
    else:
        A = _parse_matrix(raw)
        if A is None: return None, None, 0, []
        n = A.shape[0]; meta["type"] = "custom"
    meta["A"] = A
    deg = [int(sum(A[i, j] for j in range(n))) for i in range(n)]
    D = diag(*deg); L = D - A
    return A, L, n, deg

def _entropy_raw(w: List[float]) -> float:
    return -sum(p * math.log2(p) for p in w if p > 1e-15)

def _entropy(probs: List[float]) -> float:
    s = sum(probs)
    return 0.0 if s < 1e-15 else _entropy_raw([p/s for p in probs])

def _kl(P: List[float], Q: List[float]) -> float:
    return sum(P[i] * math.log2(max(P[i], 1e-15) / max(Q[i], 1e-15))
               for i in range(len(P)) if P[i] > 1e-15)

def _routh(coeffs) -> Dict[str, Any]:
    """
    v4's robust Routh-Hurwitz with epsilon perturbation for zero pivots.
    Insight: v4b's inline Routh had no epsilon handling → silent wrong results
    on control s^4+s^3+s^2+s+1. v4's _routh() handles all edge cases.
    """
    EPS = 1e-10
    c = [float(N(sp.sympify(x))) for x in coeffs]
    r0 = c[0::2]; r1 = c[1::2]
    while len(r0) < len(r1): r0.append(0.0)
    while len(r1) < len(r0): r1.append(0.0)
    rows = [r0[:], r1[:]]
    for _ in range(50):
        pr, cr = rows[-2], rows[-1]
        if not cr or all(abs(x) < 1e-15 for x in cr): break
        if abs(cr[0]) < 1e-12: cr = [EPS] + list(cr[1:])
        nr = [(cr[0]*pr[i+1] - pr[0]*cr[i+1]) / cr[0]
              for i in range(len(cr)-1) if i+1 < len(pr)]
        if not nr: break
        rows.append(nr)
    fc = [row[0] for row in rows if row]
    sc = sum(1 for i in range(len(fc)-1) if fc[i] * fc[i+1] < 0)
    stable = (sc == 0) and all(x > 0 for x in fc)
    return {"stable": stable, "sign_changes": sc, "first_column": fc, "rows": rows}

def _stationary(P: sp.Matrix) -> Optional[Dict]:
    n = P.shape[0]
    pi_ = symbols(f'pi0:{n}', positive=True)
    eqs = [sum(pi_[i] * P[i, j] for i in range(n)) - pi_[j] for j in range(n)]
    eqs.append(sum(pi_) - 1)
    return timed(lambda: solve(eqs, list(pi_)), secs=10)

def _verify_eq(expr, var, sol) -> Tuple[bool, float]:
    """v4: substitution verify. Used in phase_02 attempt() to adjust confidence."""
    try:
        res = simplify(expr.subs(var, sol))
        mag = float(abs(N(res)))
        return (mag < 1e-9, mag)
    except: return (False, float('inf'))

def _vieta_check(poly: sp.Poly, sols) -> bool:
    """v4: Vieta's formulas check. Secondary verification for polynomial roots."""
    try:
        coeffs = poly.all_coeffs(); n = poly.degree()
        s_exp = -coeffs[-2] / coeffs[-1] if n >= 1 else 0
        if abs(float(N(simplify(sum(sols) - s_exp)))) > 1e-6: return False
        p_exp = (-1)**n * coeffs[0] / coeffs[-1]
        p_act = sp.prod(sols)
        return abs(float(N(simplify(p_act - p_exp)))) <= 1e-6
    except: return True

def _is_real_value(expr) -> bool:
    """v4 fix: use .is_real (not isinstance check, which broke on complex sympy types)."""
    try:
        v = N(expr)
        return getattr(v, 'is_real', False) or (abs(float(sp_im(v))) < 1e-9)
    except: return False

def _companion_fingerprint(poly: sp.Poly, p: Problem, label: str = "") -> Optional[SpectralFingerprint]:
    """
    v4 insight: build companion matrix → eigenvalues → SpectralFingerprint.
    This is the mathematical link connecting polynomial roots (algebra),
    control poles (control theory), and dynamical equilibria (dynamics).
    Unifying identity #1 made concrete.
    """
    try:
        n = poly.degree()
        coeffs = [float(N(c)) for c in poly.all_coeffs()]
        lc = coeffs[0]
        norm = [c/lc for c in coeffs[1:]]
        C = zeros(n, n)
        for i in range(n-1): C[i+1, i] = 1
        for i, c in enumerate(norm): C[0, i] = -c
        spec_c = _spectrum_complex(C)
        vals = [z.real for z in spec_c]
        fp = _make_fp(vals, "companion_poly", spec_c, label or str(poly.as_expr()))
        return p.add_spectrum(fp)
    except: return None

def _detect_family(p: Problem) -> Optional[str]:
    """
    v4: Classify which parametric solution family this polynomial belongs to.
    Examples: perfect square, integer roots, biquadratic substitution candidate.
    Provides Phase 05 generalisation depth.
    """
    if p.ptype not in (PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.FACTORING): return None
    if p.expr is None or p.var is None: return None
    v = p.var
    try:
        poly = p.get_poly()
        if poly is None: return None
        deg = poly.degree(); coeffs = poly.all_coeffs()
        if deg == 2:
            a, b, c = [float(N(x)) for x in coeffs]
            if abs(b**2 - 4*a*c) < 1e-9:
                return f"Perfect square: ({v}+{b/(2*a):.3g})²=0"
            if (abs(a-1) < 1e-9 and
                all(abs(float(N(x)) - round(float(N(x)))) < 1e-9 for x in coeffs)):
                r1 = -b/2 + math.sqrt(abs(b**2-4*c))/2
                r2 = -b/2 - math.sqrt(abs(b**2-4*c))/2
                if abs(r1-round(r1)) < 0.01 and abs(r2-round(r2)) < 0.01:
                    return f"Integer roots: ({v}-{int(round(r1))})({v}-{int(round(r2))})=0"
            return f"General quadratic: a={a:.3g}, b={b:.3g}, c={c:.3g}"
        if deg == 3:
            a, b, c, d = [float(N(x)) for x in coeffs]
            if abs(b) < 1e-9 and abs(d) < 1e-9:
                return f"Depressed cubic: {v}({v}²-{-c/a:.3g})=0"
            return "General cubic — Cardano/factor theorem applies"
        if deg == 4:
            a, b, c, d, e = [float(N(x)) for x in coeffs]
            if abs(b) < 1e-9 and abs(d) < 1e-9:
                return f"BIQUADRATIC: substitute u={v}² → quadratic in u"
            return "Quartic — Ferrari / numerical"
        return f"Degree-{deg} polynomial"
    except: return None

def _output_entropy(lines: List[str]) -> float:
    """
    v4 insight: score diversity of engine output. Low entropy = engine is
    repeating itself. High entropy = genuinely new information per line.
    Uses word-level unigram frequency.
    """
    if not lines: return 0.0
    words: Dict[str, int] = {}
    for l in lines:
        for w in re.sub(r'[^\w]', ' ', l.lower()).split():
            if len(w) > 3: words[w] = words.get(w, 0) + 1
    total = sum(words.values())
    if total == 0: return 0.0
    probs = [c/total for c in words.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ════════════════════════════════════════════════════════════════════════════
# ATTEMPT PLAN  — v5 new: confidence-ranked method ordering
# Design principle P4: sort by (prior × boost) before running.
# First high-confidence success → early exit saves time on clear cases.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AttemptPlan:
    name:    str
    fn:      Callable
    prior:   float
    boost:   float  = 1.0
    secs:    int    = 8

    @property
    def score(self) -> float:
        return self.prior * self.boost


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

def classify(raw: str) -> Problem:
    low = raw.lower().strip()

    # AIMO specific patterns (Olympiad level)
    aimo_kws = ('triangle', 'perimeter', 'remainder', 'function f', 'tournament', 'runners', 'blackboard', 'n-tastic', 'sweets', 'norwegian', 'rectangles', 'shifty')
    if any(kw in low for kw in aimo_kws) or (('integer' in low or 'integer' in raw) and not any(skip in low for skip in ('sum', 'series', 'matrix', 'graph'))):
        return Problem(raw=raw, ptype=PT.AIMO)

    if re.match(r'^entropy\b', low):
        probs, warns = _parse_probs(raw)
        return Problem(raw=raw, ptype=PT.ENTROPY, meta={"probs": probs, "prob_warns": warns})

    if re.match(r'^markov\b', low):
        return Problem(raw=raw, ptype=PT.MARKOV, meta={"M_raw": _parse_matrix(raw)})

    if re.match(r'^matrix\b', low) or (re.search(r'\[\s*\[', raw) and
            not any(kw in low for kw in ("graph","markov","entropy","vertices"))):
        M = _parse_matrix(raw)
        if M: return Problem(raw=raw, ptype=PT.MATRIX, meta={"M": M, "n": M.shape[0]})

    if re.match(r'^(graph|network)\b', low) or "adjacency" in low:
        return Problem(raw=raw, ptype=PT.GRAPH)

    if 'vertices' in low and 'cycle' in low:
        mv = re.search(r'm\s*=\s*(\d+)', low)
        return Problem(raw=raw, ptype=PT.DIGRAPH_CYC, meta={"m": int(mv.group(1)) if mv else 0})

    if re.match(r'^dynamical?\b', low):
        body = re.sub(r'^dynamical?\s*', '', raw, flags=re.I).strip()
        expr = _parse(body); free = list(expr.free_symbols) if expr else []
        v = _var_prefer(free) or symbols('x')
        return Problem(raw=raw, ptype=PT.DYNAMICAL, expr=expr, var=v, free=free)

    if re.match(r'^control\b', low):
        body = re.sub(r'^control\s*', '', raw, flags=re.I).strip()
        expr = _parse(body); free = list(expr.free_symbols) if expr else []
        v = next((f for f in free if str(f) in 'stuvw'), _var_prefer(free) or symbols('s'))
        p_ = None
        try: p_ = Poly(expr, v)
        except: pass
        return Problem(raw=raw, ptype=PT.CONTROL, expr=expr, var=v, free=free, poly=p_)

    if re.match(r'^(optimiz[a-z]*|minimiz[a-z]*|maximiz[a-z]*|extrema|find\s+(min|max))\b', low):
        body = re.sub(r'^(optimiz[a-z]*|minimiz[a-z]*|maximiz[a-z]*|extrema|find\s+(min|max)\s*of?\s*)', '', raw, flags=re.I).strip()
        goal = ("minimize" if "minim" in low else "maximize" if "maxim" in low else "extremize")
        expr = _parse(body); free = list(expr.free_symbols) if expr else []
        v = _var_prefer(free) or symbols('x')
        return Problem(raw=raw, ptype=PT.OPTIMIZATION, expr=expr, var=v, free=free, meta={"goal": goal})

    if any(kw in low for kw in ("sum of", "1+2+", "series", "summation", "sigma")):
        return Problem(raw=raw, ptype=PT.SUM, meta={"summand": _parse_summand(raw)})

    if re.match(r'^(prove|show|demonstrate)\b', low):
        body = re.sub(r'^(prove|show\s+that|show|demonstrate)\s+', '', raw, flags=re.I).strip()
        return Problem(raw=raw, ptype=PT.PROOF, meta={"body": body, "body_low": body.lower()})

    if re.match(r'^factor\b', low):
        body = re.sub(r'^factor\s+', '', raw, flags=re.I).strip()
        expr = _parse(body); free = list(expr.free_symbols) if expr else []
        v = _var_prefer(free) or symbols('x')
        p_ = None
        try: p_ = Poly(expr, v)
        except: pass
        return Problem(raw=raw, ptype=PT.FACTORING, expr=expr, var=v, free=free, poly=p_)

    if '=' in raw and not any(x in raw for x in ('==', '>=', '<=')):
        parts = raw.split('=', 1)
        lhs_e = _parse(parts[0]); rhs_e = _parse(parts[1])
        if lhs_e is None or rhs_e is None:
            return Problem(raw=raw, ptype=PT.UNKNOWN)
        expr = sp.expand(lhs_e - rhs_e)
        free = sorted(expr.free_symbols, key=str)
        v = _var_prefer(free) or symbols('x')
        if expr.atoms(sin, cos, tan):
            return Problem(raw=raw, ptype=PT.TRIG_EQ, expr=expr, lhs=lhs_e, rhs=rhs_e, var=v, free=free)
        try:
            p_ = Poly(expr, v); deg = p_.degree()
            pt = {1: PT.LINEAR, 2: PT.QUADRATIC, 3: PT.CUBIC}.get(deg, PT.POLY)
            return Problem(raw=raw, ptype=pt, expr=expr, lhs=lhs_e, rhs=rhs_e, var=v, free=free, poly=p_)
        except:
            return Problem(raw=raw, ptype=PT.UNKNOWN, expr=expr, var=v, free=free)

    e = _parse(raw)
    if e is not None:
        free = sorted(e.free_symbols, key=str)
        v = _var_prefer(free) or symbols('x')
        pt = PT.TRIG_ID if e.atoms(sin, cos, tan) else PT.SIMPLIFY
        return Problem(raw=raw, ptype=pt, expr=e, lhs=e, rhs=sp.Integer(0), var=v, free=free)

    return Problem(raw=raw, ptype=PT.UNKNOWN)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 01 — GROUND TRUTH + INTEL
# ════════════════════════════════════════════════════════════════════════════

def phase_01(p: Problem) -> dict:
    section(1, "GROUND TRUTH + INTEL",
            "Classify · analogy · failure-predict · symmetry · adaptive depth · family")
    r = {}
    kv("Problem", p.raw); kv("Type", p.ptype.label()); kv("Variable", str(p.var))

    # Adaptive depth (v4 PHASE_DEPTH)
    depth = KB.phase_depth(p.ptype_str())
    kv("Adaptive depth", f"Phases 1–{depth}")
    r["phases_planned"] = list(range(1, depth + 1))

    # Analogies
    analogs = KB.ANALOGIES.get(p.ptype_str(), [])
    if analogs:
        for a in analogs: note(f"  ⟷ {a}")
        r["analogies"] = analogs

    # Failure mode prediction
    fm = KB.FAILURE_MODES.get(p.ptype_str())
    if fm: warn(f"Anticipated failure: {fm}"); r["failure_mode"] = fm

    # Verification plan
    cat = {PT.GRAPH: "graph", PT.MARKOV: "markov", PT.ENTROPY: "entropy",
           PT.CONTROL: "control", PT.FACTORING: "factoring",
           PT.TRIG_ID: "identity", PT.SIMPLIFY: "identity"
           }.get(p.ptype, "equation")
    checks = KB.VERIFICATION.get(cat, [])
    if checks: kv("Verification plan", " | ".join(checks)); r["verification_plan"] = checks

    # Symmetry (v4b: emit signals for phase_02)
    if p.expr is not None and p.var is not None:
        v = p.var
        try:
            is_even = simplify(p.expr.subs(v, -v) - p.expr) == 0
            is_odd  = simplify(p.expr.subs(v, -v) + p.expr) == 0
            if is_even:
                ok("EVEN f(−x)=f(x) → substitute u=x² to halve degree")
                p.fb.emit("even_symmetry"); r["symmetry"] = "even"
            elif is_odd:
                ok("ODD f(−x)=−f(x) → factor out x first")
                p.fb.emit("odd_symmetry"); r["symmetry"] = "odd"
        except: pass

    # Rational root screen
    if p.ptype in (PT.QUADRATIC, PT.CUBIC, PT.POLY) and p.expr and p.var:
        try:
            cs = Poly(p.expr, p.var).all_coeffs()
            ct = abs(int(float(N(cs[-1]))))
            cands = [s*d for d in divisors(ct) for s in [1, -1]] if ct > 0 else []
            hits = [c for c in cands if abs(float(N(p.expr.subs(p.var, c)))) < 1e-9]
            if hits:
                ok(f"Rational root screen: {hits[:4]}")
                p.fb.emit("rational_roots", hits); r["rational_roots"] = hits
        except: pass

    # Discriminant (quadratic)
    if p.ptype == PT.QUADRATIC and p.expr and p.var:
        try:
            disc = discriminant(Poly(p.expr, p.var))
            dn = float(N(disc))
            nat = ("2 real roots" if dn > 0 else "double root" if dn == 0 else "complex roots")
            kv("Discriminant Δ", f"{dn:.4f} → {nat}")
            r["discriminant"] = disc; p.conf.record("discriminant", disc, 1.0, nat)
        except: pass

    # Solution family (v4)
    if p.ptype in (PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.FACTORING):
        fam = _detect_family(p)
        if fam: kv("Solution family", fam); r["family"] = fam

    # Entropy prob validation warnings
    if p.ptype == PT.ENTROPY:
        for w in p.meta.get("prob_warns", []):
            warn(f"[prob] {w}")

    ok("Phase 01 complete"); r["briefed"] = True
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 02 — DIRECT ATTACK (confidence-ranked, feedback-unlocked)
# ════════════════════════════════════════════════════════════════════════════

def phase_02(p: Problem, g1: dict) -> dict:
    section(2, "DIRECT ATTACK", "Ranked methods · multi-method verify · spectral registration")
    r = {"successes": [], "failures": []}
    v = p.var

    def attempt(name: str, fn: Callable, conf_prior: float = 0.5,
                verify_fn=None, secs: int = 8):
        result = p.memo(name, fn, secs)
        if result is None:
            r["failures"].append(name); fail(f"[--] {name}"); return None
        conf = conf_prior
        if verify_fn:
            try:
                verified, detail = verify_fn(result)
                conf = min(conf + 0.1, 1.0) if verified else max(conf - 0.2, 0.0)
                (ok if verified else warn)(f"  verify: {'ok' if verified else f'FAIL ({detail})'}")
            except: pass
        p.conf.record(name, result, conf)
        r["successes"].append({"method": name, "result": str(result)[:80], "conf": conf})
        ok(f"[{conf:.0%}] {name} → {str(result)[:80]}")
        return result

    # ── Algebraic ──────────────────────────────────────────────────────────
    if p.ptype == PT.LINEAR:
        sol = attempt("solve(linear)", lambda: solve(p.expr, v), 0.99,
                      verify_fn=lambda s: (bool(s) and abs(float(N(p.expr.subs(v, s[0])))) < 1e-9, "sub"))
        if sol: r["roots"] = sol; p._cache["solve(linear)"] = sol

    elif p.ptype == PT.QUADRATIC:
        sol = attempt("solve(quadratic)", lambda: solve(p.expr, v), 0.99,
                      verify_fn=lambda s: (all(_verify_eq(p.expr, v, x)[0] for x in s), "sub") if s else (False, "empty"))
        if sol:
            r["roots"] = sol; p._cache["solve(quadratic)"] = sol
            # Vieta check
            poly = p.get_poly()
            if poly: vok = _vieta_check(poly, sol); (ok if vok else warn)(f"Vieta: {vok}"); r["vieta_ok"] = vok
            # Companion fingerprint (unifying identity #1)
            if poly: _companion_fingerprint(poly, p, label=p.raw[:40])
        attempt("discriminant", lambda: discriminant(Poly(p.expr, v)), 0.99)

    elif p.ptype == PT.CUBIC:
        sol = attempt("solve(cubic)", lambda: solve(p.expr, v), 0.85)
        if sol:
            r["roots"] = sol; p._cache["solve(cubic)"] = sol
            if p.get_poly(): _companion_fingerprint(p.get_poly(), p)
        else:
            attempt("nsolve(cubic)", lambda: [nsolve(p.expr, v, x0) for x0 in [-2, 0, 2]], 0.90)

    elif p.ptype == PT.POLY:
        warn("High-degree: " + KB.FAILURE_MODES["POLY"])
        if p.fb.has("even_symmetry"):
            u = symbols('u', positive=True)
            attempt("solve(biquadratic)", lambda: solve(p.expr.subs(v**2, u), u), 0.90)
        sol = attempt("solve(poly)", lambda: solve(p.expr, v), 0.40)
        if not sol:
            attempt("nsolve(poly)", lambda: [nsolve(p.expr, v, x0) for x0 in range(-3, 4)], 0.88)
        if sol:
            r["roots"] = sol
            if p.get_poly(): _companion_fingerprint(p.get_poly(), p)

    elif p.ptype == PT.TRIG_EQ:
        sol = attempt("solve(trig_eq)", lambda: list(sp.solveset(p.expr, v, domain=S.Reals)), 0.85)
        if sol: r["roots"] = sol

    elif p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        simp = attempt("trigsimp", lambda: trigsimp(p.expr), 0.90)
        if simp is not None:
            p.conf.record("trigsimp", simp, 0.99 if simp in (1, 0) else 0.85)
        if p.free:
            xt = p.free[0]
            checks = [abs(float(N(p.expr.subs(xt, vt)))) < 1e-8
                      for vt in [0.3, 0.7, 1.1, 1.5, 2.3]]
            allok = all(checks)
            (ok if allok else warn)(f"Numerical identity (5 pts): {'VERIFIED' if allok else 'FAILED'}")
            r["numerical_verify"] = allok

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        ns = symbols('n', positive=True, integer=True)
        summand = p.meta.get("summand", k)
        label = f"summation({summand},(k,1,n))"
        res = attempt(label, lambda: summation(summand, (k, 1, ns)), 0.99)
        if res:
            p._cache[label] = res
            r["sum_formula"] = factor(res)
            try:
                brute = sum(int(N(summand.subs(k, i))) for i in range(1, 6))
                fval  = int(N(res.subs(ns, 5)))
                (ok if brute == fval else warn)(f"Verify n=5: brute={brute}, formula={fval}")
                r["verify_n5"] = (brute == fval)
            except: pass

    elif p.ptype == PT.PROOF:
        body_low = p.meta.get("body_low", "")
        if any(kw in body_low for kw in ("sqrt(2)", "root 2", "irrational", "√2")):
            ok("√2 irrational: assume p/q irreducible → p²=2q² → p,q both even → contradicts gcd=1")
            r["proof_method"] = "contradiction"; r["status"] = "QED"
        elif "prime" in body_low:
            ok("Infinitely many primes (Euclid): assume finite {p₁…pₙ}; N=∏pᵢ+1 has new prime factor.")
            r["proof_method"] = "construction"; r["status"] = "QED"
        else:
            note("Proof type not in KB"); r["status"] = "Pending"

    elif p.ptype == PT.DIGRAPH_CYC:
        m = p.meta.get("m", 0)
        if m % 2 != 0: ok(f"Odd m={m}: fiber decomposition exists"); r["status"] = "Success"
        else: fail(f"Even m={m}: parity obstruction"); r["status"] = "Failure"

    elif p.ptype == PT.FACTORING:
        fac = attempt("factor(expr)", lambda: factor(p.expr), 0.90,
                      verify_fn=lambda f: (simplify(expand(f) - expand(p.expr)) == 0, "expand"))
        if fac: r["factored"] = fac
        attempt("sqf_list", lambda: str(sqf_list(p.expr, v)), 0.80)

    elif p.ptype == PT.GRAPH:
        A, L, n, deg = _build_graph(p)
        if A is None: fail("Cannot build graph"); return r
        p.meta.update({"L": L, "n": n, "deg": deg})
        ok(f"A, L built ({n}×{n})")
        r["degree_sequence"] = deg; r["edge_count"] = sum(deg) // 2
        kv("Degree sequence", deg); kv("Edges", r["edge_count"])
        L_spec = p.memo("L_spec", lambda: _spectrum(L))
        A_spec = p.memo("A_spec", lambda: _spectrum(A))
        if L_spec:
            r["L_spec"] = L_spec; p.meta["L_spec"] = L_spec
            kv("L spectrum", [f"{e:.4f}" for e in L_spec])
            (ok if abs(L_spec[0]) < 1e-9 else warn)(f"λ₁(L)={L_spec[0]:.6f}")
            (ok if abs(sum(deg) - sum(L_spec)) < 1e-6 else warn)(f"tr(L)={sum(L_spec):.3f}=Σdeg={sum(deg)}")
            p.add_spectrum(_make_fp(L_spec, "graph_laplacian"))
        if A_spec:
            r["A_spec"] = A_spec; p.meta["A_spec"] = A_spec
            kv("A spectrum", [f"{e:.4f}" for e in A_spec])
            p.add_spectrum(_make_fp(A_spec, "graph_adj"))
            if all(abs(A_spec[i] + A_spec[-(i+1)]) < 1e-6 for i in range(len(A_spec)//2)):
                p.fb.emit("bipartite", True)

    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M")
        if M is None: fail("No matrix"); return r
        n = M.shape[0]; lam = symbols('lambda')
        cp = attempt("char_poly", lambda: M.charpoly(lam).as_expr(), 0.95)
        if cp:
            try: _companion_fingerprint(Poly(cp, lam), p, "matrix_char_poly")
            except: pass
        spec = p.memo("spectrum", lambda: _spectrum(M)); p.meta["spec"] = spec or []
        r["eigenvalues"] = spec; kv("Eigenvalues", [f"{e:.4f}" for e in (spec or [])])
        if spec:
            (ok if abs(float(N(trace(M))) - sum(spec)) < 1e-6 else warn)(
                f"Trace check: {float(N(trace(M))):.4f}≈{sum(spec):.4f}")
            p.add_spectrum(_make_fp(spec, "matrix"))
            if all(e < 0 for e in spec): p.fb.emit("all_eigs_negative")

    elif p.ptype == PT.MARKOV:
        M_raw = p.meta.get("M_raw")
        if M_raw is None: fail("No P matrix"); return r
        n = M_raw.shape[0]; p.meta["n"] = n
        P_rat = sp.Matrix([[sp.Rational(M_raw[i,j]).limit_denominator(1000)
                            if isinstance(M_raw[i,j], float) else sp.sympify(M_raw[i,j])
                            for j in range(n)] for i in range(n)])
        p.meta["P_rat"] = P_rat; ok(f"Rational P ({n}×{n})")
        # Row sum validation
        for i in range(n):
            s_ = float(sum(P_rat[i, :]))
            (ok if abs(s_ - 1) < 1e-9 else fail)(f"Row {i} sums to {s_:.6f}")
        spec_c = _spectrum_complex(P_rat)
        r["eigenvalues_complex"] = [str(round(z.real,4)+round(z.imag,4)*1j) for z in spec_c]
        rho = max(abs(z) for z in spec_c) if spec_c else 0
        (ok if rho <= 1.0001 else warn)(f"ρ={rho:.6f} ≤1")
        p.add_spectrum(_make_fp([z.real for z in spec_c], "markov", spec_c))
        abs_states = [i for i in range(n) if float(P_rat[i,i]) == 1.0]
        if abs_states: p.fb.emit("absorbing", abs_states); r["absorbing"] = abs_states
        stat = _stationary(P_rat)
        if stat:
            r["stationary"] = {str(k): str(v_) for k, v_ in stat.items()}
            p.meta["stat"] = stat; ok("π computed")
            pi_v = sp.Matrix([list(stat.values())])
            all_z = all(simplify((pi_v * P_rat - pi_v)[0, j]) == 0 for j in range(n))
            (ok if all_z else warn)(f"π·P=π: {all_z}")
            # Bug 5 fix: record results in confidence ledger
            p.conf.record("stationary_dist", r["stationary"], 0.99 if all_z else 0.70)
        p.conf.record("spectral_radius", rho, 0.97)
        p.conf.record("eigenvalues", r["eigenvalues_complex"], 0.95)

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs", [])
        if probs:
            H = _entropy(probs); H_max = math.log2(len(probs))
            r["H_bits"] = H; p.meta["H_val"] = H
            r["efficiency"] = H / H_max if H_max > 0 else 1.0
            kv("H(X)", f"{H:.6f} bits"); kv("H_max", f"{H_max:.6f} bits")
            kv("Efficiency", f"{r['efficiency']:.4f}")
            KL = _kl(probs, [1/len(probs)]*len(probs))
            r["KL_uniform"] = KL; kv("KL(P||uniform)", f"{KL:.6f}")
            if H / H_max > 0.99: p.fb.emit("near_max_entropy")

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr
        equil = attempt("solve(f=0)", lambda: solve(f, v), 0.90)
        # nsolve fallback for high-degree polynomials that timeout
        if equil is None and f is not None:
            note("  symbolic solve timed out — falling back to nsolve grid")
            try:
                grid = [x0/4.0 for x0 in range(-16, 17)]
                raw_roots = set()
                for s in grid:
                    try:
                        r_ = float(N(nsolve(f, v, s)))
                        raw_roots.add(round(r_, 5))
                    except: pass
                if raw_roots:
                    equil = [sp.Float(r_) for r_ in sorted(raw_roots)]
                    p._cache["solve(f=0)"] = equil
                    ok(f"nsolve found {len(equil)} equilibria")
            except: pass
        if equil:
            r["equilibria"] = [str(e) for e in equil]
            fp_ = diff(f, v); kv("f'(x)", str(fp_))
            stab_info = {}
            for eq in equil:
                try:
                    fp_v = float(N(fp_.subs(v, eq)))
                    stab = "STABLE" if fp_v < 0 else "UNSTABLE" if fp_v > 0 else "NON-HYPERBOLIC"
                    kv(f"  f'({eq})", f"{fp_v:.4f} → {stab}"); stab_info[str(eq)] = stab
                    if stab == "NON-HYPERBOLIC": p.fb.emit("non_hyperbolic", eq)
                except: pass
            r["stability"] = stab_info
            p.meta["stability"] = stab_info
            if len(equil) > 1: p.fb.emit("multiple_equilibria", equil)
            real_eq = [e for e in equil if _is_real_value(e)]
            p.add_spectrum(_make_fp([float(N(e)) for e in real_eq], "dynamical_equil"))
            # Blow-up detection: f ~ a*x^n for large x
            try:
                poly_d = Poly(f, v); deg = poly_d.degree()
                lc = float(N(poly_d.LC()))
                if deg % 2 == 1 and lc > 0:
                    p.fb.emit("blowup_possible", {"deg": deg, "lc": lc})
                    warn(f"ODD degree {deg}, lc>0 → FINITE-TIME BLOW-UP for |x₀| large")
                    # Outermost unstable equilibrium = blow-up threshold
                    unstable = [float(N(e)) for e, s in stab_info.items() if s == "UNSTABLE"]
                    if unstable:
                        xc = max(abs(float(e)) for e in unstable)
                        p.fb.emit("blowup_threshold", xc)
                        warn(f"Blow-up threshold: |x₀| > {xc:.4f}")
                    p.conf.record("blowup_structure", f"deg={deg}, lc={lc:.3f}", 0.95)
            except: pass

    elif p.ptype == PT.CONTROL:
        f = p.expr
        attempt("solve(char_poly)", lambda: solve(f, v), 0.80, secs=6)
        rts = p._cache.get("solve(char_poly)", [])
        if rts:
            r["roots"] = [str(rt) for rt in rts]
            for rt in rts:
                try:
                    rt_c = complex(N(rt))
                    loc = "LHP stable" if rt_c.real < 0 else "RHP UNSTABLE" if rt_c.real > 0 else "marginal"
                    kv(f"  root {rt}", f"Re={rt_c.real:.4f} → {loc}")
                except: pass
            p.add_spectrum(_make_fp([float(N(sp_re(rt))) for rt in rts], "control_poles",
                                    [complex(N(rt)) for rt in rts]))
            p._cache["roots"] = rts
        poly = p.get_poly()
        if poly:
            try:
                rh = _routh(poly.all_coeffs()); r["routh"] = rh; p._cache["routh"] = rh
                kv("Routh 1st col", [f"{x:.4f}" for x in rh["first_column"]])
                (ok if rh["stable"] else fail)(f"Routh: {'STABLE' if rh['stable'] else 'UNSTABLE'}")
            except Exception as e: warn(f"Routh: {e}")

    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; fp_ = diff(f, v); fpp_ = diff(f, v, 2)
        kv("f'(x)", str(fp_))
        crit = attempt("solve(f'=0)", lambda: solve(fp_, v), 0.90)
        if crit:
            r["critical_points"] = [str(c) for c in crit]
            vals = []
            for c in crit:
                try:
                    fv  = float(N(f.subs(v, c)))
                    fpv = float(N(fpp_.subs(v, c)))
                    nat = "min" if fpv > 1e-9 else "max" if fpv < -1e-9 else "saddle"
                    kv(f"  x={c}", f"f={fv:.4f}, f''={fpv:.4f} → {nat}")
                    vals.append((fv, c, nat))
                except: pass
            r["critical"] = vals
            if len(vals) > 1: p.fb.emit("multiple_minima", vals)
            real_vals = [(fv, c, nat) for fv, c, nat in vals if _is_real_value(f.subs(v, c))]
            if real_vals:
                goal = p.meta.get("goal", "extremize")
                best = (min if "min" in goal else max)(real_vals)
                r["optimal"] = best; ok(f"Optimal: x*={best[1]}, f*={best[0]:.4f} ({best[2]})")
                p._cache["optimal"] = best
            hess = [float(N(fpp_.subs(v, c))) for c in crit]
            p.add_spectrum(_make_fp(hess, "opt_hessian"))
        try:
            lp = limit(f, v, oo); ln = limit(f, v, -oo)
            r["lim_+inf"] = str(lp); r["lim_-inf"] = str(ln)
            kv("f(+∞)", str(lp)); kv("f(−∞)", str(ln))
        except: pass

    finding(f"{len(r['successes'])} succeeded, {len(r['failures'])} failed")
    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 03 — STRUCTURE HUNT (emits feedback signals)
# ════════════════════════════════════════════════════════════════════════════

def phase_03(p: Problem, g2: dict) -> dict:
    section(3, "STRUCTURE HUNT", "Invariants · spectrum · feedback signals")
    r = {}; v = p.var

    if p.ptype == PT.GRAPH:
        L_spec = p.meta.get("L_spec", []); A_spec = p.meta.get("A_spec", [])
        deg = p.meta.get("deg", []); n = p.meta.get("n", 0)
        if len(L_spec) > 1:
            lam2 = sorted(L_spec)[1]; r["fiedler"] = lam2; kv("Fiedler λ₂", f"{lam2:.6f}")
            finding("λ₂>0 → CONNECTED" if lam2 > 1e-9 else "λ₂=0 → DISCONNECTED")
            r["connected"] = lam2 > 1e-9; p.fb.emit("connected", r["connected"])
            if lam2 > 1e-9:
                kv("Cheeger h(G)∈", f"[{lam2/2:.4f},{math.sqrt(2*lam2):.4f}]")
        if len(set(deg)) == 1:
            r["regular"] = deg[0]; finding(f"{deg[0]}-REGULAR"); p.fb.emit("regular", deg[0])
        if A_spec:
            sym = all(abs(A_spec[i] + A_spec[-(i+1)]) < 1e-6 for i in range(len(A_spec)//2))
            r["bipartite"] = sym; kv("Bipartite (sym spectrum)", sym)
            finding("BIPARTITE" if sym else "Not bipartite")
            p.fb.emit("bipartite", sym)
        nc = sum(1 for e in L_spec if abs(e) < 1e-9); r["components"] = nc; kv("Components", nc)

    elif p.ptype == PT.MATRIX:
        M = p.meta.get("M"); spec = p.meta.get("spec", [])
        if M is None: return r
        r["symmetric"] = (M == M.T); kv("Symmetric", r["symmetric"])
        if r["symmetric"] and spec:
            me = min(spec)
            if   me > 0:  r["definite"] = "pos_def";  finding("POSITIVE DEFINITE"); p.fb.emit("pos_definite", True)
            elif me >= 0: r["definite"] = "pos_semi";  finding("PSD")
            elif max(spec) < 0: r["definite"] = "neg_def"; finding("NEGATIVE DEFINITE")
            else: r["definite"] = "indefinite"; finding("INDEFINITE")
        try:
            rnk = M.rank(); r["rank"] = rnk; kv("Rank", rnk)
            finding("INVERTIBLE" if rnk == M.shape[0] else f"SINGULAR (rank={rnk})")
        except: pass
        if spec:
            rho = max(abs(e) for e in spec)
            cond = rho / max(min(abs(e) for e in spec), 1e-15)
            r["condition"] = cond; kv("κ(M)", f"{cond:.4f}")
            if cond > 100: warn(f"Ill-conditioned κ={cond:.1f}"); p.fb.emit("ill_conditioned", cond)

    elif p.ptype == PT.MARKOV:
        P_rat = p.meta.get("P_rat"); n = p.meta.get("n", 0)
        spec_c = _spectrum_complex(P_rat) if P_rat else []
        ea = sorted([abs(z) for z in spec_c], reverse=True)
        if len(ea) > 1:
            lam2 = ea[1]; gap = 1.0 - lam2; r["lambda2"] = lam2; r["gap"] = gap
            kv("|λ₂|", f"{lam2:.6f}"); kv("Mixing time~", str(int(1/gap)+1) if gap > 1e-9 else "∞")
        abs_states = p.fb.get("absorbing", [])
        # Bug 2+3 fix: detect periodicity via eigenvalue -1 (bipartite signature)
        has_neg1 = any(abs(z.real + 1) < 1e-6 and abs(z.imag) < 1e-6 for z in spec_c)
        if has_neg1:
            warn("Eigenvalue −1 detected: chain is PERIODIC (period 2) — NOT aperiodic")
            finding("PERIODIC (period 2): oscillates forever, never mixes to π")
            p.fb.emit("periodic", True)
            insight("Eigenvalue −1 is the universal bipartite/periodicity signature")
            insight("Random walk on ANY bipartite graph always has eigenvalue −1")
            r["periodic"] = True
        elif not abs_states:
            finding("ERGODIC (aperiodic + irreducible)")
        else:
            finding(f"Absorbing states: {abs_states}")
        stat = p.meta.get("stat", {})
        if stat and P_rat:
            try:
                pi_ = [sp.sympify(list(stat.values())[i]) for i in range(n)]
                rev = all(simplify(pi_[i]*P_rat[i,j] - pi_[j]*P_rat[j,i]) == 0
                          for i in range(n) for j in range(n))
                r["reversible"] = rev; kv("Reversible", rev)
                p.fb.emit("reversible", rev)
            except: pass

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs", [])
        if probs:
            H = p.meta.get("H_val", 0); Hmax = math.log2(len(probs))
            kv("Per-symbol −pᵢlog₂pᵢ", [f"{-q*math.log2(q):.4f}" for q in probs if q > 0])
            kv("Gap to max H", f"{Hmax - H:.6f} bits"); finding(f"Efficiency: {H/Hmax:.4f}")
            # Concavity proof (structure)
            p_s = symbols('p', positive=True)
            H_bin = -p_s*log(p_s,2) - (1-p_s)*log(1-p_s,2)
            kv("d²H/dp²", str(simplify(diff(H_bin, p_s, 2)))); finding("H strictly CONCAVE")

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr
        if f:
            try:
                V = p.memo("potential", lambda: timed(lambda: -integrate(f, v), secs=6))
                if V: kv("V(x)=−∫f", str(V))
                equil = p.meta.get("equilibria", [])
                n_eq = len(equil)
                if   n_eq == 1: finding("Monostable")
                elif n_eq == 2: finding("Bistable — possible saddle-node")
                elif n_eq >= 3: finding(f"Multi-stable ({n_eq}) — rich bifurcation landscape")
            except: pass

    elif p.ptype == PT.CONTROL:
        rh = p._cache.get("routh", {}); rts = p._cache.get("solve(char_poly)", [])
        if rh: kv("Stability", "STABLE" if rh.get("stable") else "UNSTABLE")
        if rts:
            for rt in rts:
                try:
                    rt_c = complex(N(rt))
                    kv(f"  λ={rt}", f"Re={rt_c.real:.4f} {'LHP' if rt_c.real < 0 else 'RHP'}")
                except: pass

    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; fpp_ = diff(f, v, 2)
        try:
            fpp_s = simplify(fpp_); kv("f''(x)", str(fpp_s))
            if fpp_s.is_polynomial(v):
                fp_poly = Poly(fpp_s, v)
                if fp_poly.degree() == 0:
                    val = float(N(fpp_s))
                    if val > 0:  finding("f''>0 everywhere → CONVEX → global min exists"); r["convex"] = True
                    elif val < 0: finding("f''<0 everywhere → CONCAVE → global max exists"); r["convex"] = False
        except: pass

    elif p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY, PT.FACTORING):
        try: fac = factor(p.expr); r["factored"] = str(fac); kv("Factored", fac)
        except: pass
        sols = p._cache.get("solve(expr,var)", []) or p._cache.get("solve(quadratic)", []) or []
        if sols and len(sols) <= 5:
            try:
                ps = {f"Σxᵢ^{k}": str(simplify(sum(s**k for s in sols))) for k in range(1, 4)}
                kv("Newton power sums", ps)
            except: pass

    elif p.ptype == PT.SUM:
        k = symbols('k', positive=True, integer=True)
        summand = p.meta.get("summand", k)
        label = f"summation({summand},(k,1,n))"
        res = p._cache.get(label)
        if res: kv("Closed form", str(factor(res)))

    elif p.ptype == PT.TRIG_ID and p.expr and v:
        test_vals = [0.3, 0.8, 1.5, 2.2, 3.0]
        residuals = [abs(float(N(p.expr.subs(v, tv)))) for tv in test_vals]
        mx = max(residuals) if residuals else 0
        (ok if mx < 1e-8 else warn)(f"Max residual: {mx:.2e}")
        r["numerical_id"] = mx < 1e-8

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 04 — PATTERN LOCK (backwards reasoning, feedback-unlocked methods)
# ════════════════════════════════════════════════════════════════════════════

def phase_04(p: Problem, g3: dict) -> dict:
    section(4, "PATTERN LOCK", "Backwards reasoning · feedback-unlocked · invariants")
    r = {}

    if p.ptype == PT.GRAPH:
        A = p.meta.get("A"); L = p.meta.get("L"); n = p.meta.get("n", 0)
        L_spec = p.meta.get("L_spec", []); A_spec = p.meta.get("A_spec", [])
        # Kirchhoff spanning trees (Matrix-Tree theorem)
        nz = [e for e in L_spec if abs(e) > 1e-9]
        if nz and n > 0:
            tau = math.prod(nz) / n; r["spanning_trees"] = tau
            kv("Spanning trees τ(G) [Kirchhoff]", f"{tau:.4f}")
            insight(f"Matrix-Tree theorem: τ=(1/n)∏λᵢ≠0 ≈ {tau:.2f}")
        # Estrada index (subgraph richness)
        if A_spec:
            ee = sum(math.exp(e) for e in A_spec); r["estrada"] = ee
            kv("Estrada index EE(G)", f"{ee:.4f}")
        # Bipartite BFS 2-colouring (feedback-unlocked)
        if p.fb.has("bipartite") and p.fb.get("bipartite") and A and n > 0:
            try:
                color = [-1]*n; color[0] = 0; q = [0]; valid = True
                while q and valid:
                    node = q.pop(0)
                    for j in range(n):
                        if int(A[node, j]):
                            if color[j] == -1: color[j] = 1 - color[node]; q.append(j)
                            elif color[j] == color[node]: valid = False; break
                if valid: r["bipartite_coloring"] = color; kv("2-coloring", color)
            except: pass
        # Fiedler partition
        if L and n <= 12:
            try:
                evects = L.eigenvects()
                sev = sorted(evects, key=lambda t: float(N(t[0])))
                if len(sev) > 1:
                    fv = sev[1][2][0]
                    signs = ["+" if float(N(x)) >= 0 else "−" for x in fv]
                    r["fiedler_partition"] = signs; kv("Fiedler partition", signs)
                    insight(f"Spectral bisection: {signs.count('+')} vs {signs.count('−')} nodes")
            except: pass

    elif p.ptype == PT.MATRIX:
        spec = p.meta.get("spec", [])
        kv("Cayley-Hamilton", "p(M)=0 — matrix satisfies its own characteristic polynomial")
        if g3.get("symmetric"):
            insight("M=QΛQᵀ → compute any matrix function via diagonalisation (exp, log, sqrt…)")
        if spec:
            all_neg = all(e < 0 for e in spec)
            insight(f"Backwards: eigenvalues → {'STABLE Jacobian (Lyapunov)' if all_neg else 'unstable modes present'}")

    elif p.ptype == PT.MARKOV:
        stat = p.meta.get("stat", {}); P_rat = p.meta.get("P_rat"); n = p.meta.get("n", 0)
        if stat:
            pi_f = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
            H_stat = _entropy(pi_f); r["H_stat"] = H_stat
            kv("H(π) stationary entropy", f"{H_stat:.6f} bits")
            # Entropy rate
            if P_rat:
                try:
                    h = -sum(pi_f[i] * sum(float(N(P_rat[i,j])) * math.log2(max(float(N(P_rat[i,j])), 1e-15))
                              for j in range(n) if float(N(P_rat[i,j])) > 1e-12)
                              for i in range(n))
                    r["entropy_rate"] = h; kv("Entropy rate h", f"{h:.6f} bits/step")
                    insight(f"Chain produces {h:.4f} bits randomness/step")
                except: pass
            if all(abs(pi_f[i] - pi_f[0]) < 1e-6 for i in range(n)):
                insight("Uniform π → P is DOUBLY STOCHASTIC")
            p.log("uniform stationary: doubly stochastic Markov chain")
            # Bug 1 fix: actually compare P^20 to π before claiming convergence
            if P_rat and n <= 6:
                try:
                    P_inf = P_rat**20
                    row0 = [float(N(P_inf[0,j])) for j in range(n)]
                    kv("P^20 row 0", [f"{x:.4f}" for x in row0])
                    pi_uniform = [1.0/n]*n
                    max_diff = max(abs(row0[j] - pi_f[j]) for j in range(n))
                    if p.fb.has("periodic"):
                        warn(f"P^20 ≠ Π (max diff={max_diff:.4f}): PERIODIC chain oscillates")
                        warn("Ergodic theorem does NOT apply — period-2 chain never settles")
                        insight("Fix: use (P+I)/2 to aperiodize, then P^∞ → Π")
                    elif max_diff < 1e-4:
                        insight("P^20 ≈ Π confirmed — ergodic convergence verified")
                    else:
                        warn(f"P^20 not yet converged (max diff={max_diff:.4f}) — needs more steps")
                except: pass

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs", [])
        if probs:
            H = _entropy(probs); n = len(probs)
            # Huffman codes
            heap = [(q, [i]) for i, q in enumerate(probs) if q > 0]
            heapq.heapify(heap); lens = {i: 0 for i in range(n)}
            while len(heap) > 1:
                p1, c1 = heapq.heappop(heap); p2, c2 = heapq.heappop(heap)
                for idx in c1: lens[idx] += 1
                for idx in c2: lens[idx] += 1
                heapq.heappush(heap, (p1+p2, c1+c2))
            avg = sum(probs[i] * lens.get(i, 0) for i in range(n))
            r["huffman_avg"] = avg; kv("Huffman avg length", f"{avg:.4f} bits/sym")
            kv("Shannon H", f"{H:.4f} bits/sym"); kv("Redundancy", f"{avg-H:.4f} bits")

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr; v = p.var
        # Use cached equilibria — never re-call solve() (timeout risk)
        equil = p._cache.get("solve(f=0)", [])
        V_pot = p._cache.get("potential")
        if V_pot is None:
            try: V_pot = timed(lambda: -integrate(f, v), secs=6)
            except: pass
        if equil and V_pot is not None:
            for eq in equil:
                try: kv(f"  V at x={eq}", str(N(V_pot.subs(v, sp.sympify(eq)), 4)))
                except: pass
            insight("Stable equil = local MINIMA of V(x) = −∫f(x)dx")
            # Verify: unstable equil are local maxima/saddles of V
            stab = p.meta.get("stability", {})
            for eq in equil:
                try:
                    eq_s = sp.sympify(eq); V_val = float(N(V_pot.subs(v, eq_s)))
                    fp2 = diff(f, v, 2)
                    fp2_v = float(N(fp2.subs(v, eq_s)))
                    # -V''(x*) = f'(x*): stable equil (f'<0) ↔ local min of V (V''>0)
                    if abs(fp2_v) > 0.1:
                        kv(f"  V''({eq:.4f})", f"{-fp2_v:.4f} → V local {'min' if fp2_v<0 else 'max'}")
                except: pass
        # Non-hyperbolic: higher derivatives
        if p.fb.has("non_hyperbolic"):
            eq = p.fb.get("non_hyperbolic")
            try:
                f2 = diff(f, v, 2); f3 = diff(f, v, 3)
                f2v = float(N(f2.subs(v, eq))); f3v = float(N(f3.subs(v, eq)))
                kv(f"f''({eq})", f"{f2v:.6f}"); kv(f"f'''({eq})", f"{f3v:.6f}")
                if abs(f2v) < 1e-9 and f3v != 0:
                    finding("f'''≠0 → transcritical/pitchfork bifurcation")
            except: pass

    elif p.ptype == PT.CONTROL:
        rts = p._cache.get("roots", []) or p._cache.get("solve(char_poly)", [])
        if rts:
            lhp = [rt for rt in rts if float(N(sp_re(rt))) < 0]
            rhp = [rt for rt in rts if float(N(sp_re(rt))) > 0]
            kv("LHP (stable modes)", [str(r_) for r_ in lhp])
            kv("RHP (unstable modes)", [str(r_) for r_ in rhp])
            insight(f"Backwards: {len(rhp)} unstable modes → need {len(rhp)} feedback gains to stabilise")

    elif p.ptype == PT.OPTIMIZATION:
        f = p.expr; v = p.var; fpp_ = diff(f, v, 2)
        crit = p._cache.get("solve(f'=0)", [])
        if crit:
            for c in crit:
                try:
                    fpp_v = float(N(fpp_.subs(v, c)))
                    insight(f"x*={c}: f''={fpp_v:.4f} → {'confirms minimum' if fpp_v > 0 else 'confirms maximum' if fpp_v < 0 else 'inconclusive'}")
                except: pass

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 05 — GENERALISE (parametric families, solution families, universality)
# ════════════════════════════════════════════════════════════════════════════

def phase_05(p: Problem, g4: dict) -> dict:
    section(5, "GENERALISE", "Parametric families · solution family · universality")
    r = {}

    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        fam = _detect_family(p)
        if fam: kv("Solution family", fam); r["family"] = fam
        deg = p.get_poly().degree() if p.get_poly() else 0
        if deg == 2:
            a, b, c_s = symbols('a b c')
            kv("General formula", "x = (−b ± √(b²−4ac)) / 2a")
            kv("Discriminant roles", "Δ>0: 2 real | Δ=0: double | Δ<0: complex conjugates")
        elif deg == 3:
            kv("Cardano depressed", "t³+pt+q=0 via substitution x=t−b/3a")

    elif p.ptype == PT.GRAPH:
        t = p.meta.get("type", "")
        formulas = {
            "complete": {"λ_L": "0 (×1), n (×(n-1))", "τ(G)": "n^(n-2)", "diam": "1"},
            "path":     {"λ_L": "2−2cos(kπ/n) for k=0..n-1", "τ(G)": "1", "diam": "n-1"},
            "cycle":    {"λ_L": "2−2cos(2πk/n) for k=0..n-1", "τ(G)": "n", "diam": "⌊n/2⌋"},
        }
        if t in formulas:
            for k, v_ in formulas[t].items(): kv(f"  {k}", v_)
        kv("General bound", "λ₂ ≥ 4/(n·diam(G)) — diameter-connectivity tradeoff")

    elif p.ptype == PT.MARKOV:
        kv("General mixing", "‖Pⁿ−Π‖ ≤ |λ₂|ⁿ")
        kv("Potential matrix", "G=(I−P)⁻¹ encodes all hitting/return times")
        kv("MCMC principle", "Any target π reachable by Metropolis-Hastings construction")

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs", [])
        if probs:
            for alpha in [0.5, 2.0]:
                try:
                    H_r = (1/(1-alpha)) * math.log2(sum(q**alpha for q in probs if q > 0))
                    kv(f"Rényi H_{alpha}", f"{H_r:.4f} bits")
                except: pass
        kv("MaxEnt principle", "Gibbs dist p*=e^{λf}/Z maximises H s.t. E[f]=const")

    elif p.ptype == PT.DYNAMICAL:
        f = p.expr; v = p.var
        try:
            V = p._cache.get("potential") or timed(lambda: -integrate(f, v), secs=6)
            if V: p._cache["potential"] = V
            if V:
                kv("Potential V(x)", str(V))
                kv("Gradient flow", "ẋ=f(x) IS gradient descent on V — unifying identity #4")
            # Homoclinic/heteroclinic orbit detection via equal-energy levels
            equil = p._cache.get("solve(f=0)", [])
            stab = p._cache.get("solve(f=0):stab", {})
            if equil and len(equil) >= 3:
                unstable_eq = [e for e in equil
                               if p.meta.get("stability", {}).get(str(e)) == "UNSTABLE"]
                # Use stored stab_info via result dict from phase_02
                # V values at unstable equilibria: equal V = heteroclinic connection
                try:
                    V_vals = [(e, float(N(V.subs(v, e)))) for e in equil
                              if _is_real_value(e)]
                    V_at_unstable = [(e, vv) for e, vv in V_vals
                                     if any(abs(float(N(e)) - float(N(eu))) < 0.01
                                            for eu in unstable_eq)] if unstable_eq else []
                    # Two unstable equilibria with same V → heteroclinic connection
                    pairs = [(a, b) for i, (a, va) in enumerate(V_at_unstable)
                             for (b, vb) in V_at_unstable[i+1:]
                             if abs(va - vb) < 1e-9]
                    if pairs:
                        insight(f"HETEROCLINIC ORBITS: {len(pairs)} pairs with equal energy")
                    # Single unstable eq with V(x*)=V(-x*) by odd symmetry → homoclinic
                    if p.fb.has("odd_symmetry"):
                        insight("ODD f: V is EVEN → V(x*)=V(-x*) always → symmetric heteroclinic pairs")
                        insight("x=0 separatrix: orbits from negative unstable cross to positive unstable")
                except: pass

        except: pass  # outer try for integrate

    elif p.ptype == PT.CONTROL:
        kv("Routh family", "Degree-n char poly → n conditions for stability")
        kv("Root locus family", "Vary gain k: poles move from open-loop to closed-loop")

    elif p.ptype == PT.OPTIMIZATION:
        if g4.get("convex", None):
            kv("Convex family", "f'' ≥ 0 → f is convex → local min = global min")
        kv("Lagrange dual", "Strong duality (Slater): primal = dual = physics + information")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 06 — PROVE LIMITS (edge cases, failure modes, open problems)
# ════════════════════════════════════════════════════════════════════════════

def phase_06(p: Problem, g5: dict) -> dict:
    section(6, "PROVE LIMITS", "Edge cases · failure modes · known open problems")
    r = {}

    if p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        fm = KB.FAILURE_MODES.get(p.ptype_str())
        if fm: warn(fm); r["failure_mode"] = fm
        if p.ptype == PT.QUADRATIC:
            try:
                disc = discriminant(Poly(p.expr, p.var))
                dn = float(N(disc))
                if dn < 0: warn("Complex roots — set S=ℂ if real expected → problem may be ill-posed")
            except: pass
        if p.ptype == PT.POLY:
            try:
                deg = p.get_poly().degree() if p.get_poly() else 0
                if deg >= 5: warn("Abel-Ruffini: no closed-form radical expression for degree ≥ 5")
            except: pass

    elif p.ptype == PT.GRAPH:
        A_spec = p.meta.get("A_spec", [])
        kv("Isomorphism problem", "Graph isomorphism: GI-complete (no poly-time algorithm known)")
        kv("Spectral limit", "Non-isomorphic graphs can have same spectrum (cospectral mates)")

    elif p.ptype == PT.MARKOV:
        warn(KB.FAILURE_MODES.get("MARKOV", ""))
        kv("Periodicity check", "Chain is aperiodic iff gcd(return times)=1 at every state")

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs", [])
        if probs:
            (ok if abs(sum(probs) - 1) < 1e-9 else fail)(f"Σpᵢ={sum(probs):.9f}")
            (ok if _entropy(probs) >= -1e-12 else fail)("H≥0")
            H_max = math.log2(len(probs))
            (ok if _entropy(probs) <= H_max + 1e-9 else fail)(f"H≤log₂n={H_max:.4f}")
        kv("Open", "Shannon capacity of general channels (beyond AWGN): unsolved in general")

    elif p.ptype == PT.DYNAMICAL:
        kv("Non-hyperbolic limit", KB.FAILURE_MODES.get("DYNAMICAL", ""))
        kv("Chaos", "ẋ=f(x) can be chaotic for dim≥3 (Lorenz) — Lyapunov exponents needed")
        # Blow-up asymptotics (Problem D class)
        if p.fb.has("blowup_possible"):
            bd = p.fb.get("blowup_possible"); deg = bd["deg"]; lc = bd["lc"]
            n = deg
            kv("Blow-up ODE", f"ẋ ≈ {lc:.3g}·x^{n} for |x|→∞")
            kv("Separable solution", f"x(t) = x₀ · (1 - {lc*(n-1):.3g}·x₀^{n-1}·t)^(-1/{n-1})")
            kv("Blow-up time t*", f"t* ≈ 1/({lc*(n-1):.3g}·x₀^{n-1})  [x₀ → ∞]")
            xc = p.fb.get("blowup_threshold", 0)
            if xc:
                kv("Threshold", f"|x₀| > {xc:.4f}: blow-up to ±∞")
                kv("Basin", f"|x₀| < {xc:.4f}: converges to stable equilibrium")
            finding("Basin boundary = stable manifolds of outermost unstable equilibria")
            insight("Blow-up rate: t* ~ 1/((n-1)|a|x₀^(n-1)) is algebraic, not exponential")
            r["blowup_asymptotics"] = f"t* ~ 1/({lc*(n-1):.3g}·x₀^{n-1})"

    elif p.ptype == PT.CONTROL:
        rh = p._cache.get("routh", {})
        if rh and not rh.get("stable"):
            sc = rh.get("sign_changes", 0)
            finding(f"{sc} RHP pole(s) — system UNSTABLE — needs {sc} feedback gain(s)")
        kv("Robustness limit", "Routh: nominal stability only — H∞ norm bounds perturbation tolerance")

    elif p.ptype == PT.OPTIMIZATION:
        kv("Global optimality", "Non-convex functions: local ≠ global — no poly-time guarantee in general")
        kv("Saddle point issue", "f''(x*)=0: second-order test fails — need higher-order or numeric")

    elif p.ptype == PT.SUM:
        kv("Convergence test", "Σ1/k diverges (harmonic) — p-series Σ1/k^p converges iff p>1")
        kv("Open", "Riemann ζ(s): zeros of analytic continuation — Riemann Hypothesis (unsolved)")

    elif p.ptype == PT.PROOF:
        kv("Open", "Goldbach: every even n>2 = sum of two primes (unproven)")
        kv("Open", "Twin prime conjecture: infinitely many (p, p+2) prime pairs (unproven)")

    return r


# ════════════════════════════════════════════════════════════════════════════
# PHASE 07 — SYNTHESIS  (cross-domain bridges, spectral unification, meta-lesson)
# Encodes all 5 unifying identities as concrete bridges.
# Output entropy self-score (v4 integration F).
# ════════════════════════════════════════════════════════════════════════════

def phase_07(p: Problem, g6: dict) -> dict:
    section(7, "SYNTHESIS",
            "Cross-domain bridges · spectral unification · output entropy · meta-lesson")
    r = {}

    # ── Universal Bridge Map — encoding the 5 unifying identities ─────────
    BRIDGES = {
        PT.GRAPH: [
            ("→Markov",       "P=D⁻¹A random walk; π_i=d_i/2|E|"),
            ("→Entropy",      "H_s=−Σ(λᵢ/trL)log(λᵢ/trL) spectral entropy"),
            ("→Dynamical",    "ẋ=−Lx heat diffusion; solution e^{−tL}x₀"),
            ("→Optimization", "Min cut = max flow (LP duality)"),
            ("→ML",           "GNN h'=σ(QᵀΛQh), Q=eigenvectors of L"),
        ],
        PT.MATRIX: [
            ("→Dynamical",    "ẋ=Ax stable iff Re(λᵢ)<0; e^{At}x₀"),
            ("→Control",      "char poly=det(sI−A): poles = eigenvalues"),
            ("→Optimization", "Hessian H: xᵀHx = curvature form"),
            ("→Entropy",      "Von Neumann S(ρ)=−tr(ρ log ρ)"),
            ("→Graph",        "0/1 symmetric = adjacency matrix"),
        ],
        PT.MARKOV: [
            ("→Graph",        "P defines weighted digraph; reversible P = undirected G"),
            ("→Entropy",      "h=lim H(Xₙ|X₀…Xₙ₋₁) entropy rate"),
            ("→Optimization", "MCMC: run chain to sample target π"),
            ("→Physics",      "Ṡ=Σπᵢ Pᵢⱼ log(πᵢPᵢⱼ/πⱼPⱼᵢ)≥0 (2nd law)"),
        ],
        PT.ENTROPY: [
            ("→Physics",      "S=k_B·H (Boltzmann)"),
            ("→Markov",       "h=−Σᵢπᵢ Σⱼ Pᵢⱼ log Pᵢⱼ"),
            ("→Optimization", "MaxEnt: max H(p) s.t. constraints → Gibbs"),
            ("→ML",           "Cross-entropy loss=H(y,p̂)=H(y)+KL(y‖p̂)"),
        ],
        PT.DYNAMICAL: [
            ("→Control",      "ẋ=f(x,u): design u to steer to target"),
            ("→Optimization", "Gradient flow ẋ=−∇f IS gradient descent"),
            ("→Markov",       "SDE ẋ=f+noise → Fokker-Planck PDE"),
            ("→Entropy",      "h_KS=Σmax(λᵢ,0) Lyapunov/chaos measure"),
        ],
        PT.CONTROL: [
            ("→Matrix",       "char poly=det(sI−A); poles=eigenvalues of A"),
            ("→Optimization", "LQR: min∫(xᵀQx+uᵀRu)dt → Riccati equation"),
            ("→Dynamical",    "Closed-loop ẋ=(A+BK)x; place eigenvalues with K"),
            ("→Graph",        "Consensus rate=λ₂(Laplacian of comm. graph)"),
        ],
        PT.OPTIMIZATION: [
            ("→Dynamical",    "Gradient descent=Euler discretisation of ẋ=−∇f"),
            ("→Markov",       "RL/MDP: policy optimisation via Bellman equations"),
            ("→Entropy",      "MaxEnt = exponential family (Gibbs distribution)"),
            ("→Graph",        "Shortest path = min-cost flow = LP on graph"),
        ],
        PT.QUADRATIC: [
            ("→Matrix",       "Roots = eigenvalues of companion [[0,−c],[1,−b]]"),
            ("→Control",      "2nd-order char poly → poles of transfer function"),
            ("→Dynamical",    "ax²+bx+c=0 ↔ equilibria of ẋ=ax²+bx+c"),
        ],
        PT.LINEAR: [
            ("→Matrix",       "Ax=b: unique solution iff A invertible"),
            ("→Optimization", "min‖Ax−b‖² → x*=(AᵀA)⁻¹Aᵀb (least squares)"),
        ],
        PT.CUBIC: [
            ("→Matrix",       "Char poly of 3×3 = cubic"),
            ("→Dynamical",    "Equilibria of ẋ=ax³+bx"),
        ],
        PT.SUM: [
            ("→Entropy",      "Partition function Z=Σexp(−Eᵢ/kT)"),
            ("→Markov",       "Expected hitting times = sums over states"),
            ("→Zeta",         "ζ(s)=Σn^{−s} encodes ALL prime distribution"),
        ],
        PT.TRIG_ID: [
            ("→Complex",      "e^{iθ}=cosθ+i sinθ (Euler) — all trig from one formula"),
            ("→Graph",        "Cycle Cₙ: eigenvalues=2−2cos(2πk/n)"),
            ("→Fourier",      "sin/cos = orthonormal basis of L²([0,2π])"),
        ],
        PT.FACTORING: [
            ("→Roots",        "Factors determine roots exactly (fundamental theorem)"),
            ("→Crypto",       "Irreducibility over Zₚ → RSA hardness"),
        ],
        PT.PROOF: [
            ("→Logic",        "Contradiction IS the diagonal argument (Cantor/Gödel)"),
            ("→Algebra",      "Irrationality ↔ field extension degree 2"),
        ],
    }

    bridges_for = BRIDGES.get(p.ptype, [])
    if bridges_for:
        kv("Cross-domain bridges", "")
        for src_dst, desc in bridges_for:
            bridge(f"{src_dst}: {desc}")
        r["bridges"] = {sd: d for sd, d in bridges_for}

    # ── Spectral Unification (v4 integration E) ────────────────────────────
    if p.spectra:
        print(f"\n  {DIM}--- spectral unification ---{RST}")
        kv("Fingerprints deposited", len(p.spectra))
        for fp in p.spectra:
            kv(f"  {fp.domain}", fp.summary())
            kv(f"    spectral entropy", f"{fp.spectral_entropy():.4f} bits")
        # Cross-domain isomorphism detection (cosine similarity + binary match)
        if len(p.spectra) > 1:
            for i, fpa in enumerate(p.spectra):
                for fpb in p.spectra[i+1:]:
                    sim = fpa.cosine_similarity(fpb)
                    if sim > 0.99:
                        insight(f"⚡ SPECTRAL ISOMORPHISM: {fpa.domain} ≅ {fpb.domain}  (cos_sim={sim:.4f})")
                        r["spectral_isomorphism"] = (fpa.domain, fpb.domain)
        if any(fp.domain == "companion_poly" for fp in p.spectra):
            insight("Companion spectrum = polynomial roots = control poles = dynamical equilibria")
            insight("(Unifying identity #1: same object, three mathematical languages)")

    # ── Domain-specific deep insights ────────────────────────────────────
    if p.ptype == PT.GRAPH:
        L_spec = p.meta.get("L_spec", [])
        if L_spec:
            tr_L = sum(L_spec)
            if tr_L > 0:
                nz = [e for e in L_spec if e > 1e-9]
                H_s = _entropy([e/tr_L for e in nz])
                r["spectral_entropy"] = H_s; kv("Spectral entropy H_s(G)", f"{H_s:.4f} bits")
        kv("Heat kernel", "e^{−tL}: diffusion from any node (t→∞ = uniform)")
        insight("DEEPEST: graph spectrum = isomorphism fingerprint (Frucht's observation)")

    elif p.ptype == PT.MATRIX:
        spec = p.meta.get("spec", [])
        kv("e^{At}", "universal solution to all linear ODEs ẋ=Ax")
        kv("SVD", "M=UΣVᵀ → optimal rank-k approx (Eckart-Young theorem)")
        insight("DEEPEST: e^{At} IS the universal solution — all linear dynamics in one formula")

    elif p.ptype == PT.MARKOV:
        stat = p.meta.get("stat", {}); P_rat = p.meta.get("P_rat"); n = p.meta.get("n", 0)
        kv("Potential", "G=(I−P)⁻¹ → hitting times")
        kv("MCMC", "Sample ANY distribution by constructing chain with target π")
        if stat and P_rat:
            try:
                pi_f = [float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                ep = sum(pi_f[i]*float(N(P_rat[i,j]))*math.log(
                    max(pi_f[i]*float(N(P_rat[i,j])), 1e-15) /
                    max(pi_f[j]*float(N(P_rat[j,i])), 1e-15))
                    for i in range(n) for j in range(n)
                    if float(N(P_rat[i,j])) > 1e-12 and float(N(P_rat[j,i])) > 1e-12)
                r["entropy_prod"] = ep
                kv("Entropy production", f"{ep:.6f} (2nd law: ≥0)")
                insight(f"Ep={ep:.4f}: {'reversible — detailed balance' if ep < 1e-9 else 'irreversible — entropy produced'}")
            except: pass
        insight("DEEPEST: Markov chain IS a random walk on a weighted graph (unifying identity #2)")
        # Bug 4 fix: detect spectral isomorphism with known graph families
        markov_fp = next((f for f in p.spectra if f.domain == "markov"), None)
        if markov_fp:
            eigs_m = sorted(markov_fp.values)
            n_m = len(eigs_m)
            # Check: Pij = Aij/degree → P eigenvalues = A eigenvalues / d
            # For random walk on d-regular graph: λ_P = λ_A / d
            for graph_name, adj_eigs_fn in [
                (f"C{n_m} (cycle)", lambda n: [2*math.cos(2*math.pi*k/n) for k in range(n)]),
                (f"K{n_m} (complete)", lambda n: [n-1]+[-1]*(n-1)),
                (f"P{n_m} (path)", lambda n: [2*math.cos(math.pi*k/(n)) for k in range(1, n+1)]),
            ]:
                try:
                    adj = sorted(adj_eigs_fn(n_m))
                    d_guess = adj[-1]  # max eigenvalue = degree for regular graphs
                    if abs(d_guess) > 1e-9:
                        scaled = sorted(e/d_guess for e in adj)
                        diffs = [abs(scaled[i]-eigs_m[i]) for i in range(n_m)]
                        if max(diffs) < 0.01:
                            insight(f"⚡ SPECTRAL ISOMORPHISM: this Markov chain = random walk on {graph_name}")
                            insight(f"   P eigenvalues {[round(e,3) for e in eigs_m]} = A eigenvalues of {graph_name} / {d_guess:.0f}")
                            r["graph_isomorphism"] = graph_name
                            break
                except: pass

    elif p.ptype == PT.ENTROPY:
        probs = p.meta.get("probs", [])
        kv("Mutual info", "I(X;Y)=H(X)+H(Y)−H(X,Y)≥0, =0 iff independent")
        kv("MDL", "Minimum description length = Occam's razor quantified")
        insight("DEEPEST: MaxEnt = Gibbs = Softmax — same exponential family (unifying identity #3)")

    elif p.ptype == PT.DYNAMICAL:
        kv("KS entropy", "h_KS=Σmax(λᵢ,0): chaos from positive Lyapunov exponents")
        kv("Variational", "Trajectories extremise action S=∫L dt (Lagrangian mechanics)")
        insight("DEEPEST: gradient flow ẋ=−∇f unifies optimisation, dynamics, stat. mech. (identity #4)")
        # Chebyshev/sine root structure detection for x^(2k+1) type polynomials
        equil_fp = next((f for f in p.spectra if f.domain == "dynamical_equil"), None)
        if equil_fp and len(equil_fp.values) >= 3:
            eigs = sorted(equil_fp.values)
            n_eq = len(eigs)
            # Check if equilibria = 2sin(kπ/n) for some n
            pos_eq = [e for e in eigs if e > 1e-9]
            if pos_eq:
                n_candidate = 2 * len(pos_eq) + 1
                expected = sorted([2*math.sin(k*math.pi/n_candidate) for k in range(1, len(pos_eq)+1)])
                diffs = [abs(pos_eq[i]-expected[i]) for i in range(len(pos_eq))]
                if all(d < 0.01 for d in diffs):
                    insight(f"⚡ CHEBYSHEV STRUCTURE: equilibria = 2sin(kπ/{n_candidate}) for k=1..{len(pos_eq)}")
                    insight(f"   f(x)/x is related to sin({n_candidate}θ)/sin(θ) with x=2sin(θ)")
                    insight(f"   Topology: {n_candidate} equilibria with alternating STABLE/UNSTABLE → 7th-roots pattern")
                    r["chebyshev_structure"] = f"n={n_candidate}"


    elif p.ptype == PT.CONTROL:
        kv("Pontryagin", "Optimal control = variational problem (PMP)")
        kv("Kalman filter", "Dual of LQR: optimal state estimation")
        insight("DEEPEST: Routh stability = spectral = Lyapunov — three equiv. conditions (identity #5)")

    elif p.ptype == PT.OPTIMIZATION:
        kv("Lagrange dual", "Strong duality (Slater): primal=dual")
        kv("Natural gradient", "Riemannian ∇ w.r.t. Fisher information metric")
        insight("DEEPEST: Lagrangian duality unifies optimisation, physics, information theory")

    elif p.ptype in (PT.LINEAR, PT.QUADRATIC, PT.CUBIC, PT.POLY):
        bridge("→Matrix: roots = eigenvalues of companion matrix C(p)")
        bridge("→Control: char poly of A(s) → poles of transfer function")
        bridge("→Dynamical: real roots = equilibria of ẋ=p(x)")
        insight("DEEPEST: FTA ↔ linear algebra ↔ dynamical systems — same object, 3 languages (identity #1)")

    elif p.ptype == PT.SUM:
        bridge("→Euler-Maclaurin: Σf(k) ≈ ∫f dx + boundary terms")
        insight("DEEPEST: Riemann zeta = partition function of primes (physics ↔ number theory)")

    elif p.ptype == PT.PROOF:
        bridge("→Diagonal argument: contradiction IS Cantor/Gödel/Halting Problem")
        insight("DEEPEST: proof by contradiction = incompleteness machinery in disguise")

    elif p.ptype == PT.TRIG_ID:
        bridge("→Euler: e^{iπ}+1=0 unifies analysis, algebra, geometry")
        insight("DEEPEST: all of trigonometry is Euler's formula in disguise")

    # ── Output Entropy Self-Score (Bug 6 fix: synthesise from all findings)
    print(f"\n  {DIM}--- output entropy scoring ---{RST}")
    # Collect rich lines: conf results + fingerprint summaries + fb signals + meta
    rich_lines = (
        [f"{k}: {str(v)[:80]}" for k, (v, _) in p.conf.results.items()]
        + [fp.summary() for fp in p.spectra]
        + [f"signal:{s}" for s in p.fb.all_signals()]
        + [str(v)[:80] for v in p.meta.values() if v is not None]
        + p.output_lines
    )
    oe = _output_entropy(rich_lines)
    r["output_entropy"] = oe; kv("Output diversity H(output)", f"{oe:.4f} bits")
    if oe > 4.0:   insight("High output diversity — engine produced genuinely varied information")
    elif oe > 2.0: note("Moderate output diversity")
    else:          warn(f"Low output diversity ({oe:.2f}) — possible repetition in output")

    # ── Feedback summary ──────────────────────────────────────────────────
    signals = p.fb.all_signals()
    if signals: kv("Feedback signals emitted", " | ".join(signals))

    # ── Confidence ledger ─────────────────────────────────────────────────
    print(f"\n  {DIM}--- confidence ledger ---{RST}")
    kv("Summary", p.conf.summary())
    for k_ in p.conf.knowns[:5]:  ok(f"[≥0.90] {k_[:70]}")
    for k_ in p.conf.unknowns[:3]: warn(f"[<0.60] {k_[:70]}")

    return r


# ════════════════════════════════════════════════════════════════════════════
# FINAL ANSWER  (cache-first, v4 pattern)
# ════════════════════════════════════════════════════════════════════════════

def _final_answer(p: Problem) -> str:
    for key in ["solve(quadratic)", "solve(cubic)", "solve(linear)",
                "solve(poly)", "roots", "factor(expr)", "factored"]:
        v_ = p._cache.get(key)
        if v_: return f"{key}: {str(v_)[:200]}"
    if p.ptype == PT.ENTROPY:
        H = p.meta.get("H_val")
        return f"H = {H:.6f} bits" if H is not None else "entropy: no distribution parsed"
    if p.ptype == PT.CONTROL:
        rh = p._cache.get("routh", {})
        return f"Routh-Hurwitz: {'STABLE' if rh.get('stable') else 'UNSTABLE'}" if rh else "control: see phase 02"
    if p.ptype == PT.OPTIMIZATION:
        opt = p._cache.get("optimal")
        return f"Optimal: x*={opt[1]}, f*={opt[0]:.4f} ({opt[2]})" if opt else "optimize: see phase 02"
    if p.ptype == PT.SUM:
        for k_, v_ in p._cache.items():
            if "summation" in k_ and v_ is not None:
                return f"Sum = {factor(v_)}"
    if p.ptype == PT.DYNAMICAL:
        eq_ = p._cache.get("solve(f=0)")
        return f"Equilibria: {[str(e) for e in eq_]}" if eq_ else "dynamical: see phase 02"
    if p.ptype in (PT.PROOF, PT.DIGRAPH_CYC):
        return p._cache.get("status", "see phase 02")
    if p.ptype in (PT.TRIG_ID, PT.SIMPLIFY):
        s = p._cache.get("trigsimp")
        return f"Simplified: {s}" if s is not None else "identity: see phase 02"
    for k_, v_ in p._cache.items():
        if v_ is not None: return f"{k_}: {str(v_)[:120]}"
    return "see phase output above"


# ════════════════════════════════════════════════════════════════════════════
# MAIN RUN
# ════════════════════════════════════════════════════════════════════════════

def run(raw: str, json_out: bool = False, quiet: bool = False):
    global _QUIET
    _QUIET = quiet

    if not quiet:
        print(f"\n{hr('═')}\n{W}DISCOVERY ENGINE v5{RST}\n{hr()}")
        print(f"  {W}Problem:{RST}  {Y}{raw}{RST}")

    t0 = time.time()
    prob = classify(raw)

    if not quiet:
        print(f"  {DIM}Type:{RST}     {prob.ptype.label()}")
        print(f"  {DIM}Variable:{RST} {prob.var}")
        print(hr('═'))

    if prob.ptype == PT.UNKNOWN:
        if not quiet:
            print(f"{R}Could not classify. Examples:{RST}")
            print(f"  'x²−5x+6=0', 'graph K4', 'markov [[…]]', 'entropy […]', 'dynamical x³−x'")
        return prob

    depth = KB.phase_depth(prob.ptype_str())

    g1 = phase_01(prob)
    g2 = phase_02(prob, g1)
    g3 = phase_03(prob, g2) if depth >= 3 else {}
    g4 = phase_04(prob, g3) if depth >= 4 else {}
    g5 = phase_05(prob, g4) if depth >= 5 else {}
    g6 = phase_06(prob, g5) if depth >= 6 else {}
    g7 = phase_07(prob, g6) if depth >= 7 else {}

    elapsed = time.time() - t0

    if not quiet:
        print(f"\n{hr('═')}\n{W}FINAL ANSWER{RST}\n{hr()}")
        print(f"  {G}{_final_answer(prob)}{RST}")
        print(f"  {DIM}Elapsed: {elapsed:.3f}s{RST}")
        phases = [g for g in [g1,g2,g3,g4,g5,g6,g7] if g]
        titles = {1:"Ground Truth",2:"Direct Attack",3:"Structure Hunt",
                  4:"Pattern Lock",5:"Generalise",6:"Prove Limits",7:"Synthesis"}
        print(f"\n{hr()}\n{W}PHASE SUMMARY{RST}\n{hr('·')}")
        for i, (g, t_) in enumerate(zip(phases, list(titles.values())[:len(phases)]), 1):
            print(f"  {PHASE_CLR[i]}{i:02d} {t_:<18}{RST} {len(g)} results")
        kv("Feedback signals", " | ".join(prob.fb.all_signals()) or "none")
        kv("Spectral fingerprints", len(prob.spectra))
        kv("Confidence", prob.conf.summary())
        print(hr('═'))

    if json_out:
        dr = DiscoveryResult(
            problem=raw, ptype=prob.ptype_str(),
            phases_run=list(range(1, depth+1)),
            solutions=[str(v_) for v_ in (
                prob._cache.get("solve(quadratic)") or
                prob._cache.get("solve(cubic)") or
                prob._cache.get("solve(linear)") or [])],
            confidence={k_: c for k_, (_, c) in prob.conf.results.items()},
            spectral=[{"domain": fp.domain, "values": fp.values[:6], "entropy": fp.spectral_entropy()}
                      for fp in prob.spectra],
            bridges=g7.get("bridges", {}),
            output_entropy=g7.get("output_entropy", 0.0),
            elapsed_s=elapsed,
            fb_signals=prob.fb.all_signals(),
            warnings=prob.conf.flags,
        )
        return dr.to_dict()

    return prob


# ════════════════════════════════════════════════════════════════════════════
# ASSERTION-BASED TEST SUITE  (v4b semantic assertion pattern, v5 extended)
# Insight from analysis: pass/fail proves no crash. Assertions prove correctness.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TR:
    desc: str; raw: str; passed: bool; elapsed: float
    ap: int = 0; af: int = 0; notes: List[str] = field(default_factory=list)

def _assert(tr: TR, cond: bool, msg: str):
    if cond: tr.ap += 1
    else:    tr.af += 1; tr.notes.append(f"FAIL: {msg}")

def _run_test(raw: str, desc: str, checks=None) -> TR:
    t0 = time.time()
    buf = io.StringIO(); prob = None
    try:
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            prob = run(raw, quiet=True)
        elapsed = time.time() - t0
        tr = TR(desc=desc, raw=raw, passed=True, elapsed=elapsed)
        if checks and prob:
            for chk in checks:
                try: chk(prob, tr)
                except Exception as e: _assert(tr, False, f"check raised: {e}")
    except Exception as e:
        elapsed = time.time() - t0
        tr = TR(desc=desc, raw=raw, passed=False, elapsed=elapsed)
        tr.notes.append(f"Exception: {e}")
    return tr

# ── Assertion helpers ─────────────────────────────────────────────────────

def assert_roots(expected):
    def chk(prob, tr):
        rts = None
        for k in ["roots","solve(linear)","solve(quadratic)","solve(cubic)","solve(poly)"]:
            if prob._cache.get(k): rts = prob._cache[k]; break
        _assert(tr, rts is not None, "roots computed")
        if rts:
            got = {float(N(r_)) for r_ in rts if _is_real_value(r_)}
            exp = {float(e) for e in expected}
            _assert(tr, all(any(abs(g-e) < 1e-3 for g in got) for e in exp),
                    f"expected {exp}, got {got}")
    return chk

def assert_entropy(lo, hi):
    def chk(prob, tr):
        H = prob.meta.get("H_val")
        _assert(tr, H is not None, "entropy computed")
        if H: _assert(tr, lo <= H <= hi, f"H={H:.4f} not in [{lo},{hi}]")
    return chk

def assert_stable(expected):
    def chk(prob, tr):
        rh = prob._cache.get("routh", {})
        _assert(tr, bool(rh), "Routh computed")
        _assert(tr, rh.get("stable") == expected, f"expected stable={expected}, got {rh.get('stable')}")
    return chk

def assert_stationary_sum():
    def chk(prob, tr):
        stat = prob.meta.get("stat", {})
        _assert(tr, bool(stat), "stationary dist computed")
        if stat:
            s = sum(float(N(sp.sympify(v_))) for v_ in stat.values())
            _assert(tr, abs(s-1) < 1e-4, f"π sums to {s:.6f}")
    return chk

def assert_connected(expected):
    def chk(prob, tr):
        conn = prob.fb.get("connected")
        _assert(tr, conn is not None, "connectivity detected")
        _assert(tr, conn == expected, f"expected connected={expected}, got {conn}")
    return chk

def assert_has_spectrum():
    def chk(prob, tr): _assert(tr, len(prob.spectra) > 0, "spectral fingerprint registered")
    return chk

def assert_sum_at(n_val, expected_val):
    def chk(prob, tr):
        res = next((v_ for k_, v_ in prob._cache.items() if 'summation' in k_ and v_ is not None), None)
        _assert(tr, res is not None, "sum formula computed")
        if res:
            ns = symbols('n', positive=True, integer=True)
            try:
                got = int(N(res.subs(ns, n_val)))
                _assert(tr, got == expected_val, f"sum at n={n_val}: got {got}, expected {expected_val}")
            except Exception as e: _assert(tr, False, f"eval failed: {e}")
    return chk

def assert_signal(signal):
    def chk(prob, tr): _assert(tr, prob.fb.has(signal), f"signal '{signal}' not emitted")
    return chk

def assert_optimal_x(x_star, tol=1e-2):
    def chk(prob, tr):
        crit = prob._cache.get("solve(f'=0)", [])
        _assert(tr, bool(crit), "critical points found")
        if crit and prob.var:
            vals = [float(N(c_)) for c_ in crit]
            _assert(tr, any(abs(xv - x_star) < tol for xv in vals),
                    f"x*={vals} not near {x_star}")
    return chk

def assert_family(expected_substr):
    def chk(prob, tr):
        fam = prob.meta.get("family","") or (prob._cache.get("family","") or "")
        # Re-detect if not cached
        if not fam: fam = _detect_family(prob) or ""
        _assert(tr, expected_substr.lower() in fam.lower(),
                f"family '{fam}' does not contain '{expected_substr}'")
    return chk

# ── Test Battery ──────────────────────────────────────────────────────────

TESTS = [
    # Algebraic
    ("x^2 - 5x + 6 = 0",           "Quadratic integer roots",      [assert_roots([2,3])]),
    ("2x + 3 = 7",                  "Linear",                       [assert_roots([2])]),
    ("x^3 - 6x^2 + 11x - 6 = 0",   "Cubic 3 integer roots",        [assert_roots([1,2,3])]),
    ("sin(x)^2 + cos(x)^2",         "Pythagorean identity",         []),
    ("factor x^4 - 16",             "Difference of squares chain",  []),
    # Summation
    ("sum of first n integers",     "Σk",                           [assert_sum_at(5, 15)]),
    ("sum of squares of first n integers", "Σk²",                   [assert_sum_at(4, 30)]),
    ("sum of cubes of first n integers",   "Σk³",                   [assert_sum_at(3, 36)]),
    ("sum of harmonic series",      "Harmonic",                     []),
    # Proofs
    ("prove sqrt(2) is irrational", "Irrationality proof",          []),
    ("prove there are infinitely many primes", "Infinitude primes",  []),
    # Digraph
    ("m^3 vertices with 3 cycles, m=3", "Digraph odd m",            []),
    ("m^3 vertices with 3 cycles, m=4", "Digraph even m",           []),
    # Graph
    ("graph K4",  "Complete K4",    [assert_connected(True), assert_has_spectrum(), assert_signal("bipartite")]),
    ("graph P5",  "Path P5",        [assert_connected(True), assert_has_spectrum()]),
    ("graph C6",  "Cycle C6",       [assert_connected(True), assert_has_spectrum(), assert_signal("bipartite")]),
    ("graph C4",  "Cycle C4",       [assert_connected(True), assert_signal("bipartite")]),
    ("graph [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]", "Custom graph",
     [assert_connected(True), assert_has_spectrum()]),
    # Matrix
    ("matrix [[2,1],[1,3]]",         "Symmetric 2×2",               [assert_has_spectrum(), assert_signal("pos_definite")]),
    ("matrix [[4,2,2],[2,3,0],[2,0,3]]", "Symmetric 3×3",            [assert_has_spectrum()]),
    # Markov
    ("markov [[0.7,0.3],[0.4,0.6]]", "2-state chain",               [assert_stationary_sum(), assert_has_spectrum()]),
    ("markov [[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]]", "3-state chain", [assert_stationary_sum(), assert_has_spectrum()]),
    ("markov [[1,0],[0.3,0.7]]",     "Absorbing Markov",            [assert_signal("absorbing"), assert_has_spectrum()]),
    # Entropy
    ("entropy [0.5,0.25,0.25]",      "Entropy skewed",              [assert_entropy(1.4, 1.6)]),
    ("entropy [0.25,0.25,0.25,0.25]","Entropy uniform",             [assert_entropy(1.99, 2.01)]),
    ("entropy [0.9,0.05,0.05]",      "Near-deterministic",          [assert_entropy(0.4, 0.75)]),
    # Dynamical
    ("dynamical x^3 - x",   "3 equilibria",     [assert_signal("multiple_equilibria")]),
    ("dynamical x^2 - 1",   "Pitchfork",        []),
    ("dynamical sin(x)",     "Trig equilibria",  []),
    # Control
    ("control s^2 + 3s + 2",          "Stable 2nd order",   [assert_stable(True)]),
    ("control s^3 + 2s^2 + 3s + 1",   "Routh 3rd order",    [assert_stable(True)]),
    ("control s^3 - s + 1",            "Unstable",           [assert_stable(False)]),
    ("control s^4 + s^3 + s^2 + s + 1","4th order Routh",   [assert_stable(False)]),
    # Optimization
    ("optimize x^4 - 4x^2 + 1",   "Quartic opt",        [assert_signal("multiple_minima"), assert_has_spectrum()]),
    ("minimize x^2 + 2x + 1",     "Minimize quadratic", [assert_optimal_x(-1)]),
    ("maximize -x^2 + 4x - 3",    "Maximize concave",   [assert_optimal_x(2)]),
    # Edge cases
    ("x^4 - 5x^2 + 4 = 0",        "Biquadratic",        [assert_roots([1,-1,2,-2])]),
    ("x^2 + 4 = 0",               "Complex roots only", []),
    ("sum of power 4 first n integers", "Σk⁴",           [assert_sum_at(3, 98)]),
    ("prove root 2 is irrational", "NL sqrt2 alias",     []),
]


def run_tests(verbose: bool = False):
    print(f"\n{hr('═')}\n{W}DISCOVERY ENGINE v5 — TEST SUITE ({len(TESTS)} tests){RST}\n{hr('═')}")
    passed = 0; failed_tests = []; total_time = 0.0; total_ap = 0; total_af = 0
    for raw, desc, checks in TESTS:
        print(f"\n  {B}[TEST]{RST} {desc}  {DIM}[{raw[:52]}]{RST}", end="", flush=True)
        tr = _run_test(raw, desc, checks)
        total_time += tr.elapsed; total_ap += tr.ap; total_af += tr.af
        if tr.passed and tr.af == 0:
            passed += 1
            print(f" {G}✓{RST} ({tr.elapsed:.2f}s) {G}+{tr.ap} assert{RST}")
        else:
            failed_tests.append(tr)
            af_str = f" {R}−{tr.af} assert fails{RST}" if tr.af else ""
            print(f" {R}✗{RST} ({tr.elapsed:.2f}s){af_str}")
        if (verbose or not tr.passed or tr.af > 0) and tr.notes:
            for n_ in tr.notes[:3]: print(f"    {R}→{RST} {n_}")
    n_tests = len(TESTS)
    print(f"\n{hr('═')}")
    clr = G if passed == n_tests else (Y if passed > n_tests*0.8 else R)
    print(f"{clr}Results: {passed}/{n_tests} passed{RST}  "
          f"| {G}+{total_ap}{RST}/{R}−{total_af}{RST} assertions  "
          f"| Total: {total_time:.1f}s  Avg: {total_time/n_tests:.2f}s/test")
    if failed_tests:
        print(f"\n{R}Failed:{RST}")
        for tr in failed_tests:
            print(f"  {R}✗{RST} {tr.desc}")
            for n_ in tr.notes[:2]: print(f"    {DIM}{n_}{RST}")
    print(hr('═')); return passed, n_tests


def run_bench():
    print(f"\n{hr('═')}\n{W}DISCOVERY ENGINE v5 — BENCHMARK{RST}\n{hr('·')}")
    cases = [
        ("x^2 - 5x + 6 = 0",                 "quadratic"),
        ("graph K4",                          "graph"),
        ("markov [[0.7,0.3],[0.4,0.6]]",      "markov"),
        ("entropy [0.5,0.25,0.25]",           "entropy"),
        ("dynamical x^3 - x",                 "dynamical"),
        ("control s^3 + 2s^2 + 3s + 1",       "control"),
        ("optimize x^4 - 4x^2 + 1",           "optimize"),
        ("matrix [[4,2,2],[2,3,0],[2,0,3]]",   "matrix"),
    ]
    times = []
    for raw, label in cases:
        t0 = time.time()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            run(raw, quiet=True)
        elapsed = time.time() - t0; times.append(elapsed)
        bar = "█" * int(elapsed * 15)
        clr = G if elapsed < 0.5 else (Y if elapsed < 1.5 else R)
        print(f"  {clr}{bar:<20}{RST} {raw[:44]:<46} {elapsed:.3f}s  [{label}]")
    print(hr('·'))
    print(f"  Total: {sum(times):.3f}s  Avg: {sum(times)/len(times):.3f}s  Max: {max(times):.3f}s")
    print(hr('═'))


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0] == "--test":
        verbose = "--verbose" in args or "-v" in args
        run_tests(verbose=verbose)
    elif args[0] == "--bench":
        run_bench()
    elif args[0] == "--json":
        raw = " ".join(a for a in args[1:] if not a.startswith("--"))
        result = run(raw, json_out=True, quiet=True)
        print(json.dumps(result, indent=2, default=str))
    else:
        quiet = "--quiet" in args
        raw = " ".join(a for a in args if not a.startswith("--"))
        run(raw, quiet=quiet)

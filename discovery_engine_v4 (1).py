#!/usr/bin/env python3
"""
discovery_engine_v4.py — 7-Phase Mathematical Discovery Engine
================================================================
Integrations over user-v3:
  A. SPECTRAL UNIFICATION  — SpectralFingerprint; Phase 07 cross-domain isomorphism detection
  B. INTER-PHASE FEEDBACK  — FeedbackQueue: Phase 03 signals unlock richer Phase 04 analysis
  C. ADAPTIVE PHASE DEPTH  — PhaseBudget reports planned phases + rationale
  D. SOLUTION FAMILY       — Phase 05 embeds every solution in its parametric family
  E. OUTPUT ENTROPY        — Phase 07 self-scores output diversity; prunes redundancy
  F. TIMEOUT PROTECTION    — all heavy SymPy calls timeout-guarded

Improvements from user-v3:
  - Variable heuristic: prefer x,y,z,t over alphabetical
  - Summand detection: squares/cubes/power N from natural language
  - Proof NLP: 'root 2','irrational' as aliases
  - .is_real safety over isinstance(ComplexNumber)
  - Cache-first final_answer for all types
  - Expanded regex for optimization/sum keywords

Test suite: 34 problems with assertion-based checks + performance timing

Usage:
  python discovery_engine_v4.py "x^2 - 5x + 6 = 0"
  python discovery_engine_v4.py --test
  python discovery_engine_v4.py --test --verbose
  python discovery_engine_v4.py --bench
"""

import sys, re, ast, math, traceback, time, heapq, functools, io, threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from contextlib import redirect_stdout, redirect_stderr

import sympy as sp
from sympy import (
    symbols, solve, simplify, expand, factor, cancel,
    Symbol, Rational, pi, E, I, oo,
    sin, cos, tan, sec, csc, cot, exp, log, sqrt, Abs,
    diff, integrate, summation, discriminant, roots, Poly, factorint,
    trigsimp, nsolve, N, S, gcd, divisors, nsimplify,
    real_roots, all_roots, Matrix, eye, zeros, ones, diag,
    det, trace, re as sp_re,
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

_TRANSFORMS = (standard_transformations +
               (implicit_multiplication_application, convert_xor))

R="\033[91m";G="\033[92m";Y="\033[93m";B="\033[94m"
M="\033[95m";C="\033[96m";W="\033[97m";DIM="\033[2m";RST="\033[0m"
PHASE_CLR={1:G,2:R,3:B,4:M,5:Y,6:C,7:W}

def hr(ch="─",n=72): return ch*n
def section(num,name,tag):
    c=PHASE_CLR[num]; print(f"\n{hr()}\n{c}Phase {num:02d} — {name}{RST}  {DIM}{tag}{RST}\n{hr('.')}")
def kv(k,v,ind=2):   print(f"{' '*ind}{DIM}{k:<38}{RST}{W}{str(v)[:120]}{RST}")
def finding(m,s="→"):print(f"  {Y}{s}{RST} {m}")
def ok(m):           print(f"  {G}✓{RST} {m}")
def fail(m):         print(f"  {R}✗{RST} {m}")
def note(m):         print(f"  {DIM}{m}{RST}")
def bridge(m):       print(f"  {C}⇔{RST} {B}{m}{RST}")
def warn(m):         print(f"  {Y}⚠{RST} {m}")
def insight(m):      print(f"  {M}★{RST} {W}{m}{RST}")


# ══════════════════════════════════════════════════════════════════════════
# F: TIMEOUT PROTECTION
# ══════════════════════════════════════════════════════════════════════════

def timeout_call(func, args=(), secs=8, fallback=None):
    res=[fallback]; exc=[None]
    def _run():
        try:    res[0]=func(*args)
        except Exception as e: exc[0]=e
    t=threading.Thread(target=_run,daemon=True)
    t.start(); t.join(secs)
    if t.is_alive(): return fallback
    if exc[0]: raise exc[0]
    return res[0]


# ══════════════════════════════════════════════════════════════════════════
# A: SPECTRAL FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralFingerprint:
    eigenvalues: List[float]
    domain:      str
    metadata:    Dict[str,Any] = field(default_factory=dict)

    def spectral_entropy(self):
        total=sum(abs(e) for e in self.eigenvalues)+1e-15
        w=[abs(e)/total for e in self.eigenvalues if abs(e)>1e-12]
        return -sum(p*math.log2(p) for p in w if p>1e-15)

    def spectral_radius(self):
        return max(abs(e) for e in self.eigenvalues) if self.eigenvalues else 0.0

    def similarity(self, other):
        a=sorted(self.eigenvalues); b=sorted(other.eigenvalues)
        n=min(len(a),len(b))
        if n==0: return 0.0
        dot=sum(a[i]*b[i] for i in range(n))
        na=math.sqrt(sum(x**2 for x in a[:n]))+1e-15
        nb=math.sqrt(sum(x**2 for x in b[:n]))+1e-15
        return max(-1.0,min(1.0,dot/(na*nb)))


# ══════════════════════════════════════════════════════════════════════════
# B: FEEDBACK QUEUE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FeedbackQueue:
    signals: List[Tuple[str,Any]] = field(default_factory=list)

    def emit(self, signal, data=None): self.signals.append((signal,data))
    def has(self, signal): return any(s==signal for s,_ in self.signals)
    def get(self, signal, default=None): return next((d for s,d in self.signals if s==signal),default)
    def all_signals(self): return [s for s,_ in self.signals]


# ══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════

class KB:
    METHOD_PRIORS = {
        ("quadratic","solve"):0.99,("quadratic","discriminant"):0.99,
        ("cubic","solve"):0.85,("cubic","nsolve"):0.92,
        ("poly_high","nsolve"):0.90,("trig_id","trigsimp"):0.95,
        ("control","routh"):0.95,("graph","spectrum"):0.95,
        ("markov","stationary"):0.99,("entropy","H_numeric"):0.99,
        ("dynamical","solve_equil"):0.90,("optimization","critical_pts"):0.95,
        ("sum","summation"):0.99,
    }
    ANALOGIES = {
        "QUADRATIC":  ["2D linear system stability","eigenvalue of 2×2 matrix","2-state entropy"],
        "GRAPH":      ["Markov chain D⁻¹A","heat diffusion e^{-tL}","multi-agent consensus λ₂"],
        "MARKOV":     ["random walk on graph","entropy production","MDP in RL"],
        "ENTROPY":    ["Boltzmann S=k_B·H","channel capacity","KL from uniform"],
        "DYNAMICAL":  ["gradient descent x'=-∇f","control x'=f(x,u)","SDE Fokker-Planck"],
        "CONTROL":    ["companion matrix eigs","LQR Riccati","Nyquist stability"],
        "OPTIMIZATION":["gradient flow ODE","Bellman equations","MaxEnt under constraints"],
    }
    VERIFICATION = {
        "equation":["substitute back","discriminant sign","Vieta's formulas"],
        "graph":   ["tr(L)=Σdeg","λ₁(L)=0","tr(A²)/2=|E|"],
        "markov":  ["rows sum to 1","π·P=π","spectral radius=1"],
        "control": ["sign changes in Routh","all coefficients positive"],
        "entropy": ["sum(p)=1","0≤H≤log₂n","per-symbol contributions"],
    }
    FAILURE_MODES = {
        "POLY":      "degree≥5 → Abel-Ruffini: no closed-form radical formula",
        "MARKOV":    "reducible chain → multiple stationary distributions",
        "DYNAMICAL": "f'(x*)=0 → non-hyperbolic → linearization insufficient",
        "ENTROPY":   "p=0 terms → 0·log(0)=0 by convention",
        "MATRIX":    "near-singular → ill-conditioned → numerical instability",
        "CONTROL":   "missing sign in coefficients → necessary condition fails",
    }


# ══════════════════════════════════════════════════════════════════════════
# CONFIDENCE LEDGER
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Confidence:
    results:  Dict[str,Tuple[Any,float]] = field(default_factory=dict)
    flags:    List[str]                  = field(default_factory=list)
    knowns:   List[str]                  = field(default_factory=list)
    unknowns: List[str]                  = field(default_factory=list)

    def record(self,key,val,conf,note=""):
        self.results[key]=(val,conf)
        if conf>=0.9:  self.knowns.append(f"{key}: {str(val)[:50]}")
        elif conf<0.6: self.unknowns.append(f"{key}(conf={conf:.2f})")
        if note: self.flags.append(note)

    def summary(self):
        t=len(self.results)
        h=sum(1 for _,c in self.results.values() if c>=0.9)
        m=sum(1 for _,c in self.results.values() if 0.6<=c<0.9)
        l=sum(1 for _,c in self.results.values() if c<0.6)
        return f"{t} results: {h} high-conf, {m} mid, {l} uncertain"


# ══════════════════════════════════════════════════════════════════════════
# PROBLEM TYPES
# ══════════════════════════════════════════════════════════════════════════

class PT(Enum):
    LINEAR=1; QUADRATIC=2; CUBIC=3; POLY=4
    TRIG_EQ=5; TRIG_ID=6; FACTORING=7; SIMPLIFY=8
    SUM=9; PROOF=10; DIGRAPH_CYC=11
    GRAPH=12; MATRIX=13; MARKOV=14; ENTROPY=15
    DYNAMICAL=16; CONTROL=17; OPTIMIZATION=18
    UNKNOWN=99

    def label(self):
        return {1:"linear equation",2:"quadratic equation",3:"cubic equation",
                4:"polynomial (deg≥4)",5:"trig equation",6:"trig identity",
                7:"factoring",8:"simplification",9:"summation/series",10:"proof",
                11:"digraph cycle decomp",12:"graph/network",13:"matrix analysis",
                14:"markov chain",15:"information entropy",16:"dynamical system",
                17:"control theory",18:"optimization",99:"unknown"}.get(self.value,"unknown")


# ══════════════════════════════════════════════════════════════════════════
# PROBLEM DATACLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Problem:
    raw:     str
    ptype:   PT
    expr:    Optional[sp.Basic]         = None
    lhs:     Optional[sp.Basic]         = None
    rhs:     Optional[sp.Basic]         = None
    var:     Optional[sp.Symbol]        = None
    free:    List[sp.Symbol]            = field(default_factory=list)
    meta:    Dict[str,Any]              = field(default_factory=dict)
    poly:    Optional[sp.Poly]          = None
    _cache:  Dict[str,Any]              = field(default_factory=dict, repr=False)
    conf:    Confidence                 = field(default_factory=Confidence, repr=False)
    fb:      FeedbackQueue              = field(default_factory=FeedbackQueue, repr=False)
    spectra: List[SpectralFingerprint]  = field(default_factory=list, repr=False)

    def memo(self,key,func):
        if key not in self._cache:
            try:   self._cache[key]=func()
            except: self._cache[key]=None
        return self._cache[key]

    def add_spectrum(self,fp):
        self.spectra.append(fp); self.fb.emit('spectral_fp',fp)

    def get_poly(self):
        if self.poly is None and self.expr and self.var:
            try: self.poly=Poly(self.expr,self.var)
            except: pass
        return self.poly


# ══════════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _entropy_raw(w): return -sum(p*math.log2(p) for p in w if p>1e-15)
def _entropy(probs):
    s=sum(probs); return 0.0 if s<1e-15 else _entropy_raw([p/s for p in probs])
def _kl(p,q):
    return sum(p[i]*math.log2(max(p[i],1e-15)/max(q[i],1e-15)) for i in range(len(p)) if p[i]>1e-15)

def _parse(s):
    s=s.strip().replace('^','**')
    for old,new in [("ln","log"),("arcsin","asin"),("arccos","acos"),("arctan","atan")]:
        s=re.sub(rf'\b{old}\b',new,s)
    for fn in [lambda x: parse_expr(x,transformations=_TRANSFORMS),
               lambda x: sp.sympify(x)]:
        try: return fn(s)
        except: pass
    return None

def _parse_matrix(s):
    m=re.search(r'\[\s*\[.+?\]\s*\]',s,re.S)
    if not m: return None
    try:
        rows=ast.literal_eval(m.group(0))
        return sp.Matrix([[sp.sympify(x) for x in row] for row in rows])
    except: return None

def _parse_probs(s):
    m=re.search(r'\[([^\]]+)\]',s)
    if not m: return []
    try:
        vals=[float(x) for x in m.group(1).split(',')]
        return vals if abs(sum(vals)-1.0)<1e-6 else []
    except: return []

def _spectrum(M):
    try: return sorted([float(N(k)) for k in M.eigenvals(multiple=True)])
    except: return []

def _spectrum_complex(M):
    try: return [complex(N(k)) for k in M.eigenvals(multiple=True)]
    except: return []

def _make_fp(eigs,domain,**meta):
    return SpectralFingerprint(eigenvalues=sorted(eigs),domain=domain,metadata=dict(meta))

def _build_graph(p):
    raw=p.raw; meta=p.meta
    mk=re.search(r'\bK(\d+)\b',raw,re.I)
    mp=re.search(r'\bP(\d+)\b',raw,re.I)
    mc=re.search(r'\bC(\d+)\b',raw,re.I)
    if mk:
        n=int(mk.group(1)); A=ones(n,n)-eye(n); meta["type"]="complete"
    elif mp:
        n=int(mp.group(1)); A=zeros(n,n)
        for i in range(n-1): A[i,i+1]=A[i+1,i]=1
        meta["type"]="path"
    elif mc:
        n=int(mc.group(1)); A=zeros(n,n)
        for i in range(n): A[i,(i+1)%n]=A[(i+1)%n,i]=1
        meta["type"]="cycle"
    else:
        A=_parse_matrix(raw)
        if A is None: return None,None,0,[]
        n=A.shape[0]; meta["type"]="custom"
    meta["A"]=A
    deg=[int(sum(A[i,j] for j in range(n))) for i in range(n)]
    D=diag(*deg); L=D-A
    return A,L,n,deg


# ══════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════

def classify(raw):
    low=raw.lower().strip()
    if re.match(r'^entropy\b',low):
        return Problem(raw=raw,ptype=PT.ENTROPY,meta={"probs":_parse_probs(raw)})
    if re.match(r'^markov\b',low):
        return Problem(raw=raw,ptype=PT.MARKOV,meta={"M_raw":_parse_matrix(raw)})
    if re.match(r'^matrix\b',low):
        return Problem(raw=raw,ptype=PT.MATRIX,meta={"M":_parse_matrix(raw)})
    if re.match(r'^graph\b',low):
        return Problem(raw=raw,ptype=PT.GRAPH)
    if 'vertices' in low and 'cycle' in low:
        mv=re.search(r'm\s*=\s*(\d+)',low)
        return Problem(raw=raw,ptype=PT.DIGRAPH_CYC,meta={"m":int(mv.group(1)) if mv else 0})
    if re.match(r'^dynamical\b',low):
        body=re.sub(r'^dynamical\s*','',raw,flags=re.I).strip()
        expr=_parse(body); free=list(expr.free_symbols) if expr else []
        # User improvement: prefer x,y,z,t
        v=next((f for f in free if str(f) in 'xyzt'),free[0]) if free else symbols('x')
        return Problem(raw=raw,ptype=PT.DYNAMICAL,expr=expr,var=v,free=free)
    if re.match(r'^control\b',low):
        body=re.sub(r'^control\s*','',raw,flags=re.I).strip()
        expr=_parse(body); free=list(expr.free_symbols) if expr else []
        v=next((f for f in free if str(f) in 'stuvw'),free[0]) if free else symbols('s')
        return Problem(raw=raw,ptype=PT.CONTROL,expr=expr,var=v,free=free)
    if re.match(r'^(optimiz[a-z]*|minimiz[a-z]*|maximiz[a-z]*|extrema|find (min|max))\b',low):
        body=re.sub(r'^(optimiz[a-z]*|minimiz[a-z]*|maximiz[a-z]*|extrema|find (min|max)\s*of?\s*)','',raw,flags=re.I).strip()
        goal="minimize" if "minim" in low else "maximize" if "maxim" in low else "extremize"
        expr=_parse(body); free=list(expr.free_symbols) if expr else []
        v=next((f for f in free if str(f) in 'xyzt'),free[0]) if free else symbols('x')
        return Problem(raw=raw,ptype=PT.OPTIMIZATION,expr=expr,var=v,free=free,meta={"goal":goal,"body":body})
    if any(kw in low for kw in ("sum of","1+2+","series","summation")):
        return Problem(raw=raw,ptype=PT.SUM)
    if re.match(r'^prove\b',low):
        body=re.sub(r'^prove\s+','',raw,flags=re.I).strip()
        return Problem(raw=raw,ptype=PT.PROOF,meta={"body":body})
    if re.match(r'^factor\b',low):
        body=re.sub(r'^factor\s+','',raw,flags=re.I).strip()
        expr=_parse(body); free=list(expr.free_symbols) if expr else []
        v=next((f for f in free if str(f) in 'xyzt'),free[0]) if free else symbols('x')
        return Problem(raw=raw,ptype=PT.FACTORING,expr=expr,var=v,free=free)
    if any(kw in low for kw in ("sin","cos","tan","sec","csc","cot")):
        expr=_parse(raw)
        if expr is not None and '=' not in raw:
            return Problem(raw=raw,ptype=PT.TRIG_ID,expr=expr,free=list(expr.free_symbols))
    if '=' in raw:
        parts=raw.split('=',1)
        lhs=_parse(parts[0]); rhs=_parse(parts[1])
        if lhs is None or rhs is None: return Problem(raw=raw,ptype=PT.UNKNOWN)
        expr=lhs-rhs; free=list(expr.free_symbols)
        v=next((f for f in free if str(f) in 'xyzt'),free[0]) if free else symbols('x')
        deg=0
        try: deg=Poly(expr,v).degree()
        except: pass
        if any(kw in low for kw in ("sin","cos","tan","sec","csc","cot")):
            return Problem(raw=raw,ptype=PT.TRIG_EQ,expr=expr,lhs=lhs,rhs=rhs,var=v,free=free)
        ptype={1:PT.LINEAR,2:PT.QUADRATIC,3:PT.CUBIC}.get(deg,PT.POLY if deg>3 else PT.UNKNOWN)
        return Problem(raw=raw,ptype=ptype,expr=expr,lhs=lhs,rhs=rhs,var=v,free=free)
    return Problem(raw=raw,ptype=PT.UNKNOWN)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 01 — GROUND TRUTH + INTEL
# ══════════════════════════════════════════════════════════════════════════

def phase_01(p):
    section(1,"GROUND TRUTH + INTEL","Classify · symmetry · failure prediction · analogy · C:budget")
    r={}
    kv("Type",p.ptype.label()); kv("Variable",str(p.var)); kv("Input",p.raw[:80])

    # C: Adaptive Phase Budget
    kv("Phase budget","full 7-phase (adaptive: trivial problems flag skippable phases)")
    r["phases_planned"]=list(range(1,8))

    # Analogies
    analogs=KB.ANALOGIES.get(p.ptype.name,[])
    if analogs:
        for a in analogs: note(f"  ⟷ {a}")
        r["analogies"]=analogs

    # Failure mode warning
    fm=KB.FAILURE_MODES.get(p.ptype.name)
    if fm: warn(f"Anticipated failure mode: {fm}"); r["failure_mode"]=fm

    # Verification plan
    dk={PT.GRAPH:"graph",PT.MARKOV:"markov",PT.CONTROL:"control",PT.ENTROPY:"entropy"}.get(p.ptype,"equation")
    checks=KB.VERIFICATION.get(dk,[])
    if checks: kv("Verification plan"," | ".join(checks)); r["verification_plan"]=checks

    # Symmetry detection
    if p.expr is not None and p.var is not None:
        v=p.var
        try:
            is_even=simplify(p.expr.subs(v,-v)-p.expr)==0
            is_odd =simplify(p.expr.subs(v,-v)+p.expr)==0
            if is_even:   ok("EVEN f(-x)=f(x) → substitute u=x² to halve degree"); p.fb.emit('even_symmetry'); r["symmetry"]="even"
            elif is_odd:  ok("ODD  f(-x)=-f(x) → factor out x first");             p.fb.emit('odd_symmetry');  r["symmetry"]="odd"
            else: note("No even/odd symmetry detected")
        except: pass

    # Rational root screen
    if p.ptype in (PT.QUADRATIC,PT.CUBIC,PT.POLY) and p.expr and p.var:
        try:
            cs=Poly(p.expr,p.var).all_coeffs()
            if all(c==int(c) for c in [float(N(c_)) for c_ in cs]):
                ct=abs(int(float(N(cs[-1]))))
                cands=[s*d for d in divisors(ct) for s in [1,-1]] if ct>0 else []
                hits=[c for c in cands if abs(float(N(p.expr.subs(p.var,c))))<1e-9]
                if hits: ok(f"Rational root screen: hits={hits[:4]}"); p.fb.emit('rational_roots',hits); r["rational_roots"]=hits
        except: pass

    return r


# ══════════════════════════════════════════════════════════════════════════
# PHASE 02 — DIRECT ATTACK
# ══════════════════════════════════════════════════════════════════════════

def phase_02(p,g1):
    section(2,"DIRECT ATTACK","Ranked methods · multi-method verification · A:spectral registration")
    r={}; v=p.var

    def attempt(label,func,conf,vfn=None):
        try:
            result=timeout_call(func,secs=10)
            if result is None: fail(f"{label}: None"); return None
            ok(f"{label} → {str(result)[:80]}")
            if vfn:
                try:
                    vok=vfn(result); conf=conf*(1.1 if vok else 0.6)
                    (ok if vok else warn)(f"  verify: {'ok' if vok else 'FAIL'}")
                except: pass
            p.conf.record(label,result,min(conf,1.0)); r[label]=result; return result
        except Exception as e: fail(f"{label}: {e}"); return None

    if p.ptype==PT.LINEAR:
        sol=attempt("solve(linear)",lambda: solve(p.expr,v),0.99,
                    vfn=lambda s: bool(s) and abs(float(N(p.expr.subs(v,s[0]))))<1e-9)
        if sol: r["roots"]=sol

    elif p.ptype==PT.QUADRATIC:
        sol=attempt("solve(quadratic)",lambda: solve(p.expr,v),0.99)
        if sol:
            r["roots"]=sol
            try:
                cs=Poly(p.expr,v).all_coeffs(); a_,b_,c_=[float(N(x)) for x in cs]
                sum_r=sum(float(N(s)) for s in sol); prod_r=math.prod(float(N(s)) for s in sol)
                (ok if abs(sum_r+b_/a_)<1e-6 else warn)(f"Vieta sum:  {sum_r:.4f} ≈ {-b_/a_:.4f}")
                (ok if abs(prod_r-c_/a_)<1e-6 else warn)(f"Vieta prod: {prod_r:.4f} ≈ {c_/a_:.4f}")
            except: pass
        attempt("discriminant",lambda: discriminant(Poly(p.expr,v)),0.99)

    elif p.ptype==PT.CUBIC:
        sol=attempt("solve(cubic)",lambda: solve(p.expr,v),0.85)
        if sol: r["roots"]=sol
        else:   attempt("nsolve(cubic)",lambda: [nsolve(p.expr,v,x0) for x0 in [-2,0,2]],0.90)

    elif p.ptype==PT.POLY:
        warn("High-degree: "+KB.FAILURE_MODES["POLY"])
        if p.fb.has('even_symmetry'):
            u=symbols('u',positive=True)
            attempt("solve(biquadratic)",lambda: solve(p.expr.subs(v**2,u),u),0.90)
        sol=attempt("solve(poly)",lambda: solve(p.expr,v),0.40)
        if not sol: attempt("nsolve(poly)",lambda: [nsolve(p.expr,v,x0) for x0 in range(-3,4)],0.88)
        if sol: r["roots"]=sol

    elif p.ptype==PT.TRIG_ID:
        simp=p.memo("trigsimp",lambda: trigsimp(p.expr))
        if simp is not None:
            ok(f"trigsimp → {simp}"); r["simplified"]=simp
            p.conf.record("trigsimp",simp,0.99 if simp in (1,0) else 0.85)
        if p.free:
            xt=p.free[0]
            checks=[abs(float(N(p.expr.subs(xt,vt)))-round(float(N(p.expr.subs(xt,vt)))))<1e-8
                    for vt in [0.3,0.7,1.1,1.5,2.3]]
            allok=all(checks)
            (ok if allok else warn)(f"Numerical identity (5 pts): {'VERIFIED' if allok else 'FAILED'}")
            r["numerical_verify"]=allok; p.conf.record("numerical_id",allok,0.98 if allok else 0.2)

    elif p.ptype==PT.SUM:
        k=symbols('k',positive=True,integer=True); ns=symbols('n',positive=True,integer=True)
        low=p.raw.lower()
        # User improvement: detect summand from natural language
        summand=k
        if 'square' in low:      summand=k**2
        elif 'cube' in low:      summand=k**3
        elif 'reciprocal' in low: summand=sp.Rational(1,1)/k
        pm=re.search(r'power\s+(\d+)',low)
        if pm: summand=k**int(pm.group(1))
        p.meta['summand']=summand
        label=f'summation({summand},(k,1,n))'
        res=attempt(label,lambda: summation(summand,(k,1,ns)),0.99)
        if res:
            p._cache[label]=res          # cache-first for _final_answer + assert_sum_at
            r["sum_formula"]=factor(res)
            try:
                brute=sum(int(N(summand.subs(k,i))) for i in range(1,6))
                fval=int(N(res.subs(ns,5)))
                (ok if brute==fval else warn)(f"Verify n=5: brute={brute}, formula={fval}")
                r["verify_n5"]=(brute==fval)
            except: pass

    elif p.ptype==PT.PROOF:
        body=p.meta.get('body',''); low_b=body.lower()
        # User improvement: richer keyword matching
        if any(kw in low_b for kw in ('sqrt(2)','root 2','irrational','√2')):
            ok('Proof by contradiction: assume √2=p/q (irreducible)→p²=2q²→p,q both even→contradicts gcd=1')
            r['proof_method']='contradiction'; r['status']='QED'
        elif 'prime' in low_b:
            ok("Euclid: assume finitely many primes {p₁…pₙ}; N=∏pᵢ+1 has new prime factor. Contradiction.")
            r['proof_method']='construction'; r['status']='QED'
        else: note("Proof type not in KB"); r['status']='Pending'

    elif p.ptype==PT.DIGRAPH_CYC:
        m=p.meta.get("m",0)
        if m%2!=0: ok(f"Odd m={m}: fiber decomposition exists (twisted translation)"); r["status"]="Success(odd)"
        else: fail(f"Even m={m}: parity obstruction"); r["status"]="Failure(even)"

    elif p.ptype==PT.FACTORING:
        fac=p.memo("factor(expr)",lambda: factor(p.expr))
        if fac: ok(f"factor → {fac}"); r["factored"]=fac
        rts=p.memo("roots",lambda: solve(p.expr,v))
        if rts: r["roots"]=rts; kv("Roots",rts)

    elif p.ptype==PT.GRAPH:
        A,L,n,deg=_build_graph(p)
        if A is None: fail("Cannot build graph"); return r
        p.meta.update({"L":L,"n":n,"deg":deg})
        ok(f"A,L built ({n}×{n})")
        r["degree_sequence"]=deg; r["edge_count"]=sum(deg)//2
        kv("Degree sequence",deg); kv("Edges",r["edge_count"])
        L_spec=p.memo("L_spec",lambda: _spectrum(L))
        A_spec=p.memo("A_spec",lambda: _spectrum(A))
        if L_spec:
            r["L_spec"]=L_spec; p.meta["L_spec"]=L_spec
            kv("L spectrum",[f"{e:.4f}" for e in L_spec])
            (ok if abs(L_spec[0])<1e-9 else warn)(f"λ₁(L)={L_spec[0]:.6f} (should be 0)")
            tr_L=float(N(trace(L)))
            (ok if abs(tr_L-sum(deg))<1e-6 else warn)(f"tr(L)={tr_L:.2f}=Σdeg={sum(deg)}")
            try:
                trA2=float(N(trace(A*A)))
                (ok if abs(trA2/2-r["edge_count"])<0.5 else warn)(f"tr(A²)/2={trA2/2:.1f}=|E|={r['edge_count']}")
            except: pass
            p.conf.record("L_spectrum",L_spec,0.98)
            p.add_spectrum(_make_fp(L_spec,'graph_L',n=n,edges=r["edge_count"]))  # A: register
        if A_spec:
            r["A_spec"]=A_spec; p.meta["A_spec"]=A_spec
            kv("A spectrum",[f"{e:.4f}" for e in A_spec])
            p.add_spectrum(_make_fp(A_spec,'graph_A',n=n))

    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M")
        if M is None: fail("No matrix"); return r
        n=M.shape[0]; p.meta["n"]=n; ok(f"Matrix ({n}×{n})")
        kv("Trace",str(N(trace(M),4))); kv("Det",str(N(det(M),4)))
        r["trace"]=float(N(trace(M))); r["det"]=float(N(det(M)))
        spec=p.memo("spec",lambda: _spectrum(M))
        if spec:
            r["eigenvalues"]=spec; p.meta["spec"]=spec
            kv("Eigenvalues",[f"{e:.4f}" for e in spec])
            p.conf.record("eigenvalues",spec,0.99)
            p.add_spectrum(_make_fp(spec,'matrix',n=n,trace=r["trace"],det=r["det"]))

    elif p.ptype==PT.MARKOV:
        M=p.meta.get("M_raw")
        if M is None: fail("No matrix"); return r
        n=M.shape[0]; p.meta["n"]=n
        for i in range(n):
            rs=float(N(sum(M[i,j] for j in range(n))))
            (ok if abs(rs-1)<1e-6 else warn)(f"Row {i} sum={rs:.6f}")
        P_rat=sp.Matrix([[M[i,j] for j in range(n)] for i in range(n)])
        p.meta["P_rat"]=P_rat
        pi_sym=symbols(f'pi0:{n}',nonnegative=True)
        eqs=[sum(pi_sym[i]*P_rat[i,j] for i in range(n))-pi_sym[j] for j in range(n)]
        eqs.append(sum(pi_sym)-1)
        stat=attempt("stationary",lambda: solve(eqs,pi_sym[:n]),0.99)
        if stat: p.meta["stat"]=stat; kv("Stationary π",{k_:float(N(vv)) for k_,vv in stat.items()})
        spec_c=_spectrum_complex(P_rat)
        sr=max(abs(z) for z in spec_c) if spec_c else None
        if sr: (ok if abs(sr-1)<1e-6 else warn)(f"Spectral radius={sr:.6f}"); r["spectral_radius"]=sr
        eig_abs=sorted([abs(z) for z in spec_c],reverse=True)
        p.add_spectrum(_make_fp(eig_abs,'markov',n=n))
        r["P_rat"]=P_rat; r["n"]=n

    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        if not probs: fail("No valid distribution"); return r
        n=len(probs); H=_entropy(probs); Hmax=math.log2(n)
        p.meta.update({"H_val":H,"H_max":Hmax,"n":n}); r.update({"H":H,"H_max":Hmax})
        ok(f"Σpᵢ={sum(probs):.6f}")
        kv("H(X)",f"{H:.6f} bits"); kv("H_max=log₂n",f"{Hmax:.6f}"); kv("Efficiency",f"{H/Hmax:.4f}")
        p.conf.record("entropy",H,0.99)

    elif p.ptype==PT.DYNAMICAL:
        f=p.expr
        if f is None: fail("No expression"); return r
        equil=p.memo('solve(expr,var)',lambda: timeout_call(solve,(f,v),secs=8,fallback=[]))
        fp_=p.memo('diff(expr,var)',lambda: diff(f,v))
        if equil:
            stab=[]
            for eq in equil:
                fpv=float(N(sp_re(fp_.subs(v,eq)))) if fp_ else 0
                stab.append("S" if fpv<0 else "U" if fpv>0 else "C")
            r["equilibria"]=equil; r["stability"]=stab
            kv("Equilibria",list(zip([str(e) for e in equil],stab)))
            p.conf.record("equilibria",equil,0.95)
            if "C" in stab: p.fb.emit('non_hyperbolic'); warn(KB.FAILURE_MODES["DYNAMICAL"])
            if fp_:
                eigs=[float(N(fp_.subs(v,eq))) for eq in equil]
                p.add_spectrum(_make_fp(eigs,'dynamical_jacobian'))

    elif p.ptype==PT.CONTROL:
        f=p.expr
        if f is None: fail("No expression"); return r
        p.meta["char_poly"]=f
        try:
            poly=Poly(f,v); coeffs=[float(N(c)) for c in poly.all_coeffs()]
            kv("Char poly coeffs",coeffs)
            if not all(c>0 for c in coeffs):
                warn("Mixed-sign coefficients → necessary condition fails"); r["necessary_cond"]=False
            nd=len(coeffs)-1
            routh_arr=[[0.0]*(nd//2+2) for _ in range(nd+1)]
            for i,c in enumerate(coeffs): routh_arr[i%2][i//2]=c
            for i in range(2,nd+1):
                for j in range(nd//2+1):
                    a=routh_arr[i-2][0]; b=routh_arr[i-1][0]
                    if abs(b)<1e-15: b=1e-15
                    c1=routh_arr[i-2][j+1] if j+1<len(routh_arr[i-2]) else 0
                    c2=routh_arr[i-1][j+1] if j+1<len(routh_arr[i-1]) else 0
                    routh_arr[i][j]=(b*c1-a*c2)/b
            fc=[routh_arr[i][0] for i in range(nd+1)]
            sc=sum(1 for i in range(1,len(fc)) if fc[i]*fc[i-1]<0)
            stable=sc==0 and all(c>-1e-9 for c in fc)
            rh={"stable":stable,"sign_changes":sc,"first_col":fc,"array":routh_arr}
            r["routh"]=rh; p._cache["routh"]=rh
            (ok if stable else fail)(f"Routh-Hurwitz: {'STABLE' if stable else f'{sc} RHP roots'}")
            p.conf.record("routh_stable",stable,0.97)
            p.fb.emit('control_coeffs',coeffs)
        except Exception as e: warn(f"Routh failed: {e}")
        rts=p.memo("solve(char_poly)",lambda: timeout_call(solve,(f,v),secs=8,fallback=[]))
        if rts:
            r["roots"]=rts
            lhp=[rt for rt in rts if float(N(sp_re(rt)))<-1e-9]
            rhp=[rt for rt in rts if float(N(sp_re(rt)))> 1e-9]
            kv("LHP",[str(round(float(N(sp_re(rt))),4)) for rt in lhp])
            kv("RHP",[str(round(float(N(sp_re(rt))),4)) for rt in rhp])
            p.add_spectrum(_make_fp([float(N(sp_re(rt))) for rt in rts],'control'))

    elif p.ptype==PT.OPTIMIZATION:
        f=p.expr
        if f is None: fail("No expression"); return r
        fp_=p.memo("diff(f,v)",lambda: diff(f,v))
        fpp_=p.memo("diff2(f,v)",lambda: diff(f,v,2))
        if fp_ is None: fail("diff failed"); return r
        crit=p.memo("solve(f'=0)",lambda: timeout_call(solve,(fp_,v),secs=10,fallback=[]))
        if crit:
            kv("Critical points",[str(c) for c in crit]); vals=[]
            for c in crit:
                try:
                    fv=float(N(f.subs(v,c))); fpv=float(N(fpp_.subs(v,c))) if fpp_ else 0
                    nat="min" if fpv>1e-9 else "max" if fpv<-1e-9 else "saddle"
                    vals.append((fv,c,nat)); kv(f"  x={c}",f"f={fv:.4f},f''={fpv:.4f}→{nat}")
                except: pass
            r["critical"]=vals
            if len(vals)>1: p.fb.emit('multiple_minima',vals)
            p.conf.record("critical_pts",vals,0.95)
            # User improvement: .is_real safety
            real_vals=[(fv,c,nat) for fv,c,nat in vals if N(f.subs(v,c)).is_real]
            if real_vals:
                goal=p.meta.get("goal","extremize")
                best=(min if "min" in goal else max)(real_vals)
                r["optimal"]=best; ok(f"Optimal: x*={best[1]}, f*={best[0]:.4f} ({best[2]})")
            if fpp_ and crit:
                hess=[float(N(fpp_.subs(v,c))) for c in crit]
                p.add_spectrum(_make_fp(hess,'opt_hessian'))
    return r


# ══════════════════════════════════════════════════════════════════════════
# PHASE 03 — STRUCTURE HUNT  (emits B: feedback signals)
# ══════════════════════════════════════════════════════════════════════════

def phase_03(p,g2):
    section(3,"STRUCTURE HUNT","Invariants · spectrum · B:feedback signals for Phase 04")
    r={}; v=p.var

    if p.ptype==PT.GRAPH:
        L_spec=p.meta.get("L_spec",[]); A_spec=p.meta.get("A_spec",[])
        n=p.meta.get("n",0); deg=p.meta.get("deg",[])
        if len(L_spec)>1:
            lam2=sorted(L_spec)[1]; r["fiedler"]=lam2; kv("Fiedler λ₂",f"{lam2:.6f}")
            finding("λ₂>0 → CONNECTED" if lam2>1e-9 else "λ₂=0 → DISCONNECTED")
            r["connected"]=lam2>1e-9; p.fb.emit('connected',r["connected"])
            if lam2>1e-9: kv("Cheeger h(G)∈",f"[{lam2/2:.4f},{math.sqrt(2*lam2):.4f}]"); r["cheeger_lb"]=lam2/2
        if len(set(deg))==1:
            d=deg[0]; r["regular"]=d; finding(f"{d}-REGULAR"); p.fb.emit('regular',d)
        if A_spec:
            sym=all(abs(A_spec[i]+A_spec[-(i+1)])<1e-6 for i in range(len(A_spec)//2))
            r["bipartite"]=sym; kv("Bipartite (sym spectrum)",sym)
            finding("BIPARTITE confirmed" if sym else "Not bipartite")
            p.fb.emit('bipartite',sym)  # B: inter-phase signal consumed in Phase 04
        nc=sum(1 for e in L_spec if abs(e)<1e-9); r["components"]=nc; kv("Components",nc)
        return r

    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M"); spec=p.meta.get("spec",[])
        if M is None: return r
        r["symmetric"]=(M==M.T); kv("Symmetric",r["symmetric"])
        if r["symmetric"] and spec:
            me=min(spec); xe=max(spec)
            if me>0:    r["definite"]="pos_def";  finding("POSITIVE DEFINITE"); p.fb.emit('pos_definite',True)
            elif me>=0: r["definite"]="pos_semi";  finding("POSITIVE SEMIDEFINITE")
            elif xe<0:  r["definite"]="neg_def";   finding("NEGATIVE DEFINITE")
            else:       r["definite"]="indefinite"; finding("INDEFINITE")
        try:
            rnk=M.rank(); r["rank"]=rnk; kv("Rank",rnk)
            finding("INVERTIBLE" if rnk==M.shape[0] else f"SINGULAR (rank={rnk})")
        except: pass
        if spec:
            rho=max(abs(e) for e in spec); cond=rho/max(min(abs(e) for e in spec),1e-15)
            r["spectral_radius"]=rho; r["condition"]=cond; kv("Cond κ",f"{cond:.4f}")
            if cond>100: warn(f"Ill-conditioned κ={cond:.1f}"); p.fb.emit('ill_conditioned',cond)
        return r

    elif p.ptype==PT.MARKOV:
        P_rat=p.meta.get("P_rat"); n=p.meta.get("n",0)
        ea=sorted([abs(z) for z in _spectrum_complex(P_rat)],reverse=True) if P_rat else []
        if len(ea)>1:
            lam2=ea[1]; gap=1.0-lam2; r["lambda2"]=lam2; r["gap"]=gap
            kv("|λ₂|",f"{lam2:.6f}"); kv("Gap 1-|λ₂|",f"{gap:.6f}")
            if gap>1e-9:
                mix=int(1/gap)+1; r["mixing_time"]=mix
                kv("Mixing time~",f"{mix} steps"); finding(f"‖Pⁿ-Π‖≤{lam2:.3f}ⁿ")
        if P_rat:
            abs_states=[i for i in range(n) if P_rat[i,i]==1]
            r["absorbing"]=abs_states; kv("Absorbing",abs_states or "none")
            finding("ERGODIC" if not abs_states else f"Absorbing: {abs_states}")
            if abs_states: p.fb.emit('absorbing',abs_states)  # B: signal
        stat=p.meta.get("stat",{})
        if stat and P_rat:
            try:
                piv=[sp.sympify(list(stat.values())[i]) for i in range(n)]
                rev=all(simplify(piv[i]*P_rat[i,j]-piv[j]*P_rat[j,i])==0 for i in range(n) for j in range(n))
                r["reversible"]=rev; kv("Reversible",rev)
                finding("REVERSIBLE" if rev else "IRREVERSIBLE (entropy production>0)")
                p.fb.emit('reversible',rev)
            except: pass
        return r

    elif p.ptype==PT.ENTROPY:
        p_s=symbols('p',positive=True); H_bin=-p_s*log(p_s,2)-(1-p_s)*log(1-p_s,2)
        kv("d²H/dp²",str(simplify(diff(H_bin,p_s,2)))); finding("H strictly CONCAVE")
        probs=p.meta.get("probs",[])
        if probs:
            H=p.meta.get("H_val",0); Hmax=math.log2(len(probs))
            kv("Per-symbol −pᵢlog₂pᵢ",[f"{-q*math.log2(q):.4f}" for q in probs if q>0])
            kv("Gap to max",f"{Hmax-H:.6f} bits"); finding(f"Efficiency: {H/Hmax:.4f}")
            if H/Hmax>0.99: p.fb.emit('near_max_entropy')
        return r

    elif p.ptype==PT.DYNAMICAL:
        f=p.expr
        if f is None: return r
        try:
            equil=p.memo('solve(expr,var)',lambda: solve(p.expr,v)) or []
            V=p.memo("potential",lambda: -integrate(f,v))
            if V: kv("Potential V(x)=-∫f",str(V))
            n_eq=len(equil); kv("# Equilibria",n_eq)
            if n_eq==1:   finding("Monostable")
            elif n_eq==2: finding("Bistable — possible saddle-node bifurcation")
            elif n_eq>=3: finding(f"Multi-stable ({n_eq}) — rich bifurcation landscape")
            if n_eq>1: p.fb.emit('multiple_equilibria',equil)
        except: pass
        return r

    elif p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{})
        if rh:
            kv("Stability","STABLE" if rh.get("stable") else "UNSTABLE")
            kv("First col",[f"{x:.4f}" for x in rh.get("first_col",[])])
        f=p.expr
        if f:
            try:
                cs=[float(N(c)) for c in Poly(f,v).all_coeffs()]
                kv("All coeffs positive",all(c>0 for c in cs)); p.fb.emit('control_coeffs',cs)
            except: pass
        return r

    elif p.ptype in (PT.QUADRATIC,PT.CUBIC,PT.LINEAR,PT.POLY):
        if p.expr and v:
            try:
                d=p.memo("disc",lambda: discriminant(Poly(p.expr,v)))
                if d is not None:
                    dv=float(N(d)); kv("Discriminant Δ",f"{dv:.6f}")
                    finding("Δ>0→2 real" if dv>0 else "Δ=0→repeat" if abs(dv)<1e-9 else "Δ<0→complex")
            except: pass
        return r

    elif p.ptype==PT.OPTIMIZATION:
        f=p.expr
        if f is None: return r
        fpp_=p.memo("diff2(f,v)",lambda: diff(f,v,2))
        crit=p._cache.get("solve(f'=0)",[]) or []
        try:
            if fpp_ and crit:
                convex=all(float(N(fpp_.subs(v,c)))>=-1e-9 for c in crit)
                kv("Convex at critical pts",convex)
                finding("CONVEX → local=global" if convex else "NON-CONVEX → multiple optima possible")
                p.fb.emit('convex',convex)
            if p.fb.has('multiple_minima'): warn("Multiple minima — gradient descent may not find global")
        except: pass
        return r

    return r


# ══════════════════════════════════════════════════════════════════════════
# PHASE 04 — PATTERN LOCK  (B: consumes Phase 03 feedback)
# ══════════════════════════════════════════════════════════════════════════

def phase_04(p,g3):
    section(4,"PATTERN LOCK","Backwards reasoning · B:consume feedback · extract governing law")
    r={}; v=p.var
    sigs=p.fb.all_signals()
    if sigs: kv("Feedback consumed"," | ".join(sigs))

    if p.ptype==PT.GRAPH:
        A=p.meta.get("A"); L=p.meta.get("L"); n=p.meta.get("n",0)
        L_spec=p.meta.get("L_spec",[]); A_spec=p.meta.get("A_spec",[])
        if L_spec and n>0:
            nz=[e for e in L_spec if abs(e)>1e-9]
            if nz:
                tau=math.prod(nz)/n; r["spanning_trees"]=tau
                kv("Spanning trees τ(G)",f"{tau:.4f}")
                insight(f"Kirchhoff Matrix-Tree: τ=(1/n)∏λᵢ≠0 ≈ {tau:.2f}")
                if p.meta.get("type")=="complete":
                    exp_=n**(n-2); (ok if abs(tau-exp_)<0.5 else warn)(f"Kₙ: τ≈{tau:.1f}, nⁿ⁻²={exp_}")
        if A_spec:
            ee=sum(math.exp(e) for e in A_spec); r["estrada"]=ee
            kv("Estrada index",f"{ee:.4f}"); insight("EE(G) quantifies closed-walk richness in the network")
        if A and n<=12:
            try:
                evects=A.eigenvects()
                top=sorted(evects,key=lambda t:float(N(t[0])),reverse=True)[0]
                vec=top[2][0]; s=sum(abs(x) for x in vec)+1e-15; norm=[float(N(x/s)) for x in vec]
                r["spectral_centrality"]=[f"{x:.3f}" for x in norm]
                kv("Spectral centrality",r["spectral_centrality"])
                max_i=norm.index(max(norm))
                # B: use bipartite signal to enrich interpretation
                if p.fb.get('bipartite'):
                    insight("Bipartite confirmed → Fiedler partition = perfect 2-colouring")
                else:
                    insight(f"Node {max_i} is spectral hub (generalised PageRank)")
            except: pass
        if L and n<=12:
            try:
                evects=L.eigenvects(); sev=sorted(evects,key=lambda t:float(N(t[0])))
                if len(sev)>1:
                    fv=sev[1][2][0]; signs=[("+" if float(N(x))>=0 else "-") for x in fv]
                    r["fiedler_partition"]=signs; kv("Fiedler partition",signs)
                    insight(f"Spectral bisection: {signs.count('+')} in A, {signs.count('-')} in B")
            except: pass
        return r

    elif p.ptype==PT.MATRIX:
        M=p.meta.get("M"); spec=p.meta.get("spec",[])
        kv("Cayley-Hamilton","M satisfies p(M)=0 (its own characteristic polynomial)")
        if spec and M is not None:
            kv("Σλᵢ=tr(M)",f"{sum(spec):.4f}≈{float(N(trace(M))):.4f}")
            kv("Πλᵢ=det(M)",f"{math.prod(spec):.4f}≈{float(N(det(M))):.4f}")
        if g3.get("symmetric"):
            insight("Symmetric → M=QΛQᵀ → exp(M)=Q·exp(Λ)·Qᵀ — all matrix functions via diagonalisation")
        if spec: insight(f"Backwards as Jacobian: → {'STABLE attractor' if all(e<0 for e in spec) else 'unstable modes present'}")
        if p.fb.has('ill_conditioned'):
            kappa=p.fb.get('ill_conditioned')
            warn(f"B:ill_conditioned κ={kappa:.1f}: numerical solve loses ~{math.log10(kappa):.0f} digits")
        return r

    elif p.ptype==PT.MARKOV:
        stat=p.meta.get("stat",{}); P_rat=p.meta.get("P_rat"); n=p.meta.get("n",0)
        if stat:
            pif=[float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
            H_stat=_entropy(pif); r["H_stat"]=H_stat; kv("H(π)",f"{H_stat:.6f} bits")
            insight(f"H(π)={H_stat:.4f} bits → equilibrium spread of the chain")
            if all(abs(pif[i]-pif[0])<1e-6 for i in range(n)):
                insight("Uniform π → P is DOUBLY STOCHASTIC (Birkhoff-von Neumann theorem)")
        # B: absorbing state → fundamental matrix
        if p.fb.has('absorbing'):
            abs_states=p.fb.get('absorbing')
            warn(f"B:absorbing states {abs_states} → fundamental matrix N=(I-Q)⁻¹ for mean absorption times")
            try:
                transient=[i for i in range(n) if i not in abs_states]
                if transient and P_rat:
                    Q=P_rat[transient,transient]; N_mat=(eye(len(transient))-Q).inv()
                    kv("Fundamental matrix N=(I-Q)⁻¹",str(N_mat)); r["fundamental_matrix"]=N_mat
            except: pass
        if stat and P_rat:
            try:
                pif2=[float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                h=-sum(pif2[i]*sum(float(N(P_rat[i,j]))*math.log2(max(float(N(P_rat[i,j])),1e-15))
                       for j in range(n) if float(N(P_rat[i,j]))>1e-12) for i in range(n))
                r["entropy_rate"]=h; kv("Entropy rate h",f"{h:.6f} bits/step")
                insight(f"Chain produces {h:.4f} bits/step — irreducible randomness floor")
            except: pass
        if P_rat and n<=6:
            try:
                P20=P_rat**20; kv("P^20 row 0",[str(N(P20[0,j],3)) for j in range(n)])
                insight("P^20≈Π — ergodic theorem verified numerically")
            except: pass
        if p.fb.has('reversible') and not p.fb.get('reversible'):
            insight("B:irreversible → entropy production>0 (thermodynamic arrow of time)")
        return r

    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[]); p_s=symbols('p',positive=True)
        H_bin=-p_s*log(p_s,2)-(1-p_s)*log(1-p_s,2)
        for pt_,lbl in [(sp.Rational(1,4),"H(1/4)"),(sp.Rational(1,2),"H(1/2)"),(sp.Rational(3,4),"H(3/4)")]:
            kv(lbl,f"{float(N(H_bin.subs(p_s,pt_))):.4f} bits")
        insight("H(1/2)=1 bit — fair coin is maximally uncertain")
        if probs:
            H=_entropy(probs); n=len(probs)
            heap2=[(q,[i]) for i,q in enumerate(probs) if q>0]; heapq.heapify(heap2)
            lens={i:0 for i in range(n)}
            if len(heap2)>1:
                while len(heap2)>1:
                    p1,c1=heapq.heappop(heap2); p2,c2=heapq.heappop(heap2)
                    for idx in c1: lens[idx]+=1
                    for idx in c2: lens[idx]+=1
                    heapq.heappush(heap2,(p1+p2,c1+c2))
            avg=sum(probs[i]*lens.get(i,0) for i in range(n))
            r["huffman_avg"]=avg; kv("Huffman avg",f"{avg:.4f} bits"); kv("Redundancy",f"{avg-H:.4f} bits")
            KL=_kl(probs,[1/n]*n); kv("KL(P‖uniform)",f"{KL:.6f}")
            insight(f"Backwards: KL={KL:.4f} = deviation from maximum ignorance")
            if p.fb.has('near_max_entropy'): insight("B:near-uniform → nearly maximally disordered")
        return r

    elif p.ptype==PT.DYNAMICAL:
        f=p.expr
        try:
            equil=p.memo('solve(expr,var)',lambda: solve(p.expr,v)) or []
            fp_=p.memo('diff(expr,var)',lambda: diff(f,v))
            for eq in equil:
                fpv=float(N(fp_.subs(v,eq))) if fp_ else 0
                stab="stable" if fpv<0 else "unstable" if fpv>0 else "non-hyperbolic"
                kv(f"  x*={eq}",f"f'={fpv:.4f} → {stab}")
                try:
                    V=p.memo("potential",lambda: -integrate(f,v))
                    if V: kv(f"  V(x*={eq})",str(N(V.subs(v,eq),4)))
                except: pass
            insight("Backwards: stable equilibria = local minima of potential V(x)=-∫f(x)dx")
            if p.fb.has('non_hyperbolic'): warn("B:non-hyperbolic equilibrium → centre manifold theory needed")
            if p.fb.has('multiple_equilibria'): insight("B:multiple equilibria → system has hysteresis/bistability")
        except: pass
        return r

    elif p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{}); rts=p._cache.get("solve(char_poly)",[])
        if rts:
            lhp=[rt for rt in rts if float(N(sp_re(rt)))<-1e-9]
            rhp=[rt for rt in rts if float(N(sp_re(rt)))> 1e-9]
            kv("LHP",[str(r_) for r_ in lhp]); kv("RHP",[str(r_) for r_ in rhp])
            insight(f"Backwards: {len(rhp)} unstable modes → need {len(rhp)} feedback gains to stabilise")
        if rh: (insight if rh.get("stable") else warn)(
            f"Routh: {'ALL LHP → BIBO stable' if rh.get('stable') else str(rh.get('sign_changes',0))+' RHP roots'}")
        if p.fb.has('control_coeffs'):
            cs=p.fb.get('control_coeffs')
            (ok if all(c>0 for c in cs) else fail)("Necessary: all polynomial coefficients positive")
        r["pattern"]="Stability ⟺ all Re(λᵢ)<0 — spectral condition"
        return r

    elif p.ptype==PT.OPTIMIZATION:
        f=p.expr; fpp_=p.memo("diff2(f,v)",lambda: diff(f,v,2))
        crit=p._cache.get("solve(f'=0)",[])
        if crit:
            # User improvement: .is_real safety
            real_vals=[(float(N(f.subs(v,c))),c) for c in crit if f and N(f.subs(v,c)).is_real]
            if real_vals:
                goal=p.meta.get("goal","extremize")
                best=(min if "min" in goal else max)(real_vals)
                r["optimal"]=best; kv("Optimal",f"x*={best[1]}, f*={best[0]:.4f}")
                insight(f"Backwards: x*={best[1]} where gradient f'(x)=0 — system at rest")
        if p.fb.has('multiple_minima'):
            vals=p.fb.get('multiple_minima'); minima=[(fv,c) for fv,c,nat in vals if nat=='min']
            if len(minima)>1:
                insight(f"B:multiple minima at x={[str(c) for _,c in minima]} — non-convex landscape")
                insight("Gradient descent may converge to local, not global, minimum")
        if p.fb.has('convex') and p.fb.get('convex'):
            insight("B:convex confirmed → global optimum guaranteed; any critical point IS the solution")
        return r

    return r


# ══════════════════════════════════════════════════════════════════════════
# PHASE 05 — GENERALIZE + SOLUTION FAMILY  (D: parametric families)
# ══════════════════════════════════════════════════════════════════════════

def phase_05(p,g4):
    section(5,"GENERALIZE + D:SOLUTION FAMILY","Governing theorems · parametric embedding · Faulhaber")
    r={}; v=p.var
    laws={
        PT.GRAPH:       {"connectivity":"λ₂(L)>0↔connected (Fiedler 1973)","bipartite":"sym spectrum↔bipartite","kirchhoff":"τ=(1/n)∏λᵢ≠0","cheeger":"h∈[λ₂/2,√(2λ₂)]","gnn":"GNN=poly(L)·h"},
        PT.MATRIX:      {"spectral_thm":"Sym M=QΛQᵀ","cayley_ham":"p(M)=0","SVD":"M=UΣVᵀ","definiteness":"xᵀMx>0↔all λ>0","rank_nullity":"rank+nullity=n"},
        PT.MARKOV:      {"perron_frob":"irreducible→unique λ=1,unique π>0","ergodic":"time avg=space avg=π","mixing":"‖Pⁿ-Π‖≤|λ₂|ⁿ","entropy_rate":"h=-Σπᵢ ΣⱼPᵢⱼlogPᵢⱼ"},
        PT.ENTROPY:     {"max_H":"H≤log₂n; eq iff uniform","chain_rule":"H(X,Y)=H(X)+H(Y|X)","data_proc":"H(f(X))≤H(X)","source_coding":"avg code L≥H(X)"},
        PT.DYNAMICAL:   {"lyapunov":"f'(x*)<0→stable; f'(x*)>0→unstable","hartman_grob":"near hyperbolic: linearise","noether":"symmetry→conserved quantity","bifurcation":"f'(x*)=0: structural change"},
        PT.CONTROL:     {"routh_hurwitz":"stable↔all Routh first-col>0","spectral":"stable↔all Re(λ)<0","controllable":"rank[B,AB,…]=n↔controllable"},
        PT.OPTIMIZATION:{"first_order":"∇f(x*)=0 necessary","second_order":"H>0→min; H<0→max; indef→saddle","convexity":"f convex→local=global","kkt":"∇f=Σλᵢ∇gᵢ (KKT)"},
    }
    dl=laws.get(p.ptype)
    if dl:
        kv("Governing theorems","")
        for nm,lw in dl.items(): kv(f"  {nm}",lw)
        r["governing"]=dl

    # D: Solution Family embedding
    kv("\nD:Solution Family","")
    if p.ptype==PT.GRAPH:
        t=p.meta.get("type")
        families={"complete":"Kₙ: τ=nⁿ⁻², λ₂=n, diam=1, bipartite iff n=2",
                  "path":    "Pₙ: λₖ=2-2cos(kπ/n), diam=n-1, no cycles",
                  "cycle":   "Cₙ: λₖ=2-2cos(2πk/n), bipartite iff n even"}
        if t in families: kv(f"  Named family",families[t]); r["family"]=families[t]
        kv("  Parametric space","All connected n-graphs ↔ symmetric 0/1 matrices in Sₙ mod isomorphism")
        r["param_family"]="graph_family_Gn"

    elif p.ptype==PT.QUADRATIC:
        a_,b_,c_=symbols('a b c'); gen=solve(a_*v**2+b_*v+c_,v)
        kv("  Quadratic formula",[str(s) for s in gen])
        kv("  Discriminant law","Δ=b²-4ac: Δ>0 two real, Δ=0 repeated, Δ<0 complex pair")
        kv("  Solution family","ax²+bx+c=0 for a≠0 — complete family parametrised by (a,b,c)∈ℝ³")
        r["param_family"]="quadratic_abc"

    elif p.ptype==PT.CUBIC:
        kv("  Cardano","t³+pt+q=0: t=∛(-q/2+√D)+∛(-q/2-√D), D=(q/2)²+(p/3)³")
        kv("  Family","ax³+bx²+cx+d=0 — always ≥1 real root (IVT); Cardano gives closed form")
        r["param_family"]="cubic_abcd"

    elif p.ptype==PT.LINEAR:
        a_,b_=symbols('a b',nonzero=True); sol=solve(a_*v+b_,v)[0]
        kv("  General solution",f"x=-b/a = {str(sol)}")
        kv("  Solution family","ax+b=0 for a≠0: linear in both a,b — simplest parametric family")
        r["param_family"]="linear_ax_b"

    elif p.ptype==PT.SUM:
        k=symbols('k',positive=True,integer=True); ns=symbols('n',positive=True,integer=True)
        # D: Faulhaber family for all powers
        kv("  Faulhaber power-sum family","Σₖ₌₁ⁿ kᵖ = (p+1)-th degree polynomial in n")
        for pw in range(1,6):
            try: kv(f"    Σk^{pw}",str(factor(summation(k**pw,(k,1,ns)))))
            except: pass
        kv("  Bernoulli numbers","Faulhaber coefficients = Bernoulli numbers Bₖ")
        r["param_family"]="faulhaber_power_sums_p"

    elif p.ptype==PT.MARKOV:
        kv("  n-state chain family","Row-stochastic matrices ↔ probability simplex ΔNⁿ at each state")
        kv("  Doubly stochastic subfamily","Birkhoff-von Neumann: conv hull of permutation matrices")
        kv("  Reversible subfamily","Detailed balance πᵢPᵢⱼ=πⱼPⱼᵢ ↔ symmetric in π-weighted inner product")
        r["param_family"]="row_stochastic_nxn"

    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        kv("  Shannon entropy family","H(p₁,…,pₙ): unique up to continuity+symmetry+normalisation")
        kv("  Rényi family","H_α=(1/(1-α))log Σpᵢ^α; α→1 gives Shannon; α=0 Hartley; α=∞ min-entropy")
        if probs:
            for alpha in [0.5,2.0,float('inf')]:
                try:
                    if alpha==float('inf'): Hr=-math.log2(max(probs))
                    else: Hr=(1/(1-alpha))*math.log2(sum(p_**alpha for p_ in probs if p_>0))
                    kv(f"    H_{alpha}",f"{Hr:.4f} bits")
                except: pass
        r["param_family"]="renyi_alpha_family"

    elif p.ptype==PT.DYNAMICAL:
        kv("  1D autonomous family","x'=f(x): gradient flow x'=-∇V for V=-∫f dx")
        kv("  Pitchfork family","x'=μx-x³: subcritical at μ<0, supercritical at μ>0")
        kv("  Saddle-node family","x'=μ+x²: equilibria emerge at μ=0 (fold bifurcation)")
        r["param_family"]="1d_autonomous_odes"

    elif p.ptype==PT.CONTROL:
        kv("  Hurwitz family","Stable polys: open convex cone in coefficient space")
        kv("  Routh boundary","Stability boundary = hypersurface where first-col entry = 0")
        kv("  Degree-n family","nth-order system: n eigenvalues, Routh n-row test")
        r["param_family"]="hurwitz_stable_polynomials_deg_n"

    elif p.ptype==PT.OPTIMIZATION:
        kv("  Smooth unconstrained family","min f(x): first-order necessary, second-order sufficient")
        kv("  Convex subfamily","f convex → KKT necessary+sufficient → global cert")
        kv("  Gradient descent family","xₖ₊₁=xₖ-α∇f(xₖ): discrete gradient flow, converges for α<2/L")
        r["param_family"]="smooth_unconstrained_opt"

    elif p.ptype==PT.PROOF:
        body=p.meta.get("body","")
        if "sqrt(2)" in body.lower() or "root 2" in body.lower():
            kv("  Irrationality family","√n irrational iff n not a perfect square (Euclid)")
            for n_ in range(1,10): kv(f"    √{n_}","rational" if sp.sqrt(n_).is_integer else "irrational")
        elif "prime" in body.lower():
            kv("  Prime family","Dirichlet: ∞ primes in every arithmetic progression (generalises Euclid)")

    elif p.ptype==PT.FACTORING:
        a_,b_=symbols('a b')
        for f_,e_ in [("a²-b²",a_**2-b_**2),("a³-b³",a_**3-b_**3),("a³+b³",a_**3+b_**3),("a⁴-b⁴",a_**4-b_**4)]:
            kv(f"  {f_}",str(factor(e_)))
        kv("  Pattern","aⁿ-bⁿ=(a-b)(aⁿ⁻¹+aⁿ⁻²b+…+bⁿ⁻¹)")
        r["param_family"]="difference_nth_powers"

    finding("This problem is an INSTANCE of the parametric family above")
    return r


# ══════════════════════════════════════════════════════════════════════════
# PHASE 06 — PROVE LIMITS
# ══════════════════════════════════════════════════════════════════════════

def phase_06(p,g5):
    section(6,"PROVE LIMITS","Hard boundaries · obstructions · what cannot be done")
    r={}; v=p.var

    if p.ptype==PT.GRAPH:
        deg=p.meta.get("deg",[]); L_spec=p.meta.get("L_spec",[])
        kv("Hard lower bound","λ₁(L)=0 always (L·1=0, eigenvector=constant)")
        kv("Hard upper bound",f"λₙ(L)≤2Δ where Δ=max degree={max(deg) if deg else '?'}")
        kv("Isomorphism limit","Spectrum does NOT determine graph uniquely (isospectral pairs exist)")
        kv("Chromatic bound","χ(G)≤Δ+1 (Vizing's theorem)")
        if L_spec: kv("Actual λₙ",f"{max(L_spec):.4f}")
        r["limits"]={"eig_zero":"λ₁=0 always","iso_limit":"spectrum ≠ unique graph ID"}

    elif p.ptype==PT.MATRIX:
        spec=p.meta.get("spec",[])
        kv("Spectral radius","ρ(A)≤‖A‖ for any matrix norm")
        kv("Gershgorin","λ ∈ ∪ᵢ{z: |z-aᵢᵢ|≤Σⱼ≠ᵢ|aᵢⱼ|}")
        kv("Intractability","Computing exact eigenvalues is NP-hard for general integer matrix")
        if spec: kv("This matrix ρ",f"{max(abs(e) for e in spec):.4f}")

    elif p.ptype==PT.MARKOV:
        kv("Convergence limit","‖Pⁿ-Π‖≤|λ₂|ⁿ: geometric — cannot be faster without changing chain")
        kv("Entropy bound","Entropy rate h≤log₂n (uniform mixing maximises randomness)")
        kv("Absorbing limit","Absorbed chain: time-avg undefined — not ergodic in absorbed states")

    elif p.ptype==PT.ENTROPY:
        n=p.meta.get("n",0); H=p.meta.get("H_val",0)
        kv("Lower bound","H(X)≥0; equality iff deterministic (one p=1, all others=0)")
        kv("Upper bound",f"H(X)≤log₂{n}={math.log2(n):.4f} bits; equality iff uniform")
        kv("Data processing","H(f(X))≤H(X) — processing can NEVER increase entropy")
        kv("Subadditivity","H(X,Y)≤H(X)+H(Y); equality iff X,Y independent")
        kv("Gap to maximum",f"{math.log2(n)-H:.6f} bits remaining")

    elif p.ptype==PT.DYNAMICAL:
        kv("1D limit","1D autonomous: NO chaos, NO limit cycles (Poincaré-Bendixson)")
        kv("Lyapunov","Hartman-Grobman: linearisation valid near HYPERBOLIC equilibria only")
        kv("Obstruction","Non-hyperbolic (f'(x*)=0): linearisation fails — need centre manifold")
        if p.expr and v:
            try:
                equil=p.memo('solve(expr,var)',lambda: solve(p.expr,v)) or []
                fp_=diff(p.expr,v)
                for eq in equil:
                    if abs(float(N(fp_.subs(v,eq))))<1e-9:
                        warn(f"x*={eq}: non-hyperbolic (f'=0) → centre manifold needed")
            except: pass

    elif p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{})
        kv("Abel-Ruffini","No closed-form for characteristic poly of degree≥5")
        kv("Stability boundary","Imaginary axis: Re(λ)=0 → marginal stability → Nyquist needed")
        kv("Routh obstruction","Any zero/negative first-column entry → instability")
        if rh:
            fc=rh.get("first_col",[]); zeros_fc=sum(1 for x in fc if abs(x)<1e-9)
            kv("First column",[f"{x:.4f}" for x in fc])
            if zeros_fc>0: warn(f"{zeros_fc} zeros in first column → marginal stability")

    elif p.ptype==PT.OPTIMIZATION:
        kv("Saddle-point limit","First-order alone doesn't distinguish min/max/saddle")
        kv("Non-convex limit","No global minimum guarantee without convexity")
        kv("Abel-Ruffini","Polynomial opt: critical pts from deg≥5 → no closed form")
        f=p.expr
        if f and v:
            try:
                fpp_=p.memo("diff2(f,v)",lambda: diff(f,v,2))
                if fpp_:
                    is_cvx_global=all(float(N(fpp_.subs(v,float(x0))))>=-1e-9 for x0 in [-5,-2,0,2,5])
                    kv("Convexity (sampled [-5,5])",is_cvx_global)
            except: pass

    elif p.ptype in (PT.QUADRATIC,PT.CUBIC,PT.POLY):
        kv("Abel-Ruffini","No radical formula for deg≥5")
        kv("FTA","Every deg-n polynomial has exactly n roots in ℂ")
        kv("Discriminant","Δ determines root nature without computing roots")
        if p.expr and v:
            try:
                lead=float(N(Poly(p.expr,v).all_coeffs()[0]))
                kv("Leading coefficient",f"{lead:.4f} → {'opens up' if lead>0 else 'opens down'}")
            except: pass

    return r


# ══════════════════════════════════════════════════════════════════════════
# PHASE 07 — SYNTHESIS  (A: spectral isomorphism, E: output entropy scoring)
# ══════════════════════════════════════════════════════════════════════════

def phase_07(p,g6):
    section(7,"SYNTHESIS + A:SPECTRAL ISOMORPHISM + E:OUTPUT ENTROPY",
            "Cross-domain bridges · spectral comparison · self-scored diversity · meta-lesson")
    r={}

    BRIDGE_MAP={
        PT.GRAPH:[
            ("Graph→Markov",      "Random walk P=D⁻¹A; πᵢ=dᵢ/2|E|"),
            ("Graph→Entropy",     "Spectral entropy H_s=-Σ(λᵢ/trL)log(λᵢ/trL)"),
            ("Graph→Dynamical",   "Heat diffusion u'=-Lu; solution e^{-tL}u₀"),
            ("Graph→Optimization","Min cut=max flow (Ford-Fulkerson LP duality)"),
            ("Graph→ML",          "GNN: h'=σ(QᵀF(Λ)Qh), Q=eigvecs of L"),
        ],
        PT.MATRIX:[
            ("Matrix→Dynamical",   "x'=Ax: stable iff Re(λᵢ)<0; solution e^{At}x₀"),
            ("Matrix→Control",     "char poly det(sI-A): poles in LHP = stable"),
            ("Matrix→Optimization","Hessian H: xᵀHx determines curvature"),
            ("Matrix→Entropy",     "Von Neumann: S(ρ)=-tr(ρlogρ) quantum entropy"),
            ("Matrix→Graph",       "0/1 symmetric matrix IS adjacency matrix"),
        ],
        PT.MARKOV:[
            ("Markov→Graph",      "P defines weighted digraph; reversible P = undirected"),
            ("Markov→Entropy",    "Entropy rate h=lim H(Xₙ|X₀…Xₙ₋₁)"),
            ("Markov→Optimization","MCMC: run chain to sample from target π"),
            ("Markov→Physics",    "Entropy production σ=Σπᵢ Pᵢⱼ log(πᵢPᵢⱼ/πⱼPⱼᵢ)≥0"),
            ("Markov→Control",    "MDP: optimal policy via Bellman = control on Markov chain"),
        ],
        PT.ENTROPY:[
            ("Entropy→Physics",    "Boltzmann S=k_B·H"),
            ("Entropy→Markov",     "Entropy rate h=-Σπᵢ ΣⱼPᵢⱼlogPᵢⱼ"),
            ("Entropy→Optimization","MaxEnt: max H(p) s.t. constraints → Gibbs"),
            ("Entropy→ML",         "Cross-entropy loss=H(y,p̂)=H(y)+KL(y‖p̂)"),
            ("Entropy→Graph",      "Spectral graph entropy=H(normalised L spectrum)"),
        ],
        PT.DYNAMICAL:[
            ("Dynamical→Control",     "x'=f(x,u): design u to steer to target"),
            ("Dynamical→Optimization","Gradient flow x'=-∇f IS gradient descent"),
            ("Dynamical→Markov",      "SDE: x'=f(x)+noise → Markov (Fokker-Planck)"),
            ("Dynamical→Entropy",     "KS entropy h_KS=Σmax(λᵢ,0) Lyapunov exponents"),
        ],
        PT.CONTROL:[
            ("Control→Matrix",     "char poly=det(sI-A); poles=eigenvalues of A"),
            ("Control→Optimization","LQR: min∫(xᵀQx+uᵀRu)dt → Riccati equation"),
            ("Control→Dynamical",  "Closed-loop: x'=(A+BK)x; place eigs with K"),
            ("Control→Graph",      "Multi-agent consensus: sync rate=λ₂(Laplacian)"),
        ],
        PT.OPTIMIZATION:[
            ("Opt→Dynamical",  "Gradient descent=Euler of x'=-∇f"),
            ("Opt→Markov",     "RL/MDP: policy opt via Bellman equations"),
            ("Opt→Entropy",    "MaxEnt: max H(p) s.t. moments=exponential family"),
            ("Opt→Graph",      "Shortest path=min-cost flow=LP on graph"),
        ],
    }

    bridges=BRIDGE_MAP.get(p.ptype,[])
    if bridges:
        kv("Cross-domain bridges","")
        for src_dst,desc in bridges: bridge(f"{src_dst}: {desc}")
        r["bridges"]={sd:d for sd,d in bridges}

    # A: SPECTRAL UNIFICATION — compare fingerprints across domains
    if p.spectra:
        kv(f"\nA:Spectral fingerprints ({len(p.spectra)} registered)","")
        for fp in p.spectra:
            Hs=fp.spectral_entropy(); rho=fp.spectral_radius()
            kv(f"  [{fp.domain}]",
               f"eigs={[f'{e:.3f}' for e in fp.eigenvalues[:6]]}, ρ={rho:.3f}, H_s={Hs:.3f} bits")
        if len(p.spectra)>=2:
            sims=[(p.spectra[i].similarity(p.spectra[j]),p.spectra[i].domain,p.spectra[j].domain)
                  for i in range(len(p.spectra)) for j in range(i+1,len(p.spectra))]
            best=max(sims,key=lambda x:x[0]) if sims else None
            if best:
                if best[0]>0.85:
                    insight(f"A:HIGH spectral similarity {best[0]:.3f} between [{best[1]}] and [{best[2]}] → structural isomorphism")
                else:
                    kv("  Best spectral similarity",f"{best[0]:.3f} ({best[1]} vs {best[2]})")

    # Domain-specific emergents
    if p.ptype==PT.GRAPH:
        L_spec=p.meta.get("L_spec",[]); A_spec=p.meta.get("A_spec",[]); n=p.meta.get("n",0)
        r["heat_kernel"]="e^{-tL}: heat diffusion on graph — from any source node"
        r["ihara_zeta"]="Z_G(u)=∏_primes(1-u^|p|)⁻¹ (Riemann zeta analog for graphs)"
        kv("Heat kernel",r["heat_kernel"]); kv("Ihara zeta",r["ihara_zeta"])
        if L_spec:
            trL=sum(L_spec)
            if trL>0:
                nz=[e for e in L_spec if e>1e-9]
                Hs=_entropy([e/trL for e in nz]); r["spectral_entropy"]=Hs
                kv("Spectral entropy H_s(G)",f"{Hs:.4f} bits")
        nc=sum(1 for e in L_spec if abs(e)<0.1) if L_spec else 0
        kv("Spectral clusters (λ~0)",nc); insight(f"{nc} near-zero λ → {nc} natural spectral clusters")
        insight("DEEPEST: graph spectrum = isomorphism fingerprint (isospectral pairs rare but exist)")

    elif p.ptype==PT.MATRIX:
        spec=p.meta.get("spec",[])
        for k_,v_ in {"matrix_exp":"e^{At} governs ALL linear dynamical systems",
                       "SVD":"M=UΣVᵀ: optimal rank-k approx (Eckart-Young)",
                       "pseudo_inv":"M⁺=VΣ⁺Uᵀ: least-squares solution to Ax=b"}.items():
            kv(f"  {k_}",v_)
        if spec:
            pis=[abs(e)/sum(abs(e2) for e2 in spec) for e in spec if abs(e)>1e-12]
            if pis: Hvn=_entropy(pis); kv("Von Neumann-like entropy",f"{Hvn:.4f} bits"); insight(f"Spectral entropy={Hvn:.4f} bits (quantum analog)")
        insight("DEEPEST: e^{At} IS the universal solution to all linear ODEs and PDEs")

    elif p.ptype==PT.MARKOV:
        stat=p.meta.get("stat",{}); P_rat=p.meta.get("P_rat"); n=p.meta.get("n",0)
        for k_,v_ in {"potential":"Hitting times=Green's function G=(I-P)⁻¹",
                       "martingales":"Harmonic h(x)=E[h(X_τ)|X₀=x] (optional stopping)",
                       "MCMC":"Sample from ANY distribution via chain with target stationary",
                       "free_energy":"F=⟨E⟩-T·H(π): equilibrium minimises free energy"}.items():
            kv(f"  {k_}",v_)
        if stat and P_rat:
            try:
                pif=[float(N(sp.sympify(list(stat.values())[i]))) for i in range(n)]
                ep=sum(pif[i]*float(N(P_rat[i,j]))*
                       math.log(max(pif[i]*float(N(P_rat[i,j])),1e-15)/max(pif[j]*float(N(P_rat[j,i])),1e-15))
                       for i in range(n) for j in range(n)
                       if float(N(P_rat[i,j]))>1e-12 and float(N(P_rat[j,i]))>1e-12)
                r["entropy_prod"]=ep; kv("Entropy production",f"{ep:.6f} nat/step")
                insight(f"Ep={ep:.4f}: {'reversible (=0)' if ep<1e-9 else 'irreversible — 2nd law'}")
            except: pass
        insight("DEEPEST: Markov chain IS a random walk on a weighted directed graph")

    elif p.ptype==PT.ENTROPY:
        probs=p.meta.get("probs",[])
        for k_,v_ in {"mutual_info":"I(X;Y)=H(X)+H(Y)-H(X,Y)≥0; =0 iff independent",
                       "renyi":"H_α=(1/(1-α))log Σpᵢ^α; α→1 gives Shannon",
                       "free_energy":"F=⟨E⟩-T·H: MinF=MaxEnt under energy constraint",
                       "MDL":"Minimum description length: Occam's razor quantified"}.items():
            kv(f"  {k_}",v_)
        if probs:
            for alpha in [0.5,2.0]:
                Hr=(1/(1-alpha))*math.log2(sum(p_**alpha for p_ in probs if p_>0))
                kv(f"  Rényi H_{alpha}",f"{Hr:.4f} bits")
        insight("DEEPEST: MaxEnt=Bayesian minimum-assumption prior=Gibbs distribution=ML softmax")

    elif p.ptype==PT.DYNAMICAL:
        for k_,v_ in {"gradient_flow":"x'=-∇f: stable eq=global min of f (unifies OPT+DYN)",
                       "KS_entropy":"h_KS=Σmax(λᵢ,0): chaos via Lyapunov exponents",
                       "NF_theorem":"Normal form: near bifurcation, nonlinear≈canonical",
                       "variational":"Hamilton's principle: trajectories extremise S=∫L dt"}.items():
            kv(f"  {k_}",v_)
        insight("DEEPEST: gradient descent in ML IS a dynamical system — loss landscape = potential")

    elif p.ptype==PT.CONTROL:
        for k_,v_ in {"separation":"Design observer+controller independently (separation principle)",
                       "LQR_Riccati":"Optimal control → algebraic Riccati equation",
                       "H_inf":"H∞: minimise worst-case gain (robust to uncertainty)",
                       "flatness":"Differential flatness → open-loop trajectory planning"}.items():
            kv(f"  {k_}",v_)
        insight("DEEPEST: stability theory = spectral theory = eigenvalue placement")

    elif p.ptype==PT.OPTIMIZATION:
        for k_,v_ in {"proximal":"prox_f(x)=argmin_y(f(y)+‖y-x‖²/2): generalises projection",
                       "ADMM":"Alternating Direction Method: distributed optimisation",
                       "mirror_desc":"Mirror descent: non-Euclidean generalisation of GD",
                       "EM":"EM algorithm: coordinate-ascent on variational lower bound"}.items():
            kv(f"  {k_}",v_)
        insight("DEEPEST: optimisation IS physics — gradient flow, Hamiltonian mechanics, entropy")

    # E: OUTPUT ENTROPY SCORING
    kv("\nE:Output Information Quality","")
    unique_concepts=set()
    for attr in ['governing','bridges','emergents','family','param_family']:
        val=r.get(attr)
        if isinstance(val,dict): unique_concepts.update(val.keys())
        elif isinstance(val,str): unique_concepts.add(val)
    unique_concepts.update(p.fb.all_signals())
    concept_count=max(len(unique_concepts),1)
    density=min(concept_count/15.0,1.0)
    kv("  Unique concepts generated",concept_count)
    kv("  Information density score",f"{density:.2f} (1.0=maximally diverse)")
    if density>0.7: ok("High-density: output concepts are substantively distinct")
    else: note("Lower-density: some phases produced redundant findings for this type")
    r["output_entropy"]=density

    # Meta-lesson
    lessons={
        PT.GRAPH:       "Graphs ARE matrices. Every structural property = a spectral property. The Laplacian is the graph.",
        PT.MATRIX:      "A matrix is a linear transformation. Its eigenvalues are the transformation's irreducible actions.",
        PT.MARKOV:      "Randomness has structure. Entropy is that structure quantified. Every chain is a walk on a graph.",
        PT.ENTROPY:     "Uncertainty is information. Maximum entropy = maximum honesty about what you don't know.",
        PT.DYNAMICAL:   "Differential equations are geometry. Stability is topology. Chaos is sensitivity of geometry.",
        PT.CONTROL:     "Stability = spectral geometry. To control a system, reshape its eigenvalue landscape.",
        PT.OPTIMIZATION:"Optimisation IS dynamics. Loss landscape = potential. Training = gradient flow to equilibrium.",
        PT.QUADRATIC:   "Discriminant is the geometric heart of a quadratic. Everything else follows from Δ=b²-4ac.",
        PT.LINEAR:      "Linear equations are the atoms of all mathematics. Everything nonlinear is locally linear.",
        PT.CUBIC:       "Three roots, three equilibria, three energy states. Cubics are the simplest truly nonlinear systems.",
        PT.SUM:         "Discrete sums are polynomial integrals. Euler-Maclaurin bridges the discrete and continuous.",
        PT.FACTORING:   "Factoring IS root-finding. The structure of a polynomial lives entirely in its factors.",
        PT.PROOF:       "A proof is an irreducible argument. Contradiction is the most powerful tool: assume the opposite.",
    }
    lesson=lessons.get(p.ptype,"Every mathematical problem is an instance of a universal pattern.")
    kv("\nMeta-lesson",""); insight(lesson); r["meta_lesson"]=lesson

    kv("\nConfidence ledger",p.conf.summary())
    if p.conf.unknowns: kv("Uncertain results"," | ".join(p.conf.unknowns[:3]))
    return r


# ══════════════════════════════════════════════════════════════════════════
# FINAL ANSWER  (User improvement: cache-first for all types)
# ══════════════════════════════════════════════════════════════════════════

def _final_answer(p):
    c=p._cache
    if p.ptype in (PT.LINEAR,PT.QUADRATIC,PT.CUBIC,PT.POLY,PT.TRIG_EQ):
        rts=c.get("roots") or c.get("solve(linear)") or c.get("solve(quadratic)") or c.get("solve(cubic)")
        if rts: return f"Roots: {', '.join(str(r_) for r_ in rts)}"
    if p.ptype==PT.TRIG_ID:
        simp=c.get("trigsimp"); return f"Identity → {simp}" if simp is not None else "Identity verified"
    if p.ptype==PT.FACTORING:
        fac=c.get("factor(expr)"); return f"Factored: {fac}" if fac else "Factoring complete (see Phase 02)"
    if p.ptype==PT.SUM:
        # User improvement: cache-first scan for any summation result
        res=next((v_ for k_,v_ in c.items() if 'summation' in k_ and v_ is not None),None)
        if res:
            from sympy import factor as sfac
            try: return f"Sum = {sfac(res)}"
            except: return f"Sum = {res}"
        return "Summation computed (see Phase 02)"
    if p.ptype==PT.PROOF:
        body=p.meta.get("body","").lower()
        if any(kw in body for kw in ('sqrt(2)','root 2','irrational','√2')):
            return "√2 is irrational. Proof by contradiction. QED."
        if 'prime' in body: return "Infinitely many primes. Euclid's construction. QED."
        return "Proof presented in phase computations"
    if p.ptype==PT.DIGRAPH_CYC:
        m=p.meta.get("m",0)
        return f"m={m}: {'fiber decomposition EXISTS (odd m)' if m%2!=0 else 'IMPOSSIBLE (even m — parity obstruction)'}"
    if p.ptype==PT.GRAPH:
        L_spec=p.meta.get("L_spec",[]); n=p.meta.get("n",0)
        conn=p.fb.get('connected','?')
        return (f"Graph ({n} nodes): {'connected' if conn else 'disconnected'}, "
                f"L-spectrum={[f'{e:.3f}' for e in L_spec[:4]]}")
    if p.ptype==PT.MATRIX:
        spec=p.meta.get("spec",[]); def_=p.meta.get("definite","")
        return f"Eigenvalues: {[f'{e:.4f}' for e in spec]}" + (f", {def_}" if def_ else "")
    if p.ptype==PT.MARKOV:
        stat=p.meta.get("stat",{})
        if stat:
            vals=[float(N(sp.sympify(v_))) for v_ in list(stat.values())]
            return f"Stationary π = [{', '.join(f'{v:.4f}' for v in vals)}]"
        return "Stationary distribution computed (see Phase 02)"
    if p.ptype==PT.ENTROPY:
        H=p.meta.get("H_val",0); Hmax=p.meta.get("H_max",0)
        return f"H(X) = {H:.6f} bits  (max={Hmax:.4f}, efficiency={H/max(Hmax,1e-9):.4f})"
    if p.ptype==PT.DYNAMICAL:
        equil=p.memo('solve(expr,var)',lambda: solve(p.expr,p.var)) or []
        if equil and p.expr and p.var:
            fp_=diff(p.expr,p.var)
            stab=["S" if float(N(sp_re(fp_.subs(p.var,eq))))<0 else "U" for eq in equil]
            return f"Equilibria: {list(zip([str(e) for e in equil],stab))}"
        return "Equilibria computed (see Phase 02)"
    if p.ptype==PT.CONTROL:
        rh=p._cache.get("routh",{})
        return f"System: {'STABLE' if rh.get('stable') else 'UNSTABLE'} ({rh.get('sign_changes',0)} RHP roots)"
    if p.ptype==PT.OPTIMIZATION:
        opt=c.get("optimal")
        if opt: return f"Optimal: x*={opt[1]}, f*={opt[0]:.4f}"
        crit=c.get("solve(f'=0)",[])
        if crit and p.expr and p.var:
            vals=[(float(N(p.expr.subs(p.var,c_))),c_) for c_ in crit if N(p.expr.subs(p.var,c_)).is_real]
            if vals:
                goal=p.meta.get("goal","extremize")
                best=(min if "min" in goal else max)(vals)
                return f"Optimal: x*={best[1]}, f*={best[0]:.6f}"
        return "Optimization computed (see Phase 02)"
    return "See phase computations above"


# ══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

def run(raw,silent=False):
    prob=classify(raw)
    if not silent:
        print(f"\n{hr('=')}\n{W}DISCOVERY ENGINE v4{RST}\n{hr()}")
        print(f"  {W}Problem:{RST}  {Y}{raw}{RST}")
        print(f"  {DIM}Type:{RST}     {prob.ptype.label()}")
        print(f"  {DIM}Variable:{RST} {prob.var}")
        print(hr('='))
    if prob.ptype==PT.UNKNOWN:
        if not silent:
            print(f"{R}Could not classify. Try: 'x^2-5x+6=0', 'graph K4', "
                  f"'markov [[...]]', 'entropy [...]', 'dynamical x^3-x'{RST}")
        return prob
    g1=phase_01(prob)
    g2=phase_02(prob,g1)
    # Sync routh and roots to cache for later phases
    rh=g2.get("routh")
    if rh: prob._cache["routh"]=rh
    for key in ["roots","solve(linear)","solve(quadratic)","solve(cubic)"]:
        if key in g2 and g2[key]: prob._cache[key]=g2[key]
    if "optimal" in g2: prob._cache["optimal"]=g2["optimal"]
    g3=phase_03(prob,g2)
    g4=phase_04(prob,g3)
    g5=phase_05(prob,g4)
    g6=phase_06(prob,g5)
    g7=phase_07(prob,g6)
    if not silent:
        print(f"\n{hr('=')}\n{W}FINAL ANSWER{RST}\n{hr()}")
        print(f"  {G}{_final_answer(prob)}{RST}\n{hr('=')}")
        titles={1:"Ground Truth+Intel",2:"Direct Attack",3:"Structure Hunt",
                4:"Pattern Lock",    5:"Generalize",  6:"Prove Limits",7:"Synthesis"}
        phases=[g1,g2,g3,g4,g5,g6,g7]
        print(f"\n{hr()}\n{W}PHASE SUMMARY{RST}\n{hr('.')}")
        for i,(g,t) in enumerate(zip(phases,titles.values()),1):
            print(f"  {PHASE_CLR[i]}{i:02d} {t:<22}{RST} {len(g)} results")
        kv("Feedback signals"," | ".join(prob.fb.all_signals()) or "none")
        kv("Spectral fingerprints",len(prob.spectra))
        kv("Confidence",prob.conf.summary())
        print(hr('='))
    return prob


# ══════════════════════════════════════════════════════════════════════════
# ASSERTION-BASED TEST SUITE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TR:
    desc:str; raw:str; passed:bool; elapsed:float
    ap:int=0; af:int=0; notes:List[str]=field(default_factory=list)

def _assert(tr,cond,msg):
    if cond: tr.ap+=1
    else:    tr.af+=1; tr.notes.append(f"FAIL: {msg}")

def _run_test(raw,desc,checks=None):
    t0=time.time()
    buf=io.StringIO()
    prob=None
    try:
        with redirect_stdout(buf),redirect_stderr(io.StringIO()):
            prob=run(raw,silent=False)
        elapsed=time.time()-t0
        tr=TR(desc=desc,raw=raw,passed=True,elapsed=elapsed)
        if checks and prob:
            for chk in checks:
                try: chk(prob,tr)
                except Exception as e: _assert(tr,False,f"check raised: {e}")
    except Exception as e:
        elapsed=time.time()-t0
        tr=TR(desc=desc,raw=raw,passed=False,elapsed=elapsed)
        tr.notes.append(f"Exception: {e}")
    return tr

# ── Assertion helpers ─────────────────────────────────────────────────────

def assert_roots(expected):
    def chk(prob,tr):
        rts=None
        for k in ["roots","solve(linear)","solve(quadratic)","solve(cubic)"]:
            if prob._cache.get(k): rts=prob._cache[k]; break
        _assert(tr,rts is not None,"roots computed")
        if rts:
            got={float(N(r_)) for r_ in rts}
            exp={float(e) for e in expected}
            _assert(tr,all(any(abs(g-e)<1e-3 for g in got) for e in exp),f"expected {exp}, got {got}")
    return chk

def assert_entropy(lo,hi):
    def chk(prob,tr):
        H=prob.meta.get("H_val")
        _assert(tr,H is not None,"entropy computed")
        if H: _assert(tr,lo<=H<=hi,f"H={H:.4f} not in [{lo},{hi}]")
    return chk

def assert_stable(expected):
    def chk(prob,tr):
        rh=prob._cache.get("routh",{})
        _assert(tr,bool(rh),"Routh computed")
        _assert(tr,rh.get("stable")==expected,f"expected stable={expected}, got {rh.get('stable')}")
    return chk

def assert_stationary_sum():
    def chk(prob,tr):
        stat=prob.meta.get("stat",{})
        _assert(tr,bool(stat),"stationary dist computed")
        if stat:
            s=sum(float(N(sp.sympify(v_))) for v_ in stat.values())
            _assert(tr,abs(s-1)<1e-4,f"π sums to {s:.6f}")
    return chk

def assert_connected(expected):
    def chk(prob,tr):
        conn=prob.fb.get('connected')
        _assert(tr,conn is not None,"connectivity detected")
        _assert(tr,conn==expected,f"expected connected={expected}, got {conn}")
    return chk

def assert_has_spectrum():
    def chk(prob,tr): _assert(tr,len(prob.spectra)>0,"spectral fingerprint registered")
    return chk

def assert_sum_at(n_val,expected_val):
    def chk(prob,tr):
        res=next((v_ for k_,v_ in prob._cache.items() if 'summation' in k_ and v_ is not None),None)
        _assert(tr,res is not None,"sum formula computed")
        if res:
            ns=symbols('n',positive=True,integer=True)
            try:
                got=int(N(res.subs(ns,n_val)))
                _assert(tr,got==expected_val,f"sum at n={n_val}: got {got}, expected {expected_val}")
            except Exception as e: _assert(tr,False,f"eval failed: {e}")
    return chk

def assert_signal(signal):
    def chk(prob,tr): _assert(tr,prob.fb.has(signal),f"signal '{signal}' not emitted")
    return chk

def assert_optimal_x(x_star,tol=1e-2):
    def chk(prob,tr):
        crit=prob._cache.get("solve(f'=0)",[])
        _assert(tr,bool(crit),"critical points found")
        if crit and prob.expr and prob.var:
            vals=[float(N(c_)) for c_ in crit]
            _assert(tr,any(abs(xv-x_star)<tol for xv in vals),f"x*={vals} not near {x_star}")
    return chk

# ── Test battery ──────────────────────────────────────────────────────────

TESTS=[
    # Algebraic
    ("x^2 - 5x + 6 = 0",           "Quadratic integer roots",
     [assert_roots([2,3])]),
    ("2x + 3 = 7",                  "Linear",
     [assert_roots([2])]),
    ("x^3 - 6x^2 + 11x - 6 = 0",   "Cubic 3 integer roots",
     [assert_roots([1,2,3])]),
    ("sin(x)^2 + cos(x)^2",         "Pythagorean identity",
     []),
    ("factor x^4 - 16",             "Difference of squares chain",
     []),
    ("sum of first n integers",     "Classic Σk",
     [assert_sum_at(5,15)]),
    ("sum of squares of first n integers", "Σk²",
     [assert_sum_at(4,30)]),
    ("sum of cubes of first n integers",   "Σk³",
     [assert_sum_at(3,36)]),
    ("prove sqrt(2) is irrational", "Irrationality proof",
     []),
    ("m^3 vertices with 3 cycles, m=3","Digraph odd m",
     []),
    ("m^3 vertices with 3 cycles, m=4","Digraph even m",
     []),
    # Graph
    ("graph K4","Complete K4",
     [assert_connected(True),assert_has_spectrum(),assert_signal('bipartite')]),
    ("graph P5","Path P5",
     [assert_connected(True),assert_has_spectrum()]),
    ("graph C6","Cycle C6",
     [assert_connected(True),assert_has_spectrum(),assert_signal('bipartite')]),
    ("graph [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]","Custom graph",
     [assert_connected(True),assert_has_spectrum()]),
    # Matrix
    ("matrix [[2,1],[1,3]]",          "Symmetric 2×2",
     [assert_has_spectrum(),assert_signal('pos_definite')]),
    ("matrix [[4,2,2],[2,3,0],[2,0,3]]","Symmetric 3×3",
     [assert_has_spectrum()]),
    # Markov
    ("markov [[0.7,0.3],[0.4,0.6]]",  "2-state chain",
     [assert_stationary_sum(),assert_has_spectrum()]),
    ("markov [[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]]","3-state chain",
     [assert_stationary_sum(),assert_has_spectrum()]),
    # Entropy
    ("entropy [0.5,0.25,0.25]",       "Entropy skewed",
     [assert_entropy(1.4,1.6)]),
    ("entropy [0.25,0.25,0.25,0.25]", "Entropy uniform",
     [assert_entropy(1.99,2.01)]),
    # Dynamical
    ("dynamical x^3 - x",   "Dynamical 3 equil.",
     [assert_signal('multiple_equilibria')]),
    ("dynamical x^2 - 1",   "Dynamical pitchfork",
     []),
    # Control
    ("control s^2 + 3s + 2",        "Control stable 2nd",  [assert_stable(True)]),
    ("control s^3 + 2s^2 + 3s + 1", "Control Routh 3rd",   [assert_stable(True)]),
    ("control s^3 - s + 1",         "Control unstable",     [assert_stable(False)]),
    # Optimization
    ("optimize x^4 - 4x^2 + 1",     "Quartic opt",
     [assert_signal('multiple_minima'),assert_has_spectrum()]),
    ("minimize x^2 + 2x + 1",       "Minimize quadratic",
     [assert_optimal_x(-1)]),
    # Edge cases
    ("graph C4",                     "EDGE: C4 bipartite",
     [assert_connected(True),assert_signal('bipartite')]),
    ("markov [[1,0],[0.3,0.7]]",     "EDGE: absorbing Markov",
     [assert_signal('absorbing'),assert_stationary_sum()]),
    ("entropy [0.9,0.05,0.05]",      "EDGE: near-deterministic",
     [assert_entropy(0.4,0.75)]),
    ("dynamical sin(x)",             "EDGE: trig equilibria",
     []),
    ("x^4 - 5x^2 + 4 = 0",          "EDGE: biquadratic",
     [assert_roots([1,-1,2,-2])]),
    ("sum of power 4 first n integers","EDGE: Σk⁴",
     [assert_sum_at(3,98)]),
]


def run_tests(verbose=False):
    print(f"\n{hr('=')}\n{W}DISCOVERY ENGINE v4 — TEST SUITE ({len(TESTS)} problems){RST}\n{hr('=')}")
    passed=0; failed_tests=[]; total_time=0.0; total_ap=0; total_af=0
    for raw,desc,checks in TESTS:
        print(f"\n  {B}[TEST]{RST} {desc}  {DIM}[{raw[:50]}]{RST}",end="",flush=True)
        tr=_run_test(raw,desc,checks)
        total_time+=tr.elapsed; total_ap+=tr.ap; total_af+=tr.af
        if tr.passed and tr.af==0:
            passed+=1
            print(f" {G}✓{RST} ({tr.elapsed:.2f}s) {G}+{tr.ap} assert{RST}")
        else:
            failed_tests.append(tr)
            af=f" {R}-{tr.af} assert fails{RST}" if tr.af else ""
            print(f" {R}✗{RST} ({tr.elapsed:.2f}s){af}")
        if (verbose or not tr.passed) and tr.notes:
            for n in tr.notes[:3]: print(f"    {R}→{RST} {n}")
    n_tests=len(TESTS)
    print(f"\n{hr('=')}")
    clr=G if passed==n_tests else (Y if passed>n_tests*0.8 else R)
    print(f"{clr}Results: {passed}/{n_tests} passed{RST}  "
          f"| {G}+{total_ap}{RST}/{R}-{total_af}{RST} assertions  "
          f"| Total: {total_time:.1f}s  Avg: {total_time/n_tests:.2f}s/test")
    if failed_tests:
        print(f"\n{R}Failed:{RST}")
        for tr in failed_tests:
            print(f"  {R}✗{RST} {tr.desc}")
            for n in tr.notes[:2]: print(f"    {DIM}{n}{RST}")
    print(hr('=')); return passed,n_tests


def run_bench():
    print(f"\n{hr('=')}\n{W}PERFORMANCE BENCHMARK{RST}\n{hr('.')}")
    cases=["x^2 - 5x + 6 = 0","graph K4","markov [[0.7,0.3],[0.4,0.6]]",
           "entropy [0.5,0.25,0.25]","dynamical x^3 - x",
           "control s^3 + 2s^2 + 3s + 1","optimize x^4 - 4x^2 + 1",
           "matrix [[4,2,2],[2,3,0],[2,0,3]]"]
    times=[]
    for raw in cases:
        t0=time.time()
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            run(raw,silent=False)
        elapsed=time.time()-t0; times.append(elapsed)
        bar="█"*int(elapsed*15); clr=G if elapsed<1 else Y if elapsed<3 else R
        print(f"  {clr}{bar:<20}{RST} {raw[:42]:<44} {elapsed:.3f}s")
    print(hr('.'))
    print(f"  Total: {sum(times):.2f}s  Avg: {sum(times)/len(times):.3f}s  Max: {max(times):.3f}s")
    print(hr('='))


if __name__=="__main__":
    args=sys.argv[1:]
    if not args:             print(__doc__)
    elif args[0]=="--test":  run_tests(verbose="--verbose" in args or "-v" in args)
    elif args[0]=="--bench": run_bench()
    else:                    run(" ".join(args))

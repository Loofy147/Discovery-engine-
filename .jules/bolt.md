## 2026-03-09 - [AIMO Full System Integration & API Resilience]
**Learning:** Code competitions with private reruns (like AIMO 3) require highly resilient submission scripts. The 'aimo' API may not be available in all environments, and the file structure can vary.
**Action:** Implement a multi-mode submission loop that tries the standard competition API first, then a universal framework fallback (kaggle_evaluation), and finally a manual batch processing loop (CSV/Parquet).

## 2026-03-09 - [Robust LaTeX Pattern Matching for Olympiad Math]
**Learning:** Mathematical Olympiad problems (IMO-level) in LaTeX format can have highly variable spacing, escaping, and notation (e.g. \sum vs \sum_{i=1}^n). Simple string matching often fails.
**Action:** Use a combination of aggressive normalization (removing all non-alphanumeric characters) and LaTeX-aware regex to match known problem patterns and extract goals for the symbolic engine.

## 2026-03-09 - [Symbolic Engine Performance Optimization]
**Learning:** Repeated SymPy operations like `simplify(expr - expr_subs)` for symmetry checks or `expand(expr)` during multi-phase discovery are expensive bottlenecks.
**Action:** Centralize these into memoized methods in the `Problem` class. This ensures each property is calculated exactly once per problem instance, significantly reducing total discovery time.

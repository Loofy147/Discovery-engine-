## 2025-03-07 - [SymPy Discovery Engine Optimization]
**Learning:** SymPy objects like `Poly` and expensive symbolic operations (`solve`, `summation`, `trigsimp`) are often recreated or recomputed across different phases of the application. Caching these results in the `Problem` object and reusing the `Poly` object significantly reduces redundant computations.
**Action:** Always check if a symbolic expression can be converted to a `Poly` once and reused for degree, coefficient, and root analysis. Use a cache for expensive operations that might be repeated across different modules or phases.

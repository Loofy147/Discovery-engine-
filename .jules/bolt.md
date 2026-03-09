## 2025-03-07 - [SymPy Discovery Engine Optimization]
**Learning:** SymPy objects like `Poly` and expensive symbolic operations (`solve`, `summation`, `trigsimp`) are often recreated or recomputed across different phases of the application. Caching these results in the `Problem` object and reusing the `Poly` object significantly reduces redundant computations.
**Action:** Always check if a symbolic expression can be converted to a `Poly` once and reused for degree, coefficient, and root analysis. Use a cache for expensive operations that might be repeated across different modules or phases.

## 2025-03-07 - [Python dict.get() Default Evaluation Pitfall]
**Learning:** Using `dict.get(key, expensive_call())` in Python always evaluates `expensive_call()` even if the key is present in the dictionary. For expensive symbolic computations in SymPy, this negates the benefits of caching.
**Action:** Use a dedicated `memo` method or a pattern like `if key not in cache: cache[key] = func()` to ensure lazy evaluation of cached values.

## 2025-03-07 - [SymPy 1.14 Incompatibility: ComplexNumber]
**Learning:** In SymPy 1.14, `sp.core.numbers.ComplexNumber` is no longer available or has moved. Using `isinstance(x, sp.core.numbers.ComplexNumber)` causes an AttributeError.
**Action:** Use `x.is_real` or `x.is_complex` instead of `isinstance` with the internal `ComplexNumber` class for checking the nature of numerical results.

## 2026-03-08 - [Eager Evaluation in dict.get() and Redundant SymPy Calls]
**Learning:** Python's `dict.get(key, default)` evaluates the `default` argument eagerly. When the default is an expensive SymPy function like `factor()` or `simplify()`, it causes significant performance overhead even if the value is already cached.
**Action:** Use `Problem.memo(key, lambda: expensive_call())` or explicit membership checks (`if key not in cache: cache[key] = ...`) to ensure lazy evaluation. Consolidate repetitive symbolic operations across phases into the central cache.

## 2026-03-08 - [Centralized Parity and Expansion Caching]
**Learning:** Repetitive symbolic properties like parity (even/odd) and expanded forms were being recomputed multiple times across different phases. Even with a per-problem cache, the overhead of re-invoking the logic and the `_timed` wrapper adds up.
**Action:** Centralize these common properties into methods on the `Problem` class that leverage the `memo` method. This ensures they are computed at most once per problem and provides a clean API for all phases to consume.

## 2026-03-09 - [Optimizing Matrix Characteristic Polynomials]
**Learning:** SymPy's `det(M - lam*I)` is extremely slow for symbolic matrices because it uses naive determinant expansion. `M.charpoly(lam).as_expr()` uses the Berkowitz algorithm, which is O(n^4) and vastly more efficient for large or symbolic matrices.
**Action:** Always prefer `M.charpoly(var).as_expr()` over `det(M - var*I)` when the characteristic polynomial is needed.

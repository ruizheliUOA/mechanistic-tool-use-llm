#!/usr/bin/env python3
"""Minimum certifiable target-hit rate given N exits.

Test: one-sided exact binomial, H0: p = 0.50, alpha = 0.05, power = 0.80.
Given N exits, find the smallest true target-hit rate p such that the test
rejects H0 with >= 80% probability.
"""
from scipy.stats import binom

def critical_k(n, alpha=0.05):
    """Smallest k such that P(X >= k | p=0.50) <= alpha."""
    k = binom.ppf(1 - alpha, n, 0.50) + 1
    return int(k) if k <= n else None

def power_at(n, p, alpha=0.05):
    k = critical_k(n, alpha)
    if k is None:
        return 0.0
    return float(binom.sf(k - 1, n, p))

def min_certifiable(n, alpha=0.05, power=0.80):
    """Weakest true target-hit rate certifiable at N exits. None if impossible."""
    if critical_k(n, alpha) is None:
        return None
    lo, hi = 0.5001, 0.9999
    if power_at(n, hi, alpha) < power:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(n, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi

def exits_needed(p, alpha=0.05, power=0.80, cap=20000):
    for n in range(5, cap):
        if power_at(n, p, alpha) >= power:
            return n
    return None

if __name__ == "__main__":
    print("VERIFICATION — reproduce the project's frozen exchange rate")
    for p, claim in [(0.73,30),(0.68,49),(0.63,93),(0.60,158),(0.58,245),(0.55,620)]:
        got = exits_needed(p)
        print(f"  target-hit {p:.2f} -> {got:>5} exits  (frozen {claim})"
              f"  {'MATCH' if got==claim else 'MISMATCH'}")
    print("\nMINIMUM CERTIFIABLE TARGET-HIT BY N")
    for n in [10,20,25,30,40,50,60,80,100,120,150,200,250,300,400,500,1000]:
        m = min_certifiable(n)
        print(f"  N={n:>5}  ->  {m:.4f}" if m else f"  N={n:>5}  ->  none (test cannot reject at any p)")

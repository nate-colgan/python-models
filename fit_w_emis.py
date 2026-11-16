"""
Generate degree-5 polynomial fit for Planck-weighted total W emissivity.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from thermal.properties import w_emis_total

T_grid  = np.linspace(200, 3100, 400)
eps_ref = np.array([w_emis_total(T) for T in T_grid])

x      = T_grid / 1000.0
coeffs = np.polyfit(x, eps_ref, 5)
eps_fit = np.polyval(coeffs, x)
rms    = np.sqrt(np.mean((eps_fit - eps_ref)**2))
maxerr = np.max(np.abs(eps_fit - eps_ref))

print(f"Degree-5 poly(T/1000):  rms={rms:.5f}  max_err={maxerr:.5f}")
print()
print("Coefficients:")
for i, c in enumerate(coeffs):
    print(f"  {c:.10e}")
print()

print(f"{'T [K]':>8}  {'reference':>10}  {'fit':>10}  {'err':>8}")
for T in [200, 300, 500, 700, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 2800, 3000, 3100]:
    ref = w_emis_total(T)
    fit = np.polyval(coeffs, T / 1000.0)
    print(f"{T:8d}  {ref:10.5f}  {fit:10.5f}  {fit-ref:+8.5f}")

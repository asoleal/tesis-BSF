import sys
sys.path.insert(0, '.')

import numpy as np
from modelo.modelo_bsf import (
    ParametrosEriksen, ParametrosLecho, estado_inicial_desde_dw,
    rhs, carbono_a_ppm_min
)

p_larva = ParametrosEriksen(
    amax=0.5, alpha=1.0, beta=1.0, YB=0.44, YL=0.15, m=0.15,
    Bmax0=39.69, tp_min=13.0, rho=1.0, delta_I=1.25,
    delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
    f_mant_prepupa=0.3, k_cat_prepupa=0.02
)

p_lecho = ParametrosLecho(
    f_dieta=1.0, k_ref=0.1, K_SM=100.0, Y_X=0.4,
    delta_CS=0.2, k_d=0.05, gamma=0.1, K_G=0.1
)

y0 = estado_inicial_desde_dw(
    XDW0_mg=50.0*0.325, D_C0=500.0*0.325*0.49*0.9,
    X0=1.0, p=p_larva
)

t, N = 0.0, 100
dydt = rhs(t, y0, p_larva, p_lecho, N, 27.0)

print("=== VERIFICACIONES ===")
print(f"dB/dt > 0? {dydt[0] > 0}")
print(f"dD_C/dt < 0? {dydt[2] < 0}")
print(f"dX/dt < 0? {dydt[3] < 0}")
print(f"dCO2/dt > 0? {dydt[4] > 0}")
print(f"dR/dt > 0? {dydt[5] > 0}")

# ppm (T_K = 300.15)
R_C_total = 0.25125  # valor de prueba
ppm = carbono_a_ppm_min(R_C_total, V_aire_L=12.1, T_K=300.15)
print(f"\nPendiente CO2: {ppm:.4f} ppm/min")

print("\n=== SIMULACION OK ===")

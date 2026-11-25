#!/usr/bin/env python3
"""
Calibracion fina en 2 etapas:
1. k_ref, delta_CS contra controles PINN
2. a_max, rho, f_D4, gamma_D1, gamma_D4 contra tratamientos PINN
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from parametros import PARAMETROS
from modelo_eriksen import estado_inicial_desde_dw
from modelo_completo import rhs_completo, tasas_lecho, carbono_a_ppm_min

BASE = __import__('pathlib').Path(__file__).resolve().parents[1]

# Cargar datos PINN
pinn = pd.read_csv(BASE / "datos" / "procesados" / "datos_finales_PINN_corregidos.csv")
pinn = pinn[pinn["Gas"] == "co2"].copy()
pinn["dia_larvario"] = pinn["Dia_Experimento"] - PARAMETROS["montaje"]["DIA_MONTAJE_SEP"]

# Separar controles y tratamientos
ctrl = pinn[pinn["Etiqueta"].str.contains("Control")].copy()
ctrl["dieta"] = ctrl["Etiqueta"].str.replace("Control_Alimento_", "")
trat = pinn[pinn["Etiqueta"].str.contains("Tratamiento|Generacion_Base")].copy()
trat["dieta"] = trat["Etiqueta"].str.extract(r"(D[14])")[0]

p_montaje = PARAMETROS["montaje"]
p_larva = PARAMETROS["eriksen"]
V = p_montaje["V_aire_L"]
T_K = p_montaje["T_K"]
T_C = p_montaje["T_C"]
T_FIN = p_montaje["T_FIN"]


# ============================================================
# ETAPA 1: Calibrar controles (solo lecho, sin larvas)
# ============================================================
def simular_control(dieta, log10_k_ref, delta_CS, t_eval):
    k_ref = 10.0 ** log10_k_ref
    p_lecho = PARAMETROS["lecho"].copy()
    p_lecho["k_ref"] = k_ref
    p_lecho["delta_CS"] = delta_CS
    p_lecho["f_dieta"] = 0.0
    p_lecho["gamma"] = 0.0
    
    y0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    y0[2] += 500.0 * p_montaje["ms_dieta"][dieta] * p_montaje["delta_C_alim"] * delta_CS
    
    sol = solve_ivp(
        lambda t, y: rhs_completo(t, y, p_larva, p_lecho, p_montaje, dieta, 0),
        (0, T_FIN), y0, t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-10,
    )
    if not sol.success:
        return None
    
    ppm = np.array([carbono_a_ppm_min(tasas_lecho(y[2], y[3], T_C, p_lecho)["rCmic"], V, T_K) 
                    for y in sol.y.T])
    return sol.t, ppm


def residuos_control(x, dieta):
    log10_k, delta_CS = x
    obs = ctrl[ctrl["dieta"] == dieta]
    if len(obs) == 0:
        return np.array([1e6])
    dias = obs["dia_larvario"].values
    vals = obs["Tasa_Produccion"].values
    t_eval = np.linspace(0, T_FIN, 100)
    res = simular_control(dieta, log10_k, delta_CS, t_eval)
    if res is None:
        return np.array([1e6] * len(vals))
    t, ppm = res
    pred = np.interp(dias, t, ppm)
    return pred - vals


print("=" * 60)
print("ETAPA 1: Calibracion controles")
print("=" * 60)

etapa1 = {}
for dieta in ["D1", "D4"]:
    x0 = np.array([-0.33, 0.6])  # log10(k_ref)≈0.47, delta_CS≈0.6
    lo = np.array([-2.0, 0.3])
    hi = np.array([1.0, 0.9])
    sol = least_squares(residuos_control, x0, bounds=(lo, hi), method="trf", args=(dieta,))
    k_ref = 10.0 ** sol.x[0]
    delta_CS = sol.x[1]
    etapa1[dieta] = (k_ref, delta_CS)
    rmse = np.sqrt(np.mean(sol.fun ** 2))
    print(f"\n{dieta}: k_ref={k_ref:.4f}, delta_CS={delta_CS:.4f}, RMSE={rmse:.2f}")
    
    # Verificacion
    t_eval = np.linspace(0, T_FIN, 100)
    t, ppm = simular_control(dieta, sol.x[0], delta_CS, t_eval)
    obs = ctrl[ctrl["dieta"] == dieta]
    print("  Dia   Obs    Mod")
    for _, r in obs.iterrows():
        d = r["dia_larvario"]
        p = np.interp(d, t, ppm)
        print(f"  {d:5.1f} {r['Tasa_Produccion']:6.1f} {p:6.1f}")

print("\n" + "=" * 60)
print("ETAPA 1 COMPLETADA")
print("=" * 60)
for d, (k, cs) in etapa1.items():
    print(f"{d}: k_ref={k:.4f}, delta_CS={cs:.4f}")

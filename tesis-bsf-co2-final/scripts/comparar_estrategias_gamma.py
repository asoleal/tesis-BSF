#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from modelo import ParametrosEriksen, ParametrosLecho, estado_inicial_desde_dw, rhs, tasas_larva, tasas_lecho, carbono_a_ppm_min

P = ParametrosEriksen(amax=1.200, rho=3.0, tp_min=8.5)
PULSOS = [(0.0, 250.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
N_LARVAS = 700
T_K = 300.0
T_C = 27.0
V_AIRE_L = 13.08
DELTA_C_ALIM = 0.47
F_MS = 0.325
T_FIN = 14.0
DIA_MONTAJE_SEP = 3
K_D = 2.0
Y_X = 0.4
Y_CMIC = 0.4
K_SM = 50.0
PESO_GAS = 25.0

micro = {
    "D1": (1.0, 0.665),
    "D4": (1.0, 0.538),
}

crec = pd.read_csv("datos/procesados/crecimiento_canonico.csv")
cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
gas = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") & (cierres["condicion"] == "tratamiento")].copy()
gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP

PH0 = {d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0]) for d, g in crec.groupby("dieta")}

def simular_dieta(dieta, amax, rho, f_d4, gamma, k_ref, delta_CS):
    f_dieta = 1.0 if dieta == "D1" else f_d4
    p_larva = ParametrosEriksen(amax=amax, rho=rho, tp_min=8.5)
    p_lecho = ParametrosLecho(f_dieta=f_dieta, k_ref=k_ref, K_SM=K_SM, Y_X=Y_X, delta_CS=delta_CS, k_d=K_D, gamma=gamma, K_G=0.1)
    y = estado_inicial_desde_dw(PH0[dieta] * F_MS, 0.0, 1.0, p_larva)
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    t_eval = np.linspace(0, T_FIN, 300)
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a in pulsos:
            y[2] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * delta_CS
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            continue
        sol = solve_ivp(rhs, (a, b), y, args=(p_larva, p_lecho, N_LARVAS, T_C), method="LSODA", t_eval=t_seg, rtol=1e-6, atol=1e-9)
        if not sol.success:
            return None, None, None
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    R_C = np.empty_like(t)
    for k, (tk, yk) in enumerate(zip(t, Y.T)):
        B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in yk]
        mic = tasas_lecho(D_C, X, T_C, p_lecho)
        lar = tasas_larva(tk, B, Ltot, p_larva, p_lecho.f_dieta)
        R_C[k] = N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic
    ppm = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = ((1.0 - p_larva.delta_LB) * B / (p_larva.delta_CB * p_larva.delta_O) + Ltot / p_larva.delta_CL) / F_MS
    return t, ppm, ph

def rmse_gas(dieta, amax, rho, f_d4, gamma, k_ref, delta_CS):
    t, ppm, ph = simular_dieta(dieta, amax, rho, f_d4, gamma, k_ref, delta_CS)
    if t is None:
        return 1e6
    g = gas[gas["dieta"] == dieta]
    obs = g["tasa_ppm_min_tiempo"].values
    dias = g["dia_larvario"].values
    pred = np.interp(dias, t, ppm)
    r = pred - obs
    # Saturados: solo penaliza por debajo
    for i, sat in enumerate(g["Saturado"]):
        if sat and r[i] > 0:
            r[i] = 0
    return np.sqrt(np.mean(r**2))

def rmse_crec(dieta, amax, rho, f_d4, gamma, k_ref, delta_CS):
    t, ppm, ph = simular_dieta(dieta, amax, rho, f_d4, gamma, k_ref, delta_CS)
    if t is None:
        return 1e6
    c = crec[crec["dieta"] == dieta]
    obs = c["ph_mg"].values
    dias = c["dia_larvario"].values
    pred = np.interp(dias, t, ph)
    w = np.maximum(c["de_mg"].values, 0.1 * obs)
    return np.sqrt(np.mean(((pred - obs) / w)**2))

print("=== ESTRATEGIA A: gamma por dieta ===")
best_A = None
best_rmse_A = 1e9
for g1 in np.arange(0.0, 0.51, 0.02):
    for g4 in np.arange(0.0, 0.61, 0.02):
        r1g = rmse_gas("D1", 1.2, 3.0, 0.7, g1, 1.0, 0.665)
        r4g = rmse_gas("D4", 1.2, 3.0, 0.7, g4, 1.0, 0.538)
        r1c = rmse_crec("D1", 1.2, 3.0, 0.7, g1, 1.0, 0.665)
        r4c = rmse_crec("D4", 1.2, 3.0, 0.7, g4, 1.0, 0.538)
        total = r1g + r4g + r1c + r4c
        if total < best_rmse_A:
            best_rmse_A = total
            best_A = (g1, g4, r1g, r4g, r1c, r4c, total)

print(f"  gamma_D1={best_A[0]:.3f}, gamma_D4={best_A[1]:.3f}")
print(f"  RMSE gas D1={best_A[2]:.1f}, D4={best_A[3]:.1f}")
print(f"  RMSE crec D1={best_A[4]:.1f}, D4={best_A[5]:.1f}")
print(f"  Total={best_A[6]:.1f}")

print("\n=== ESTRATEGIA B: gamma = gamma_0 * f_dieta ===")
best_B = None
best_rmse_B = 1e9
for g0 in np.arange(0.0, 0.71, 0.02):
    g1 = g0 * 1.0
    g4 = g0 * 0.7
    r1g = rmse_gas("D1", 1.2, 3.0, 0.7, g1, 1.0, 0.665)
    r4g = rmse_gas("D4", 1.2, 3.0, 0.7, g4, 1.0, 0.538)
    r1c = rmse_crec("D1", 1.2, 3.0, 0.7, g1, 1.0, 0.665)
    r4c = rmse_crec("D4", 1.2, 3.0, 0.7, g4, 1.0, 0.538)
    total = r1g + r4g + r1c + r4c
    if total < best_rmse_B:
        best_rmse_B = total
        best_B = (g0, g1, g4, r1g, r4g, r1c, r4c, total)

print(f"  gamma_0={best_B[0]:.3f} -> gamma_D1={best_B[1]:.3f}, gamma_D4={best_B[2]:.3f}")
print(f"  RMSE gas D1={best_B[3]:.1f}, D4={best_B[4]:.1f}")
print(f"  RMSE crec D1={best_B[5]:.1f}, D4={best_B[6]:.1f}")
print(f"  Total={best_B[7]:.1f}")

print(f"\n=== GANADOR ===")
if best_rmse_A < best_rmse_B:
    print(f"Estrategia A gana por {best_rmse_B - best_rmse_A:.1f}")
else:
    print(f"Estrategia B gana por {best_rmse_A - best_rmse_B:.1f}")

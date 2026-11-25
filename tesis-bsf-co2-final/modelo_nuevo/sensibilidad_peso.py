#!/usr/bin/env python3
"""Sensibilidad del peso del crecimiento en la calibracion."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from modelo_eriksen import estado_inicial_desde_dw, tasas_larva
from modelo_completo import rhs_completo, tasas_lecho, carbono_a_ppm_min

K_REF = {"D1": 0.3994, "D4": 0.3961}
DELTA_CS = {"D1": 0.8855, "D4": 0.9000}
N_LARVAS, PULSOS, MS_DIETA = 700, [(0.0, 500.0), (5.0, 250.0)], {"D1": 0.863, "D4": 0.803}
T_K, T_C, V_AIRE_L, DELTA_C_ALIM, F_MS, DIA_MONTAJE_SEP, T_FIN = 300.15, 27.0, 12.1, 0.47, 0.325, 3, 14.0

P_LARVA = {"amax": 1.2, "alpha": 1.0, "beta": 1.0, "YB": 0.44, "YL": 0.42, "m": 0.08,
    "Bmax0": 90.0 * 0.49 * 0.9, "tp_min": 8.5, "rho": 3.0,
    "delta_I": 5.0, "delta_LB": 0.1, "delta_CB": 0.49, "delta_CL": 0.79, "delta_O": 0.9,
    "f_mant_prepupa": 1.0, "k_cat_prepupa": 0.0}
P_LECHO_BASE = {"K_D": 50.0, "K_SM": 50.0, "Y_X": 0.4, "Y_Cmic": 0.4,
    "k_d": 0.05, "K_G": 0.1, "Q10": 2.0, "T_ref_C": 25.0, "theta": 0.7, "K_theta": 0.15}

BASE = Path(__file__).resolve().parents[1]
pinn = pd.read_csv(BASE / "datos" / "procesados" / "datos_finales_PINN_corregidos.csv")
pinn = pinn[pinn["Gas"] == "co2"].copy()
pinn["dia_larvario"] = pinn["Dia_Experimento"] - DIA_MONTAJE_SEP
trat = pinn[pinn["Etiqueta"].str.contains("Tratamiento|Generacion_Base")].copy()
trat["dieta"] = trat["Etiqueta"].str.extract(r"(D[14])")[0]
crec = pd.read_csv(BASE / "datos" / "procesados" / "crecimiento_canonico.csv")

obs_gas = {d: [(float(r["dia_larvario"]), float(r["Tasa_Produccion"]), bool(r["Saturado"]))
               for _, r in trat[trat["dieta"] == d].iterrows()] for d in ["D1", "D4"]}
obs_crec = {d: [(float(r["dia_larvario"]), float(r["ph_mg"]))
                for _, r in crec[crec["dieta"] == d].iterrows()] for d in ["D1", "D4"]}

t_eval = np.array(sorted(set([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    + [dia for dia, _, _ in obs_gas["D1"]] + [dia for dia, _ in obs_crec["D1"]]
    + [dia for dia, _, _ in obs_gas["D4"]] + [dia for dia, _ in obs_crec["D4"]])))

def simular(dieta_nom, a_max, rho, f_D4, gamma_D1, gamma_D4):
    p_larva = dict(P_LARVA, amax=a_max, rho=rho)
    f_dieta = f_D4 if dieta_nom == "D4" else 1.0
    gamma = gamma_D4 if dieta_nom == "D4" else gamma_D1
    p_lecho = dict(P_LECHO_BASE, f_dieta=f_dieta, k_ref=K_REF[dieta_nom],
                   delta_CS=DELTA_CS[dieta_nom], gamma=gamma)
    ph0 = crec[(crec["dieta"] == dieta_nom) & (crec["dia_larvario"] == 0)]["ph_mg"].iloc[0]
    y0_eri = estado_inicial_desde_dw(ph0 * F_MS, p_larva)
    y0 = np.array([y0_eri[0], y0_eri[1], 0.0, 1.0, 0.0, 0.0])
    y0[2] += PULSOS[0][1] * MS_DIETA[dieta_nom] * DELTA_C_ALIM * DELTA_CS[dieta_nom]
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0 and a in pulsos:
            y0[2] += pulsos[a] * MS_DIETA[dieta_nom] * DELTA_C_ALIM * DELTA_CS[dieta_nom]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(lambda t, y: rhs_completo(t, y, p_larva, p_lecho, {"T_C": T_C}, dieta_nom, N_LARVAS),
                        (a, b), y0, t_eval=t_seg, method="LSODA", rtol=1e-7, atol=1e-10)
        if not sol.success:
            raise RuntimeError(f"fallo [{a},{b}]")
        ts.append(sol.t); ys.append(sol.y); y0 = sol.y[:, -1].copy()
    t = np.concatenate(ts); Y = np.concatenate(ys, axis=1)
    ppm = np.array([carbono_a_ppm_min(N_LARVAS * tasas_larva(ti, yi[0], yi[1], p_larva, f_dieta)["rCO2_C"] / 1000.0
                    + tasas_lecho(yi[2], yi[3], T_C, p_lecho)["rCmic"], V_AIRE_L, T_K)
                    for ti, yi in zip(t, Y.T)])
    ph = np.array([((1.0 - p_larva["delta_LB"]) * Bi / (p_larva["delta_CB"] * p_larva["delta_O"]) + Li / p_larva["delta_CL"]) / F_MS
                   for Bi, Li in zip(Y[0], Y[1])])
    return t, ph, ppm

def calibrar(peso_crec):
    def residuos(theta):
        a_max, rho, f_D4, gamma_D1, gamma_D4 = theta
        res = []
        for dieta_nom in ["D1", "D4"]:
            t, ph, ppm = simular(dieta_nom, a_max, rho, f_D4, gamma_D1, gamma_D4)
            for dia, ob, sat in obs_gas[dieta_nom]:
                m = float(np.interp(dia, t, ppm)); r = m - ob
                if sat and r > 0: r = 0.0
                res.append(r / 25.0)
            for dia, ob in obs_crec[dieta_nom]:
                m = float(np.interp(dia, t, ph))
                res.append((m - ob) / peso_crec)
        return np.array(res)

    lo = np.array([1.0, 0.5, 0.4, 0.0, 0.0])
    hi = np.array([1.5, 15.0, 1.0, 1.0, 1.0])
    x0 = np.array([1.2, 3.0, 0.7, 0.10, 0.40])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    sol = least_squares(residuos, x0, bounds=(lo, hi), method="trf", max_nfev=500)

    a_max, rho, f_D4, gamma_D1, gamma_D4 = sol.x
    t_fino = np.linspace(0, T_FIN, 200)
    resultados = []
    for dieta_nom in ["D1", "D4"]:
        t, ph, ppm = simular(dieta_nom, a_max, rho, f_D4, gamma_D1, gamma_D4)
        rmse_g = np.sqrt(np.mean([(float(np.interp(dia, t, ppm)) - ob)**2 for dia, ob, _ in obs_gas[dieta_nom]]))
        rmse_c = np.sqrt(np.mean([(float(np.interp(dia, t, ph)) - ob)**2 for dia, ob in obs_crec[dieta_nom]]))
        resultados.append((dieta_nom, rmse_g, rmse_c))
    return a_max, rho, f_D4, gamma_D1, gamma_D4, sol.cost, resultados

print(f"{'peso':>8} {'a_max':>7} {'rho':>6} {'f_D4':>7} {'g_D1':>7} {'g_D4':>7} {'cost':>8} {'D1_g':>7} {'D1_c':>7} {'D4_g':>7} {'D4_c':>7}")
print("-" * 95)

for peso in [100.0, 50.0, 30.0, 20.0, 15.0, 10.0, 5.0]:
    try:
        a, r, f, g1, g4, cost, res = calibrar(peso)
        r1g, r1c = res[0][1], res[0][2]
        r4g, r4c = res[1][1], res[1][2]
        print(f"{peso:8.1f} {a:7.4f} {r:6.3f} {f:7.4f} {g1:7.4f} {g4:7.4f} {cost:8.2f} {r1g:7.1f} {r1c:7.2f} {r4g:7.1f} {r4c:7.2f}")
    except Exception as e:
        print(f"{peso:8.1f} ERROR: {e}")

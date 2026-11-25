#!/usr/bin/env python3
"""Validacion E0 vs C con parametros finales de calibracion — CORREGIDO."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from modelo_eriksen import estado_inicial_desde_dw, rhs_eriksen, tasas_larva, B_Ltot_a_PH
from modelo_completo import rhs_completo, tasas_lecho, carbono_a_ppm_min

P = {
    "montaje": {
        "V_aire_L": 12.1, "T_K": 300.15, "T_C": 27.0, "T_FIN": 14.0,
        "DIA_MONTAJE_SEP": 3, "N_LARVAS": 700,
        "ms_dieta": {"D1": 0.863, "D4": 0.803},
        "delta_C_alim": 0.47,
    },
    "eriksen": {
        "amax": 1.2912, "alpha": 1.0, "beta": 1.0,
        "YB": 0.44, "YL": 0.42, "m": 0.08,
        "Bmax0": 90.0 * 0.49 * 0.9,
        "tp_min": 8.5, "rho": 4.331,
        "delta_I": 5.0, "delta_LB": 0.1,
        "delta_CB": 0.49, "delta_CL": 0.79, "delta_O": 0.9,
        "f_mant_prepupa": 1.0, "k_cat_prepupa": 0.0,
    },
    "lecho": {
        "K_D": 50.0, "K_SM": 50.0, "Y_X": 0.4, "Y_Cmic": 0.4,
        "k_d": 0.05, "K_G": 0.1, "Q10": 2.0,
        "T_ref_C": 25.0, "theta": 0.7, "K_theta": 0.15,
    },
    "dietas": {
        "D1": {"k_ref": 0.3994, "delta_CS": 0.8855, "f_dieta": 1.0, "gamma": 0.0784},
        "D4": {"k_ref": 0.3961, "delta_CS": 0.9000, "f_dieta": 0.6546, "gamma": 0.5416},
    }
}

F_MS = 0.325
PULSOS = [(0.0, 500.0), (5.0, 250.0)]

BASE = Path(__file__).resolve().parents[1]
pinn = pd.read_csv(BASE / "datos" / "procesados" / "datos_finales_PINN_corregidos.csv")
pinn = pinn[pinn["Gas"] == "co2"].copy()
pinn["dia_larvario"] = pinn["Dia_Experimento"] - P["montaje"]["DIA_MONTAJE_SEP"]

ctrl = pinn[pinn["Etiqueta"].str.contains("Control")].copy()
ctrl["dieta"] = ctrl["Etiqueta"].str.replace("Control_Alimento_", "")
trat = pinn[pinn["Etiqueta"].str.contains("Tratamiento|Generacion_Base")].copy()
trat["dieta"] = trat["Etiqueta"].str.extract(r"(D[14])")[0]
crec = pd.read_csv(BASE / "datos" / "procesados" / "crecimiento_canonico.csv")


def sim_E0(dieta, t_eval):
    p_larva = dict(P["eriksen"])
    f_dieta = P["dietas"][dieta]["f_dieta"]
    ph0 = crec[(crec["dieta"] == dieta) & (crec["dia_larvario"] == 0)]["ph_mg"].iloc[0]
    y0 = estado_inicial_desde_dw(ph0 * F_MS, p_larva)
    sol = solve_ivp(lambda t, y: rhs_eriksen(t, y, p_larva, f_dieta),
                    (0, P["montaje"]["T_FIN"]), y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-7, atol=1e-10)
    B, Ltot = sol.y[0], sol.y[1]
    ph = np.array([B_Ltot_a_PH(Bi, Li, p_larva, F_MS) for Bi, Li in zip(B, Ltot)])
    rCO2 = np.array([tasas_larva(ti, Bi, Li, p_larva, f_dieta)["rCO2_C"]
                     for ti, Bi, Li in zip(sol.t, B, Ltot)])
    ppm = np.array([carbono_a_ppm_min(P["montaje"]["N_LARVAS"] * r / 1000.0,
                    P["montaje"]["V_aire_L"], P["montaje"]["T_K"]) for r in rCO2])
    return sol.t, ph, ppm


def sim_C(dieta, t_eval, N_larvas):
    p_larva = dict(P["eriksen"])
    f_dieta = P["dietas"][dieta]["f_dieta"]
    p_lecho = dict(P["lecho"],
                   f_dieta=f_dieta,
                   k_ref=P["dietas"][dieta]["k_ref"],
                   delta_CS=P["dietas"][dieta]["delta_CS"],
                   gamma=P["dietas"][dieta]["gamma"])

    if N_larvas > 0:
        ph0 = crec[(crec["dieta"] == dieta) & (crec["dia_larvario"] == 0)]["ph_mg"].iloc[0]
        y0_eri = estado_inicial_desde_dw(ph0 * F_MS, p_larva)
        y0 = np.array([y0_eri[0], y0_eri[1], 0.0, 1.0, 0.0, 0.0])
    else:
        y0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    y0[2] += PULSOS[0][1] * P["montaje"]["ms_dieta"][dieta] * P["montaje"]["delta_C_alim"] * P["dietas"][dieta]["delta_CS"]

    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [P["montaje"]["T_FIN"]]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0 and a in pulsos:
            y0[2] += pulsos[a] * P["montaje"]["ms_dieta"][dieta] * P["montaje"]["delta_C_alim"] * P["dietas"][dieta]["delta_CS"]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            lambda t, y: rhs_completo(t, y, p_larva, p_lecho, {"T_C": P["montaje"]["T_C"]}, dieta, N_larvas),
            (a, b), y0, t_eval=t_seg, method="LSODA", rtol=1e-7, atol=1e-10)
        ts.append(sol.t); ys.append(sol.y); y0 = sol.y[:, -1].copy()

    t = np.concatenate(ts); Y = np.concatenate(ys, axis=1)
    ppm = np.empty_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        if N_larvas > 0:
            lar = tasas_larva(ti, yi[0], yi[1], p_larva, f_dieta)
            rCO2_larva = N_larvas * lar["rCO2_C"] / 1000.0
        else:
            rCO2_larva = 0.0
        mic = tasas_lecho(yi[2], yi[3], P["montaje"]["T_C"], p_lecho)
        rCmic_g_d = mic["rCmic"] * 24.0 * 60.0 / 1000.0
        R_C = rCO2_larva + rCmic_g_d
        ppm[i] = carbono_a_ppm_min(R_C, P["montaje"]["V_aire_L"], P["montaje"]["T_K"])

    if N_larvas > 0:
        ph = np.array([B_Ltot_a_PH(Bi, Li, p_larva, F_MS) for Bi, Li in zip(Y[0], Y[1])])
    else:
        ph = np.zeros_like(t)
    return t, ph, ppm, Y


t_eval = np.array(sorted(set(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    + list(ctrl[ctrl["dieta"] == "D1"]["dia_larvario"].values)
    + list(ctrl[ctrl["dieta"] == "D4"]["dia_larvario"].values)
    + list(trat[trat["dieta"] == "D1"]["dia_larvario"].values)
    + list(trat[trat["dieta"] == "D4"]["dia_larvario"].values)
    + list(crec[crec["dieta"] == "D1"]["dia_larvario"].values)
    + list(crec[crec["dieta"] == "D4"]["dia_larvario"].values)
)))

print("=" * 70)
print("VALIDACION E0 vs C — Parametros finales (peso 30)")
print("=" * 70)

for dieta in ["D1", "D4"]:
    print(f"\n--- {dieta} ---")

    t, ph_E0, ppm_E0 = sim_E0(dieta, t_eval)
    t, ph_C, ppm_C, Y_C = sim_C(dieta, t_eval, P["montaje"]["N_LARVAS"])
    t_ctrl, ph_ctrl, ppm_ctrl, Y_ctrl = sim_C(dieta, t_eval, 0)

    sub_t = trat[trat["dieta"] == dieta]
    obs_g = [(float(r["dia_larvario"]), float(r["Tasa_Produccion"]), bool(r["Saturado"])) for _, r in sub_t.iterrows()]

    print("Gas tratamientos (dia, obs, E0, C):")
    for dia, ob, sat in obs_g:
        p_E0 = float(np.interp(dia, t, ppm_E0))
        p_C = float(np.interp(dia, t, ppm_C))
        print(f"  {dia:5.1f} {ob:8.1f} {p_E0:8.1f} {p_C:8.1f} {'SAT' if sat else ''}")

    rmse_g_E0 = np.sqrt(np.mean([(float(np.interp(d, t, ppm_E0)) - ob)**2 for d, ob, _ in obs_g]))
    rmse_g_C = np.sqrt(np.mean([(float(np.interp(d, t, ppm_C)) - ob)**2 for d, ob, _ in obs_g]))
    print(f"  RMSE gas trat: E0={rmse_g_E0:.1f}, C={rmse_g_C:.1f}")

    sub_c = ctrl[ctrl["dieta"] == dieta]
    obs_c = [(float(r["dia_larvario"]), float(r["Tasa_Produccion"])) for _, r in sub_c.iterrows()]
    if obs_c:
        print("Controles (dia, obs, C-control):")
        for dia, ob in obs_c:
            p_Cc = float(np.interp(dia, t_ctrl, ppm_ctrl))
            print(f"  {dia:5.1f} {ob:8.1f} {p_Cc:8.1f}")
        rmse_c_ctrl = np.sqrt(np.mean([(float(np.interp(d, t_ctrl, ppm_ctrl)) - ob)**2 for d, ob in obs_c]))
        print(f"  RMSE control: C={rmse_c_ctrl:.1f}")

    sub_cr = crec[crec["dieta"] == dieta]
    obs_cr = [(float(r["dia_larvario"]), float(r["ph_mg"])) for _, r in sub_cr.iterrows()]
    print("Crecimiento (dia, obs, E0, C):")
    for dia, ob in obs_cr:
        p_E0 = float(np.interp(dia, t, ph_E0))
        p_C = float(np.interp(dia, t, ph_C))
        print(f"  {dia:5.1f} {ob:8.1f} {p_E0:8.1f} {p_C:8.1f}")
    rmse_cr_E0 = np.sqrt(np.mean([(float(np.interp(d, t, ph_E0)) - ob)**2 for d, ob in obs_cr]))
    rmse_cr_C = np.sqrt(np.mean([(float(np.interp(d, t, ph_C)) - ob)**2 for d, ob in obs_cr]))
    print(f"  RMSE crec: E0={rmse_cr_E0:.2f}, C={rmse_cr_C:.2f}")

print("\n" + "=" * 70)
print("CIERRE DE CARBONO (solo tratamientos)")
print("=" * 70)

for dieta in ["D1", "D4"]:
    t, ph, ppm, Y = sim_C(dieta, t_eval, P["montaje"]["N_LARVAS"])
    p_larva = dict(P["eriksen"])
    f_dieta = P["dietas"][dieta]["f_dieta"]
    p_lecho = dict(P["lecho"], f_dieta=f_dieta, k_ref=P["dietas"][dieta]["k_ref"],
                   delta_CS=P["dietas"][dieta]["delta_CS"], gamma=P["dietas"][dieta]["gamma"])

    B0, L0 = Y[0, 0], Y[1, 0]
    C_larva_ini = (B0 + L0) / 1000.0 * P["montaje"]["N_LARVAS"]

    Bf, Lf = Y[0, -1], Y[1, -1]
    C_larva_fin = (Bf + Lf) / 1000.0 * P["montaje"]["N_LARVAS"]
    C_larva_acum = C_larva_fin - C_larva_ini

    C_alim_mg = sum(ms * P["montaje"]["ms_dieta"][dieta] * P["montaje"]["delta_C_alim"] * P["dietas"][dieta]["delta_CS"] for _, ms in PULSOS)
    C_alim_g = C_alim_mg / 1000.0

    CO2_total_g = 0.0
    for i in range(len(t)-1):
        dt_dias = t[i+1] - t[i]
        lar = tasas_larva(t[i], Y[0, i], Y[1, i], p_larva, f_dieta)
        mic = tasas_lecho(Y[2, i], Y[3, i], P["montaje"]["T_C"], p_lecho)
        CO2_larva_mg = P["montaje"]["N_LARVAS"] * lar["rCO2_C"] * dt_dias
        CO2_lecho_mg = mic["rCmic"] * 24.0 * 60.0 * dt_dias
        CO2_total_g += (CO2_larva_mg + CO2_lecho_mg) / 1000.0

    C_mic_muerto_g = Y[4, -1] / 1000.0
    D_C_final_g = Y[2, -1] / 1000.0

    print(f"\n{dieta}:")
    print(f"  C alimento:        {C_alim_g:10.4f} g C")
    print(f"  C larva acumulado: {C_larva_acum:10.4f} g C")
    print(f"  CO2 emitido:       {CO2_total_g:10.4f} g C")
    print(f"  C microbio muerto: {C_mic_muerto_g:10.4f} g C")
    print(f"  C sustrato res:    {D_C_final_g:10.4f} g C")
    C_salidas = C_larva_acum + CO2_total_g + C_mic_muerto_g + D_C_final_g
    print(f"  C no explicado:    {C_alim_g - C_salidas:10.4f} g C")
    print(f"  Balance:           {C_salidas / C_alim_g * 100:9.2f}%")

print("\nOK")

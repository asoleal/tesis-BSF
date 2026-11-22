#!/usr/bin/env python3
"""Compara E0 (Eriksen puro) vs C (modelo completo): controles + tratamientos."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from modelo.modelo_bsf import (
    ParametrosEriksen, ParametrosLecho, estado_inicial_desde_dw,
    rhs, tasas_larva, tasas_lecho, carbono_a_ppm_min,
)

A_MAX = 1.1798
RHO = 3.0
F_D4 = 0.6978
GAMMA_C = 0.1703

N_LARVAS = 700
PULSOS = [(0.0, 500.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.15; T_C = T_K - 273.15
V_AIRE_L = 12.1; DELTA_C_ALIM = 0.47; F_MS = 0.325
DIA_MONTAJE_SEP = 3; T_FIN = 14.0
X0_G_C = 1.0; Y_X = 0.4; K_SM = 50.0; K_G = 0.1

K_REF_C = {"D1": 1.0, "D4": 1.0}
DELTA_CS_C = {"D1": 0.665, "D4": 0.538}


def simular_control_C(dieta, t_eval):
    """Control con modelo C: SOLO lecho (2 estados), sin larva."""
    p_lecho = ParametrosLecho(
        f_dieta=1.0, k_ref=K_REF_C[dieta], K_SM=K_SM,
        Y_X=Y_X, delta_CS=DELTA_CS_C[dieta], k_d=0.05,
        gamma=0.0, K_G=K_G,
    )

    def rhs_lecho(t, y):
        D_C, X = [max(float(v), 0.0) for v in y]
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        return [micro.dD_C, micro.dX]

    y = np.array([0.0, X0_G_C])
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0:
            y[0] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS_C[dieta]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs_lecho, (a, b), y, method="LSODA", t_eval=t_seg,
            rtol=1e-5, atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"fallo [{a},{b}]: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1].copy()

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    R_C = np.array([
        tasas_lecho(max(float(y[0]), 0.0), max(float(y[1]), 0.0), T_C, p_lecho).rCmic
        for y in Y.T
    ])
    return t, carbono_a_ppm_min(R_C, V_AIRE_L, T_K)


def simular_tratamiento(config, dieta, t_eval):
    """Tratamiento: E0 (sin lecho) o C (con lecho + pastoreo)."""
    gamma = 0.0 if config == "E0" else GAMMA_C
    k_ref = 0.0 if config == "E0" else K_REF_C[dieta]
    dcs = 1.0 if config == "E0" else DELTA_CS_C[dieta]
    x0 = 0.0 if config == "E0" else X0_G_C

    p_larva = ParametrosEriksen(
        amax=A_MAX, alpha=1.0, beta=1.0, YB=0.44, YL=0.15, m=0.15,
        Bmax0=39.69, tp_min=13.0, rho=RHO, delta_I=1.25,
        delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
        f_mant_prepupa=0.3, k_cat_prepupa=0.02,
    )
    f_dieta = F_D4 if dieta == "D4" else 1.0
    p_lecho = ParametrosLecho(
        f_dieta=f_dieta, k_ref=k_ref, K_SM=K_SM,
        Y_X=Y_X, delta_CS=dcs, k_d=0.05,
        gamma=gamma, K_G=K_G,
    )

    XDW0 = 4.0 * F_MS
    D_C0 = 1e6 if config == "E0" else PULSOS[0][1] * MS_DIETA[dieta] * DELTA_C_ALIM * dcs
    y0 = estado_inicial_desde_dw(XDW0_mg=XDW0, D_C0=D_C0, X0=x0, p=p_larva)

    def f(t, y): return rhs(t, y, p_larva, p_lecho, N_LARVAS, T_C)

    if config == "E0":
        sol = solve_ivp(f, (0, T_FIN), y0, method="LSODA", t_eval=t_eval, rtol=1e-5, atol=1e-8)
        if not sol.success: raise RuntimeError(sol.message)
        t = sol.t; Y = sol.y
    else:
        cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
        pulsos = dict(PULSOS)
        ts, ys = [], []
        y = y0.copy()
        for a, b in zip(cortes[:-1], cortes[1:]):
            if a > 0.0:
                y[2] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * dcs
            t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
            if len(t_seg) < 2:
                t_seg = np.linspace(a, b, 3)
            sol = solve_ivp(f, (a, b), y, method="LSODA", t_eval=t_seg, rtol=1e-5, atol=1e-8)
            if not sol.success:
                raise RuntimeError(f"fallo [{a},{b}]")
            ts.append(sol.t); ys.append(sol.y); y = sol.y[:, -1].copy()
        t = np.concatenate(ts); Y = np.concatenate(ys, axis=1)

    ppm = np.zeros_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        lar = tasas_larva(ti, yi[0], yi[1], p_larva, factor_alimento=f_dieta)
        mic = tasas_lecho(yi[2], yi[3], T_C, p_lecho)
        R_C = N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic
        ppm[i] = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)

    B = np.maximum(Y[0], 0.0); Ltot = np.maximum(Y[1], 0.0)
    ph = ((1.0 - p_larva.delta_LB) * B / (p_larva.delta_CB * p_larva.delta_O)
          + Ltot / p_larva.delta_CL) / F_MS
    return t, ph, ppm


def main():
    cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")

    ctrl = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") & (cierres["condicion"] == "control")].copy()
    trata = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") & (cierres["condicion"] == "tratamiento")].copy()
    ctrl["dia_larvario"] = ctrl["Dia_Experimento"] - DIA_MONTAJE_SEP
    trata["dia_larvario"] = trata["Dia_Experimento"] - DIA_MONTAJE_SEP

    t_eval = np.linspace(0, T_FIN, 71)

    filas = []
    for config in ("E0", "C"):
        # Controles: C usa lecho de 2 estados, E0 usa lecho apagado (sin microbios)
        for dieta, g in ctrl.groupby("dieta"):
            if config == "C":
                t, ppm = simular_control_C(dieta, t_eval)
            else:
                # E0 control: sin larvas, sin lecho = 0 CO2
                t = t_eval
                ppm = np.zeros_like(t_eval)
            for _, r in g.iterrows():
                dia = float(r["dia_larvario"])
                mod = float(np.interp(dia, t, ppm))
                filas.append(dict(config=config, tipo="control", dieta=dieta, dia=dia,
                                  obs=float(r["tasa_ppm_min_tiempo"]), mod=mod,
                                  residuo=mod-float(r["tasa_ppm_min_tiempo"])))

        # Tratamientos
        for dieta, g in trata.groupby("dieta"):
            t, ph, ppm = simular_tratamiento(config, dieta, t_eval)
            for _, r in g.iterrows():
                dia = float(r["dia_larvario"])
                mod = float(np.interp(dia, t, ppm))
                filas.append(dict(config=config, tipo="tratamiento", dieta=dieta, dia=dia,
                                  obs=float(r["tasa_ppm_min_tiempo"]), mod=mod,
                                  residuo=mod-float(r["tasa_ppm_min_tiempo"]),
                                  saturado=bool(r.get("Saturado", False))))

    comp = pd.DataFrame(filas)
    comp.to_csv("resultados/comparacion_E0_vs_C.csv", index=False)

    print("=== COMPARACION E0 vs C ===")
    for (config, tipo, dieta), g in comp.groupby(["config", "tipo", "dieta"]):
        rmse = float(np.sqrt(np.mean(g["residuo"]**2)))
        mae = float(np.mean(np.abs(g["residuo"])))
        print(f"{config} {tipo} {dieta}: RMSE={rmse:.2f}, MAE={mae:.2f} (n={len(g)})")

    print("\nTabla resumen:")
    resumen = comp.groupby(["config", "tipo", "dieta"]).agg(
        rmse=("residuo", lambda x: float(np.sqrt(np.mean(x**2)))),
        mae=("residuo", lambda x: float(np.mean(np.abs(x)))),
        n=("residuo", "count"),
    ).reset_index()
    print(resumen.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

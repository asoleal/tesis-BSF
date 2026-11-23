#!/usr/bin/env python3
"""Etapa 3: prepupa por dieta."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from modelo.modelo_bsf import (
    ParametrosEriksen, ParametrosLecho, estado_inicial_desde_dw,
    rhs, tasas_larva, tasas_lecho, carbono_a_ppm_min,
)

A_MAX = 1.1798; RHO = 3.0; F_D4 = 0.6978; GAMMA = 0.1703
K_REF = {"D1": 1.0, "D4": 1.0}; DELTA_CS = {"D1": 0.664521, "D4": 0.537609}
N = 700; PULSOS = [(0.0, 500.0), (5.0, 250.0)]
MS = {"D1": 0.863, "D4": 0.803}; T_K = 300.15; T_C = T_K - 273.15
V = 12.1; DELTA_C = 0.47; F_MS = 0.325; T_FIN = 14.0; X0 = 1.0
Y_X = 0.4; K_SM = 50.0; K_G = 0.1; k_d = 0.05


def simular(dieta, f_mant, k_cat, t_eval):
    p_larva = ParametrosEriksen(
        amax=A_MAX, alpha=1.0, beta=1.0, YB=0.44, YL=0.42, m=0.08,
        Bmax0=39.69, tp_min=13.0, rho=RHO, delta_I=5.0,
        delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
         
    )
    f_d = F_D4 if dieta == "D4" else 1.0
    p_lecho = ParametrosLecho(
        f_dieta=f_d, k_ref=K_REF[dieta], K_SM=K_SM,
        Y_X=Y_X, delta_CS=DELTA_CS[dieta], k_d=k_d,
        gamma=GAMMA, K_G=K_G,
    )
    y0 = estado_inicial_desde_dw(
        XDW0_mg=4.0 * F_MS,
        D_C0=PULSOS[0][1] * MS[dieta] * DELTA_C * DELTA_CS[dieta],
        X0=X0, p=p_larva,
    )

    def f(t, y): return rhs(t, y, p_larva, p_lecho, N, T_C)

    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    y = y0.copy()
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0: y[2] += pulsos[a] * MS[dieta] * DELTA_C * DELTA_CS[dieta]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2: t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(f, (a, b), y, method="LSODA", t_eval=t_seg, rtol=1e-5, atol=1e-8)
        if not sol.success: raise RuntimeError(f"fallo [{a},{b}]")
        ts.append(sol.t); ys.append(sol.y); y = sol.y[:, -1].copy()
    t = np.concatenate(ts); Y = np.concatenate(ys, axis=1)

    ppm = np.zeros_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        lar = tasas_larva(ti, yi[0], yi[1], p_larva, factor_alimento=p_lecho.f_dieta)
        mic = tasas_lecho(yi[2], yi[3], T_C, p_lecho)
        ppm[i] = carbono_a_ppm_min(N * lar.rCO2_C / 1000.0 + mic.rCmic, V, T_K)
    return t, ppm


def main():
    cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
    gas = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") &
                  (cierres["condicion"] == "tratamiento")].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - 3

    obs = {}
    for d in ("D1", "D4"):
        obs[d] = [(float(r["dia_larvario"]), float(r["tasa_ppm_min_tiempo"]), bool(r.get("Saturado", False)))
                  for _, r in gas[(gas["dieta"] == d) & (gas["dia_larvario"] >= 10)].iterrows()]

    t_eval = np.array(sorted(set([0.0, 5.0, 8.0, 10.0, 12.0, 14.0] +
                                 [dia for d in obs.values() for dia, _, _ in d])))

    lo = np.array([0.05, 0.0])
    hi = np.array([1.0, 0.3])
    resultados = {}

    for dieta in ("D1", "D4"):
        print(f"\n=== {dieta} ===")
        arranques = [(0.3, 0.05), (0.1, 0.15), (0.5, 0.1)]
        mejor = None
        for i, arr in enumerate(arranques):
            print(f"  Arranque {i+1}...", end=" ", flush=True)
            x0 = np.clip(np.array(arr), lo + 1e-9, hi - 1e-9)

            def residuos(theta):
                f_mant, k_cat = theta
                t, ppm = simular(dieta, f_mant, k_cat, t_eval)
                res = []
                for dia, ob, sat in obs[dieta]:
                    m = float(np.interp(dia, t, ppm))
                    r = m - ob
                    if sat: r = max(r, 0.0)
                    res.append(r / 25.0)
                return np.array(res)

            sol = least_squares(residuos, x0, bounds=(lo, hi), method="trf", max_nfev=200)
            print(f"cost={sol.cost:.2f}")
            if mejor is None or sol.cost < mejor.cost: mejor = sol

        f_mant, k_cat = mejor.x
        fr = [n for n, v, l, h in zip(["f_mant", "k_cat"], mejor.x, lo, hi) if abs(v - l) < 1e-3 or abs(v - h) < 1e-3]
        print(f"  f_mant={f_mant:.4f}, k_cat={k_cat:.4f} [{';'.join(fr) if fr else 'interior'}]")

        t, ppm = simular(dieta, f_mant, k_cat, t_eval)
        res = []
        for dia, ob, sat in obs[dieta]:
            m = float(np.interp(dia, t, ppm))
            r = m - ob
            if sat: r = max(r, 0.0)
            res.append(r)
        rmse = np.sqrt(np.mean(np.array(res)**2))
        print(f"  RMSE tardio = {rmse:.2f}")
        resultados[dieta] = (f_mant, k_cat, rmse)

    print(f"\nD1: f_mant={resultados['D1'][0]:.4f}, k_cat={resultados['D1'][1]:.4f}, RMSE={resultados['D1'][2]:.2f}")
    print(f"D4: f_mant={resultados['D4'][0]:.4f}, k_cat={resultados['D4'][1]:.4f}, RMSE={resultados['D4'][2]:.2f}")

    pd.DataFrame([
        dict(dieta="D1", parametro="f_mant", valor=resultados["D1"][0]),
        dict(dieta="D1", parametro="k_cat", valor=resultados["D1"][1]),
        dict(dieta="D4", parametro="f_mant", valor=resultados["D4"][0]),
        dict(dieta="D4", parametro="k_cat", valor=resultados["D4"][1]),
    ]).to_csv("resultados/parametros_etapa3_por_dieta.csv", index=False)
    print("\nOK")


if __name__ == "__main__":
    main()

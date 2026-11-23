#!/usr/bin/env python3
"""Etapa 2 recalibrada: gamma por dieta."""

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

# Fijos etapa 1
K_REF = {"D1": 1.0, "D4": 1.0}
DELTA_CS = {"D1": 0.664521, "D4": 0.537609}

# Montaje
N_LARVAS = 700
PULSOS = [(0.0, 500.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.15; T_C = T_K - 273.15
V_AIRE_L = 12.1; DELTA_C_ALIM = 0.47; F_MS = 0.325
DIA_MONTAJE_SEP = 3
T_FIN = 14.0; DIA_AJUSTE_MAX = 8.0
X0_G_C = 1.0; Y_X = 0.4; K_SM = 50.0; K_G = 0.1; k_d = 0.05

BOUNDS = {
    "a_max": (1.0, 1.5),
    "rho": (0.5, 15.0),
    "f_D4": (0.4, 1.0),
    "gamma_D1": (0.0, 1.0),
    "gamma_D4": (0.0, 1.0),
}

ARRANQUES = [
    (1.2, 3.0, 0.7, 0.17, 0.17),
    (1.0, 5.0, 0.6, 0.1, 0.3),
    (1.4, 1.0, 0.8, 0.2, 0.1),
]


def simular_tratamiento(dieta, a_max, rho, f_D4, gamma, t_eval):
    p_larva = ParametrosEriksen(
        amax=a_max, alpha=1.0, beta=1.0, YB=0.44, YL=0.42, m=0.08,
        Bmax0=39.69, tp_min=13.0, rho=rho, delta_I=5.0,
        delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
         
    )
    f_dieta = f_D4 if dieta == "D4" else 1.0
    p_lecho = ParametrosLecho(
        f_dieta=f_dieta, k_ref=K_REF[dieta], K_SM=K_SM,
        Y_X=Y_X, delta_CS=DELTA_CS[dieta], k_d=k_d,
        gamma=gamma, K_G=K_G,
    )
    XDW0 = 4.0 * F_MS
    D_C0 = PULSOS[0][1] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS[dieta]
    y0 = estado_inicial_desde_dw(XDW0_mg=XDW0, D_C0=D_C0, X0=X0_G_C, p=p_larva)

    def f(t, y):
        return rhs(t, y, p_larva, p_lecho, N_LARVAS, T_C)

    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    y = y0.copy()
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0:
            y[2] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS[dieta]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(f, (a, b), y, method="LSODA", t_eval=t_seg, rtol=1e-5, atol=1e-8)
        if not sol.success:
            raise RuntimeError(f"fallo [{a},{b}]")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1].copy()
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)

    ph_mg = np.zeros_like(t)
    ppm = np.zeros_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        ph_mg[i] = (yi[0] / (p_larva.delta_CB * p_larva.delta_O) +
                    yi[1] / (p_larva.delta_CL / (p_larva.delta_CB * p_larva.delta_O)))
        lar = tasas_larva(ti, yi[0], yi[1], p_larva, factor_alimento=f_dieta)
        mic = tasas_lecho(yi[2], yi[3], T_C, p_lecho)
        R_C = N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic
        ppm[i] = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    return t, ph_mg, ppm


def main():
    crec = pd.read_csv("datos/procesados/crecimiento_canonico.csv")
    crec["dia_larvario"] = crec["dia_larvario"].astype(float)
    crec_et2 = crec[crec["dia_larvario"] <= DIA_AJUSTE_MAX].copy()

    cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
    gas = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") &
                  (cierres["condicion"] == "tratamiento")].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP
    gas_et2 = gas[gas["dia_larvario"] <= DIA_AJUSTE_MAX].copy()

    obs_crec = {}
    obs_gas = {}
    for d in ("D1", "D4"):
        obs_crec[d] = [(float(r["dia_larvario"]), float(r["ph_mg"]))
                       for _, r in crec_et2[crec_et2["dieta"] == d].iterrows()]
        obs_gas[d] = [(float(r["dia_larvario"]), float(r["tasa_ppm_min_tiempo"]), bool(r.get("Saturado", False)))
                      for _, r in gas_et2[gas_et2["dieta"] == d].iterrows()]

    t_eval = np.array(sorted(set([0.0, 5.0, 8.0] +
                                 [dia for d in obs_crec.values() for dia, _ in d] +
                                 [dia for d in obs_gas.values() for dia, _, _ in d])))

    def residuos(theta):
        a_max, rho, f_D4, gamma_D1, gamma_D4 = theta
        res = []
        for d in ("D1", "D4"):
            gamma = gamma_D1 if d == "D1" else gamma_D4
            t, ph, ppm = simular_tratamiento(d, a_max, rho, f_D4, gamma, t_eval)
            for dia, ob in obs_crec[d]:
                m = float(np.interp(dia, t, ph))
                res.append((m - ob) / 50.0)
            for dia, ob, sat in obs_gas[d]:
                m = float(np.interp(dia, t, ppm))
                r = m - ob
                if sat:
                    r = max(r, 0.0)
                res.append(r / 25.0)
        return np.array(res)

    lo = np.array([1.0, 0.5, 0.4, 0.0, 0.0])
    hi = np.array([1.5, 15.0, 1.0, 1.0, 1.0])

    mejor = None
    for i, arr in enumerate(ARRANQUES):
        print(f"Arranque {i+1}...", end=" ", flush=True)
        x0 = np.clip(np.array(arr), lo + 1e-9, hi - 1e-9)
        sol = least_squares(residuos, x0, bounds=(lo, hi), method="trf", max_nfev=400)
        print(f"cost={sol.cost:.2f}, nfev={sol.nfev}")
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol

    a_max, rho, f_D4, gamma_D1, gamma_D4 = mejor.x
    fr = [n for n, v, l, h in zip(["a_max", "rho", "f_D4", "gamma_D1", "gamma_D4"],
                                    mejor.x, lo, hi) if abs(v - l) < 1e-3 or abs(v - h) < 1e-3]
    print(f"\na_max={a_max:.4f}, rho={rho:.4f}, f_D4={f_D4:.4f}")
    print(f"gamma_D1={gamma_D1:.4f}, gamma_D4={gamma_D4:.4f}")
    print(f"[{';'.join(fr) if fr else 'interior'}]")

    # Resultados completos
    t_fino = np.linspace(0, T_FIN, 71)
    filas = []
    for d in ("D1", "D4"):
        gamma = gamma_D1 if d == "D1" else gamma_D4
        t, ph, ppm = simular_tratamiento(d, a_max, rho, f_D4, gamma, t_fino)
        for _, r in crec[crec["dieta"] == d].iterrows():
            dia = float(r["dia_larvario"])
            m = float(np.interp(dia, t, ph))
            filas.append(dict(dieta=d, tipo="crec", dia=dia, obs=float(r["ph_mg"]),
                              mod=m, residuo=m - float(r["ph_mg"])))
        for _, r in gas[gas["dieta"] == d].iterrows():
            dia = float(r["dia_larvario"])
            ob = float(r["tasa_ppm_min_tiempo"])
            m = float(np.interp(dia, t, ppm))
            filas.append(dict(dieta=d, tipo="gas", dia=dia, obs=ob, mod=m,
                              residuo=m - ob, saturado=bool(r.get("Saturado", False))))

    comp = pd.DataFrame(filas)
    comp.to_csv("resultados/calibracion_etapa2_gamma_dieta.csv", index=False)

    for (dieta, bloque), g in comp.groupby(["dieta", "tipo"]):
        rmse = float(np.sqrt(np.mean(g["residuo"] ** 2)))
        print(f"RMSE {dieta} {bloque}: {rmse:.2f}")

    pd.DataFrame([
        dict(parametro="a_max", valor=a_max),
        dict(parametro="rho", valor=rho),
        dict(parametro="f_D4", valor=f_D4),
        dict(parametro="gamma_D1", valor=gamma_D1),
        dict(parametro="gamma_D4", valor=gamma_D4),
    ]).to_csv("resultados/parametros_etapa2_gamma_dieta.csv", index=False)
    print("\nOK")


if __name__ == "__main__":
    main()

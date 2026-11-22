#!/usr/bin/env python3
"""Etapa 2: nucleo larvario + lecho + pastoreo contra crecimiento y gas."""

import argparse
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
DELTA_CS = {"D1": 0.665, "D4": 0.538}

# Montaje
N_LARVAS = 700
PULSOS = [(0.0, 500.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.15
T_C = T_K - 273.15
V_AIRE_L = 12.1
DELTA_C_ALIM = 0.47
F_MS = 0.325
DIA_MONTAJE_SEP = 3
T_FIN = 14.0

# Fijos estructurales
X0_G_C = 1.0
Y_X = 0.4
K_SM = 50.0
K_G = 0.1

# Cotas
BOUNDS = {
    "a_max": (1.0, 1.5),
    "rho": (0.5, 15.0),
    "f_D4": (0.4, 1.0),
    "gamma": (0.0, 1.0),
}
# Solo 2 arranques para velocidad
ARRANQUES = [(1.1, 2.0, 0.8, 0.1), (1.3, 5.0, 0.6, 0.3)]


def pulso_carbono(dieta, g_fresco, dcs):
    return g_fresco * MS_DIETA[dieta] * DELTA_C_ALIM * dcs


def simular_tratamiento(dieta, a_max, rho, f_D4, gamma, t_eval):
    p_larva = ParametrosEriksen(
        amax=a_max, alpha=1.0, beta=1.0, YB=0.44, YL=0.15, m=0.15,
        Bmax0=39.69, tp_min=13.0, rho=rho, delta_I=1.25,
        delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
        f_mant_prepupa=0.3, k_cat_prepupa=0.02,
    )
    f_dieta = f_D4 if dieta == "D4" else 1.0
    p_lecho = ParametrosLecho(
        f_dieta=f_dieta, k_ref=K_REF[dieta], K_SM=K_SM,
        Y_X=Y_X, delta_CS=DELTA_CS[dieta], k_d=0.05,
        gamma=gamma, K_G=K_G,
    )
    # PH inicial = 4 mg
    XDW0_mg = 4.0 * F_MS
    D_C0 = pulso_carbono(dieta, PULSOS[0][1], DELTA_CS[dieta])
    y0 = estado_inicial_desde_dw(XDW0_mg=XDW0_mg, D_C0=D_C0, X0=X0_G_C, p=p_larva)

    def rhs_full(t, y):
        return rhs(t, y, p_larva, p_lecho, N_LARVAS, T_C)

    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    y = y0.copy()
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0:
            y[2] += pulso_carbono(dieta, pulsos[a], DELTA_CS[dieta])
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs_full, (a, b), y, method="LSODA", t_eval=t_seg,
            rtol=1e-6, atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"fallo en [{a},{b}]: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1].copy()
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)

    # CO2 (ppm/min)
    ppm = np.zeros_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        larva = tasas_larva(ti, yi[0], yi[1], p_larva, factor_alimento=f_dieta)
        micro = tasas_lecho(yi[2], yi[3], T_C, p_lecho)
        R_C_total = N_LARVAS * larva.rCO2_C / 1000.0 + micro.rCmic
        ppm[i] = carbono_a_ppm_min(R_C_total, V_AIRE_L, T_K)

    # PH (mg/larva)
    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = ((1.0 - p_larva.delta_LB) * B / (p_larva.delta_CB * p_larva.delta_O) + Ltot / p_larva.delta_CL) / F_MS

    return t, ph, ppm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cierres", default="datos/procesados/cierres_etiquetados.csv")
    parser.add_argument("--crecimiento", default="datos/procesados/crecimiento_canonico.csv")
    parser.add_argument("--salida", default="resultados/calibracion_etapa2.csv")
    args = parser.parse_args()

    cierres = pd.read_csv(args.cierres)
    crec = pd.read_csv(args.crecimiento)

    # Gas tratamientos
    gas = cierres[
        (~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") &
        (cierres["condicion"] == "tratamiento")
    ].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP

    # Observaciones por dieta
    obs_gas = {}
    obs_crec = {}
    for dieta in ("D1", "D4"):
        g = gas[gas["dieta"] == dieta]
        obs_gas[dieta] = [(float(r["dia_larvario"]), float(r["tasa_ppm_min_tiempo"]), bool(r.get("Saturado", False))) for _, r in g.iterrows()]
        c = crec[crec["dieta"] == dieta]
        obs_crec[dieta] = [(float(r["dia_larvario"]), float(r["ph_mg"])) for _, r in c.iterrows()]

    # Evaluacion rapida: 40 puntos (no 400)
    t_eval = np.unique(np.concatenate([
        np.linspace(0, T_FIN, 41),
        [d for d, _, _ in obs_gas["D1"]],
        [d for d, _, _ in obs_gas["D4"]],
        [d for d, _ in obs_crec["D1"]],
        [d for d, _ in obs_crec["D4"]],
    ]))

    def residuos(theta):
        a_max, rho, f_D4, gamma = theta
        res = []
        for dieta in ("D1", "D4"):
            t, ph, ppm = simular_tratamiento(dieta, a_max, rho, f_D4, gamma, t_eval)
            # Gas
            for dia, obs, sat in obs_gas[dieta]:
                mod = float(np.interp(dia, t, ppm))
                r = mod - obs
                if sat:
                    r = max(r, 0.0)  # censurado: solo penaliza subprediccion
                res.append(r / 25.0)  # peso 25 ppm/min
            # Crecimiento
            for dia, obs in obs_crec[dieta]:
                mod = float(np.interp(dia, t, ph))
                res.append((mod - obs) / 50.0)  # peso ~50 mg
        return np.array(res)

    lo = np.array([BOUNDS["a_max"][0], BOUNDS["rho"][0], BOUNDS["f_D4"][0], BOUNDS["gamma"][0]])
    hi = np.array([BOUNDS["a_max"][1], BOUNDS["rho"][1], BOUNDS["f_D4"][1], BOUNDS["gamma"][1]])

    mejor = None
    for i, arranque in enumerate(ARRANQUES):
        print(f"Arranque {i+1}/{len(ARRANQUES)}...", end=" ", flush=True)
        x0 = np.clip(np.array(arranque, dtype=float), lo + 1e-9, hi - 1e-9)
        sol = least_squares(residuos, x0, bounds=(lo, hi), method="trf", max_nfev=500)
        print(f"cost={sol.cost:.2f}, nfev={sol.nfev}")
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol

    a_max, rho, f_D4, gamma = mejor.x
    frontera = []
    for nombre, val, l, h in zip(
        ["a_max", "rho", "f_D4", "gamma"], mejor.x, lo, hi
    ):
        if abs(val - l) < 1e-3 or abs(val - h) < 1e-3:
            frontera.append(nombre)

    marca = f" [frontera: {','.join(frontera)}]" if frontera else ""
    print(f"\n{a_max=:.4f}, {rho=:.3f}, {f_D4=:.4f}, {gamma=:.4f}{marca}")

    # Resultados detallados
    t_fino = np.linspace(0, T_FIN, 141)
    filas = []
    for dieta in ("D1", "D4"):
        t, ph, ppm = simular_tratamiento(dieta, a_max, rho, f_D4, gamma, t_fino)
        for dia, obs in obs_crec[dieta]:
            mod = float(np.interp(dia, t, ph))
            filas.append(dict(dieta=dieta, bloque="crec", dia=dia, obs=obs, mod=mod, residuo=mod-obs))
        for dia, obs, sat in obs_gas[dieta]:
            mod = float(np.interp(dia, t, ppm))
            filas.append(dict(dieta=dieta, bloque="gas", dia=dia, obs=obs, mod=mod, residuo=mod-obs, saturado=sat))

    comp = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(destino, index=False)

    print(f"\nArchivo: {destino}")
    for (bloque, dieta), g in comp.groupby(["bloque", "dieta"]):
        rmse = float(np.sqrt(np.mean(g["residuo"] ** 2)))
        print(f"RMSE {bloque} {dieta}: {rmse:.2f}")

    # Guardar parametros
    pdest = destino.with_name("parametros_etapa2.csv")
    pd.DataFrame([
        dict(parametro="a_max", valor=a_max),
        dict(parametro="rho", valor=rho),
        dict(parametro="f_D4", valor=f_D4),
        dict(parametro="gamma", valor=gamma),
    ]).to_csv(pdest, index=False)
    print(f"\nParametros: {pdest}")


if __name__ == "__main__":
    main()

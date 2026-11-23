#!/usr/bin/env python3
"""Genera tabla de cierres para E0, S, C con parametros actuales."""

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

N_TRAT = 700
PULSOS = [(0.0, 500.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.15
T_C = T_K - 273.15
V_AIRE_L = 12.1
DELTA_C_ALIM = 0.47
F_MS = 0.325
DIA_MONTAJE_SEP = 3
T_FIN = 14.0
X0 = 1.0
Y_X = 0.4
K_SM = 50.0
K_G = 0.1
k_d = 0.05

K_REF = {"D1": 1.0, "D4": 1.0}
DELTA_CS = {"D1": 0.664521, "D4": 0.537609}
F_D4 = 0.6977750069098937
GAMMA_C = 0.17033807072065274

PARAMS_ERIKSEN_ORIG = dict(amax=1.0, alpha=1.0, beta=1.0, YB=0.44, YL=0.42, m=0.08,
                           Bmax0=39.69, tp_min=13.0, rho=1.0, delta_I=5.0,
                           delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9)
PARAMS_ERIKSEN_CAL = dict(amax=1.179810618529265, alpha=1.0, beta=1.0, YB=0.44, YL=0.42, m=0.08,
                           Bmax0=39.69, tp_min=13.0, rho=3.0, delta_I=5.0,
                           delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9)


def simular_tratamiento(config, dieta, t_eval):
    if config == "E0":
        p_larva = ParametrosEriksen(**PARAMS_ERIKSEN_ORIG)
        f_d = 1.0 if dieta == "D1" else F_D4
        p_lecho = ParametrosLecho(f_dieta=f_d, k_ref=0.0, K_SM=K_SM, Y_X=Y_X,
                                   delta_CS=0.5, k_d=k_d, gamma=0.0, K_G=K_G)
    else:
        p_larva = ParametrosEriksen(**PARAMS_ERIKSEN_CAL)
        f_d = 1.0 if dieta == "D1" else F_D4
        gamma = 0.0 if config == "S" else GAMMA_C
        p_lecho = ParametrosLecho(f_dieta=f_d, k_ref=K_REF[dieta], K_SM=K_SM, Y_X=Y_X,
                                   delta_CS=DELTA_CS[dieta], k_d=k_d, gamma=gamma, K_G=K_G)
    
    XDW0 = 4.0 * F_MS
    D_C0 = PULSOS[0][1] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS[dieta]
    y0 = estado_inicial_desde_dw(XDW0_mg=XDW0, D_C0=D_C0, X0=X0, p=p_larva)
    
    def f(t, y):
        return rhs(t, y, p_larva, p_lecho, N_TRAT, T_C)
    
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
    
    ppm = np.zeros_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        lar = tasas_larva(ti, yi[0], yi[1], p_larva, factor_alimento=p_lecho.f_dieta)
        mic = tasas_lecho(yi[2], yi[3], T_C, p_lecho)
        R_C = N_TRAT * lar.rCO2_C / 1000.0 + mic.rCmic
        ppm[i] = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    return t, ppm


def simular_control(config, dieta, t_eval):
    if config == "E0":
        return t_eval, np.zeros_like(t_eval)
    
    f_d = 1.0 if dieta == "D1" else F_D4
    # Controles sin larvas: gamma=0 siempre
    p_lecho = ParametrosLecho(f_dieta=f_d, k_ref=K_REF[dieta], K_SM=K_SM, Y_X=Y_X,
                               delta_CS=DELTA_CS[dieta], k_d=k_d, gamma=0.0, K_G=K_G)
    
    def rhs_lecho(t, y):
        D_C, X = [max(float(v), 0.0) for v in y]
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        return [micro.dD_C, micro.dX]
    
    # D_C inicial = 0, como en compare_E0_vs_C.py
    y = np.array([0.0, X0])
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0:
            y[0] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS[dieta]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(rhs_lecho, (a, b), y, method="LSODA", t_eval=t_seg, rtol=1e-5, atol=1e-8)
        if not sol.success:
            raise RuntimeError(f"fallo control [{a},{b}]")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1].copy()
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    
    ppm = np.zeros_like(t)
    for i, (D_C, X) in enumerate(zip(Y[0], Y[1])):
        mic = tasas_lecho(D_C, X, T_C, p_lecho)
        R_C = mic.rCmic
        ppm[i] = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    return t, ppm


def main():
    cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
    gas = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2")].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP
    
    # Controles: asignar dieta
    control_mask = gas["condicion"] == "control"
    gas.loc[control_mask, "dieta"] = "D1"  # por defecto, luego corregimos
    
    configs = ["E0", "S", "C"]
    t_eval = np.array(sorted(set([0.0, 5.0, 8.0, 10.0, 12.0, 14.0] +
                                  gas["dia_larvario"].tolist())))
    
    filas = []
    for _, r in gas.iterrows():
        dieta = r["dieta"]
        tipo = r["condicion"]
        dia = float(r["dia_larvario"])
        obs = float(r["tasa_ppm_min_tiempo"])
        sat = bool(r.get("Saturado", False))
        
        if tipo == "control":
            if dieta not in ("D1", "D4"):
                continue
            fila = dict(dieta=dieta, tipo=tipo, dia=dia, obs=obs, saturado=sat)
            for cfg in configs:
                t, ppm = simular_control(cfg, dieta, t_eval)
                pred = float(np.interp(dia, t, ppm))
                fila[f"{cfg}_pred"] = pred
                err = (pred - obs) / obs * 100 if obs != 0 else 0
                fila[f"{cfg}_err"] = err
            filas.append(fila)
        else:
            if dieta not in ("D1", "D4"):
                continue
            fila = dict(dieta=dieta, tipo=tipo, dia=dia, obs=obs, saturado=sat)
            for cfg in configs:
                t, ppm = simular_tratamiento(cfg, dieta, t_eval)
                pred = float(np.interp(dia, t, ppm))
                fila[f"{cfg}_pred"] = pred
                err = (pred - obs) / obs * 100 if obs != 0 else 0
                fila[f"{cfg}_err"] = err
            filas.append(fila)
    
    df = pd.DataFrame(filas)
    df.to_csv("resultados/tabla_cierres.csv", index=False)
    
    print("\n=== CONTROLES ===")
    ctrl = df[df["tipo"] == "control"]
    print(f"{'Dieta':<5} {'Dia':<4} {'Obs':<10} {'Sat':<4} {'E0':<10} {'S':<10} {'C':<10}")
    for _, r in ctrl.iterrows():
        sat_mark = "sí" if r["saturado"] else "no"
        print(f"{r['dieta']:<5} {int(r['dia']):<4} {r['obs']:<10.2f} {sat_mark:<4} {r['E0_pred']:<10.2f} {r['S_pred']:<10.2f} {r['C_pred']:<10.2f}")
    
    print("\n=== TRATAMIENTOS ===")
    trat = df[df["tipo"] == "tratamiento"]
    print(f"{'Dieta':<5} {'Dia':<4} {'Obs':<10} {'Sat':<4} {'E0':<10} {'errE0':<8} {'S':<10} {'errS':<8} {'C':<10} {'errC':<8}")
    for _, r in trat.iterrows():
        sat_mark = "sí" if r["saturado"] else "no"
        print(f"{r['dieta']:<5} {int(r['dia']):<4} {r['obs']:<10.2f} {sat_mark:<4} {r['E0_pred']:<10.2f} {r['E0_err']:<8.1f} {r['S_pred']:<10.2f} {r['S_err']:<8.1f} {r['C_pred']:<10.2f} {r['C_err']:<8.1f}")
    
    # RMSE por tipo
    for tipo, g_tipo in df.groupby("tipo"):
        print(f"\n--- {tipo.upper()} ---")
        for cfg in configs:
            res_cens = []
            for _, r in g_tipo.iterrows():
                resid = r[f"{cfg}_pred"] - r["obs"]
                if r["saturado"]:
                    resid = max(resid, 0.0)
                res_cens.append(resid)
            rmse_cens = np.sqrt(np.mean(np.array(res_cens)**2))
            print(f"  {cfg}: RMSE censurado = {rmse_cens:.2f}")
    
    print("\nOK")


if __name__ == "__main__":
    main()

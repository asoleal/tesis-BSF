#!/usr/bin/env python3
"""
validar.py — Simula E0 (Eriksen puro) y C (completo) para D1/D4,
compara contra datos_finales_PINN_corregidos.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from parametros import PARAMETROS
from modelo_eriksen import rhs_eriksen, estado_inicial_desde_dw, B_Ltot_a_PH, tasas_larva
from modelo_completo import rhs_completo, tasas_lecho, carbono_a_ppm_min

BASE = Path(__file__).resolve().parents[1]

# Mapeo de etiquetas PINN a dieta/condicion
MAPA_ETIQUETAS = {
    "D1_Tratamiento": ("D1", "tratamiento"),
    "Control_Alimento_D1": ("D1", "control"),
    "D4_Tratamiento": ("D4", "tratamiento"),
    "Control_Alimento_D4": ("D4", "control"),
    "Larvas_Generacion_Base": ("D1", "tratamiento"),
    "Larvas_Ayuno": (None, "ayuno"),
}


def cargar_datos():
    pinn = pd.read_csv(BASE / "datos" / "procesados" / "datos_finales_PINN_corregidos.csv")
    # Filtrar solo CO2
    pinn = pinn[pinn["Gas"] == "co2"].copy()
    # Mapear
    pinn["dieta"] = pinn["Etiqueta"].map(lambda e: MAPA_ETIQUETAS.get(e, (None, None))[0])
    pinn["condicion"] = pinn["Etiqueta"].map(lambda e: MAPA_ETIQUETAS.get(e, (None, None))[1])
    pinn["dia_larvario"] = pinn["Dia_Experimento"] - PARAMETROS["montaje"]["DIA_MONTAJE_SEP"]
    
    gas = pinn[pinn["condicion"] == "tratamiento"].copy()
    control = pinn[pinn["condicion"] == "control"].copy()
    crec = pd.read_csv(BASE / "datos" / "procesados" / "crecimiento_canonico.csv")
    return gas, control, crec


def simular_E0(dieta: str, t_eval: np.ndarray):
    p_larva = PARAMETROS["eriksen"]
    p_montaje = PARAMETROS["montaje"]
    cal = PARAMETROS["calibrados"][dieta]
    f_dieta = cal["f_dieta"]
    N = p_montaje["N_larvas"]
    V = p_montaje["V_aire_L"]
    T_K = p_montaje["T_K"]

    crec = pd.read_csv(BASE / "datos" / "procesados" / "crecimiento_canonico.csv")
    ph0 = crec[(crec["dieta"] == dieta) & (crec["dia_larvario"] == 0)]["ph_mg"].iloc[0]
    y0 = estado_inicial_desde_dw(ph0 * p_montaje["f_MS"], p_larva)

    sol = solve_ivp(
        lambda t, y: rhs_eriksen(t, y, p_larva, f_dieta),
        (0, p_montaje["T_FIN"]), y0, t_eval=t_eval, method="LSODA",
        rtol=1e-7, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"E0 fallo: {sol.message}")

    t = sol.t
    B, Ltot = sol.y[0], sol.y[1]
    ph = np.array([B_Ltot_a_PH(Bi, Li, p_larva, p_montaje["f_MS"]) for Bi, Li in zip(B, Ltot)])
    ppm = np.array([carbono_a_ppm_min(N * tasas_larva(ti, B[i], Ltot[i], p_larva, f_dieta)["rCO2_C"] / 1000.0, V, T_K) for i, ti in enumerate(t)])
    return t, ph, ppm


def simular_C(dieta: str, t_eval: np.ndarray):
    p_larva = PARAMETROS["eriksen"]
    p_montaje = PARAMETROS["montaje"]
    p_lecho_base = PARAMETROS["lecho"].copy()
    cal = PARAMETROS["calibrados"][dieta]
    p_lecho_base["k_ref"] = cal["k_ref"]
    p_lecho_base["delta_CS"] = cal["delta_CS"]
    p_lecho_base["f_dieta"] = cal["f_dieta"]
    p_lecho_base["gamma"] = cal["gamma"]

    N = p_montaje["N_larvas"]
    V = p_montaje["V_aire_L"]
    T_K = p_montaje["T_K"]
    T_C = p_montaje["T_C"]
    T_FIN = p_montaje["T_FIN"]
    pulsos = p_montaje["pulsos"]
    ms = p_montaje["ms_dieta"][dieta]
    delta_C = p_montaje["delta_C_alim"]

    crec = pd.read_csv(BASE / "datos" / "procesados" / "crecimiento_canonico.csv")
    ph0 = crec[(crec["dieta"] == dieta) & (crec["dia_larvario"] == 0)]["ph_mg"].iloc[0]
    y0_eri = estado_inicial_desde_dw(ph0 * p_montaje["f_MS"], p_larva)
    y0 = np.array([y0_eri[0], y0_eri[1], 0.0, 1.0, 0.0, 0.0])

    cortes = sorted(set([0.0] + [d for d, _ in pulsos] + [T_FIN]))
    pulsos_dict = dict(pulsos)
    ts, ys = [], []

    for a, b in zip(cortes[:-1], cortes[1:]):
        if a in pulsos_dict:
            y0[2] += pulsos_dict[a] * ms * delta_C * cal["delta_CS"]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            lambda t, y: rhs_completo(t, y, p_larva, p_lecho_base, p_montaje, dieta, N),
            (a, b), y0, t_eval=t_seg, method="LSODA",
            rtol=1e-7, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"C fallo en [{a},{b}]: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        y0 = sol.y[:, -1]

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    B, Ltot = Y[0], Y[1]
    ph = np.array([B_Ltot_a_PH(Bi, Li, p_larva, p_montaje["f_MS"]) for Bi, Li in zip(B, Ltot)])
    ppm = np.array([carbono_a_ppm_min(N * tasas_larva(ti, B[i], Ltot[i], p_larva, cal["f_dieta"])["rCO2_C"] / 1000.0 + tasas_lecho(Y[2, i], Y[3, i], T_C, p_lecho_base)["rCmic"], V, T_K) for i, ti in enumerate(t)])
    return t, ph, ppm


def rmse_gas(dieta: str, t_mod, ppm_mod, gas_df):
    g = gas_df[gas_df["dieta"] == dieta]
    if len(g) == 0:
        return np.nan
    pred = np.interp(g["dia_larvario"].values, t_mod, ppm_mod)
    obs = g["Tasa_Produccion"].values
    return np.sqrt(np.mean((pred - obs) ** 2))


def rmse_crec(dieta: str, t_mod, ph_mod, crec_df):
    c = crec_df[(crec_df["dieta"] == dieta)]
    if len(c) == 0:
        return np.nan
    pred = np.interp(c["dia_larvario"].values, t_mod, ph_mod)
    obs = c["ph_mg"].values
    w = np.maximum(c["de_mg"].values, 0.1 * obs)
    return np.sqrt(np.mean(((pred - obs) / w) ** 2))


def main():
    gas, control, crec = cargar_datos()
    t_eval = np.linspace(0, PARAMETROS["montaje"]["T_FIN"], 300)

    print("=" * 60)
    print("COMPARACION E0 (Eriksen puro) vs C (completo)")
    print("Datos: datos_finales_PINN_corregidos.csv (CO2)")
    print("=" * 60)

    # Verificar controles
    for dieta in ["D1", "D4"]:
        c = control[control["dieta"] == dieta]
        print(f"\n--- Controles {dieta} ---")
        print(c[["Dia_Experimento", "Etiqueta", "Tasa_Produccion"]].to_string(index=False))

    resultados = []
    for dieta in ["D1", "D4"]:
        t0, ph0, ppm0 = simular_E0(dieta, t_eval)
        tC, phC, ppmC = simular_C(dieta, t_eval)

        rg0 = rmse_gas(dieta, t0, ppm0, gas)
        rgC = rmse_gas(dieta, tC, ppmC, gas)
        rc0 = rmse_crec(dieta, t0, ph0, crec)
        rcC = rmse_crec(dieta, tC, phC, crec)

        resultados.append({
            "dieta": dieta,
            "E0_gas": rg0, "C_gas": rgC,
            "E0_crec": rc0, "C_crec": rcC,
        })

        print(f"\n--- {dieta} ---")
        print(f"  E0  RMSE gas={rg0:.1f}  crec={rc0:.2f}")
        print(f"  C   RMSE gas={rgC:.1f}  crec={rcC:.2f}")
        if rgC < rg0:
            print(f"  Gas mejora: {(rg0 - rgC) / rg0 * 100:.1f}%")
        else:
            print(f"  Gas empeora: {(rgC - rg0) / rg0 * 100:.1f}%")

        # Detalle por dia
        g = gas[gas["dieta"] == dieta]
        print(f"\n  Dia   Obs      E0       C")
        for _, r in g.iterrows():
            d = r["dia_larvario"]
            obs = r["Tasa_Produccion"]
            p0 = np.interp(d, t0, ppm0)
            pC = np.interp(d, tC, ppmC)
            print(f"  {d:5.1f} {obs:8.1f} {p0:8.1f} {pC:8.1f}")

    print("\n" + "=" * 60)
    print("TABLA RESUMEN")
    print("=" * 60)
    print(f"{'Dieta':<6} {'E0_gas':<10} {'C_gas':<10} {'E0_crec':<10} {'C_crec':<10}")
    for r in resultados:
        print(f"{r['dieta']:<6} {r['E0_gas']:<10.1f} {r['C_gas']:<10.1f} {r['E0_crec']:<10.2f} {r['C_crec']:<10.2f}")


if __name__ == "__main__":
    main()

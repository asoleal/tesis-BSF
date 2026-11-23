#!/usr/bin/env python3
"""Simula el nucleo de Eriksen puro (sin lecho, sin pastoreo)."""

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

# Parametros calibrados etapa 2
A_MAX = 1.1798
RHO = 3.0
F_D4 = 0.6978
GAMMA = 0.0  # nucleo puro: sin pastoreo

# Montaje
N_LARVAS = 700
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.15; T_C = T_K - 273.15
V_AIRE_L = 12.1; DELTA_C_ALIM = 0.47; F_MS = 0.325
DIA_MONTAJE_SEP = 3; T_FIN = 14.0

# Lecho APAGADO: k_ref=0, gamma=0, X0=0, D_C0 enorme (alimento ilimitado)
X0_G_C = 0.0  # sin microbios
K_REF = {"D1": 0.0, "D4": 0.0}  # sin degradacion
DELTA_CS = {"D1": 1.0, "D4": 1.0}  # irrelevante
Y_X = 0.0; K_SM = 1.0; K_G = 0.1


def simular_eriksen(dieta, t_eval):
    """Nucleo Eriksen: lecho apagado, alimento ilimitado."""
    p_larva = ParametrosEriksen(
        amax=A_MAX, alpha=1.0, beta=1.0, YB=0.44, YL=0.15, m=0.15,
        Bmax0=39.69, tp_min=13.0, rho=RHO, delta_I=1.25,
        delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
         
    )
    f_dieta = F_D4 if dieta == "D4" else 1.0
    p_lecho = ParametrosLecho(
        f_dieta=f_dieta, k_ref=K_REF[dieta], K_SM=K_SM,
        Y_X=Y_X, delta_CS=DELTA_CS[dieta], k_d=0.05,
        gamma=GAMMA, K_G=K_G,
    )
    # PH inicial = 4 mg
    XDW0 = 4.0 * F_MS
    D_C0 = 1e6  # alimento "infinito"
    y0 = estado_inicial_desde_dw(XDW0_mg=XDW0, D_C0=D_C0, X0=X0_G_C, p=p_larva)

    def f(t, y): return rhs(t, y, p_larva, p_lecho, N_LARVAS, T_C)

    sol = solve_ivp(f, (0, T_FIN), y0, method="LSODA", t_eval=t_eval,
                    rtol=1e-5, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t; Y = sol.y

    # CO2 solo larvario (lecho apagado)
    ppm = np.zeros_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        lar = tasas_larva(ti, yi[0], yi[1], p_larva, factor_alimento=f_dieta)
        ppm[i] = carbono_a_ppm_min(N_LARVAS * lar.rCO2_C / 1000.0, V_AIRE_L, T_K)

    # PH
    B = np.maximum(Y[0], 0.0); Ltot = np.maximum(Y[1], 0.0)
    ph = ((1.0 - p_larva.delta_LB) * B / (p_larva.delta_CB * p_larva.delta_O)
          + Ltot / p_larva.delta_CL) / F_MS

    return t, ph, ppm


def main():
    crec = pd.read_csv("datos/procesados/crecimiento_canonico.csv")
    cierres = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
    gas = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2") &
                  (cierres["condicion"] == "tratamiento")].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP

    t_eval = np.linspace(0, T_FIN, 141)

    filas = []
    for dieta in ("D1", "D4"):
        t, ph, ppm = simular_eriksen(dieta, t_eval)

        # Crecimiento
        for _, r in crec[crec["dieta"] == dieta].iterrows():
            dia = float(r["dia_larvario"])
            mod = float(np.interp(dia, t, ph))
            filas.append(dict(dieta=dieta, bloque="crec", dia=dia,
                              obs=float(r["ph_mg"]), mod=mod, residuo=mod-float(r["ph_mg"])))

        # Gas
        for _, r in gas[gas["dieta"] == dieta].iterrows():
            dia = float(r["dia_larvario"])
            mod = float(np.interp(dia, t, ppm))
            filas.append(dict(dieta=dieta, bloque="gas", dia=dia,
                              obs=float(r["tasa_ppm_min_tiempo"]), mod=mod,
                              residuo=mod-float(r["tasa_ppm_min_tiempo"]),
                              saturado=bool(r.get("Saturado", False))))

    comp = pd.DataFrame(filas)
    comp.to_csv("resultados/simulacion_eriksen.csv", index=False)

    print("=== NUCLEO DE ERIKSEN PURO ===")
    for (b, d), g in comp.groupby(["bloque", "dieta"]):
        rmse = float(np.sqrt(np.mean(g["residuo"]**2)))
        print(f"RMSE {b} {d}: {rmse:.2f}")

    print("\nTabla:")
    print(comp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

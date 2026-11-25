#!/usr/bin/env python3
"""Etapa 2 fina: a_max, rho, f_D4, gamma_D1, gamma_D4 contra PINN."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from modelo_eriksen import estado_inicial_desde_dw, tasas_larva
from modelo_completo import rhs_completo, tasas_lecho, carbono_a_ppm_min

# ============================================================
# Parametros fijos
# ============================================================
K_REF = {"D1": 0.3994, "D4": 0.3961}
DELTA_CS = {"D1": 0.8855, "D4": 0.9000}
N_LARVAS = 700
PULSOS = [(0.0, 500.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.15
T_C = 27.0
V_AIRE_L = 12.1
DELTA_C_ALIM = 0.47
F_MS = 0.325
DIA_MONTAJE_SEP = 3
T_FIN = 14.0
K_SM = 50.0
Y_X = 0.4
K_G = 0.1

# Parametros Eriksen fijos
P_LARVA = {
    "amax": 1.2, "alpha": 1.0, "beta": 1.0,
    "YB": 0.44, "YL": 0.42, "m": 0.08,
    "Bmax0": 90.0 * 0.49 * 0.9,
    "tp_min": 8.5, "rho": 3.0,
    "delta_I": 5.0, "delta_LB": 0.1,
    "delta_CB": 0.49, "delta_CL": 0.79, "delta_O": 0.9,
    "f_mant_prepupa": 1.0, "k_cat_prepupa": 0.0,
}

# Parametros lecho fijos
P_LECHO_BASE = {
    "K_D": 50.0, "K_SM": 50.0, "Y_X": 0.4, "Y_Cmic": 0.4,
    "k_d": 0.05, "K_G": 0.1, "Q10": 2.0,
    "T_ref_C": 25.0, "theta": 0.7, "K_theta": 0.15,
}


def simular(dieta, a_max, rho, f_D4, gamma_D1, gamma_D4, t_eval):
    p_larva = dict(P_LARVA, amax=a_max, rho=rho)
    f_dieta = f_D4 if dieta == "D4" else 1.0
    gamma = gamma_D4 if dieta == "D4" else gamma_D1
    
    p_lecho = dict(P_LECHO_BASE,
                   f_dieta=f_dieta, k_ref=K_REF[dieta],
                   delta_CS=DELTA_CS[dieta], gamma=gamma)
    
    # Estado inicial desde peso del experimento
    crec = pd.read_csv(Path(__file__).resolve().parents[1] / "datos" / "procesados" / "crecimiento_canonico.csv")
    ph0 = crec[(crec["dieta"] == dieta) & (crec["dia_larvario"] == 0)]["ph_mg"].iloc[0]
    y0_eri = estado_inicial_desde_dw(ph0 * F_MS, p_larva)
    y0 = np.array([y0_eri[0], y0_eri[1], 0.0, 1.0, 0.0, 0.0])
    
    # Primer pulso ya aplicado al inicializar
    y0[2] += PULSOS[0][1] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS[dieta]
    
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0 and a in pulsos:
            y0[2] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS[dieta]
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            lambda t, y: rhs_completo(t, y, p_larva, p_lecho, {"T_C": T_C}, dieta, N_LARVAS),
            (a, b), y0, t_eval=t_seg, method="LSODA", rtol=1e-7, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"fallo [{a},{b}]")
        ts.append(sol.t)
        ys.append(sol.y)
        y0 = sol.y[:, -1].copy()
    
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    
    # CO2 total
    ppm = np.empty_like(t)
    for i, (ti, yi) in enumerate(zip(t, Y.T)):
        lar = tasas_larva(ti, yi[0], yi[1], p_larva, f_dieta)
        mic = tasas_lecho(yi[2], yi[3], T_C, p_lecho)
        R_C = N_LARVAS * lar["rCO2_C"] / 1000.0 + mic["rCmic"]
        ppm[i] = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    
    # Crecimiento
    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = np.array([
        ((1.0 - p_larva["delta_LB"]) * Bi / (p_larva["delta_CB"] * p_larva["delta_O"]) + Li / p_larva["delta_CL"]) / F_MS
        for Bi, Li in zip(B, Ltot)
    ])
    
    return t, ph, ppm


def main():
    # Cargar datos PINN
    BASE = Path(__file__).resolve().parents[1]
    pinn = pd.read_csv(BASE / "datos" / "procesados" / "datos_finales_PINN_corregidos.csv")
    pinn = pinn[pinn["Gas"] == "co2"].copy()
    pinn["dia_larvario"] = pinn["Dia_Experimento"] - DIA_MONTAJE_SEP
    
    # Filtrar tratamientos
    trat = pinn[pinn["Etiqueta"].str.contains("Tratamiento|Generacion_Base")].copy()
    trat["dieta"] = trat["Etiqueta"].str.extract(r"(D[14])")[0]
    
    # Crecimiento
    crec = pd.read_csv(BASE / "datos" / "procesados" / "crecimiento_canonico.csv")
    
    # Observaciones
    obs_gas = {}
    obs_crec = {}
    for d in ["D1", "D4"]:
        sub = trat[trat["dieta"] == d]
        obs_gas[d] = [(float(r["dia_larvario"]), float(r["Tasa_Produccion"]), bool(r["Saturado"]))
                      for _, r in sub.iterrows()]
        obs_crec[d] = [(float(r["dia_larvario"]), float(r["ph_mg"]))
                       for _, r in crec[crec["dieta"] == d].iterrows()]
    
    t_eval = np.array(sorted(set(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        + [d for d, _, _ in obs_gas["D1"]] + [d for d, _ in obs_crec["D1"]]
        + [d for d, _, _ in obs_gas["D4"]] + [d for d, _ in obs_crec["D4"]]
    )))
    
    def residuos(theta):
        a_max, rho, f_D4, gamma_D1, gamma_D4 = theta
        res = []
        for d in ["D1", "D4"]:
            t, ph, ppm = simular(d, a_max, rho, f_D4, gamma_D1, gamma_D4, t_eval)
            # Gas
            for dia, ob, sat in obs_gas[d]:
                m = float(np.interp(dia, t, ppm))
                r = m - ob
                if sat and r > 0:
                    r = 0.0
                res.append(r / 25.0)
            # Crecimiento
            for dia, ob in obs_crec[d]:
                m = float(np.interp(dia, t, ph))
                res.append((m - ob) / 15.0)
        return np.array(res)
    
    lo = np.array([1.0, 0.5, 0.4, 0.0, 0.0])
    hi = np.array([1.5, 15.0, 1.0, 1.0, 1.0])
    
    # Arranques
    arranques = [
        (1.2, 3.0, 0.7, 0.10, 0.40),
        (1.1, 2.0, 0.6, 0.05, 0.30),
        (1.3, 4.0, 0.8, 0.15, 0.50),
    ]
    
    mejor = None
    for i, arr in enumerate(arranques):
        print(f"Arranque {i+1}...", end=" ", flush=True)
        x0 = np.clip(np.array(arr), lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(residuos, x0, bounds=(lo, hi), method="trf", max_nfev=500)
            print(f"cost={sol.cost:.2f}, nfev={sol.nfev}")
            if mejor is None or sol.cost < mejor.cost:
                mejor = sol
        except Exception as e:
            print(f"ERROR: {e}")
    
    a_max, rho, f_D4, gamma_D1, gamma_D4 = mejor.x
    fr = [n for n, v, l, h in zip(["a_max", "rho", "f_D4", "gamma_D1", "gamma_D4"], mejor.x, lo, hi)
          if abs(v - l) < 1e-3 or abs(v - h) < 1e-3]
    print(f"\na_max={a_max:.4f}, rho={rho:.3f}, f_D4={f_D4:.4f}")
    print(f"gamma_D1={gamma_D1:.4f}, gamma_D4={gamma_D4:.4f} [{';'.join(fr) if fr else 'interior'}]")
    
    # Resultados detallados
    t_fino = np.linspace(0, T_FIN, 200)
    print("\n" + "=" * 60)
    for d in ["D1", "D4"]:
        t, ph, ppm = simular(d, a_max, rho, f_D4, gamma_D1, gamma_D4, t_fino)
        print(f"\n--- {d} ---")
        print("Gas (dia, obs, mod, residuo):")
        for dia, ob, sat in obs_gas[d]:
            m = float(np.interp(dia, t, ppm))
            print(f"  {dia:5.1f} {ob:8.1f} {m:8.1f} {m-ob:8.1f} {'SAT' if sat else ''}")
        print("Crec (dia, obs, mod, residuo):")
        for dia, ob in obs_crec[d]:
            m = float(np.interp(dia, t, ph))
            print(f"  {dia:5.1f} {ob:8.1f} {m:8.1f} {m-ob:8.1f}")
        
        # RMSE
        rmse_g = np.sqrt(np.mean([(float(np.interp(d, t, ppm)) - ob)**2 for d, ob, _ in obs_gas[d]]))
        rmse_c = np.sqrt(np.mean([(float(np.interp(d, t, ph)) - ob)**2 for d, ob in obs_crec[d]]))
        print(f"RMSE gas={rmse_g:.1f}, crec={rmse_c:.2f}")
    
    # Guardar
    Path("resultados").mkdir(exist_ok=True)
    pd.DataFrame([
        dict(parametro="a_max", valor=a_max),
        dict(parametro="rho", valor=rho),
        dict(parametro="f_D4", valor=f_D4),
        dict(parametro="gamma_D1", valor=gamma_D1),
        dict(parametro="gamma_D4", valor=gamma_D4),
    ]).to_csv("resultados/parametros_etapa2_fina.csv", index=False)
    print("\nOK - Guardado en resultados/parametros_etapa2_fina.csv")


if __name__ == "__main__":
    main()

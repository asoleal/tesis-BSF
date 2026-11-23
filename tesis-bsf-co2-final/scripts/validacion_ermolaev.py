#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "modelo"))

from modelo_bsf import ParametrosEriksen, ParametrosLecho, estado_inicial_desde_dw, tasas_larva, tasas_lecho, rhs

FC_SUSTRATO = 0.45
T_C = 28.0
N_LARVAS = 700
MS_FRAC = 0.249
KG_FW = 291.7 / MS_FRAC / 1000.0
PULSOS = {0: 42.0, 2: 105.0, 7: 84.0, 11: 63.0}
MG_C_A_MG_CO2 = 44.01 / 12.01

P_LARVA = ParametrosEriksen(
    amax=1.180, alpha=1.0, beta=1.0, YB=0.44, YL=0.42, m=0.08,
    Bmax0=90.0 * 0.49 * 0.9, tp_min=13.0, rho=3.0,
    delta_I=5.0, delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
    f_mant_prepupa=0.300, k_cat_prepupa=0.050,
)

P_LECHO_BASE = dict(k_ref=1.0, K_SM=50.0, Y_X=0.4,
                    delta_CS=0.665, k_d=0.05, gamma=0.170, K_G=0.1)
X0_G_C = 1.0
y0_base = estado_inicial_desde_dw(1.5, 0.0, 0.0, P_LARVA)
B0, L0 = y0_base[0], y0_base[1]

def _solve(f, t_span, y0, args, tev, method="LSODA"):
    sol = solve_ivp(f, t_span, y0, args=args, t_eval=tev, method=method,
                    rtol=1e-7, atol=1e-10, dense_output=True)
    if not sol.success:
        sol = solve_ivp(f, t_span, y0, args=args, t_eval=tev, method="BDF",
                        rtol=1e-7, atol=1e-10, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Integracion fallo: {sol.message}")
    sol.y = np.asarray(sol.y)
    if sol.y.ndim == 1:
        sol.y = sol.y.reshape(-1, 1)
    sol.t = np.asarray(sol.t)
    return sol

def simular_E0(f_dieta, t_span):
    def rhs_E0(t, y):
        B, Ltot = max(y[0], 0.0), max(y[1], 0.0)
        larva = tasas_larva(t, B, Ltot, P_LARVA, f_dieta)
        return np.array([larva.dB, larva.dLtot])
    tev = np.linspace(t_span[0], t_span[1], 800)
    sol = _solve(rhs_E0, t_span, np.array([B0, L0]), (), tev)
    return sol.t, sol.y

def flujo_CO2_E0(t, Y, f_dieta):
    fl = np.empty(len(t))
    for i, ti in enumerate(t):
        larva = tasas_larva(ti, Y[0, i], Y[1, i], P_LARVA, f_dieta)
        fl[i] = N_LARVAS * larva.rCO2_C * MG_C_A_MG_CO2 / 1440.0
    return fl

def simular_C(f_dieta, t_span):
    p_lecho = ParametrosLecho(f_dieta=f_dieta, **P_LECHO_BASE)
    eventos = sorted([d for d in PULSOS.keys() if t_span[0] < d <= t_span[1]])
    cortes = [t_span[0]] + eventos + [t_span[1]]
    ts, Ys = [], []
    t0 = t_span[0]
    D_C0 = PULSOS.get(0, 0.0) * FC_SUSTRATO
    y = np.array([B0, L0, D_C0, X0_G_C, 0.0, 0.0])
    for j in range(len(cortes) - 1):
        te = cortes[j+1]
        if te <= t0:
            continue
        tev = np.linspace(t0, te, 300)
        sol = _solve(rhs, (t0, te), y, (P_LARVA, p_lecho, N_LARVAS, T_C), tev)
        ts.append(sol.t if j == 0 else sol.t[1:])
        Ys.append(sol.y if j == 0 else sol.y[:, 1:])
        y = np.asarray(sol.y)[:, -1].copy()
        if te in PULSOS:
            y[2] += PULSOS[te] * FC_SUSTRATO
        t0 = te
    if not ts:
        raise RuntimeError("No se generaron puntos de integracion")
    return np.concatenate(ts), np.concatenate(Ys, axis=1), p_lecho

def flujo_CO2_C(t, Y, p_lecho):
    fl = np.empty(len(t))
    for i, ti in enumerate(t):
        B, Ltot, D_C, X, _, _ = [max(v, 0.0) for v in Y[:, i]]
        larva = tasas_larva(ti, B, Ltot, P_LARVA, p_lecho.f_dieta)
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        fl[i] = (N_LARVAS * larva.rCO2_C / 1000.0 + micro.rCmic) * 1000.0 * MG_C_A_MG_CO2 / 1440.0
    return fl

def main():
    HERE = Path(__file__).resolve().parent
    obs = pd.read_csv(HERE / "../datos/crudos/externos/ermolaev_co2.csv", comment="#")
    Lobs = obs[obs["tratamiento"] == "L"].dropna(subset=["co2_mg_min"])
    dias_obs = Lobs["dia"].to_numpy(float)
    co2_obs = Lobs["co2_mg_min"].to_numpy(float)
    sd_obs = Lobs["sd_mg_min"].to_numpy(float)
    t_span = (0.0, 14.0)

    print("Ajustando E0...")
    def resid_E0(f):
        t, Y = simular_E0(float(f[0]), t_span)
        fl = flujo_CO2_E0(t, Y, float(f[0]))
        mod = np.interp(dias_obs, t, fl)
        return (mod - co2_obs) / sd_obs
    sol_E0 = least_squares(resid_E0, [0.8], bounds=([0.2], [1.2]))
    f_E0 = float(sol_E0.x[0])
    print(f"  f_dieta={f_E0:.3f} resid={np.linalg.norm(sol_E0.fun):.2f}")

    print("Ajustando C...")
    def resid_C(f):
        t, Y, p = simular_C(float(f[0]), t_span)
        fl = flujo_CO2_C(t, Y, p)
        mod = np.interp(dias_obs, t, fl)
        return (mod - co2_obs) / sd_obs
    sol_C = least_squares(resid_C, [0.8], bounds=([0.2], [1.2]))
    f_C = float(sol_C.x[0])
    print(f"  f_dieta={f_C:.3f} resid={np.linalg.norm(sol_C.fun):.2f}")

    t_E0, Y_E0 = simular_E0(f_E0, t_span)
    fl_E0 = flujo_CO2_E0(t_E0, Y_E0, f_E0)
    t_C, Y_C, p_C = simular_C(f_C, t_span)
    fl_C = flujo_CO2_C(t_C, Y_C, p_C)

    cum_E0 = cumulative_trapezoid(fl_E0 * 1440.0 / 1000.0, t_E0, initial=0.0) / KG_FW
    cum_C = cumulative_trapezoid(fl_C * 1440.0 / 1000.0, t_C, initial=0.0) / KG_FW

    print(f"\ndia  obs     E0      C")
    for d, co in zip(dias_obs, co2_obs):
        fl_e = float(np.interp(d, t_E0, fl_E0))
        fl_c = float(np.interp(d, t_C, fl_C))
        print(f"{int(d):>3} {co:>6.2f} {fl_e:>6.2f} {fl_c:>6.2f}")

    print(f"\nAcum dia 14: obs=92.4  E0={float(np.interp(14.0,t_E0,cum_E0)):.1f}  C={float(np.interp(14.0,t_C,cum_C)):.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t_E0, fl_E0, "b--", lw=1.5, label=f"E0 (f={f_E0:.2f})")
    ax[0].plot(t_C, fl_C, "r-", lw=1.5, label=f"C (f={f_C:.2f})")
    ax[0].errorbar(dias_obs, co2_obs, yerr=sd_obs, fmt="ks", capsize=3, label="Obs")
    ax[0].set_xlabel("Dia"); ax[0].set_ylabel("CO2 (mg/min/caja)"); ax[0].legend(fontsize=8)
    ax[1].plot(t_E0, cum_E0, "b--", lw=1.5, label="E0")
    ax[1].plot(t_C, cum_C, "r-", lw=1.5, label="C")
    Lac = obs[obs["tratamiento"]=="L"].dropna(subset=["acum_g_kg_fw"])
    ax[1].errorbar(Lac["dia"], Lac["acum_g_kg_fw"], yerr=Lac["sd_acum"].fillna(0), fmt="ks", capsize=3)
    ax[1].set_xlabel("Dia"); ax[1].set_ylabel("Acumulado (g/kg FW)"); ax[1].legend(fontsize=8)
    fig.suptitle("Validacion Ermolaev 2019: E0 vs C")
    fig.tight_layout()
    out = RAIZ / "figuras" / "validacion_ermolaev_E0_vs_C.png"
    fig.savefig(out, dpi=200)
    print(f"\nFigura: {out}")

if __name__ == "__main__":
    main()

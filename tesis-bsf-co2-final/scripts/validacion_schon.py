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
T_C = 27.0
N_LARVAS = 600
DIAS = 7.0
DM_CAMA_G = 80.0
ML_CO2_POR_G_C_D = 24617.0 / (12.01 * 1440.0)

DIETAS = ["REF", "P:C 1:3", "P:C 1:5", "P:C 1:9"]
VCO2_TOTAL_TABLA = {"REF": 19.6, "P:C 1:3": 16.9, "P:C 1:5": 16.2, "P:C 1:9": 10.8}

P_LARVA = ParametrosEriksen(
    amax=1.180, alpha=1.0, beta=1.0, YB=0.44, YL=0.42, m=0.08,
    Bmax0=90.0 * 0.49 * 0.9, tp_min=13.0, rho=3.0,
    delta_I=5.0, delta_LB=0.1, delta_CB=0.49, delta_CL=0.79, delta_O=0.9,
    f_mant_prepupa=0.300, k_cat_prepupa=0.050,
)

P_LECHO_BASE = dict(k_ref=1.0, K_SM=50.0, Y_X=0.4,
                    delta_CS=0.665, k_d=0.05, gamma=0.170, K_G=0.1)

y0_base = estado_inicial_desde_dw(1.3, 0.0, 0.0, P_LARVA)
B0, L0 = y0_base[0], y0_base[1]
D_C0 = DM_CAMA_G * FC_SUSTRATO
X0_G_C = 1.0
y0 = np.array([B0, L0, D_C0, X0_G_C, 0.0, 0.0])

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
    tev = np.linspace(t_span[0], t_span[1], 700)
    sol = _solve(rhs_E0, t_span, np.array([B0, L0]), (), tev)
    return sol.t, sol.y

def flujo_CO2_E0(t, Y, f_dieta):
    fl = np.empty(len(t))
    for i, ti in enumerate(t):
        larva = tasas_larva(ti, Y[0, i], Y[1, i], P_LARVA, f_dieta)
        fl[i] = N_LARVAS * larva.rCO2_C * ML_CO2_POR_G_C_D / 1000.0
    return fl

def simular_C(f_dieta, t_span):
    p_lecho = ParametrosLecho(f_dieta=f_dieta, **P_LECHO_BASE)
    tev = np.linspace(t_span[0], t_span[1], 700)
    sol = _solve(rhs, t_span, y0, (P_LARVA, p_lecho, N_LARVAS, T_C), tev)
    return sol.t, sol.y, p_lecho

def flujo_CO2_C(t, Y, p_lecho):
    fl = np.empty(len(t))
    for i, ti in enumerate(t):
        B, Ltot, D_C, X, _, _ = [max(v, 0.0) for v in Y[:, i]]
        larva = tasas_larva(ti, B, Ltot, P_LARVA, p_lecho.f_dieta)
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        fl[i] = (N_LARVAS * larva.rCO2_C / 1000.0 + micro.rCmic) * ML_CO2_POR_G_C_D
    return fl

def main():
    HERE = Path(__file__).resolve().parent
    obs = pd.read_csv(HERE / "../datos/crudos/externos/schn_gas.csv", comment="#")
    crec = pd.read_csv(HERE / "../datos/crudos/externos/schn_crecimiento.csv", comment="#")
    t_span = (0.0, DIAS)

    resultados_E0, resultados_C = {}, {}

    print("=== E0: Eriksen puro ===")
    for dieta in DIETAS:
        o = obs[obs["dieta"] == dieta]
        do, fo = o["dia"].to_numpy(float), o["vco2_ml_min"].to_numpy(float)

        def resid(f):
            t, Y = simular_E0(float(f[0]), t_span)
            fl = flujo_CO2_E0(t, Y, float(f[0]))
            return np.interp(do, t, fl) - fo

        sol = least_squares(resid, [0.8], bounds=([0.2], [1.2]))
        f_d = float(sol.x[0])
        t, Y = simular_E0(f_d, t_span)
        fl = flujo_CO2_E0(t, Y, f_d)
        total_L = float(cumulative_trapezoid(fl * 1440.0 / 1000.0, t, initial=0)[-1])
        masa_fin = float(Y[0, -1] + Y[1, -1])
        resultados_E0[dieta] = dict(f=f_d, total_L=total_L, masa_fin=masa_fin)
        print(f"  {dieta:>8}: f={f_d:.3f}  VCO2={total_L:5.1f}L  masa={masa_fin:.1f}mgC")

    print("\n=== C: Modelo completo ===")
    for dieta in DIETAS:
        o = obs[obs["dieta"] == dieta]
        do, fo = o["dia"].to_numpy(float), o["vco2_ml_min"].to_numpy(float)

        def resid(f):
            t, Y, p = simular_C(float(f[0]), t_span)
            fl = flujo_CO2_C(t, Y, p)
            return np.interp(do, t, fl) - fo

        sol = least_squares(resid, [0.8], bounds=([0.2], [1.2]))
        f_d = float(sol.x[0])
        t, Y, p = simular_C(f_d, t_span)
        fl = flujo_CO2_C(t, Y, p)
        total_L = float(cumulative_trapezoid(fl * 1440.0 / 1000.0, t, initial=0)[-1])
        masa_fin = float(Y[0, -1] + Y[1, -1])
        resultados_C[dieta] = dict(f=f_d, total_L=total_L, masa_fin=masa_fin, t=t, fl=fl)
        print(f"  {dieta:>8}: f={f_d:.3f}  VCO2={total_L:5.1f}L  masa={masa_fin:.1f}mgC")

    print("\n=== CRITERIOS ===")
    orden_E0 = [resultados_E0[d]["f"] for d in DIETAS]
    orden_C = [resultados_C[d]["f"] for d in DIETAS]
    ok_ord_E0 = all(orden_E0[i] >= orden_E0[i+1] - 0.05 for i in range(3))
    ok_ord_C = all(orden_C[i] >= orden_C[i+1] - 0.05 for i in range(3))
    print(f"Orden f E0: {'CUMPLE' if ok_ord_E0 else 'NO CUMPLE'}")
    print(f"Orden f C:  {'CUMPLE' if ok_ord_C else 'NO CUMPLE'}")

    print(f"\n{'Dieta':>8} {'Obs':>6} {'E0':>6} {'err%':>6} | {'C':>6} {'err%':>6}")
    for dieta in DIETAS:
        obs_t = VCO2_TOTAL_TABLA[dieta]
        e0_t = resultados_E0[dieta]["total_L"]
        c_t = resultados_C[dieta]["total_L"]
        print(f"{dieta:>8} {obs_t:>6.1f} {e0_t:>6.1f} {100*abs(e0_t-obs_t)/obs_t:>5.1f}% | {c_t:>6.1f} {100*abs(c_t-obs_t)/obs_t:>5.1f}%")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k, dieta in enumerate(DIETAS):
        a = ax[k // 2][k % 2]
        r = resultados_C[dieta]
        o = obs[obs["dieta"] == dieta]
        a.plot(r["t"], r["fl"], "r-", lw=1.6, label=f"C (f={r['f']:.2f})")
        t_e0, Y_e0 = simular_E0(resultados_E0[dieta]["f"], t_span)
        fl_e0 = flujo_CO2_E0(t_e0, Y_e0, resultados_E0[dieta]["f"])
        a.plot(t_e0, fl_e0, "b--", lw=1.2, alpha=0.7, label=f"E0 (f={resultados_E0[dieta]['f']:.2f})")
        a.plot(o["dia"], o["vco2_ml_min"], "s", color="tab:green", ms=3, alpha=0.7, label="obs")
        a.set_title(dieta, fontsize=10)
        a.set_xlabel("Dia"); a.set_ylabel("VCO2 (mL/min)")
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle("Validacion Schon 2026: E0 vs C")
    fig.tight_layout()
    out = RAIZ / "figuras" / "validacion_schon_E0_vs_C.png"
    fig.savefig(out, dpi=200)
    print(f"\nFigura: {out}")

if __name__ == "__main__":
    main()

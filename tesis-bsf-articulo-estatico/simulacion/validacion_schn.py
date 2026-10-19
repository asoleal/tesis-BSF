#!/usr/bin/env python3
# validacion_schn.py - Validacion externa del modelo con Schon et al. (2026)
# J. Insect Physiol. 170:104961 (dieta P:C, respirometria de flujo abierto).
# Archivo NUEVO: no modifica validacion_estatica.py (se importa como V).
# Escenario Schon:
#   * 600 larvas de 5-6 dias, 5.0 mg WW (~1.3 mg MS) por larva al inicio
#   * lecho unico: 80 g MS + 261.3 g agua (341.3 g dieta fresca, ~77 % agua)
#   * 27 C, 7 dias, sin realimentacion; control = camara vacia (no alimento
#     solo), asi que el CO2 microbiano del lecho NO esta separado en obs
#   * flujo abierto: el flujo medido es la produccion total (larvas + lecho),
#     en mL/min por camara -> comparamos VCO2 total directamente
# Transferencia: nucleo DEB, (B0, tp1, wp) y modulo microbiano D1 fijos;
# libre: f por dieta (el factor de calidad dieta-especifico del articulo).
# Declarado: RER>1 (lipogenesis) es estructuralmente ajeno al modelo; la
# validacion es de VCO2 (VO2 queda como referencia con RQ asumido).
import io
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares

import validacion_estatica as V

HERE = Path(__file__).resolve().parent
FIG = V.RAIZ / "figuras"

N_LARVAS = 600
DM_CAMA = 80.0          # g MS (premix seco por camara)
W_CAMA = 261.3          # g agua (341.3 g dieta fresca - 80 g MS)
DIAS = 7.0
MAS_INI_MS_MG = 1.3     # 5.0 mg WW x 26.3 % MS (su Table 2, larvas al inicio)
# volumen molar del CO2 a 27 C, 1 atm: Rg*T/P = 24617 mL/mol; 44.01 g/mol
MG_A_ML = 0.5596   # mL de CO2 por mg (Vmolar=24617 mL/mol a 27 C, M=44.01 g/mol)
DIETAS = ["REF", "P:C 1:3", "P:C 1:5", "P:C 1:9"]
VCO2_TOTAL_TABLA = {"REF": 19.6, "P:C 1:3": 16.9, "P:C 1:5": 16.2, "P:C 1:9": 10.8}  # L


def calibrar_modelo_propio():
    """Modulo microbiano D1 (en silencio) + nucleo calibrado aceptado del
    articulo (calibracion_conjunta con instares): ya NO se recalibra aqui."""
    datos, _ = V.cargar_datos()
    for dieta in ["D1", "D4"]:
        datos[dieta]["neto"] = (datos[dieta]["trat"] - datos[dieta]["ctrl"]).clip(lower=0)
    kmic, ymic, xmic = {}, {}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        for dieta in ["D1", "D4"]:
            dc = datos[dieta].dropna(subset=["ctrl"])
            kmic[dieta], ymic[dieta], xmic[dieta] = V.calibrar_micro(
                dc["dia"].values, dc["ctrl"].values, dieta)
    B0, amax, Bmax, rho = 1.097, 1.499, 48.93, 2.0   # calibracion conjunta aceptada (750 g)
    return kmic, ymic, xmic, B0, amax, Bmax, rho


def estado_inicial(B0, amax, Bmax, rho, kref, YCO2, masa_objetivo=MAS_INI_MS_MG):
    """(B, L) de nuestra simulacion D1 en el instante en que B+L alcanza la
    masa inicial de Schon (~1.3 mg MS): sus larvas son 5-6 dias pero de una
    colonia distinta (llegan mas pequenas que las nuestras a esa edad)."""
    p = dict(V.P); p.update(f=1.0, amax=amax, Bmax=Bmax, rho=rho)
    xb = V.X0["B"]
    V.X0["B"] = B0
    t, Y = V.simular(p, kref, YCO2)
    V.X0["B"] = xb
    total = Y[0] + Y[1]
    i = int(np.argmax(total >= masa_objetivo))
    return float(Y[0, i]), float(Y[1, i]), float(t[i])


def simular_schn(f, p_base, kref, YCO2, Bs, Ls, X0m):
    """Lote unico, 7 dias, sin cortes (no hay pulsos)."""
    p = dict(p_base); p["f"] = f; p["N"] = N_LARVAS
    y0 = [Bs, Ls, DM_CAMA, W_CAMA, X0m]
    sol = solve_ivp(V.rhs, (0.0, DIAS), y0, args=(p, kref, YCO2), method="LSODA",
                    t_eval=np.linspace(0.0, DIAS, 700), rtol=1e-6, atol=1e-9)
    return sol.t, sol.y, p


def flujo_ml_min(t, Y, p, kref, YCO2):
    fl = np.empty(len(t))
    for i, ti in enumerate(t):
        r = V.tasas(ti, *Y[:, i], p, kref, YCO2)
        fl[i] = (p["N"] * r["rCO2lar"] + 1000.0 * r["rCmic"]) / 1440.0
    return fl * MG_A_ML   # mg/min -> mL/min (CO2 a 27 C)


def main():
    print("Calibracion del modelo propio en silencio (puede tardar unos minutos)...")
    kmic, ymic, xmic, B0, amax, Bmax, rho = calibrar_modelo_propio()
    print(f"  micro D1: kref={kmic['D1']:.4g}, YCO2={ymic['D1']:.4g}, X0={xmic['D1']:.3g} g")
    print(f"  nucleo transferido: B0={B0:.4g} mg, amax={amax:.3g}, Bmax={Bmax:.3g}, rho={rho:.3g}")
    Bs, Ls, ts = estado_inicial(B0, amax, Bmax, rho, kmic["D1"], ymic["D1"])
    print(f"  estado inicial transplantado (t*={ts:.2f} d de nuestra D1): "
          f"B={Bs:.3g} mg, L={Ls:.3g} mg, B+L={Bs+Ls:.3g} mg MS")

    V.TC = 27.0   # escenario Schon (27 C): Q10 solo en el modulo microbiano
    obs = pd.read_csv(HERE / "schn_gas.csv", comment="#")
    crec = pd.read_csv(HERE / "schn_crecimiento.csv", comment="#")
    # t_pm corregido por el transplante: el dia 0 de Schon es nuestro t*,
    # y t_pm=8.5 esta definido en el eje del montaje.
    p_base = dict(V.P); p_base.update(amax=amax, Bmax=Bmax, rho=rho,
                                      t_pm=max(V.P["t_pm"] - ts, 1.0))

    resultados, fig_filas = {}, []
    print("\nAjuste de f por dieta contra la curva VCO2 horaria (mL/min/camara):")
    for dieta in DIETAS:
        o = obs[obs["dieta"] == dieta]
        do, fo = o["dia"].to_numpy(float), o["vco2_ml_min"].to_numpy(float)

        def resid(fpar, do=do, fo=fo):
            t, Y, p = simular_schn(float(fpar[0]), p_base, kmic["D1"], ymic["D1"],
                                   Bs, Ls, xmic["D1"])
            fl = flujo_ml_min(t, Y, p, kmic["D1"], ymic["D1"])
            return np.interp(do, t, fl) - fo

        sol = least_squares(resid, [0.8], bounds=([0.2], [1.2]))
        f_d = float(sol.x[0])
        t, Y, p = simular_schn(f_d, p_base, kmic["D1"], ymic["D1"], Bs, Ls, xmic["D1"])
        fl = flujo_ml_min(t, Y, p, kmic["D1"], ymic["D1"])
        total_L = float(cumulative_trapezoid(fl * 1440.0 / 1000.0, t, initial=0)[-1])
        pico_d = float(t[int(np.argmax(fl))])
        masa_fin = float(Y[0, -1] + Y[1, -1])
        cr = crec[crec["dieta"] == dieta]
        masa_obs = float(cr["masa_mg_ms"].iloc[-1])
        resultados[dieta] = dict(f=f_d, total_L=total_L, pico_d=pico_d,
                                 masa_mod=masa_fin, masa_obs=masa_obs, t=t, fl=fl,
                                 obs=(do, fo))
        print(f"  {dieta:>8}: f={f_d:.3f}  VCO2 total={total_L:5.1f} L "
              f"(tabla {VCO2_TOTAL_TABLA[dieta]:4.1f})  pico dia {pico_d:.2f}  "
              f"masa final {masa_fin:.1f} mg MS (obs {masa_obs:.1f})")

    print("\nCriterios de validacion:")
    orden_f = [resultados[d]["f"] for d in DIETAS]
    ok_ord = all(orden_f[i] >= orden_f[i+1] - 0.05 for i in range(3))
    print(f"  ordenamiento f coherente con P:C (REF>=1:3>=1:5>=1:9): "
          f"{'CUMPLE' if ok_ord else 'NO CUMPLE'}")
    for dieta in DIETAS:
        r = resultados[dieta]
        ok_t = abs(r["total_L"] - VCO2_TOTAL_TABLA[dieta]) / VCO2_TOTAL_TABLA[dieta] <= 0.25
        ok_m = r["masa_mod"] <= 2.0 * r["masa_obs"]
        print(f"  {dieta:>8}: total +-25 % {'CUMPLE' if ok_t else 'NO CUMPLE'}   "
              f"masa dentro de factor 2 {'CUMPLE' if ok_m else 'NO CUMPLE'}")
        fig_filas.append(dict(dieta=dieta, f=r["f"], vco2_total_L=r["total_L"],
                              vco2_total_obs_L=VCO2_TOTAL_TABLA[dieta],
                              pico_dia=r["pico_d"], masa_final_mod_mg=r["masa_mod"],
                              masa_final_obs_mg=r["masa_obs"]))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k, dieta in enumerate(DIETAS):
        a = ax[k // 2][k % 2]
        r = resultados[dieta]
        do, fo = r["obs"]
        a.plot(r["t"], r["fl"], "k-", lw=1.6, label="modelo")
        a.plot(do, fo, "s", color="tab:blue", ms=3, alpha=0.7, label="obs (digitalizado)")
        cr = crec[crec["dieta"] == dieta]
        a2 = a.twinx()
        a2.plot(cr["dia"], cr["masa_mg_ms"], "o-", color="tab:orange", ms=3, lw=1,
                alpha=0.8, label="masa obs (mg MS)")
        a2.set_ylabel("mg MS/larva", fontsize=8)
        a.set_title(f"{dieta}  (f={r['f']:.2f})", fontsize=10)
        a.set_xlabel("dias"); a.set_ylabel("VCO2 (mL/min)")
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle("Validacion externa con Schon et al. (2026), respirometria de flujo abierto")
    fig.tight_layout()
    out = FIG / "validacion_schn.png"
    fig.savefig(out, dpi=200)
    pd.DataFrame(fig_filas).to_csv(HERE / "resultados_schn.csv", index=False)
    print(f"\nFigura: {out}")
    print(f"Tabla:  {HERE / 'resultados_schn.csv'}")


if __name__ == "__main__":
    main()

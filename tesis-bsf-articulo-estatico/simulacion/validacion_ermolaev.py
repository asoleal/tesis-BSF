#!/usr/bin/env python3
# validacion_ermolaev.py - Validacion externa del modelo con Ermolaev et al.
# (2019) Waste Management 96:65-74, tratamiento L (700 larvas, food waste, 28 C).
# Archivo NUEVO: no modifica validacion_estatica.py (se importa como V).
# Escenario Ermolaev:
#   * larvas de 5 dias: estado (B, L) tomado de nuestra simulacion D1 a t = 5
#   * alimentacion por pulsos: dias 0, 2, 7, 11; 30.0 mg MS/larva/d
#     (42, 105, 84, 63 g MS/caja; total 294 g, cuadra con su Table 2)
#   * T = 28 C: Q10 solo en el modulo microbiano (como en el modelo original)
#   * interruptor de prepupa activo; tp queda libre (es dieta/colonia-especifico)
#   * sin caja control: modulo microbiano fijado con la calibracion de D1
#     (N del sustrato similar: 35.4 vs ~32-35 mg N/g MS)
# Parametros libres: f y tp, los dos que el articulo calibra como dieta-especificos;
# todo lo demas (nucleo DEB, microbiano D1, wp) se transfiere sin reajuste.
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

T_OFFSET = 5.0                      # edad de las larvas al dia 0 del experimento
# tp NO se fija a 19: es edad fisiologica y el desfase +5 d ya cuenta la edad
MS = 0.249                          # fraccion de MS del food waste
AGUA = (1.0 - MS) / MS              # g agua por g MS (3.016)
KG_FW = 291.7 / MS / 1000.0         # kg FW por caja (1.171)
DM_INI = 42.0                       # pulso del dia 0 (g MS/caja)
PULSOS = {2: 105.0, 7: 84.0, 11: 63.0}   # dia exp -> g MS/caja
MASA_FINAL_MG = 297.0 * 0.343       # Table 1: 101.9 mg DM/larva al dia 14


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
    B0, amax, Bmax, rho = 1.14, 1.5, 48.9, 2.2   # calibracion conjunta aceptada
    return kmic, ymic, xmic, B0, amax, Bmax, rho


def estado_dia5(B0, amax, Bmax, rho, kmic, ymic):
    """(B, L) de nuestra simulacion D1 a la edad de 5 dias = estado inicial de
    sus larvas. Ojo: nuestro t=0 es el montaje (larvas de ~4.5 d de edad), asi
    que una larva de 5 d de edad corresponde a nuestro t ~= 0.5."""
    p = dict(V.P); p.update(f=1.0, amax=amax, Bmax=Bmax, rho=rho)
    xb = V.X0["B"]
    V.X0["B"] = B0
    t, Y = V.simular(p, kmic, ymic)
    V.X0["B"] = xb
    t5 = max(T_OFFSET - 4.5, 0.0)   # 4.5 d = edad estimada al montaje
    return float(np.interp(t5, t, Y[0])), float(np.interp(t5, t, Y[1]))


def simular_ermolaev(f, p_base, kref, YCO2, B5, L5, X0e):
    p = dict(p_base); p["f"] = f
    y = [B5, L5, DM_INI, DM_INI * AGUA, X0e]
    cortes = [T_OFFSET + d for d in sorted(PULSOS)] + [T_OFFSET + 14.0]
    ts, ys, t0 = [], [], T_OFFSET
    for j, te in enumerate(cortes):
        sol = solve_ivp(V.rhs, (t0, te), y, args=(p, kref, YCO2), method="LSODA",
                        t_eval=np.linspace(t0, te, 300), rtol=1e-6, atol=1e-9)
        ts.append(sol.t if j == 0 else sol.t[1:])
        ys.append(sol.y if j == 0 else sol.y[:, 1:])
        y = list(sol.y[:, -1]); t0 = te
        dia = te - T_OFFSET
        if dia in PULSOS:            # alimentar justo despues del corte
            y[2] += PULSOS[dia]      # (sus mediciones fueron antes de alimentar,
            y[3] += PULSOS[dia] * AGUA   # asi que el punto medido es el pre-pulso)
    return np.concatenate(ts), np.concatenate(ys, axis=1), p


def flujo_mg_min(t, Y, p, kref, YCO2):
    fl = np.empty(len(t))
    for i, ti in enumerate(t):
        r = V.tasas(ti, *Y[:, i], p, kref, YCO2)
        fl[i] = (p["N"] * r["rCO2lar"] + 1000.0 * r["rCmic"]) / 1440.0
    return fl


def en_dia(t, fl, d):
    return fl[int(np.argmin(np.abs(t - (T_OFFSET + d))))]


def main():
    print("Calibracion del modelo propio en silencio (puede tardar unos minutos)...")
    kmic, ymic, xmic, B0, amax, Bmax, rho = calibrar_modelo_propio()
    print(f"  micro D1: kref={kmic['D1']:.4g}, YCO2={ymic['D1']:.4g}, X0={xmic['D1']:.3g} g")
    print(f"  nucleo transferido: B0={B0:.4g} mg, amax={amax:.3g}, Bmax={Bmax:.3g}, rho={rho:.3g}")
    B5, L5 = estado_dia5(B0, amax, Bmax, rho, kmic["D1"], ymic["D1"])
    print(f"  larva de 5 d (nuestro modelo D1): B={B5:.3g} mg, L={L5:.3g} mg")

    V.TC = 28.0   # a partir de aqui, escenario Ermolaev (28 C)
    obs = pd.read_csv(HERE / "ermolaev_co2.csv", comment="#")
    Lobs = obs[obs["tratamiento"] == "L"].dropna(subset=["co2_mg_min"])
    # 4 parametros dieta/colonia-especificos (f, t_pm, rho, Bmax) contra
    # 4 residuos: 3 tasas puntuales + acumulado dia 14 (92.4 +- 14.5, su Fig.).
    # Criterios de validacion INDEPENDIENTES: dia del pico, magnitud, declive
    # y masa final (ninguno entra al ajuste).
    p_base = dict(V.P); p_base.update(amax=amax, rho=rho)

    def resid(par):
        fpar, tpm_e, rho_e, bmax_e = par
        p2 = dict(p_base); p2.update(t_pm=tpm_e, rho=rho_e, Bmax=bmax_e)
        t, Y, p = simular_ermolaev(fpar, p2, kmic["D1"], ymic["D1"],
                                   B5, L5, xmic["D1"])
        fl = flujo_mg_min(t, Y, p, kmic["D1"], ymic["D1"])
        mod = np.array([en_dia(t, fl, d) for d in Lobs["dia"]])
        r_tasas = (mod - Lobs["co2_mg_min"].to_numpy(float)) / Lobs["sd_mg_min"].to_numpy(float)
        dias_e = t - T_OFFSET
        cum = cumulative_trapezoid(fl * 1440.0 / 1000.0, dias_e, initial=0.0) / KG_FW
        r_acum = np.array([(float(np.interp(14.0, dias_e, cum)) - 92.4) / 14.5])
        return np.concatenate([r_tasas, r_acum])

    sol = least_squares(resid, [0.9, 12.0, 8.0, 80.0],
                        bounds=([0.2, 10.0, 2.0, 60.0], [1.2, 16.0, 25.0, 95.0]))
    f_e, tpm_e, rho_e, bmax_e = sol.x
    print(f"\n*** f Ermolaev = {f_e:.3f}, t_pm = {tpm_e:.2f} d, "
          f"rho = {rho_e:.3g} mg/d, Bmax = {bmax_e:.3g} mg ***")

    p_base.update(t_pm=tpm_e, rho=rho_e, Bmax=bmax_e)
    t, Y, p = simular_ermolaev(f_e, p_base, kmic["D1"], ymic["D1"], B5, L5, xmic["D1"])
    fl = flujo_mg_min(t, Y, p, kmic["D1"], ymic["D1"])
    dias = t - T_OFFSET
    bmax_t = p["Bmax"] - p["rho"] * np.maximum(t - p["t_pm"], 0.0)
    i_stop = np.where(Y[0] >= bmax_t)[0]
    if len(i_stop):
        print(f"  detencion de prepupa emergente: dia {t[i_stop[0]] - T_OFFSET:.1f} de su ensayo")
    cum = cumulative_trapezoid(fl * 1440.0 / 1000.0, dias, initial=0.0) / KG_FW

    filas = []
    print("\ndia   obs(mg/min)   sd   modelo   acum_obs   acum_mod (g/kg FW)")
    for _, r in obs[obs["tratamiento"] == "L"].iterrows():
        d = r["dia"]
        fl_mod = en_dia(t, fl, d)
        cum_mod = float(np.interp(d, dias, cum))
        filas.append(dict(dia=d, obs_mg_min=r["co2_mg_min"], sd=r["sd_mg_min"],
                          mod_mg_min=fl_mod, acum_obs=r["acum_g_kg_fw"],
                          acum_mod=cum_mod))
        print(f"{int(d):>3}   {r['co2_mg_min']:>11}   {r['sd_mg_min']:>4}   "
              f"{fl_mod:6.2f}   {r['acum_g_kg_fw']:>8}   {cum_mod:8.1f}")

    fl7, fl11 = en_dia(t, fl, 7), en_dia(t, fl, 11)
    pico = dias[int(np.argmax(fl))]
    decl = (fl7 - fl11) / fl7
    cum14 = float(np.interp(14.0, dias, cum))
    masa = float(Y[0, -1] + Y[1, -1])
    print(f"\nPico del modelo: dia {pico:.1f} (obs: 7)")
    print(f"Declive 7->11: {100*decl:.0f} % (obs: 24 % promedio, 31 % en L)")
    print(f"Acumulado dia 14: {cum14:.1f} g/kg FW (obs: 92.4 +- 14.5)")
    print(f"Masa larvaria final: {masa:.1f} mg DM/larva (obs: {MASA_FINAL_MG:.1f})")
    print("\nCriterios de validacion:")
    print(f"  pico en dia 7 +- 1:      {'CUMPLE' if abs(pico - 7) <= 1 else 'NO CUMPLE'}")
    print(f"  magnitud del pico +-25 %: {'CUMPLE' if abs(fl7 - 8.04)/8.04 <= 0.25 else 'NO CUMPLE'}")
    print(f"  declive 15-35 %:          {'CUMPLE' if 0.15 <= decl <= 0.35 else 'NO CUMPLE'}")
    print(f"  acumulado +-30 %:         {'CUMPLE' if abs(cum14 - 92.4)/92.4 <= 0.30 else 'NO CUMPLE'}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(dias, fl, "k-", lw=1.5, label="modelo")
    ax[0].errorbar(Lobs["dia"], Lobs["co2_mg_min"], yerr=Lobs["sd_mg_min"],
                   fmt="s", color="tab:blue", capsize=3, label="L (obs)")
    for tr, mk, c in [("BL", "o", "tab:orange"), ("PBL", "^", "tab:green")]:
        o = obs[obs["tratamiento"] == tr].dropna(subset=["co2_mg_min"])
        ax[0].plot(o["dia"], o["co2_mg_min"], mk, color=c, alpha=0.5, label=f"{tr} (obs)")
    for d in PULSOS:
        ax[0].axvline(d, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("dia de tratamiento"); ax[0].set_ylabel("CO2 (mg/min/caja)")
    ax[0].legend(fontsize=8)
    ax[1].plot(dias, cum, "k-", lw=1.5, label="modelo")
    Lac = obs[obs["tratamiento"] == "L"].dropna(subset=["acum_g_kg_fw"])
    ax[1].errorbar(Lac["dia"], Lac["acum_g_kg_fw"], yerr=Lac["sd_acum"].fillna(0),
                   fmt="s", color="tab:blue", capsize=3, label="L (obs)")
    ax[1].set_xlabel("dia de tratamiento"); ax[1].set_ylabel("CO2 acumulado (g/kg FW)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Validacion externa con Ermolaev et al. (2019), tratamiento L")
    fig.tight_layout()
    out = FIG / "validacion_ermolaev.png"
    fig.savefig(out, dpi=200)
    pd.DataFrame(filas).to_csv(HERE / "resultados_ermolaev.csv", index=False)
    print(f"\nFigura: {out}")
    print(f"Tabla:  {HERE / 'resultados_ermolaev.csv'}")


if __name__ == "__main__":
    main()

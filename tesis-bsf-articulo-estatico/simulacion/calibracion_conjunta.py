#!/usr/bin/env python3
# calibracion_conjunta.py - Calibracion conjunta gas + biomasa.
# Etapa 1 (microbiana): igual que validacion_estatica (controles de alimento solo).
# Etapa 2 (larvaria): ajuste simultaneo contra tasas netas de CO2 (puntos limpios
# + cotas inferiores de saturados) y contra las curvas de crecimiento del ensayo
# paralelo (crecimiento_propio.csv).
# Nucleo larvario: instares de Eriksen (2022), ecs. 5, 15 y 16 (ver
# validacion_estatica.tasas): la detencion de prepupa EMERGE cuando B alcanza
# Bmax(t) = Bmax - rho*(t - t_pm); ya no hay t_p ni w_p calibrados.
# Libres: B0, f_D4, a_max (acotado al rango de Eriksen 1.1-1.5 d^-1), B_max,
# rho, DM_f y tau por dieta (anticipo de la rampa de %MS respecto al t_stop).
# Residuos ponderados: gas por el rango entre cierres (piso 10 ppm/min),
# crecimiento por la DE de las replicas (piso 10% del valor). L0 = 0.25 B0.
# Conversion MS->PH con %MS variable: 26% en crecimiento, subiendo a DM_f hacia
# la prepupa; la rampa se ancla al t_stop EMERGENTE, tau dias antes, con
# anchura fija W_DM = 1 d (vaciado de intestino / perdida de agua de prepupa).
# Desfase delta (2 d, notas de laboratorio): los dias del CSV de gas se evaluan
# en (dia - delta).
from pathlib import Path
import io, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import validacion_estatica as V

HERE = Path(__file__).resolve().parent
DM_INI, DM_BANDA, W_DM = 0.26, 0.04, 1.0
DELTA = 2.0  # desfase fijo: dias del CSV de gas = dia de montaje + 2

def t_stop_emergente(t, Y, pd_):
    """Primer dia en que B alcanza Bmax(t) (prepupa emergente, ecs. 15-16)."""
    bmax_t = pd_["Bmax"] - pd_["rho"] * np.maximum(t - pd_["t_pm"], 0.0)
    cruza = np.where(Y[0] >= bmax_t)[0]
    return float(t[cruza[0]]) if len(cruza) else float(t[-1])

def dm_frac(t, t_stop, dm_f, tau=0.0):
    return DM_INI + (dm_f - DM_INI) / (1.0 + np.exp(-(t - (t_stop - tau)) / W_DM))

def cargar_todo():
    datos, ayuno = V.cargar_datos()
    for d in ["D1", "D4"]:
        datos[d]["neto"] = (datos[d]["trat"] - datos[d]["ctrl"]).clip(lower=0)
    kmic, ymic, xmic = {}, {}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        for d in ["D1", "D4"]:
            dc = datos[d].dropna(subset=["ctrl"])
            kmic[d], ymic[d], xmic[d] = V.calibrar_micro(dc["dia"].values,
                                                         dc["ctrl"].values, d)
    crec = pd.read_csv(HERE / "crecimiento_propio.csv", comment="#")
    return datos, ayuno, kmic, ymic, xmic, crec

def simular_dieta(dieta, pars, kmic, ymic, xmic):
    pd_ = dict(V.P); pd_.update(f=pars["f"], amax=pars["amax"],
                                Bmax=pars["Bmax"], rho=pars["rho"])
    xb, xl, xx = V.X0["B"], V.X0["L"], V.X0["X"]
    V.X0["B"], V.X0["L"], V.X0["X"] = pars["B0"], 0.25 * pars["B0"], xmic[dieta]
    previo = V.poner_lecho(dieta)
    t, Y = V.simular(pd_, kmic[dieta], ymic[dieta])
    V.quitar_lecho(previo)
    V.X0["B"], V.X0["L"], V.X0["X"] = xb, xl, xx
    return t, Y, pd_

def resid_conjunto(x, datos, crec, kmic, ymic, xmic):
    B0 = 10.0**x[0]
    f4, amax, Bmax, rho, dm_f, tau1, tau4 = x[1:]
    delta = DELTA
    r = []
    for dieta, f, tau in [("D1", 1.0, tau1), ("D4", f4, tau4)]:
        pars = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
        t, Y, pd_ = simular_dieta(dieta, pars, kmic, ymic, xmic)
        lar, _ = V.componentes_ppm(t, Y, pd_, kmic[dieta], ymic[dieta])
        t_stop = t_stop_emergente(t, Y, pd_)
        d = datos[dieta].dropna(subset=["neto"])
        mod = np.interp(d["dia"].to_numpy(float) - delta, t, lar)
        sig = ((d["trat_max"] - d["trat_min"]) / 2.0).to_numpy(float)
        sig = np.where(np.isfinite(sig) & (sig > 10.0), sig, 10.0)
        ri = (mod - d["neto"].to_numpy(float)) / sig
        sat = d["sat"].to_numpy(bool)
        ri[sat] = np.minimum(ri[sat], 0.0)
        r.append(ri)
        g = crec[crec["dieta"] == dieta]
        dmf = dm_frac(t, t_stop, dm_f, tau)
        modg = np.interp(g["dia"].to_numpy(float), t, (Y[0] + Y[1]) / dmf)
        sig_g = np.maximum(g["de_mg"].to_numpy(float), 0.10 * g["ph_mg"].to_numpy(float))
        r.append((modg - g["ph_mg"].to_numpy(float)) / sig_g)
    return np.concatenate(r)

def calibrar(datos, crec, kmic, ymic, xmic):
    arranques = [
        [np.log10(0.8), 0.8, 1.3, 40.0, 1.0, 0.35, 3.0, 3.0],
        [np.log10(0.4), 0.8, 1.4, 65.0, 1.0, 0.38, 2.0, 2.0],
        [np.log10(1.0), 1.0, 1.2, 30.0, 4.0, 0.32, 4.0, 1.0],
        [np.log10(0.6), 0.9, 1.5, 50.0, 8.0, 0.40, 1.0, 4.0],
    ]
    cotas = ([np.log10(0.05), 0.2, 1.1, 20.0, 0.5, 0.28, 0.0, 0.0],
             [np.log10(2.0),  1.2, 1.5, 80.0, 15.0, 0.65, 6.0, 6.0])
    mejor = None
    for x0 in arranques:
        sol = least_squares(resid_conjunto, x0, args=(datos, crec, kmic, ymic, xmic),
                            bounds=cotas)
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol
    B0 = 10.0**mejor.x[0]
    return B0, *mejor.x[1:]

def main():
    datos, ayuno, kmic, ymic, xmic, crec = cargar_todo()
    B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = calibrar(datos, crec, kmic, ymic, xmic)
    delta = DELTA
    print(f"\n*** Calibracion conjunta (instares Eriksen): B0={B0:.3g} mg, "
          f"f_D4={f4:.3g}, a_max={amax:.3g} d^-1, B_max={Bmax:.3g} mg, "
          f"rho={rho:.3g} mg/d, DM_f={dm_f:.2f}, "
          f"tau D1={tau1:.2f} d, tau D4={tau4:.2f} d, delta={delta:.1f} d (fijo) ***")
    print(f"    (microbiano sin cambios: k_ref {kmic['D1']:.3g}/{kmic['D4']:.3g} d^-1, "
          f"Y_CO2 {ymic['D1']:.3g}/{ymic['D4']:.3g})")
    filas = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for j, (dieta, f, tau) in enumerate([("D1", 1.0, tau1), ("D4", f4, tau4)]):
        pars = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
        t, Y, pd_ = simular_dieta(dieta, pars, kmic, ymic, xmic)
        lar, mic = V.componentes_ppm(t, Y, pd_, kmic[dieta], ymic[dieta])
        t_stop = t_stop_emergente(t, Y, pd_)
        d = datos[dieta].copy()
        d["neto_lo"] = (d["trat_min"] - d["ctrl"]).clip(lower=0)
        d["neto_hi"] = (d["trat_max"] - d["ctrl"]).clip(lower=0)
        d["dia_m"] = d["dia"] - delta
        limpios = d[~d["sat"]].dropna(subset=["neto"])
        ml = np.interp(limpios["dia_m"].to_numpy(float), t, lar)
        if len(limpios) >= 2:
            r2 = 1 - np.sum((limpios["neto"] - ml)**2) / np.sum((limpios["neto"] - limpios["neto"].mean())**2)
        else:
            r2 = float("nan")
        rmse_g = float(np.sqrt(np.mean((limpios["neto"] - ml)**2))) if len(limpios) else float("nan")
        ax = axes[0][j]
        ax.plot(t, lar, color="tab:blue", label="Larvario (modelo)")
        ax.plot(t, mic, color="tab:green", ls="--", label="Microbiano (modelo)")
        ax.plot(t, lar + mic, color="tab:gray", lw=2, alpha=0.7, label="Total (modelo)")
        for m, mk, lab in [(~d["sat"], "o", "Neto obs."), (d["sat"], "^", "Neto obs. saturado")]:
            dd = d[m].dropna(subset=["neto"])
            if len(dd):
                ax.errorbar(dd["dia_m"], dd["neto"], yerr=[dd["neto"] - dd["neto_lo"], dd["neto_hi"] - dd["neto"]],
                            fmt=mk, color="tab:blue", capsize=3, label=lab)
        ax.plot(d["dia_m"], d["ctrl"], "s", color="tab:green", label="Control obs.")
        ax.set_title(f"Gas {dieta} | R2 {r2:.3f}, RMSE {rmse_g:.1f} ppm/min" if np.isfinite(r2) else f"Gas {dieta} | RMSE {rmse_g:.1f} ppm/min")
        ax.set_xlabel("Día"); ax.legend(fontsize=7)
        g = crec[crec["dieta"] == dieta]
        ms = Y[0] + Y[1]
        dmf = dm_frac(t, t_stop, dm_f, tau)
        ph = ms / dmf
        modg = np.interp(g["dia"].to_numpy(float), t, ph)
        rmse_b = float(np.sqrt(np.mean((modg - g["ph_mg"])**2)))
        pico_m = float(t[np.argmax(ph)]); pico_o = float(g.loc[g["ph_mg"].idxmax(), "dia"])
        d14m = float(np.interp(14, t, ph)); d14o = float(g.loc[g["dia"] == 14, "ph_mg"].iloc[0])
        i14 = int(np.argmin(np.abs(t - 14.0)))
        lip14 = float(Y[1][i14] / ms[i14])
        print(f"  {dieta}: gas R2={r2:.3f} RMSE={rmse_g:.1f} ppm/min | "
              f"biomasa RMSE={rmse_b:.1f} mg, pico modelo {pico_m:.1f} vs obs {pico_o:.0f}, "
              f"dia14 modelo {d14m:.0f} vs obs {d14o:.0f} mg PH | "
              f"t_stop {t_stop:.1f} d, lipidos d14 {100*lip14:.0f}% MS")
        ax = axes[1][j]
        ax.fill_between(t, ms / (dmf + DM_BANDA), ms / (dmf - DM_BANDA), color="tab:blue", alpha=0.15, label="Banda MS ±4%")
        ax.plot(t, ph, color="tab:blue", label="Modelo B+L (MS variable)")
        ax.errorbar(g["dia"], g["ph_mg"], yerr=g["de_mg"], fmt="o", color="k", capsize=3, label="Observado")
        ax.axvline(t_stop, color="gray", ls=":", lw=1, label=f"t_stop {t_stop:.1f} d")
        ax.set_title(f"Biomasa {dieta} | RMSE {rmse_b:.1f} mg PH")
        ax.set_xlabel("Día"); ax.legend(fontsize=7)
        for _, r in d.iterrows():
            filas.append(dict(dieta=dieta, dia=r["dia"], dia_montaje=r["dia_m"], neto=r["neto"],
                              lar_mod=float(np.interp(r["dia_m"], t, lar)), sat=r["sat"]))
        for _, r in g.iterrows():
            filas.append(dict(dieta=dieta, dia=r["dia"], ph_obs=r["ph_mg"],
                              ph_mod=float(np.interp(r["dia"], t, ph))))
    axes[0][0].set_ylabel("CO$_2$ (ppm/min)"); axes[1][0].set_ylabel("Masa (mg PH/larva)")
    fig.tight_layout()
    fig.savefig(HERE.parent / "figuras" / "calibracion_conjunta.png", dpi=150)
    pd.DataFrame(filas).to_csv(HERE / "resultados_conjunta.csv", index=False)
    print("\nFigura en figuras/calibracion_conjunta.png y tabla en simulacion/resultados_conjunta.csv")

if __name__ == "__main__":
    main()

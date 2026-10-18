#!/usr/bin/env python3
# validacion_biomasa.py - Validacion directa de biomasa: compara la trayectoria
# B+L del modelo (mg MS/larva) contra las curvas de crecimiento del ensayo
# paralelo (crecimiento_propio.csv, mg PH/larva).
# El estado inicial se ancla al peso medio del dia 0 (3.82 mg PH, 15 cajas):
# MS0 = 3.82 x 0.27 = 1.03 mg MS, repartido B0 = 0.8 MS0, L0 = 0.2 MS0.
# La conversion MS -> PH usa 27% MS nominal (rango 25-33% como banda),
# tomado de la literatura (Schon et al. 2026 reporta 26.3% al inicio y
# 27.7-32.7% al final para larvas de 5-6 d); no hay %MS propio por edad.
# Los demas parametros son los calibrados con los datos de gas (validacion_estatica).
from pathlib import Path
import io, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import validacion_estatica as V

HERE = Path(__file__).resolve().parent
MS_FRAC, MS_LO, MS_HI = 0.27, 0.25, 0.33   # fraccion MS nominal y banda
PH0 = 3.82                                 # mg PH/larva dia 0 (media de 15 cajas)
MS0 = PH0 * MS_FRAC                        # 1.03 mg MS/larva
B0A, L0A = 0.8 * MS0, 0.2 * MS0            # reparto nominal estructura/lipidos

def calibrar():
    datos, ayuno = V.cargar_datos()
    for d in ["D1", "D4"]:
        datos[d]["neto"] = (datos[d]["trat"] - datos[d]["ctrl"]).clip(lower=0)
    kmic, ymic, xmic = {}, {}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        for d in ["D1", "D4"]:
            dc = datos[d].dropna(subset=["ctrl"])
            kmic[d], ymic[d], xmic[d] = V.calibrar_micro(dc["dia"].values,
                                                         dc["ctrl"].values, d)
        B0, f4, tp1, tp4, wp = V.calibrar_conjunto(datos, kmic, ymic, xmic)
    return kmic, ymic, xmic, f4, tp1, tp4, wp

def simular_masa(dieta, tp, f, wp, kmic, ymic, xmic):
    pd_ = dict(V.P); pd_["tp"], pd_["f"], pd_["wp"] = tp, f, wp
    xb, xl, xx = V.X0["B"], V.X0["L"], V.X0["X"]
    V.X0["B"], V.X0["L"], V.X0["X"] = B0A, L0A, xmic[dieta]
    previo = V.poner_lecho(dieta)
    t, Y = V.simular(pd_, kmic[dieta], ymic[dieta])
    V.quitar_lecho(previo)
    V.X0["B"], V.X0["L"], V.X0["X"] = xb, xl, xx
    return t, Y[0] + Y[1]   # mg MS/larva

def main():
    kmic, ymic, xmic, f4, tp1, tp4, wp = calibrar()
    obs = pd.read_csv(HERE / "crecimiento_propio.csv", comment="#")
    print(f"Estado inicial anclado: MS0={MS0:.2f} mg (B0={B0A:.2f}, L0={L0A:.2f}), "
          f"MS%={MS_FRAC} (banda {MS_LO}-{MS_HI})")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    filas = []
    for ax, (dieta, tp, f) in zip(axes, [("D1", tp1, 1.0), ("D4", tp4, f4)]):
        t, ms = simular_masa(dieta, tp, f, wp, kmic, ymic, xmic)
        ph, ph_lo, ph_hi = ms / MS_FRAC, ms / MS_HI, ms / MS_LO
        d = obs[obs["dieta"] == dieta]
        mod_en_dias = np.interp(d["dia"], t, ph)
        rmse = float(np.sqrt(np.mean((mod_en_dias - d["ph_mg"])**2)))
        pico_m = float(t[np.argmax(ph)]); pico_o = float(d.loc[d["ph_mg"].idxmax(), "dia"])
        d14_m = float(np.interp(14, t, ph)); d14_o = float(d.loc[d["dia"] == 14, "ph_mg"].iloc[0])
        print(f"\n=== {dieta} (f={f:.3g}, t_p={tp:.2f} d) ===")
        print(f"  RMSE trayectoria: {rmse:.1f} mg PH | pico modelo dia {pico_m:.1f} vs obs dia {pico_o:.0f}"
              f" | dia 14: modelo {d14_m:.0f} vs obs {d14_o:.0f} mg PH")
        for _, r in d.iterrows():
            filas.append(dict(dieta=dieta, dia=r["dia"], obs=r["ph_mg"], de=r["de_mg"],
                              modelo=float(np.interp(r["dia"], t, ph))))
        ax.fill_between(t, ph_lo, ph_hi, color="tab:blue", alpha=0.15,
                        label=f"Banda MS {MS_LO}-{MS_HI}")
        ax.plot(t, ph, color="tab:blue", label="Modelo B+L (MS 27%)")
        ax.errorbar(d["dia"], d["ph_mg"], yerr=d["de_mg"], fmt="o", color="k",
                    capsize=3, label="Observado (media ± DE)")
        ax.axvline(tp, color="gray", ls=":", lw=1)
        ax.set_xlabel("Día desde el montaje"); ax.set_title(f"Dieta {dieta}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Masa larvaria (mg PH/larva)")
    fig.tight_layout()
    fig.savefig(HERE.parent / "figuras" / "validacion_biomasa.png", dpi=150)
    pd.DataFrame(filas).to_csv(HERE / "resultados_biomasa.csv", index=False)
    print("\nFigura en figuras/validacion_biomasa.png y tabla en simulacion/resultados_biomasa.csv")

if __name__ == "__main__":
    main()

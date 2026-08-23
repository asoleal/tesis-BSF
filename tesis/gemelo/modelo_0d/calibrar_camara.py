"""Calibracion del gemelo con experimentos de camara cerrada.
Lee datos/experimentos/{batch}/experimento_{D1,D4}[_alimento]/experimento_{co2,ch4}.csv,
ajusta pendiente de acumulacion (ppm/min) enmascarando saturacion del sensor,
convierte a flujo (mL/h) con V_aire = 11.4 L, y cruza larvas vs control."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuracion ---
EXP = Path("/mnt/Compartida/Descargas_HDD/tesis-doctorado/tesis-BSF/datos/experimentos")
V_AIRE_L = 11.4                      # panera 11.9 L - ~0.5 L de sustrato
ML_H_POR_PPPMIN = 60 * V_AIRE_L / 1000.0
SAT = {"co2": 4800, "ch4": 48000}    # mascara: 96 % del fondo de escala
BATCHES = ["9", "11", "13", "17"]
EXPS = ["experimento_D1", "experimento_D1_alimento",
        "experimento_D4", "experimento_D4_alimento"]

def cargar(f):
    df = pd.read_csv(f, sep="\t", header=None, names=["hora", "ppm"])
    df["ppm"] = pd.to_numeric(df["ppm"], errors="coerce")
    df = df.dropna()
    t = pd.to_timedelta(df["hora"]).dt.total_seconds() / 60
    return (t - t.iloc[0]).values, df["ppm"].values

filas, curvas = [], {}
for b in BATCHES:
    for e in EXPS:
        for gas in ["co2", "ch4"]:
            f = EXP / b / e / f"experimento_{gas}.csv"
            if not f.exists():
                continue
            t, ppm = cargar(f)
            m = ppm < SAT[gas]
            if m.sum() < 5:
                m = np.ones(len(ppm), bool)
            slope, ice = np.polyfit(t[m], ppm[m], 1)
            filas.append(dict(batch=b, exp=e, gas=gas,
                              pend_ppm_min=round(slope, 1),
                              flujo_mL_h=round(slope * ML_H_POR_PPPMIN, 1),
                              saturado=bool((ppm >= SAT[gas]).any()),
                              n_usados=int(m.sum()), n_total=len(ppm)))
            curvas[(b, e, gas)] = (t, ppm, m, slope)

res = pd.DataFrame(filas)
OUT = Path(__file__).resolve().parent.parent
(OUT / "datos").mkdir(exist_ok=True); (OUT / "resultados").mkdir(exist_ok=True)
res.to_csv(OUT / "datos" / "flujos_reales.csv", index=False)
print(res[res.gas == "co2"].to_string(index=False))

print("\nFlujo neto CO2 atribuible a larvas (mL/h):")
for b in BATCHES:
    for dia in ["D1", "D4"]:
        lar = res[(res.batch == b) & (res.exp == f"experimento_{dia}")
                  & (res.gas == "co2")]["flujo_mL_h"]
        ali = res[(res.batch == b) & (res.exp == f"experimento_{dia}_alimento")
                  & (res.gas == "co2")]["flujo_mL_h"]
        if len(lar) and len(ali):
            print(f"  batch {b:>2} {dia}: {lar.iloc[0] - ali.iloc[0]:7.1f} mL/h "
                  f"(larvas {lar.iloc[0]}, control {ali.iloc[0]})")

fig, axs = plt.subplots(4, 4, figsize=(14, 12))
for i, b in enumerate(BATCHES):
    for j, e in enumerate(EXPS):
        ax = axs[i, j]
        key = (b, e, "co2")
        if key not in curvas:
            ax.axis("off"); continue
        t, ppm, m, slope = curvas[key]
        ax.plot(t[m], ppm[m], ".", ms=3)
        ax.plot(t[~m], ppm[~m], ".", ms=3, color="r")
        tt = np.linspace(0, t.max(), 10)
        ax.plot(tt, np.polyval([slope, ppm[m].mean() - slope * t[m].mean()], tt),
                "-", lw=1, color="k")
        ax.set_title(f"b{b} {e.replace('experimento_','')}\n{slope:.0f} ppm/min",
                     fontsize=8)
        ax.grid(alpha=0.3)
fig.suptitle("Ajuste de acumulacion CO2 (rojo = saturacion enmascarada)")
fig.tight_layout()
fig.savefig(OUT / "resultados" / "calibracion_camara_co2.png", dpi=130)
print(f"\nCSV: {OUT/'datos'/'flujos_reales.csv'}")
print(f"Figura: {OUT/'resultados'/'calibracion_camara_co2.png'}")

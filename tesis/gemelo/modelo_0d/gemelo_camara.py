"""Gemelo 0D en modo camara cerrada: validacion contra datos reales
y tabla de diseno de ventilacion (que ACH evita saturar el MH-410D)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EXP = Path("/mnt/Compartida/Descargas_HDD/tesis-doctorado/tesis-BSF/datos/experimentos")
GEM = Path(__file__).resolve().parent.parent
V_AIRE_L = 11.4
C_AMB_PPM = 420.0
SAT_PPM = 4800.0

flujos = pd.read_csv(GEM / "datos" / "flujos_reales.csv")

def cargar(f):
    df = pd.read_csv(f, sep="\t", header=None, names=["hora", "ppm"])
    df["ppm"] = pd.to_numeric(df["ppm"], errors="coerce")
    df = df.dropna()
    t = pd.to_timedelta(df["hora"]).dt.total_seconds() / 60
    return (t - t.iloc[0]).values, df["ppm"].values

# --- validacion: gemelo (linea calibrada) vs datos ---
sel = flujos[(flujos.gas == "co2") & (~flujos.exp.str.contains("alimento"))].copy()
fig, axs = plt.subplots(3, 4, figsize=(14, 9))
r2s = []
for ax, (_, r) in zip(axs.flat, sel.iterrows()):
    t, ppm = cargar(EXP / str(r.batch) / r.exp / "experimento_co2.csv")
    modelo = ppm[0] + r.pend_ppm_min * t
    m = ppm < SAT_PPM
    ss_res = np.sum((ppm[m] - modelo[m])**2)
    ss_tot = np.sum((ppm[m] - ppm[m].mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2s.append(r2)
    ax.plot(t, ppm, ".", ms=3)
    ax.plot(t, modelo, "-", lw=1, color="r")
    ax.axhline(5000, ls="--", color="gray", lw=0.8)
    ax.set_title(f"b{r.batch} {r.exp.replace('experimento_','')}: R2={r2:.3f}", fontsize=8)
    ax.grid(alpha=0.3)
for ax in list(axs.flat)[len(sel):]:
    ax.axis("off")
fig.suptitle("Gemelo modo camara cerrada (rojo) vs datos reales (puntos)")
fig.tight_layout()
fig.savefig(GEM / "resultados" / "gemelo_vs_datos.png", dpi=130)

# --- tabla de diseno: CO2 estacionario segun ACH ---
print("CO2 estacionario esperado en cabecera con ventilacion continua [ppm]")
print("(flujo calibrado con larvas; ! = satura el MH-410D)\n")
print(f"{'batch':>5} {'dia':>4} {'mL/h':>7} | {'1':>7} {'2':>7} {'4':>7} {'8':>7} ACH")
for _, r in sel.iterrows():
    q = r.flujo_mL_h / 1000  # L/h
    est = [C_AMB_PPM + q / (ach * V_AIRE_L) * 1e6 for ach in [1, 2, 4, 8]]
    dia = r.exp.replace("experimento_", "")
    marca = ["!" if e > 5000 else " " for e in est]
    print(f"{r.batch:>5} {dia:>4} {r.flujo_mL_h:>7.1f} | "
          + " ".join(f"{e:6.0f}{m}" for e, m in zip(est, marca)))

print(f"\nR2 medio del gemelo en camara cerrada: {np.mean(r2s):.3f}")
print(f"Figura: {GEM/'resultados'/'gemelo_vs_datos.png'}")

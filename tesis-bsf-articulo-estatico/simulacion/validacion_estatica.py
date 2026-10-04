#!/usr/bin/env python3
# validacion_estatica.py - Modelo reducido (extension del DEB de Eriksen)
# validado con camara estatica. Nucleo DEB + interruptor de prepupa +
# modulo microbiano con crecimiento de biomasa + balance de gas.
# Carga unica de 250 g (100/150 g), sin reposicion.
# Experimento iniciado con ~700 huevos: B0 compartido entre dietas;
# calidad de dieta f (D1=1 referencia Eriksen, D4 calibrado) y t_p por dieta.
# Puntos saturados del NDIR = cotas inferiores (restriccion, no igualdad).
# Salidas: figuras/validacion_co2_D1.png, figuras/validacion_co2_D4.png,
#          simulacion/resultados_validacion.csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

RAIZ = Path(__file__).resolve().parent.parent
CSV = Path(__file__).resolve().parent / "datos_finales_PINN_corregidos.csv"

P = dict(amax=1.4, alpha=1.0, beta=2.0, YB=0.44, YL=0.42, m=0.08, Bmax=65.0,
         tp=12.0, wp=1.0, m_suelo=0.13, N=700, f=1.0,
         Vair=12.1, T=300.15, Patm=101325.0, Rg=8.314,
         Q10=2.0, Tref=25.0, Kth=0.15, kmax=1.0, gw=0.41, Ks=2.0, YX=0.4, Ksm=5.0)
TC = P["T"] - 273.15
X0 = dict(B=0.012, L=0.003, DM=100.0, W=150.0, X=1.0)  # mg/larva ; g de lecho
T_FIN, REPON = 18.0, 2.0
DIAS_MED = [9, 11, 13, 17]

def tasas(t, B, L, DM, W, X, p, kref, YCO2):
    S = 1.0 / (1.0 + np.exp(-(t - p["tp"]) / p["wp"]))
    a = (p["f"] * p["amax"] / (1.0 + (B / p["Bmax"]) ** p["alpha"])
         * (1.0 - S) * DM / (DM + p["Ks"]))
    rA = a * B
    G = max(1.0 - (B / p["Bmax"]) ** p["beta"], 0.0)
    muB = max((a - p["m"]) * G / (1.0 + p["YB"] * G), 0.0)
    rB = muB * B
    rCm = p["m"] * B * ((1.0 - S) + p["m_suelo"] * S)
    rL = rA - (1.0 + p["YB"]) * rB - rCm
    rCO2lar = rCm + p["YB"] * rB + p["YL"] * max(rL, 0.0)
    ths = W / (DM + W) if DM + W > 0 else 0.0
    k = min(kref * p["Q10"] ** ((TC - p["Tref"]) / 10.0) * ths / (ths + p["Kth"]),
            p["kmax"])
    rdeg = k * X * DM / (DM + p["Ksm"])  # g sustrato degradado / d
    return dict(rA=rA, rB=rB, rCm=rCm, rL=rL, rCO2lar=rCO2lar, k=k,
                rdeg=rdeg, rCmic=YCO2 * rdeg)

def rhs(t, y, p, kref, YCO2):
    B, L, DM, W, X = [max(v, 0.0) for v in y]
    r = tasas(t, B, L, DM, W, X, p, kref, YCO2)
    dL = r["rL"] if (L > 1e-12 or r["rL"] > 0) else 0.0
    dDM = -p["N"] * r["rA"] / 1000.0 - r["rdeg"]
    if DM <= 0:
        dDM = max(dDM, 0.0)
    dW = p["gw"] * (p["N"] * r["rCO2lar"] / 1000.0 + r["rCmic"])
    dX = p["YX"] * r["rdeg"]
    return [r["rB"], dL, dDM, dW, dX]

def simular(p, kref, YCO2, repon=False):
    y = [X0["B"], X0["L"], X0["DM"], X0["W"], X0["X"]]
    cortes = list(np.arange(REPON, T_FIN, REPON)) + [T_FIN] if repon else [T_FIN]
    ts, ys, t0 = [], [], 0.0
    for te in cortes:
        sol = solve_ivp(rhs, (t0, te), y, args=(p, kref, YCO2), method="LSODA",
                        t_eval=np.linspace(t0, te, 200), rtol=1e-6, atol=1e-9)
        ts.append(sol.t); ys.append(sol.y); y = list(sol.y[:, -1]); t0 = te
        if repon and te < T_FIN:
            y[2], y[3], y[4] = X0["DM"], X0["W"], X0["X"]
    return np.concatenate(ts), np.concatenate(ys, axis=1)

def ppm_min(mol_d):
    return mol_d / (P["Vair"] / 1000.0) * (P["Rg"] * P["T"] / P["Patm"]) * 1e6 / 1440.0

def componentes_ppm(t, Y, p, kref, YCO2):
    lar, mic = [], []
    for i, ti in enumerate(t):
        r = tasas(ti, *Y[:, i], p, kref, YCO2)
        lar.append(ppm_min(p["N"] * r["rCO2lar"] / 1000.0 / 44.0))
        mic.append(ppm_min(r["rCmic"] / 44.0))
    return np.array(lar), np.array(mic)

def curva_lar(dias, B0, tp, f, kref, YCO2, X0mic, wp=1.0):
    pd = dict(P); pd["tp"] = tp; pd["f"] = f; pd["wp"] = wp
    xb, xx = X0["B"], X0["X"]
    X0["B"], X0["X"] = B0, X0mic
    t, Y = simular(pd, kref, YCO2)
    lar, _ = componentes_ppm(t, Y, pd, kref, YCO2)
    X0["B"], X0["X"] = xb, xx
    return t, lar, np.interp(dias, t, lar)

def cargar_datos():
    df = pd.read_csv(CSV)
    df = df[(df["Gas"] == "co2") & (df["R2"] >= 0.5)]
    df = df.drop_duplicates(subset=["Dia_Experimento", "Etiqueta", "Tasa_Produccion"])
    datos = {}
    for dieta in ["D1", "D4"]:
        trat = df[df["Etiqueta"] == f"{dieta}_Tratamiento"]
        ctrl = df[df["Etiqueta"] == f"Control_Alimento_{dieta}"]
        filas = []
        for dia in DIAS_MED:
            tt = trat[trat["Dia_Experimento"] == dia]["Tasa_Produccion"]
            cc = ctrl[ctrl["Dia_Experimento"] == dia]["Tasa_Produccion"]
            sat = bool(trat[trat["Dia_Experimento"] == dia]["Saturado"].any())
            if len(tt) == 0 and len(cc) == 0:
                continue
            filas.append(dict(dia=dia,
                              trat=float(tt.mean()) if len(tt) else np.nan,
                              trat_min=float(tt.min()) if len(tt) else np.nan,
                              trat_max=float(tt.max()) if len(tt) else np.nan,
                              ctrl=float(cc.mean()) if len(cc) else np.nan,
                              sat=sat))
        datos[dieta] = pd.DataFrame(filas)
    ay = df[df["Etiqueta"] == "Larvas_Ayuno"]["Tasa_Produccion"]
    return datos, (float(ay.mean()) if len(ay) else np.nan)

def calibrar_micro(dias, ctrl_obs):
    p0 = dict(P); p0["N"] = 0
    def resid(logpar):
        kref, YCO2 = 10.0**logpar
        t, Y = simular(p0, kref, YCO2, repon=False)
        _, mic = componentes_ppm(t, Y, p0, kref, YCO2)
        return np.interp(dias, t, mic) - ctrl_obs
    sol = least_squares(resid, [np.log10(0.05), np.log10(0.3)], verbose=1,
                        bounds=([np.log10(1e-3), np.log10(0.05)],
                                [np.log10(1.0), np.log10(2.0)]))
    return 10.0**sol.x[0], 10.0**sol.x[1], X0["X"]

def calibrar_conjunto(datos, kmic, ymic, xmic):
    """B0 compartido (misma postura de huevos) + f_D4 (calidad de dieta)
    + t_p por dieta. Puntos saturados entran como restriccion (cota inferior)."""
    def resid(par):
        B0 = 10.0**par[0]; f4, tp1, tp4, wp = par[1], par[2], par[3], par[4]
        r = []
        for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
            d = datos[dieta].dropna(subset=["neto"])
            _, _, mod = curva_lar(d["dia"].to_numpy(float), B0, tp, f,
                                  kmic[dieta], ymic[dieta], xmic[dieta], wp)
            ri = mod - d["neto"].to_numpy(float)
            ri[d["sat"].to_numpy(bool)] = np.minimum(ri[d["sat"].to_numpy(bool)], 0.0)
            r.append(ri)
        print(".", end="", flush=True)
        return np.concatenate(r)
    mejor = None
    for x0 in [[np.log10(0.012), 0.8, 11.0, 11.0, 1.0],
               [np.log10(0.05), 0.5, 10.0, 12.0, 0.5],
               [np.log10(0.1), 0.7, 12.0, 10.5, 0.7]]:
        sol = least_squares(resid, x0, verbose=1,
                            bounds=([np.log10(0.008), 0.2, 9.0, 9.0, 0.1],
                                    [np.log10(0.5), 1.2, 17.0, 17.0, 2.0]))
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol
    return 10.0**mejor.x[0], mejor.x[1], mejor.x[2], mejor.x[3], mejor.x[4]

def r2(obs, mod):
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    return float(1 - np.sum((obs - mod)**2) / np.sum((obs - obs.mean())**2))

def rmse(obs, mod):
    return float(np.sqrt(np.mean((np.asarray(obs, float) - np.asarray(mod, float))**2)))

def evaluar(dieta, d, B0, tp, f, wp, kref, YCO2, X0mic, ayuno):
    d = d.copy()
    d["neto"] = (d["trat"] - d["ctrl"]).clip(lower=0)
    d["neto_lo"] = (d["trat_min"] - d["ctrl"]).clip(lower=0)
    d["neto_hi"] = (d["trat_max"] - d["ctrl"]).clip(lower=0)
    t, lar, mod_en_dias = curva_lar(d["dia"].to_numpy(float), B0, tp, f,
                                    kref, YCO2, X0mic, wp)
    pd = dict(P); pd["tp"] = tp; pd["f"] = f; pd["wp"] = wp
    xb, xx = X0["B"], X0["X"]
    X0["B"], X0["X"] = B0, X0mic
    tt, Y = simular(pd, kref, YCO2)
    lar_c, mic_c = componentes_ppm(tt, Y, pd, kref, YCO2)
    X0["B"], X0["X"] = xb, xx
    p0c = dict(pd); p0c["N"] = 0
    tc, Yc = simular(p0c, kref, YCO2)
    _, mic_ctrl = componentes_ppm(tc, Yc, p0c, kref, YCO2)
    d["lar_mod"] = mod_en_dias
    d["mic_mod"] = np.interp(d["dia"].to_numpy(float), tc, mic_ctrl)
    d["mic_trat"] = np.interp(d["dia"].to_numpy(float), tt, mic_c)
    d["tot_mod"] = d["lar_mod"] + d["mic_trat"]
    dn = d.dropna(subset=["neto"])
    limpios = dn[~dn["sat"]]
    print(f"\n=== Dieta {dieta} | k_ref={kref:.4g} d^-1, Y_CO2={YCO2:.3g}, "
          f"X0={X0mic:.3g} g, B0={B0:.3g} mg, f={f:.3g}, t_p={tp:.2f} d, "
          f"w_p={wp:.2f} d ===")
    print(d[["dia", "ctrl", "mic_mod", "neto", "lar_mod", "trat", "tot_mod", "sat"]].round(1).to_string(index=False))
    for _, r in dn[dn["sat"]].iterrows():
        margen = (r["lar_mod"] - r["neto"]) / r["neto"] * 100.0
        estado = "OK" if margen >= 0 else ("OK (dentro de tolerancia 3%)" if margen >= -3.0 else "NO CUMPLE")
        print(f"  cota inferior dia {int(r['dia'])}: modelo {r['lar_mod']:.1f} vs cota {r['neto']:.1f} ({margen:+.1f}%) -> {estado}")
    if len(limpios) >= 2:
        print(f"  R2 larvario (puntos limpios): {r2(limpios['neto'], limpios['lar_mod']):.3f} | "
              f"RMSE {rmse(limpios['neto'], limpios['lar_mod']):.1f} ppm/min")
    else:
        print(f"  puntos limpios: {len(limpios)} | residuos: "
              + ", ".join(f"dia {int(r['dia'])}: {r['lar_mod']-r['neto']:+.1f}" for _, r in limpios.iterrows()))
    dc = d.dropna(subset=["ctrl"])
    print(f"  R2 microbiano vs control: {r2(dc['ctrl'], dc['mic_mod']):.3f} | "
          f"RMSE {rmse(dc['ctrl'], dc['mic_mod']):.1f} ppm/min")
    d["dieta"] = dieta; d["B0"] = B0; d["f"] = f; d["tp"] = tp; d["wp"] = wp
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(tt, lar_c, color="tab:blue", label="Larvario (modelo)")
    ax.plot(tt, mic_c, color="tab:green", ls="--", label="Microbiano (modelo)")
    ax.plot(tt, lar_c + mic_c, color="tab:gray", lw=2, alpha=0.7, label="Total (modelo)")
    for mask, mk, lab in [(~d["sat"], "o", "Neto obs. (trat - control)"),
                          (d["sat"], "^", "Neto obs. saturado (cota inferior)")]:
        dd = d[mask]
        if len(dd):
            ax.errorbar(dd["dia"].to_numpy(), dd["neto"].to_numpy(),
                        yerr=[(dd["neto"] - dd["neto_lo"]).to_numpy(),
                              (dd["neto_hi"] - dd["neto"]).to_numpy()],
                        fmt=mk, color="tab:blue", capsize=3, label=lab)
    ax.plot(d["dia"].to_numpy(), d["ctrl"].to_numpy(), "s", color="tab:green",
            label="Control obs. (alimento solo)")
    ax.plot(d["dia"].to_numpy(), d["trat"].to_numpy(), "x", color="tab:gray",
            label="Tratamiento obs.")
    if dieta == "D4" and not np.isnan(ayuno):
        ax.plot([17], [ayuno], "d", color="tab:red", label="Ayuno obs. (larvas solas)")
    ax.set_xlabel("Dia"); ax.set_ylabel(r"Tasa de CO$_2$ (ppm/min)")
    ax.set_title(f"Validacion por fuentes - dieta {dieta}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(RAIZ / "figuras" / f"validacion_co2_{dieta}.png", dpi=150)
    plt.close(fig)
    return d

def main():
    datos, ayuno = cargar_datos()
    for dieta in ["D1", "D4"]:
        datos[dieta]["neto"] = (datos[dieta]["trat"] - datos[dieta]["ctrl"]).clip(lower=0)
    kmic, ymic, xmic = {}, {}, {}
    for dieta in ["D1", "D4"]:
        dc = datos[dieta].dropna(subset=["ctrl"])
        kmic[dieta], ymic[dieta], xmic[dieta] = calibrar_micro(dc["dia"].values,
                                                                dc["ctrl"].values)
    B0, f4, tp1, tp4, wp = calibrar_conjunto(datos, kmic, ymic, xmic)
    print(f"\n*** Ajuste conjunto: B0={B0:.3g} mg/larva (postura de huevos), "
          f"f_D4={f4:.3g} (D1=1 referencia), t_p D1={tp1:.2f} d, t_p D4={tp4:.2f} d, "
          f"w_p={wp:.2f} d ***")
    filas = []
    for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
        filas.append(evaluar(dieta, datos[dieta], B0, tp, f, wp,
                             kmic[dieta], ymic[dieta], xmic[dieta], ayuno))
    pd.concat(filas).to_csv(Path(__file__).resolve().parent / "resultados_validacion.csv", index=False)
    print("\nFiguras en figuras/ y tabla en simulacion/resultados_validacion.csv")

if __name__ == "__main__":
    main()

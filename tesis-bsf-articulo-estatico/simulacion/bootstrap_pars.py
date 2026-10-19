#!/usr/bin/env python3
# bootstrap_pars.py - Incertidumbre de los parametros de la calibracion conjunta
# por bootstrap de residuos: residuos crudos (obs - ajuste) remuestreados con
# reemplazo por dieta y bloque (gas / biomasa); datos sinteticos = ajuste +
# residuo permutado, y recalibracion completa de los 8 grados de libertad en
# cada replica (mismos arranques multiples de calibracion_conjunta.calibrar).
# Los puntos saturados conservan su tratamiento de cota inferior en la
# recalibracion. El modulo microbiano (k_ref, Y_CO2) queda fijo en su ajuste a
# los controles; delta y demas fijos, sin perturbar.
# Salida: simulacion/bootstrap_pars.csv (una fila por replica) y tabla de
# percentiles 2.5 / 50 / 97.5 por consola.
from pathlib import Path
import sys, time, os
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validacion_estatica as V
import calibracion_conjunta as C

N_BOOT, SEMILLA = 200, 10000
NOM = ["B0", "f_D4", "a_max", "B_max", "rho", "DM_f", "tau_D1", "tau_D4"]


def ajustados(datos, crec, kmic, ymic, xmic, pars):
    """Valores ajustados del caso base en cada punto de observacion."""
    B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = pars
    fit = {}
    for dieta, f, tau in [("D1", 1.0, tau1), ("D4", f4, tau4)]:
        p = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
        t, Y, pd_ = C.simular_dieta(dieta, p, kmic, ymic, xmic)
        lar, _ = V.componentes_ppm(t, Y, pd_, kmic[dieta], ymic[dieta])
        t_stop = C.t_stop_emergente(t, Y, pd_)
        d = datos[dieta].dropna(subset=["neto"])
        mod_gas = np.interp(d["dia"].to_numpy(float) - C.DELTA, t, lar)
        g = crec[crec["dieta"] == dieta]
        dmf = C.dm_frac(t, t_stop, dm_f, tau)
        mod_bio = np.interp(g["dia"].to_numpy(float), t, (Y[0] + Y[1]) / dmf)
        fit[dieta] = dict(gas_mod=mod_gas, bio_mod=mod_bio,
                          gas_obs=d["neto"].to_numpy(float),
                          bio_obs=g["ph_mg"].to_numpy(float),
                          idx_gas=d.index, idx_bio=g.index)
    return fit


def replica(args):
    """Una replica del bootstrap: datos sinteticos + recalibracion completa."""
    seed, datos, crec, fit, err, kmic, ymic, xmic = args
    try:
        rng = np.random.default_rng(seed)
        datos_b, crec_b = {}, crec.copy()
        for d in ["D1", "D4"]:
            db = datos[d].copy()
            eg = rng.choice(err[(d, "gas")], size=len(err[(d, "gas")]), replace=True)
            db.loc[fit[d]["idx_gas"], "neto"] = np.clip(fit[d]["gas_mod"] + eg, 0.0, None)
            datos_b[d] = db
            eb = rng.choice(err[(d, "bio")], size=len(err[(d, "bio")]), replace=True)
            crec_b.loc[fit[d]["idx_bio"], "ph_mg"] = np.clip(fit[d]["bio_mod"] + eb, 0.1, None)
        B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = C.calibrar(datos_b, crec_b, kmic, ymic, xmic)
        ts = {}
        for d, f, tau in [("D1", 1.0, tau1), ("D4", f4, tau4)]:
            p = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
            t, Y, pd_ = C.simular_dieta(d, p, kmic, ymic, xmic)
            ts[d] = C.t_stop_emergente(t, Y, pd_)
        return [B0, f4, amax, Bmax, rho, dm_f, tau1, tau4, ts["D1"], ts["D4"]]
    except Exception:
        return [np.nan] * 10


def main():
    t0 = time.time()
    datos, ayuno, kmic, ymic, xmic, crec = C.cargar_todo()
    pars = C.calibrar(datos, crec, kmic, ymic, xmic)
    print("Base: " + " ".join(f"{n}={v:.4g}" for n, v in zip(NOM, pars)), flush=True)
    fit = ajustados(datos, crec, kmic, ymic, xmic, pars)
    err = {(d, b): fit[d][f"{b}_obs"] - fit[d][f"{b}_mod"]
           for d in ["D1", "D4"] for b in ["gas", "bio"]}
    args = [(SEMILLA + i, datos, crec, fit, err, kmic, ymic, xmic)
            for i in range(N_BOOT)]
    nproc = max(1, (os.cpu_count() or 2) - 1)
    min_est = int(15 * N_BOOT / nproc / 60) + 1
    print(f"{N_BOOT} replicas en {nproc} procesos (~{min_est} min estimados)...", flush=True)
    with Pool(nproc) as pool:
        filas = pool.map(replica, args)
    cols = NOM + ["t_stop_D1", "t_stop_D4"]
    df = pd.DataFrame(filas, columns=cols)
    df.to_csv(HERE / "bootstrap_pars.csv", index=False)
    malas = int(df[NOM[0]].isna().sum())
    df = df.dropna()
    print(f"\nReplicas validas: {len(df)}/{N_BOOT} (fallidas: {malas})")
    print(f"a_max en su cota superior (1.5): {(df['a_max'] > 1.499).mean() * 100:.0f} % de las replicas")
    for d in ["D1", "D4"]:
        cierra = (df[f"t_stop_{d}"] < 17.5).mean() * 100
        print(f"cierre emergente dentro de la ventana en {d} (t_stop < 17.5 d): {cierra:.0f} % de las replicas")
    print("\nPercentiles del bootstrap (2.5 / 50 / 97.5):")
    for c in cols:
        q = df[c].quantile([0.025, 0.5, 0.975]).values
        print(f"  {c:>9}: {q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g}")
    print(f"\nTabla en simulacion/bootstrap_pars.csv ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()

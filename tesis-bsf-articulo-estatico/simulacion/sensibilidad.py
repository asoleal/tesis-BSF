#!/usr/bin/env python3
# sensibilidad.py - Sensibilidad uno-a-la-vez (+-20%) sobre los parametros
# FIJOS del nucleo (m, YB, N, t_pm), recalibrando el ajuste conjunto completo
# (8 gdl: B0, f_D4, a_max, B_max, rho, DM_f, tau1, tau4) en cada escenario.
import numpy as np
import validacion_estatica as V
import calibracion_conjunta as C

CLAVES = ["m", "YB", "N", "t_pm"]
FACTORES = [0.8, 1.2]

def metricas(datos, crec, kmic, ymic, xmic, sol):
    B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = sol
    out = {}
    for dieta, f, tau in [("D1", 1.0, tau1), ("D4", f4, tau4)]:
        pars = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
        t, Y, pd_ = C.simular_dieta(dieta, pars, kmic, ymic, xmic)
        lar, _ = V.componentes_ppm(t, Y, pd_, kmic[dieta], ymic[dieta])
        d = datos[dieta].dropna(subset=["neto"])
        obs = d["neto"].to_numpy(float)
        mod = np.interp(d["dia"].to_numpy(float) - C.DELTA, t, lar)
        limp = ~d["sat"].to_numpy(bool)
        r2 = (1.0 - np.sum((obs[limp] - mod[limp])**2)
              / np.sum((obs[limp] - obs[limp].mean())**2)) if limp.sum() >= 2 else float("nan")
        rmse_g = float(np.sqrt(np.mean((obs[limp] - mod[limp])**2))) if limp.sum() else float("nan")
        sat = d["sat"].to_numpy(bool)
        cota = float(np.min((mod[sat] - obs[sat]) / obs[sat] * 100.0)) if sat.sum() else float("nan")
        g = crec[crec["dieta"] == dieta]
        t_stop = C.t_stop_emergente(t, Y, pd_)
        dmf = C.dm_frac(t, t_stop, dm_f, tau)
        ph = (Y[0] + Y[1]) / dmf
        modg = np.interp(g["dia"].to_numpy(float), t, ph)
        rmse_b = float(np.sqrt(np.mean((modg - g["ph_mg"].to_numpy(float))**2)))
        out[dieta] = dict(r2=r2, rmse_g=rmse_g, cota=cota, rmse_b=rmse_b, t_stop=t_stop)
    return out

def correr(etiqueta):
    datos, ayuno, kmic, ymic, xmic, crec = C.cargar_todo()
    sol = C.calibrar(datos, crec, kmic, ymic, xmic)
    B0, f4, amax, Bmax, rho = sol[:5]
    m = metricas(datos, crec, kmic, ymic, xmic, sol)
    print("%-10s B0=%.3g f4=%.3g amax=%.3g Bmax=%.4g rho=%.3g | "
          "D1: RMSEg=%.1f RMSEb=%.1f cota=%.1f%% tstop=%.1f | "
          "D4: R2=%.3f RMSEg=%.1f RMSEb=%.1f cota=%.1f%% tstop=%.1f"
          % (etiqueta, B0, f4, amax, Bmax, rho,
             m["D1"]["rmse_g"], m["D1"]["rmse_b"], m["D1"]["cota"], m["D1"]["t_stop"],
             m["D4"]["r2"], m["D4"]["rmse_g"], m["D4"]["rmse_b"], m["D4"]["cota"], m["D4"]["t_stop"]),
          flush=True)

def main():
    faltan = [k for k in CLAVES if k not in V.P]
    if faltan:
        print("AVISO: no estan en V.P y se omiten:", faltan)
    correr("base")
    for k in CLAVES:
        if k not in V.P:
            continue
        v0 = V.P[k]
        for fct in FACTORES:
            V.P[k] = v0 * fct
            try:
                correr("%s x%.2g" % (k, fct))
            finally:
                V.P[k] = v0

if __name__ == "__main__":
    main()

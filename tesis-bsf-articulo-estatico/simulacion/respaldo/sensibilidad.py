#!/usr/bin/env python3
# sensibilidad.py - analisis de sensibilidad uno-a-la-vez (+-20%) de m, amax y N.
# Recalibra (B0, f_D4, t_p D1/D4, w_p) y reporta metricas por perturbacion.
import io, contextlib
import numpy as np
import validacion_estatica as V

def preparar():
    datos, _ = V.cargar_datos()
    for dieta in ["D1", "D4"]:
        datos[dieta]["neto"] = (datos[dieta]["trat"] - datos[dieta]["ctrl"]).clip(lower=0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kmic, ymic, xmic = {}, {}, {}
        for dieta in ["D1", "D4"]:
            dc = datos[dieta].dropna(subset=["ctrl"])
            kmic[dieta], ymic[dieta], xmic[dieta] = V.calibrar_micro(
                dc["dia"].values, dc["ctrl"].values)
    return datos, kmic, ymic, xmic

def metricas(datos, kmic, ymic, xmic, B0, f4, tp1, tp4, wp):
    fila = {}
    for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
        d = datos[dieta]
        _, _, mod = V.curva_lar(d["dia"].to_numpy(float), B0, tp, f,
                                kmic[dieta], ymic[dieta], xmic[dieta], wp)
        d = d.copy(); d["lar_mod"] = mod
        dn = d.dropna(subset=["neto"]); li = dn[~dn["sat"]]
        fila[f"R2_{dieta}"] = V.r2(li["neto"], li["lar_mod"]) if len(li) >= 2 else float("nan")
        fila[f"RMSE_{dieta}"] = V.rmse(li["neto"], li["lar_mod"]) if len(li) else float("nan")
        sa = dn[dn["sat"]]
        fila[f"margen_{dieta}"] = float((((sa["lar_mod"] - sa["neto"]) / sa["neto"]) * 100).min()) if len(sa) else float("nan")
    return fila

def correr(datos, kmic, ymic, xmic, par=None, fac=1.0):
    viejo = None
    if par:
        viejo = V.P[par]; V.P[par] = viejo * fac
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        B0, f4, tp1, tp4, wp = V.calibrar_conjunto(datos, kmic, ymic, xmic)
    if par:
        V.P[par] = viejo
    m = metricas(datos, kmic, ymic, xmic, B0, f4, tp1, tp4, wp)
    m.update(B0=B0, f4=f4, tp1=tp1, tp4=tp4, wp=wp)
    return m

def main():
    datos, kmic, ymic, xmic = preparar()
    casos = [("base", None, 1.0)]
    for par in ["m", "amax", "N"]:
        casos += [(f"{par} -20%", par, 0.8), (f"{par} +20%", par, 1.2)]
    filas = []
    for nombre, par, fac in casos:
        print(f"corriendo {nombre} ...", flush=True)
        filas.append((nombre, correr(datos, kmic, ymic, xmic, par, fac)))
    enc = ["caso", "B0", "f_D4", "tp_D1", "tp_D4", "w_p",
           "R2_D4", "RMSE_D4", "RMSE_D1", "cotD1%", "cotD4%"]
    print(("{:>10s}" + "{:>9s}" * 10).format(*enc))
    for nombre, m in filas:
        print("{:>10s}".format(nombre)
              + "{:>9.3g}{:>9.3g}{:>9.2f}{:>9.2f}{:>9.2f}".format(
                  m["B0"], m["f4"], m["tp1"], m["tp4"], m["wp"])
              + "{:>9.3f}{:>9.2f}{:>9.2f}{:>9.1f}{:>9.1f}".format(
                  m["R2_D4"], m["RMSE_D4"], m["RMSE_D1"],
                  m["margen_D1"], m["margen_D4"]))

if __name__ == "__main__":
    main()

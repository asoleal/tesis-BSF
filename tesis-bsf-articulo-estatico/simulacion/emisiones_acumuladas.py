#!/usr/bin/env python3
# emisiones_acumuladas.py - Emisiones acumuladas de CO2 del ciclo (0-15 d) en
# unidades de ingenieria: g por caja, g/kg de alimento fresco y g/kg de materia
# seca, con la separacion larvaria/microbiana, para el anclaje con factores de
# emision de la literatura (Ermolaev 2019: 96 g/kg fresco; Lindberg 2022:
# 47-147 g/kg fresco; Parodi 2020: fraccion microbiana ~34 %).
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validacion_estatica as V
import calibracion_conjunta as C

T_FIN_CICLO = 15.0  # el experimento se dio por finalizado el dia 15


def acumulada(dieta, f, pars, kmic, ymic, xmic, tfin=T_FIN_CICLO):
    B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = pars
    p = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
    t, Y, pd_ = C.simular_dieta(dieta, p, kmic, ymic, xmic)
    lar, mic = V.componentes_ppm(t, Y, pd_, kmic[dieta], ymic[dieta])
    m = t <= tfin
    conv = (V.P["Vair"] / 1000.0) * V.P["Patm"] / (V.P["Rg"] * V.P["T"]) / 1e6 * 1440.0  # ppm/min -> mol/d
    mol_lar = np.trapezoid(lar[m] * conv, t[m])
    mol_mic = np.trapezoid(mic[m] * conv, t[m])
    return 44.0 * mol_lar, 44.0 * mol_mic  # g CO2 por caja


def main():
    datos, ayuno, kmic, ymic, xmic, crec = C.cargar_todo()
    pars = C.calibrar(datos, crec, kmic, ymic, xmic)
    print("Base: B0=%.3g f=%.3g amax=%.3g Bmax=%.3g rho=%.3g DMf=%.3g tau=%.2g/%.2g" % tuple(pars))
    for dieta, f in [("D1", 1.0), ("D4", pars[1])]:
        gl, gm = acumulada(dieta, f, pars, kmic, ymic, xmic)
        dm_g, w_g = V.LECHO[dieta]
        fresco_kg = (dm_g + w_g) / 1000.0
        print(f"{dieta}: larvaria {gl:.1f} g, microbiana {gm:.1f} g, total {gl + gm:.1f} g/caja | "
              f"{(gl + gm) / fresco_kg:.1f} g/kg fresco | {(gl + gm) / (dm_g / 1000):.1f} g/kg MS | "
              f"fraccion microbiana {100 * gm / (gl + gm):.0f} %")


if __name__ == "__main__":
    main()

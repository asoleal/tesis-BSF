#!/usr/bin/env python3
# verificacion_pulso.py - Robustez de la aproximacion de carga agregada.
# El experimento cargo cada unidad con 500 g de alimento al montaje y 250 g
# adicionales el dia 5 del experimento (= dia 4 desde el montaje, siembra dia 1).
# El modelo las representa agregadas al montaje (750 g x MS bromatologica).
# Aqui la segunda carga entra como pulso explicito en T_PULSO: el lecho arranca
# con 2/3 de su materia seca y agua y el tercio restante se suma en t = 4 d.
# Se recalibra todo (modulo microbiano + 8 g.d.l. conjuntos) y se comparan.
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validacion_estatica as V
import calibracion_conjunta as C

T_PULSO, FRAC_INI = 4.0, 2.0 / 3.0  # pulso en t=4 d; 500 g de 750 g al montaje
NOM = ["B0", "f_D4", "a_max", "B_max", "rho", "DM_f", "tau_D1", "tau_D4"]


def simular_pulso(p, kref, YCO2, repon=False):
    """V.simular con la segunda carga como pulso en T_PULSO."""
    y = [V.X0["B"], V.X0["L"], V.X0["DM"] * FRAC_INI, V.X0["W"] * FRAC_INI, V.X0["X"]]
    ts, ys, t0 = [], [], 0.0
    for te, sumar in [(T_PULSO, True), (V.T_FIN, False)]:
        sol = solve_ivp(V.rhs, (t0, te), y, args=(p, kref, YCO2), method="LSODA",
                        t_eval=np.linspace(t0, te, 160), rtol=1e-6, atol=1e-9)
        ts.append(sol.t); ys.append(sol.y); y = list(sol.y[:, -1]); t0 = te
        if sumar:
            y[2] += V.X0["DM"] * (1.0 - FRAC_INI)
            y[3] += V.X0["W"] * (1.0 - FRAC_INI)
    return np.concatenate(ts), np.concatenate(ys, axis=1)


def corridas(etiqueta):
    datos, ayuno, kmic, ymic, xmic, crec = C.cargar_todo()
    pars = C.calibrar(datos, crec, kmic, ymic, xmic)
    B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = pars
    ts = {}
    for d, f, tau in [("D1", 1.0, tau1), ("D4", f4, tau4)]:
        p = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
        t, Y, pd_ = C.simular_dieta(d, p, kmic, ymic, xmic)
        ts[d] = C.t_stop_emergente(t, Y, pd_)
    print(f"[{etiqueta}] " + " ".join(f"{n}={v:.4g}" for n, v in zip(NOM, pars))
          + f" | k_ref {kmic['D1']:.4g}/{kmic['D4']:.4g}, Y_CO2 {ymic['D1']:.4g}/{ymic['D4']:.4g}"
          + f" | t_stop {ts['D1']:.2f}/{ts['D4']:.2f}", flush=True)
    return pars, kmic, ymic, ts


def main():
    base = corridas("AGREGADA")
    sim_original = V.simular
    V.simular = simular_pulso
    try:
        pulso = corridas("PULSO t=4")
    finally:
        V.simular = sim_original
    print("\nDesplazamiento relativo (pulso vs agregada):")
    for n, b, q in zip(NOM, base[0], pulso[0]):
        print(f"  {n:>7}: {b:.4g} -> {q:.4g}  ({100 * (q - b) / b:+.1f} %)")
    for i, d in enumerate(["D1", "D4"]):
        b, q = base[1][d], pulso[1][d]
        print(f"  k_ref {d}: {b:.4g} -> {q:.4g}  ({100 * (q - b) / b:+.1f} %)")
        b, q = base[2][d], pulso[2][d]
        print(f"  Y_CO2 {d}: {b:.4g} -> {q:.4g}  ({100 * (q - b) / b:+.1f} %)")
        b, q = base[3][d], pulso[3][d]
        print(f" t_stop {d}: {b:.2f} -> {q:.2f} d")


if __name__ == "__main__":
    main()

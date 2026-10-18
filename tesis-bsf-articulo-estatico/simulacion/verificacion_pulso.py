#!/usr/bin/env python3
# verificacion_pulso.py - Robustez ante la segunda carga de 250 g (dia 5).
import numpy as np
from scipy.integrate import solve_ivp
import validacion_estatica as V
import calibracion_conjunta as C

T_PULSO = 4.0  # "250 g al dia 5" contando la siembra como dia 1

def simular_pulso(p, kref, YCO2, repon=False):
    y = [V.X0["B"], V.X0["L"], V.X0["DM"] / 2.0, V.X0["W"] / 2.0, V.X0["X"]]
    ts, ys, t0 = [], [], 0.0
    for te in [T_PULSO, V.T_FIN]:
        sol = solve_ivp(V.rhs, (t0, te), y, args=(p, kref, YCO2), method="LSODA",
                        t_eval=np.linspace(t0, te, 200), rtol=1e-6, atol=1e-9)
        ts.append(sol.t); ys.append(sol.y); y = list(sol.y[:, -1]); t0 = te
        if te < V.T_FIN:
            y[2] += V.X0["DM"] / 2.0
            y[3] += V.X0["W"] / 2.0
    return np.concatenate(ts), np.concatenate(ys, axis=1)

def correr(etiqueta):
    datos, ayuno, kmic, ymic, xmic, crec = C.cargar_todo()
    par = C.calibrar(datos, crec, kmic, ymic, xmic)
    B0, f4, amax, Bmax, rho, dm_f, tau1, tau4 = par
    print(f"{etiqueta}: B0={B0:.3g}, f_D4={f4:.3g}, a_max={amax:.3g}, "
          f"B_max={Bmax:.3g}, rho={rho:.3g}, DM_f={dm_f:.3g}, tau={tau1:.2f}/{tau4:.2f}")
    print(f"   k_ref {kmic['D1']:.3g}/{kmic['D4']:.3g}, Y_CO2 {ymic['D1']:.3g}/{ymic['D4']:.3g}")
    for dieta, f in [("D1", 1.0), ("D4", f4)]:
        pars = dict(B0=B0, f=f, amax=amax, Bmax=Bmax, rho=rho)
        t, Y, pd_ = C.simular_dieta(dieta, pars, kmic, ymic, xmic)
        print(f"   {dieta}: t_stop emergente {C.t_stop_emergente(t, Y, pd_):.1f} d")
    return par

print("== Carga agregada (como esta en el articulo) ==")
base = correr("BASE ")
V.simular = simular_pulso
print(f"\n== Dos cargas: 250 g al montaje + 250 g en t={T_PULSO:.0f} d ==")
pulso = correr("PULSO")
print("\nDesplazamientos relativos:")
for n, b, q in zip(["B0","f_D4","a_max","B_max","rho","DM_f","tau_D1","tau_D4"], base, pulso):
    print(f"  {n}: {b:.4g} -> {q:.4g} ({100*(q-b)/b:+.1f} %)")

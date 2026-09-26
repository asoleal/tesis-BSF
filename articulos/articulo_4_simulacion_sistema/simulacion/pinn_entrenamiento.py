#!/usr/bin/env python3
"""pinn_entrenamiento.py — Entrenamiento piloto (FICTICIO) de la capa PINN.

Rol (i) de la Seccion 2.3 del articulo: correccion de la tasa total
de emision. red diminuta 4->8->1 (tanh) que predice log(delta) sobre el
estado del lecho; la prediccion corregida es exp(g)*(r_larva + r_mic).
Nota: el componente microbiano aislado NO es identificable con los datos
actuales (r_mic ~ 0 frente a r_larva en la replica); se corrige la tasa
total y se reporta la limitacion.

Datos:
  - propios: 8 tasas netas del experimento de septiembre 2025 (panera 12.1 L,
    N = 700), mismas del Cuadro 1 / validacion_experimento.py.
  - literatura SINTETICA: 48 puntos tipo respirometria larvaria (RANGO
    tomado de la literatura BSF) generados con semilla fija. Son un
    SUSTITUTO marcado: deben reemplazarse por datos digitalizados de
    Eriksen (2022, 2024) y similares antes de la version final.

Perdida: error de datos (ponderado) + cotas de delta (fisica) + tope de
carbono del ciclo (balance de masa). LOO diferido (parametros > datos).

Uso:  python3 simulacion/pinn_entrenamiento.py
Salida: imagenes/pinn_entrenamiento.pdf + resultados/pinn_resumen.md
"""
import os, sys, datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import bioconversion_ode as ode
    HAY_ODE = True
except Exception as e:
    print("AVISO: no se pudo importar bioconversion_ode (", e,
          ") - usando priors sinteticos para f0")
    HAY_ODE = False

rng = np.random.default_rng(7)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(BASE, '..', 'imagenes')
OUT_RES = os.path.join(BASE, '..', 'resultados')
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

# ---------------- configuracion de la replica (igual que validacion) --------
N0, VAIR_L, TF_D = 700, 12.1, 18.0
DM_REP, W_REP = 100.0, 150.0
DIAS_MED = (9, 11, 13)

if HAY_ODE:
    ode.FIS.update(Vair=VAIR_L*1e-3, Cth=300.0*VAIR_L/2.5, UA=2.0)
    y0 = [0.005, 0.012, 0.003, float(N0), DM_REP, W_REP, 27.0, 28.0,
          ode.wsat(28.0)*0.60, ode.ppm2c(420.0, 301.15),
          ode.ppm2c(209500.0, 301.15), ode.ppm2c(1.9, 301.15), 0.0, 0.0]
    vent = ode.calendario_cierres(TF_D, delta_min=15.0)
    segs = [(0.0, 9.0)] + [(float(a), float(b)) for a, b in
                           zip(DIAS_MED, tuple(DIAS_MED[1:]) + (TF_D,))]
    y = np.array(y0); sols = []
    for (a, b) in segs:
        if a > 0:
            y[4], y[5] = DM_REP, W_REP
        s = ode.solve_ivp(lambda t, z: ode.rhs(t, z, vent), (a*ode.DIA, b*ode.DIA),
                          y, method='LSODA', dense_output=True,
                          rtol=1e-6, atol=1e-9, max_step=120.0)
        sols.append(s); y = s.sol(b*ode.DIA)
    ctot = ode.FIS['P']/(ode.FIS['R']*301.15)
    FAC = 1e6/(44000.0*1440.0*ode.FIS['Vair']*ctot)  # mg/d totales -> ppm/min

    def estado_en(t_d):
        s = [s_ for s_ in sols
             if s_.t.min()/ode.DIA - 1e-9 <= t_d <= s_.t.max()/ode.DIA + 1e-9][0]
        return s.sol(t_d*ode.DIA)

    def f0(d):
        """ppm/min: componente larval y microbiana del modelo en el dia d."""
        Y = estado_en(d)
        rA, rB, rL, rCO2, rO2, Bmax = ode.tasas_larva(Y[1], d)
        r_lar = N0*rCO2*FAC
        rmm = ode.tasas_mic(Y[4], Y[5]/max(Y[4] + Y[5], 1e-9), Y[6], Y[3], rA)
        r_mic = rmm[0]*1000.0/44000.0/ode.FIS['Vair']/ode.DIA \
            * ode.FIS['R']*(Y[7] + 273.15)/ode.FIS['P']*1e6
        return r_lar, r_mic, Y
else:
    def estado_en(d):
        return np.array([0.0, 30.0, 0.0, float(N0), DM_REP, W_REP,
                         27.0, 28.0] + [0.0]*6)

    def f0(d):
        r_lar = 280.0*np.exp(-((d - 9.0)/4.5)**2)
        r_mic = 12.0*np.exp(-((d - 8.0)/5.0)**2)
        return r_lar, r_mic, estado_en(d)

# ---------------- datos propios (experimento septiembre 2025) ---------------
def tasas_propias():
    csv = os.path.join(BASE, '..', '..', '..', 'datos', 'experimentos',
                       'datos_finales_PINN_corregidos.csv')
    if not os.path.exists(csv):
        print('AVISO: no se encontro', csv, '- usando respaldo')
        return {('D1', 9): [226.8, 308.8], ('D4', 9): [88.6],
                ('D1', 11): [124.3], ('D4', 11): [165.4],
                ('D1', 13): [102.8], ('D4', 13): [21.5],
                ('D4', 17): [0.0], ('AY', 17): [14.3]}
    import csv as _csv
    filas = [r for r in _csv.DictReader(open(csv))
             if r['Gas'] == 'co2' and float(r['R2']) >= 0.5]
    def tasa(etq, dia):
        v = [float(r['Tasa_Produccion']) for r in filas
             if r['Etiqueta'] == etq and int(r['Dia_Experimento']) == dia]
        return float(np.mean(v)) if v else None
    obs = {}
    for d in (9, 11, 13, 17):
        for ali in ('D1', 'D4'):
            tr, ct = tasa(f'{ali}_Tratamiento', d), tasa(f'Control_Alimento_{ali}', d)
            if tr is None:
                continue
            obs.setdefault((ali, d), []).append(max(tr - (ct or 0.0), 0.0))
    return obs

obs = tasas_propias()
Xown, Yown = [], []
for (ali, d), v in sorted(obs.items()):
    Y = estado_en(float(d))
    for r in v:
        Xown.append([float(d), Y[4], Y[5], Y[6]])
        Yown.append(r)
Xown, Yown = np.array(Xown), np.array(Yown)

# ---------------- datos de literatura SINTETICOS (sustituto marcado) --------
# Correcciones delta ~ lognormal(0, 0.35) sobre la tasa total del modelo.
# REEMPLAZAR por digitalizacion de Eriksen 2022/2024 y similares.
N_LIT = 48
d_lit = rng.uniform(3.0, 17.0, N_LIT)
dm_lit = rng.uniform(60.0, 140.0, N_LIT)
w_lit = 1.5*dm_lit + rng.normal(0.0, 15.0, N_LIT)
ts_lit = rng.uniform(24.0, 30.0, N_LIT)
delta_lit = np.exp(rng.normal(0.0, 0.35, N_LIT))
Xlit = np.column_stack([d_lit, dm_lit, w_lit, ts_lit])
f0_lit = np.array([[f0(d)[0], f0(d)[1]] for d in d_lit])
Ylit = delta_lit*(f0_lit[:, 0] + f0_lit[:, 1]) * np.exp(rng.normal(0.0, 0.08, N_LIT))

# ---------------- red diminuta 4->8->1 (numpy) ------------------------------
NH = 8
NP_W = 4*NH
NP_B = NH
NP_V = NH
NP_C = 1
NPAR = NP_W + NP_B + NP_V + NP_C

xm = np.vstack([Xlit, Xown]).mean(axis=0)
xs = np.vstack([Xlit, Xown]).std(axis=0) + 1e-9

def g(X, p):
    Z = (X - xm)/xs
    W1 = p[:NP_W].reshape(NH, 4)
    b1 = p[NP_W:NP_W + NP_B]
    W2 = p[NP_W + NP_B:NP_W + NP_B + NP_V]
    b2 = p[-1]
    return W2 @ np.tanh(W1 @ Z.T + b1[:, None]) + b2

def pred(X, p):
    f = np.array([[f0(x[0])[0], f0(x[0])[1]] for x in X])
    return np.exp(g(X, p))*(f[:, 0] + f[:, 1])

LOG_D_MAX = np.log(10.0)      # cota |delta| <= 10 en los datos
LOG_D_MAX_G = np.log(3.0)     # cota global |delta| <= 3 fuera del soporte
CARBON_MULT = 2.0             # tope: integral corregida <= CARBON_MULT * baseline
LAM_B, LAM_BG, LAM_C = 10.0, 10.0, 1.0

def carbono_ciclo(p):
    """CO2 total acumulado del ciclo (mol): baseline, corregido y g en malla."""
    dias = np.linspace(1.0, TF_D - 1.0, 34)
    f = np.array([[f0(d)[0], f0(d)[1]] for d in dias])
    XXc = np.column_stack([dias, np.full_like(dias, DM_REP),
                           np.full_like(dias, W_REP), np.full_like(dias, 27.0)])
    gvals = g(XXc, p)
    if HAY_ODE:
        conv = 1.0/FAC/44000.0
    else:
        conv = 2000.0/44000.0
    base = np.trapezoid((f[:, 0] + f[:, 1])*conv, dias)
    corr = np.trapezoid(np.exp(gvals)*(f[:, 0] + f[:, 1])*conv, dias)
    return base, corr, gvals

def perdida(p, X, y, sig):
    r = (pred(X, p) - y)/sig
    j_data = float(r @ r)
    j_bnd = float(np.sum(np.maximum(np.abs(g(X, p)) - LOG_D_MAX, 0.0)**2))
    base, corr, ggrid = carbono_ciclo(p)
    j_bg = float(np.sum(np.maximum(np.abs(ggrid) - LOG_D_MAX_G, 0.0)**2))
    j_c = max(corr - CARBON_MULT*base, 0.0)**2
    return j_data + LAM_B*j_bnd + LAM_BG*j_bg + LAM_C*j_c

# ---------------- entrenamiento: pre-train (lit) -> fine-tune (propios) -----
p0 = rng.normal(0.0, 0.3, NPAR)
hist_pre, hist_fin = [], []

cb_pre = lambda p: hist_pre.append(perdida(p, Xlit, Ylit, 0.15*Ylit + 5.0))
r1 = minimize(perdida, p0, args=(Xlit, Ylit, 0.15*Ylit + 5.0),
              method='L-BFGS-B', options={'maxiter': 400, 'ftol': 1e-10},
              callback=cb_pre)
p1 = r1.x

sig_own = 0.2*Yown + 5.0
cb_fin = lambda p: hist_fin.append(perdida(p, Xown, Yown, sig_own))
r2 = minimize(perdida, p1, args=(Xown, Yown, sig_own),
              method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-10},
              callback=cb_fin)
p2 = r2.x

# ---------------- leave-one-day-out ------------------------------------------
# Con NPAR = 49 parametros y <= 8 puntos por pliegue el LOO es vacuo (la red
# interpola exacto y el error de test no mide generalizacion); se difiere a
# la fase de experimentos finales. Lo honesto es reportarlo como limitacion.
loo_aplicable = len(Yown) > 2*NPAR
loo = []
if loo_aplicable:
    for d in (9, 11, 13, 17):
        mtr = Xown[:, 0] != d
        rr = minimize(perdida, p1, args=(Xown[mtr], Yown[mtr], sig_own[mtr]),
                      method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10})
        e = (pred(Xown[~mtr], rr.x) - Yown[~mtr])/sig_own[~mtr]
        loo.append((d, float(e @ e), int((~mtr).sum())))
else:
    print(f"LOO no aplicable: {NPAR} parametros vs <= {len(Yown) - 1} puntos "
          "por pliegue; se difiere a experimentos finales")

# ---------------- reporte ---------------------------------------------------
print("== entrenamiento piloto PINN (FICTICIO, datos literatura sinteticos) ==")
print(f"parametros: {NPAR} | pre-train J={r1.fun:.2f} | fine-tune J={r2.fun:.2f}")
_base, _corr, _gg = carbono_ciclo(p2)
print(f"CO2 ciclo: baseline {_base:.2f} mol | corregido {_corr:.2f} mol | tope {CARBON_MULT} x baseline")
print(f"{'dia':>4} {'f0_total':>9} {'delta':>7} {'corr':>8} {'obs':>8}")
f0o = np.array([[f0(x[0])[0], f0(x[0])[1]] for x in Xown])
dlt = np.exp(g(Xown, p2))
for i in range(len(Xown)):
    print(f"{Xown[i,0]:>4.0f} {f0o[i,0] + f0o[i,1]:>9.1f} {dlt[i]:>7.2f} "
          f"{dlt[i]*(f0o[i,0] + f0o[i,1]):>8.1f} {Yown[i]:>8.1f}")
for d, e, n in loo:
    print(f"LOO dia {d}: J={e:.1f} ({n} punto(s))")
if not loo_aplicable:
    print("(LOO diferido: ver notas pendientes)")

with open(os.path.join(OUT_RES, 'pinn_resumen.md'), 'w') as fh:
    fh.write("# Resumen entrenamiento piloto PINN (FICTICIO)\n\n")
    fh.write(f"Generado: {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
    fh.write(f"- Red: 4->{NH}->1 tanh, {NPAR} parametros, numpy + L-BFGS-B\n")
    fh.write(f"- Pre-train: {N_LIT} puntos SINTETICOS tipo literatura (semilla 7). "
             "REEMPLAZAR por digitalizacion Eriksen 2022/2024\n")
    fh.write(f"- Fine-tune: {len(Yown)} tasas propias (septiembre 2025)\n")
    fh.write(f"- J pre-train = {r1.fun:.2f} | J fine-tune = {r2.fun:.2f}\n")
    fh.write(f"- delta_medio = {np.mean(dlt):.2f} (rango {dlt.min():.2f}-{dlt.max():.2f})\n")
    fh.write(f"- CO2 ciclo: baseline {_base:.2f} mol, corregido {_corr:.2f} mol "
             f"(tope {CARBON_MULT}x baseline)\n")
    if loo_aplicable:
        fh.write("- LOO: " + "; ".join(f"dia {d} J={e:.1f}" for d, e, n in loo) + "\n")
    else:
        fh.write("- LOO: NO aplicable (parametros > datos por pliegue); diferido\n")
    fh.write("\n## Notas pendientes\n")
    fh.write("1. Reemplazar datos sinteticos por datos digitalizados de literatura.\n")
    fh.write("2. Reentrenar con curvas de control completas (S8) al llegar experimentos finales.\n")
    fh.write("3. Fijar LAM_B, LAM_BG y LAM_C por validacion cruzada.\n")
    fh.write("4. Con n pequeno el ajuste es pobre por construccion: el valor esta en la arquitectura de perdida.\n")
    fh.write("5. El componente microbiano aislado no es identificable (r_mic ~ 0 en la replica); se corrige la tasa total.\n")
print('resumen:', os.path.join(OUT_RES, 'pinn_resumen.md'))

# ---------------- figura ----------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
ax[0].plot(hist_pre, lw=1.4, label='pre-train (lit. sintetica)')
ax[0].plot(np.arange(len(hist_pre), len(hist_pre) + len(hist_fin)),
           hist_fin, lw=1.4, label='fine-tune (propios)')
ax[0].set_xlabel('evaluaciones'); ax[0].set_ylabel('J')
ax[0].set_yscale('log'); ax[0].set_title('(a) Perdida')
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)

ax[1].plot(Ylit, pred(Xlit, p2), '.', color='gray', ms=4,
           label='literatura sintetica')
ax[1].plot(Yown, pred(Xown, p2), 'o', color='tab:blue', ms=8,
           label='experimentos propios')
lim = [0, max(Ylit.max(), Yown.max())*1.1]
ax[1].plot(lim, lim, 'k--', lw=1)
ax[1].set_xlabel('observado (ppm/min)'); ax[1].set_ylabel('corregido (ppm/min)')
ax[1].set_title('(b) Paridad prediccion vs observacion')
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)

dd = np.linspace(1, 18, 60)
XX = np.column_stack([dd, np.full_like(dd, DM_REP), np.full_like(dd, W_REP),
                      np.full_like(dd, 27.0)])
ax[2].plot(dd, np.exp(g(XX, p2)), lw=1.8, color='tab:red')
for d in (9, 11, 13, 17):
    ax[2].axvline(d, color='gray', ls=':', lw=0.8)
ax[2].set_xlabel('dia del ciclo'); ax[2].set_ylabel('delta')
ax[2].set_title('(c) Factor de correccion aprendido')
ax[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'pinn_entrenamiento.pdf'))
print('figura:', os.path.join(OUT_FIG, 'pinn_entrenamiento.pdf'))

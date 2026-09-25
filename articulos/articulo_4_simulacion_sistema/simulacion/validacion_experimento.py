#!/usr/bin/env python3
"""validacion_experimento.py — Replica del experimento preliminar (panera)
y figura modelo-vs-datos para el articulo.

Condiciones: N=700, alimento 250 g repuesto en cada medicion (ad libitum, S3),
V_air = 12.1 L (panera 30x19 -> 36x25 x 17 cm, S1), dias de medicion 9/11/13/17.
Compara la tasa LARVAL neta del gemelo contra (tratamiento - control) medido.
El modulo microbiano NO entra a esta figura (subestima 5-20x, pendiente S8).

Uso:  python3 simulacion/validacion_experimento.py
Salida: imagenes/validacion_preliminar.pdf + tabla en consola.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bioconversion_ode as ode

# ---------------- configuracion de la replica (supuestos.md) ----------------
N0     = 700                 # S2
VAIR_L = 12.1                # S1
TF_D   = 18.0
DM_REP, W_REP = 100.0, 150.0 # S3: 250 g totales (40/60)
DIAS_MED = (9, 11, 13)       # reposicion de alimento (protocolo)

ode.FIS.update(Vair=VAIR_L*1e-3, Cth=300.0*VAIR_L/2.5, UA=2.0)

# ---------------- tasas observadas netas (repo de datos) ----------------
def tasas_observadas():
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', '..', 'datos', 'experimentos',
                       'datos_finales_PINN_corregidos.csv')
    if not os.path.exists(csv):          # respaldo si el CSV se movio
        print('AVISO: no se encontro', csv, '- usando valores de respaldo')
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
            if tr is None: continue
            net = tr - ct if ct is not None else tr
            obs.setdefault((ali, d), []).append(max(net, 0.0))
        ay = tasa('Larvas_Ayuno', d)
        if ay is not None:
            obs[('AY', d)] = [ay]
    return obs

# ---------------- integracion por segmentos (reposiciones) ----------------
y0 = [0.005, 0.012, 0.003, float(N0), DM_REP, W_REP, 27.0, 28.0,
      ode.wsat(28.0)*0.60, ode.ppm2c(420.0, 301.15), ode.ppm2c(209500.0, 301.15),
      ode.ppm2c(1.9, 301.15), 0.0, 0.0]
vent = ode.calendario_cierres(TF_D, delta_min=15.0)
segs = [(0.0, 9.0)] + [(float(a), float(b))
                       for a, b in zip(DIAS_MED, tuple(DIAS_MED[1:]) + (TF_D,))]
y = np.array(y0); sols = []
for (a, b) in segs:
    if a > 0:
        y[4], y[5] = DM_REP, W_REP        # reposicion alimento
    s = ode.solve_ivp(lambda t, z: ode.rhs(t, z, vent), (a*ode.DIA, b*ode.DIA), y,
                      method='LSODA', dense_output=True,
                      rtol=1e-6, atol=1e-9, max_step=120.0)
    sols.append(s); y = s.sol(b*ode.DIA)

ctot = ode.FIS['P']/(ode.FIS['R']*301.15)
FAC = 1e6/(44000.0*1440.0*ode.FIS['Vair']*ctot)   # mg/d totales -> ppm/min

def modelo_en(t_d):
    s = [s_ for s_ in sols
         if s_.t.min()/ode.DIA - 1e-9 <= t_d <= s_.t.max()/ode.DIA + 1e-9][0]
    Y = s.sol(t_d*ode.DIA)
    rA, rB, rL, rCO2, rO2, Bmax = ode.tasas_larva(Y[1], t_d)
    return N0*rCO2*FAC, Y

tt = np.linspace(0.2, TF_D - 0.1, 800)
mod = np.array([modelo_en(x)[0] for x in tt])
Bm  = np.array([modelo_en(x)[1][1] for x in tt])
S_lecho = []
for x in tt:
    s = [s_ for s_ in sols
         if s_.t.min()/ode.DIA - 1e-9 <= x <= s_.t.max()/ode.DIA + 1e-9][0]
    Y = s.sol(x*ode.DIA); S_lecho.append(Y[4] + Y[5])

obs = tasas_observadas()

# ---------------- tabla ----------------
print(f"{'dia':>4} {'modelo':>8}  observado neto (trat - control, ppm/min)")
for d in (9, 11, 13, 17):
    m = modelo_en(float(d))[0]
    det = ' | '.join(f"{k[0]}: {min(v):.0f}-{max(v):.0f}" if len(v) > 1
                     else f"{k[0]}: {v[0]:.0f}" for k, v in sorted(obs.items())
                     if k[1] == d)
    print(f"{d:>4} {m:>8.0f}  {det}")

# ---------------- figura ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
MK = {'D1': ('o', 'tab:blue', 'alimento D1'),
      'D4': ('^', 'tab:orange', 'alimento D4'),
      'AY': ('*', 'tab:green', 'ayuno (solo larvas)')}
for k, v in sorted(obs.items()):
    ali, d = k
    mk, col, lb = MK[ali]
    ax[0].plot([d]*len(v), v, mk, color=col, ms=9 if ali != 'AY' else 13,
               ls='none', label=lb)
ax[0].plot(tt, mod, color='k', lw=1.8, label='gemelo 0D (larval neto)')
ax[0].set_xlabel('dia del ciclo'); ax[0].set_ylabel('tasa neta de CO2 (ppm/min)')
ax[0].set_title('(a) Validacion: modelo vs experimento (panera 12.1 L)')
ax[0].grid(alpha=0.3); ax[0].set_xlim(0, TF_D)
by_lbl = {}
for h, l in zip(*ax[0].get_legend_handles_labels()):
    by_lbl.setdefault(l, h)
ax[0].legend(by_lbl.values(), by_lbl.keys(), fontsize=8, loc='upper left')

ax[1].plot(tt, Bm, color='tab:blue', lw=1.8, label='B (biomasa estructural)')
ax[1].plot(tt, np.array(S_lecho)/10, color='tab:red', lw=1.4, label='S/10 (lecho)')
ax[1].set_xlabel('dia del ciclo'); ax[1].set_ylabel('mg / (g/10)')
ax[1].set_title('(b) Trayectorias del gemelo (sin verificar: no hay biomasa medida)')
ax[1].grid(alpha=0.3); ax[1].set_xlim(0, TF_D); ax[1].legend(fontsize=8)
for a in DIAS_MED:
    ax[1].axvline(a, color='gray', ls=':', lw=0.8)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'imagenes')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'validacion_preliminar.pdf'))
print('figura:', os.path.join(out, 'validacion_preliminar.pdf'))

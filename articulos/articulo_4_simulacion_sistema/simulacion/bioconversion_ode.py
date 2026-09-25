#!/usr/bin/env python3
"""
bioconversion_ode.py - Gemelo digital 0D del sistema de bioconversion.
Estados: A,B,L (mg/larva), N, DM (materia seca lecho, g), W (agua lecho, g),
Ts, T (°C), w (kg/kg), cCO2,cO2,cCH4 (mol/m3), IT,IH (integradores PI).
Frass: la fraccion no asimilada del sustrato permanece en el lecho (Obs. 5.3).
Salida: figuras PDF en ../imagenes/ + resumen en consola.
"""
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'imagenes')

PHY = dict(amax=1.4, alpha=1.0, beta=2.0, YB=0.44, YL=0.42, m=0.08,
           Bmax0=65.0, rho=1.0, tp_min=13.0, RQ=0.9)
MIC = dict(kref=3.0e-3, kmax=2.0e-2, Q10=2.0, Tref=25.0, Kth=0.15,
           YCO2=0.30, RQmic=0.85, YCH4=5.0e-4,
           xi_max=0.12, th_cs=0.55, s_th=0.05, a_biot=5e-4)
FIS = dict(Vair=2.5e-3, P=101325.0, R=8.314, keAs=1.0e-5, lam=2.45e6,
           Cths=2000.0, Cth=300.0, hsAs=2.0, UA=0.8, gammaq=470e3,
           gammaw=0.41, Pmax=15.0, Jmax=5e-6,
           Ta=25.0, Tref=28.0, HRref=0.70, rhoair=1.18, cpair=1005.0)
DIA = 86400.0

def esat(T):
    return 610.94*np.exp(17.625*T/(T+243.04))
def wsat(T):
    e = esat(T)
    return 0.622*e/(FIS['P']-e)
def c2ppm(c, TK):
    return c*FIS['R']*TK/FIS['P']*1e6
def ppm2c(ppm, TK):
    return ppm*1e-6*FIS['P']/(FIS['R']*TK)

def calendario_cierres(tf_d, delta_min=3.0):
    vent = []
    t = 0.25*DIA
    d = delta_min*60.0
    while t < tf_d*DIA:
        vent.append((t, t+d))
        t += (24.0 if 3.0 <= t/DIA <= 10.0 else 48.0)*3600.0
    return vent

def caudal(t, ventanas):
    for a, b in ventanas:
        if a <= t < b:
            return 0.0
    return 1.6667e-5          # 1 L/min en m3/s

def calendario_adaptativo(tf_d, sol, cmax=5000.0):
    """Delta_k: tiempo hasta acercarse a la saturacion del NDIR (0-5000 ppm),
    acotado a la ventana operativa de 2 a 30 min (2.5 L); 10-30 min a escala panera (dos pasadas)."""
    vent, deltas = [], []
    t = 0.25*DIA
    while t < tf_d*DIA:
        Y = sol.sol(t)
        rA, rB, rL, rCO2, rO2, Bmax = tasas_larva(Y[1], t/DIA)
        ths = Y[5]/max(Y[4] + Y[5], 1e-9)
        rCmic, rOmic, rCH4g, kmic = tasas_mic(Y[4], ths, Y[6], Y[3], rA)
        Rmol = (Y[3]*rCO2 + rCmic*1000.0)/44000.0          # mol/d
        TK = Y[7] + 273.15
        ppm_s = Rmol/FIS['Vair']/DIA * FIS['R']*TK/FIS['P'] * 1e6
        margen = max(cmax - c2ppm(Y[9], TK), 50.0)        # ppm
        dk = min(max(margen/max(ppm_s, 1e-9), 120.0), 1800.0)
        tm = t + dk/2.0
        Ym = sol.sol(tm)
        rm = tasas_larva(Ym[1], tm/DIA)
        rmm = tasas_mic(Ym[4], Ym[5]/max(Ym[4] + Ym[5], 1e-9), Ym[6], Ym[3], rm[0])
        ppm_sm = (Ym[3]*rm[3] + rmm[0]*1000.0)/44000.0/FIS['Vair']/DIA*FIS['R']*(Ym[7] + 273.15)/FIS['P']*1e6
        dk = min(max(margen/max(ppm_sm, 1e-9), 120.0), 1800.0)
        vent.append((t, t + dk)); deltas.append(dk)
        t += (24.0 if 3.0 <= t/DIA <= 10.0 else 48.0)*3600.0
    return vent, deltas

PREP = dict(t_p=12.0, ancho=1.0, m_suelo=0.13)  # S5: switch prepupa (supuestos.md)

def tasas_larva(B, t_d):
    Bmax = PHY['Bmax0'] if t_d <= PHY['tp_min'] else \
           max(PHY['Bmax0']-PHY['rho']*(t_d-PHY['tp_min']), 1.0)
    S_prep = 1.0/(1.0+np.exp(-(t_d-PREP['t_p'])/PREP['ancho']))  # S5
    a = PHY['amax']/(1.0+(B/Bmax)**PHY['alpha'])*(1.0-S_prep)      # d^-1 (Ec. 5)
    rA = a*B                                           # asimilacion mg/d (Ec. 4)
    logistic = max(1.0-(B/Bmax)**PHY['beta'], 0.0)     # Ec. 10
    muB = max((a-PHY['m'])*logistic/(1.0+PHY['YB']*logistic), 0.0)
    rB = muB*B
    rCm = PHY['m']*B*((1.0-S_prep)+PREP['m_suelo']*S_prep)
    rCB = PHY['YB']*rB
    rL = rA - rB - rCm - rCB                           # Ec. 12 (A en SS)
    rCL = PHY['YL']*rL if rL > 0 else 0.0
    rCO2 = rCm + rCB + rCL                             # mg CO2/larva/d
    return rA, rB, rL, rCO2, rCO2/PHY['RQ'], Bmax

def tasas_mic(DM, ths, Ts, N, rA):
    kmic = min(MIC['kref']*MIC['Q10']**((Ts-MIC['Tref'])/10.0)
               * ths/(ths+MIC['Kth']), MIC['kmax'])
    xi = MIC['xi_max']/(1.0+np.exp(-(ths-MIC['th_cs'])/MIC['s_th'])) \
         / (1.0+MIC['a_biot']*N*rA)
    return MIC['YCO2']*kmic*DM, MIC['YCO2']*kmic*DM/MIC['RQmic'], \
           MIC['YCH4']*xi*kmic*DM, kmic

def rhs(t, y, ventanas):
    A, B, L, N, DM, W, Ts, T, w, cCO2, cO2, cCH4, IT, IH = y
    t_d = t/DIA
    rA, rB, rL, rCO2, rO2, Bmax = tasas_larva(B, t_d)
    S = DM + W
    ths = W/max(S, 1e-9)
    rCmic, rOmic, rCH4g, kmic = tasas_mic(DM, ths, Ts, N, rA)
    Q = caudal(t, ventanas)
    phi = min(ths/0.60, 1.0)
    E = FIS['keAs']*max(phi*wsat(Ts)-w, 0.0)           # kg/s
    comida = 0.0 if DM <= 1e-9 else N*rA/1000.0        # S3: tope por DM
    DM_p = (-comida - kmic*DM)/DIA                # g/s (frass queda)
    W_p = -E/DIA
    RCO2_mol = (N*rCO2 + rCmic*1000.0)/44000.0/DIA
    RO2_mol = (N*rO2 + rOmic*1000.0)/32000.0/DIA
    RCH4_mol = rCH4g*1000.0/16000.0/DIA
    Ts_p = (FIS['gammaq']*RCO2_mol - FIS['hsAs']*(Ts-T)
            - FIS['lam']*E)/FIS['Cths']
    eT = FIS['Tref']-T
    Ppel = np.clip(3.0*eT + 0.05*IT, -FIS['Pmax'], FIS['Pmax'])
    T_p = (FIS['hsAs']*(Ts-T) + Ppel
           - FIS['rhoair']*Q*FIS['cpair']*(T-FIS['Ta'])
           - FIS['UA']*(T-FIS['Ta']))/FIS['Cth']
    HR = w/max(wsat(T), 1e-9)
    eH = FIS['HRref']-HR
    Jhum = np.clip(2.0e-5*eH + 1.0e-6*IH, 0.0, FIS['Jmax'])
    Emet = FIS['gammaw']*(N*rCO2 + rCmic*1000.0)/1000.0/DIA
    w_p = (FIS['rhoair']*Q*(wsat(FIS['Ta'])*0.55-w) + E + Emet
           + Jhum)/(FIS['Vair']*FIS['rhoair'])
    TK = T+273.15
    cCO2in = ppm2c(420.0, TK)
    cO2in = ppm2c(209500.0, TK)
    cCH4in = ppm2c(1.9, TK)
    cCO2_p = (Q*(cCO2in-cCO2)+RCO2_mol)/FIS['Vair']
    cO2_p = (Q*(cO2in-cO2)-RO2_mol)/FIS['Vair']
    cCH4_p = (Q*(cCH4in-cCH4)+RCH4_mol)/FIS['Vair']
    return [0.0, rB/DIA, rL/DIA, 0.0, DM_p, W_p, Ts_p, T_p, w_p,
            cCO2_p, cO2_p, cCH4_p, eT, eH]

def correr(N0, tf_d=16.0, nombre='E2'):
    S0 = 1.4*N0
    DM0, W0 = 0.40*S0, 0.60*S0
    y0 = [0.005, 0.012, 0.003, float(N0), DM0, W0, 27.0, 28.0,
          wsat(28.0)*0.60, ppm2c(420.0, 301.15), ppm2c(209500.0, 301.15),
          ppm2c(1.9, 301.15), 0.0, 0.0]
    vent1 = calendario_cierres(tf_d, delta_min=15.0)
    sol1 = solve_ivp(lambda t, y: rhs(t, y, vent1), (0, tf_d*DIA), y0,
                     method='LSODA', dense_output=True,
                     rtol=1e-6, atol=1e-9, max_step=120.0)
    vent, deltas = calendario_adaptativo(tf_d, sol1)
    print(f"  [{nombre}] Delta_k (min): "
          + " ".join(f"{d/60:.1f}" for d in deltas))
    sol = solve_ivp(lambda t, y: rhs(t, y, vent), (0, tf_d*DIA), y0,
                   method='LSODA', dense_output=True,
                   rtol=1e-6, atol=1e-9, max_step=120.0)
    t = np.linspace(0, tf_d*DIA, 4000)
    Y = sol.sol(t)
    td = t/DIA
    TK = Y[7]+273.15
    co2 = c2ppm(Y[9], TK)
    o2 = c2ppm(Y[10], TK)
    ch4 = c2ppm(Y[11], TK)
    S = Y[4]+Y[5]
    ths = Y[5]/S
    rCO2_v = np.array([tasas_larva(b, x)[3] for x, b in zip(td, Y[1])])
    em_lar = np.trapezoid(N0*rCO2_v/44000.0, td)
    em_mic = np.trapezoid([tasas_mic(dm, th, tss, N0, ra)[0]
                           for dm, th, tss, ra
                           in zip(Y[4], ths, Y[6], rCO2_v)], td)/1000.0
    print(f"[{nombre}] N0={N0}")
    print(f"  CO2 pico={co2.max():6.0f} ppm | O2 min={o2.min()/1e4:5.2f}% | "
          f"CH4 pico={ch4.max():4.1f} ppm")
    print(f"  B final={Y[1,-1]:5.1f} mg | L final={Y[2,-1]:5.1f} mg | "
          f"S final={S[-1]:5.0f} g | ths final={ths[-1]:.3f}")
    print(f"  Emisiones: CO2 larval={em_lar:6.2f} mol | "
          f"CO2 microbiano={em_mic:5.2f} mol")
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a, b in vent:
        for k in range(4):
            ax.flat[k].axvspan(a/DIA, b/DIA, color='gray', alpha=0.25, lw=0)
    ax[0, 0].plot(td, co2, lw=1)
    ax[0, 0].set_ylabel('CO2 (ppm)')
    ax[0, 0].set_title('(a) CO2 en camara')
    ax[0, 1].plot(td, o2/1e4, lw=1)
    ax[0, 1].set_ylabel('O2 (% vol)')
    ax[0, 1].set_ylim(15, 22)
    ax[0, 1].set_title('(b) O2 en camara')
    ax[1, 0].plot(td, Y[7], lw=1, label='T aire')
    ax[1, 0].plot(td, Y[6], lw=1, label='Ts lecho')
    ax[1, 0].plot(td, 30*ths, lw=1, label='30*ths')
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_ylabel('°C / (30x1)')
    ax[1, 0].set_title('(c) Termica e hidrica')
    ax[1, 1].plot(td, Y[1], lw=1, label='B')
    ax[1, 1].plot(td, Y[2], lw=1, label='L')
    ax[1, 1].plot(td, S/10, lw=1, label='S/10')
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_ylabel('mg / (g/10)')
    ax[1, 1].set_title('(d) Biomasa y sustrato')
    for k in range(4):
        ax.flat[k].set_xlim(0, tf_d)
        ax.flat[k].set_xlabel('t (dias)')
    fig.suptitle(f'Escenario {nombre} (N0={N0})')
    fig.tight_layout()
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(os.path.join(IMG, f'sim_{nombre}.pdf'))
    plt.close(fig)

if __name__ == '__main__':
    correr(400, nombre='E2')
    correr(700, nombre='E5')
    print('Figuras en imagenes/sim_E2.pdf y sim_E5.pdf')

"""Gemelo 0D continuo CALIBRADO con datos reales (camara cerrada).
Escala real: 750 larvas, 250 g sustrato, camara 11.9 L (11.4 L de aire).
Calibracion: m_co2 anclado al flujo neto D1 (174 mL/h, ~11 g larvas);
m_mic anclado a controles (~50 mL/h por 250 g).
Tres escenarios: ACH fijo=2, adaptativo por CO2+T (cabecera),
adaptativo + aireacion de cama (hA aumenta con ACH)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

P = dict(
    X0=11.0, K=150.0, r_max=0.5/24,      # 750 larvas ~15 mg -> 150 g potencial
    T_opt=30.0, T_w=8.0,
    m_co2=0.0158,                        # CALIBRADO: 174 mL/h / 11 g [L/(g h)]
    a_co2=0.10,                          # pendiente: ajustar al pesar larvas
    m_mic=0.20,                          # CALIBRADO: 50 mL/h / 0.25 kg [L/(kg h)]
    S0=0.25, k_s=0.05/24,
    oxical=20e3/3600,
    C_cama=750.0, C_aire=13.7,           # 250 g sustrato humedo, 11.4 L aire
    hA=0.3, U_pared=0.7,
    T_amb=25.0, V_aire_L=11.4,
    H0=0.70, M_seco=0.075,
    e0=0.002, lam=2260e3/3600,
    k_ch4=0.15, H_crit=0.55,             # placeholder: pendiente validacion GC
    C_CO2_AMB=0.042, C_CH4_AMB=0.0002,
    CO2_META=0.40,                       # % : meta de control (tope sensor 0.5 %)
    ACH_MIN=2.0, ACH_MAX=15.0,
    T_MAX=33.0,                          # proteccion termica de larvas
)
T_DIAS = 14

def f_T(T, p):
    return np.exp(-((T - p["T_opt"]) / p["T_w"])**2)

def ach_control(q_co2, Tb, p):
    """ACH para mantener CO2 bajo la meta Y la cama bajo T_MAX."""
    ach = 100 * q_co2 / (p["V_aire_L"] * p["CO2_META"])
    if Tb > p["T_MAX"]:
        ach = p["ACH_MAX"]
    return float(np.clip(ach, p["ACH_MIN"], p["ACH_MAX"]))

def rhs_factory(p, modo):
    def rhs(t, y):
        X, S, Tb, Ta, H, cco2, cch4 = y
        fT = f_T(Tb, p)
        dX = p["r_max"] * fT * X * (1 - X / p["K"])
        dS = -p["k_s"] * fT * S
        q_co2 = p["a_co2"] * dX + p["m_co2"] * X + p["m_mic"] * max(S, 0) * fT
        anaerob = max(0.0, (H - p["H_crit"]) / p["H_crit"])
        q_ch4 = p["k_ch4"] * p["m_mic"] * max(S, 0) * fT * anaerob
        ach = p["ACH_MIN"] if modo == "fijo" else ach_control(q_co2, Tb, p)
        hA = p["hA"] * (1 + ach / 4) if modo == "aeracion" else p["hA"]
        E = p["e0"] * H * max(Tb - 15, 0) / 10 * (0.5 + ach / 8)
        Q_gen = p["oxical"] * q_co2
        dTb = (Q_gen - hA * (Tb - Ta) - p["lam"] * E) * 3600 / p["C_cama"]
        dTa = (hA * (Tb - Ta) - p["U_pared"] * (Ta - p["T_amb"])
               - 1200 * p["V_aire_L"] / 1000 * ach / 3600 * (Ta - p["T_amb"])) \
              * 3600 / p["C_aire"]
        dH = -E / p["M_seco"]
        dco2 = 100 * q_co2 / p["V_aire_L"] - ach * (cco2 - p["C_CO2_AMB"])
        dch4 = 100 * q_ch4 / p["V_aire_L"] - ach * (cch4 - p["C_CH4_AMB"])
        return [dX, dS, dTb, dTa, dH, dco2, dch4]
    return rhs

def correr(modo):
    y0 = [P["X0"], P["S0"], P["T_amb"] + 1, P["T_amb"], P["H0"],
          P["C_CO2_AMB"], P["C_CH4_AMB"]]
    t_eval = np.arange(0, T_DIAS * 24 + 0.5, 0.5)
    sol = solve_ivp(rhs_factory(P, modo), (0, T_DIAS * 24), y0,
                    t_eval=t_eval, method="LSODA", rtol=1e-6)
    X, S, Tb, Ta, H, cco2, cch4 = sol.y
    dX = np.gradient(X, sol.t)
    q = P["a_co2"] * dX + P["m_co2"] * X + P["m_mic"] * S * f_T(Tb, P)
    ach = (np.full_like(q, P["ACH_MIN"]) if modo == "fijo"
           else np.array([ach_control(qi, tbi, P) for qi, tbi in zip(q, Tb)]))
    return sol.t / 24, X, Tb, Ta, H, cco2, cch4, q, ach

RES = Path(__file__).resolve().parent.parent / "resultados"
RES.mkdir(exist_ok=True)
t, X, Tb, Ta, H, c_ad, cch4, q, ach = correr("adaptativo")
_, _, _, _, _, c_fijo, _, _, _ = correr("fijo")
t2, X2, Tb2, Ta2, H2, c_ae, cch4_2, q2, ach2 = correr("aeracion")

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
axs[0, 0].plot(t, X, label="cabecera"); axs[0, 0].plot(t2, X2, label="aireacion")
axs[0, 0].set_title("Biomasa (750 larvas) [g]"); axs[0, 0].legend()
axs[0, 1].plot(t, c_fijo, label="ACH fijo = 2")
axs[0, 1].plot(t, c_ad, label="Adaptativo (cabecera)")
axs[0, 1].plot(t2, c_ae, label="Adaptativo + aireacion cama")
axs[0, 1].axhline(0.5, color="r", ls="--", label="tope MH-410D")
axs[0, 1].set_title("CO2 cabecera [%]"); axs[0, 1].legend(fontsize=8)
axs[1, 0].plot(t, ach, label="cabecera"); axs[1, 0].plot(t2, ach2, label="aireacion")
axs[1, 0].set_title("Ventilacion adaptativa [ACH]"); axs[1, 0].legend()
axs[1, 1].plot(t, Tb, label="T cama (cabecera)")
axs[1, 1].plot(t2, Tb2, label="T cama (aireacion)")
axs[1, 1].axhline(P["T_MAX"], color="r", ls="--", label="T_MAX larvas")
axs[1, 1].axhline(P["T_amb"], color="k", ls=":")
axs[1, 1].set_title("Temperaturas [C]"); axs[1, 1].legend(fontsize=8)
for ax in axs.flat:
    ax.set_xlabel("t [dias]"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RES / "gemelo_continuo_adaptativo.png", dpi=150)

print(f"CO2 pico  | fijo ACH=2:   {c_fijo.max():.2f} % ({c_fijo.max()*1e4:.0f} ppm)")
print(f"CO2 pico  | adaptativo:   {c_ad.max():.2f} % ({c_ad.max()*1e4:.0f} ppm)")
print(f"ACH medio: {ach.mean():.1f} | ACH max: {ach.max():.1f}")
print(f"T cama pico: {Tb.max():.1f} C | Biomasa final: {X[-1]:.0f} g")
print(f"CH4 pico (placeholder): {cch4.max()*1e4:.0f} ppm")
print(f"\n--- Modo aireacion de cama ---")
print(f"CO2 pico: {c_ae.max():.2f} % | T cama pico: {Tb2.max():.1f} C "
      f"| Biomasa final: {X2[-1]:.0f} g")
print(f"Figura: {RES/'gemelo_continuo_adaptativo.png'}")

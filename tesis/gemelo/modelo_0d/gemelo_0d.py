"""Gemelo digital 0D del batch de bioconversion con H. illucens.
Estados: X biomasa [g], S sustrato humedo [kg], T_cama/T_aire [C],
H humedad cama [kg/kg base seca], C_CO2/C_CH4 en cabecera [%].
Tiempo en horas. CO2 metabolico: Herbert-Pirt (a*dX/dt + m*X + microbiano).
Calor: equivalencia oxicalorica ~20 kJ por L de CO2 (RQ~1)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

# --- Parametros del batch (placeholder, calibrar con datos reales) ---
P = dict(
    X0=100.0, K=4000.0, r_max=0.35/24,   # crecimiento logistico larvas [/h]
    T_opt=30.0, T_w=8.0,                 # campana termica del crecimiento
    a_co2=0.10, m_co2=1.2e-3,            # Herbert-Pirt [L/g], [L/(g h)]
    m_mic=0.01,                          # CO2 microbiano [L/(kg sustrato h)]
    S0=8.0, k_s=0.02/24,                 # sustrato [kg], consumo [/h]
    oxical=20e3/3600,                    # 20 kJ/L CO2 -> W por (L/h)
    C_cama=28000.0, C_aire=72.0,         # capacidades termicas [J/K]
    hA=2.0, U_pared=2.5,                 # transferencia [W/K]
    T_amb=25.0, ACH=2.0,                 # ambiente y ventilacion [1/h]
    V_aire_L=60.0,                       # volumen de cabecera [L]
    H0=0.70, M_seco=3.0,                 # humedad inicial [kg/kg], masa seca [kg]
    e0=0.008,                            # coef evaporacion [kg/h]
    lam=2260e3/3600,                     # 2260 kJ/kg -> W por (kg/h)
    k_ch4=0.15, H_crit=0.55,             # cinetica CH4 anaerobio
    C_CO2_AMB=0.042, C_CH4_AMB=0.0002,   # ambiente [%]
)
T_DIAS = 14

def f_T(T, p):
    return np.exp(-((T - p["T_opt"]) / p["T_w"])**2)

def rhs(t, y, p):
    X, S, Tb, Ta, H, cco2, cch4 = y
    fT = f_T(Tb, p)
    dX = p["r_max"] * fT * X * (1 - X / p["K"])          # [g/h]
    dS = -p["k_s"] * fT * S                              # [kg/h]
    # --- generacion de gases [L/h] ---
    q_co2 = p["a_co2"] * dX + p["m_co2"] * X + p["m_mic"] * max(S, 0) * fT
    anaerob = max(0.0, (H - p["H_crit"]) / p["H_crit"])  # humedad -> anaerobiosis
    q_ch4 = p["k_ch4"] * p["m_mic"] * max(S, 0) * fT * anaerob
    # --- evaporacion y calor (W -> J/h con el factor 3600) ---
    E = p["e0"] * H * max(Tb - 15, 0) / 10 * (0.5 + p["ACH"] / 8)   # [kg/h]
    Q_gen = p["oxical"] * q_co2                          # [W]
    dTb = (Q_gen - p["hA"] * (Tb - Ta) - p["lam"] * E) * 3600 / p["C_cama"]
    dTa = (p["hA"] * (Tb - Ta) - p["U_pared"] * (Ta - p["T_amb"])
           - 1200 * p["V_aire_L"] / 1000 * p["ACH"] / 3600 * (Ta - p["T_amb"])) \
          * 3600 / p["C_aire"]
    dH = -E / p["M_seco"]
    # --- gases en cabecera (mezcla perfecta, 0D) ---
    dco2 = 100 * q_co2 / p["V_aire_L"] - p["ACH"] * (cco2 - p["C_CO2_AMB"])
    dch4 = 100 * q_ch4 / p["V_aire_L"] - p["ACH"] * (cch4 - p["C_CH4_AMB"])
    return [dX, dS, dTb, dTa, dH, dco2, dch4]

y0 = [P["X0"], P["S0"], P["T_amb"] + 1, P["T_amb"], P["H0"],
      P["C_CO2_AMB"], P["C_CH4_AMB"]]
t_eval = np.arange(0, T_DIAS * 24 + 1, 1.0)
sol = solve_ivp(rhs, (0, T_DIAS * 24), y0, t_eval=t_eval,
                args=(P,), method="LSODA", rtol=1e-6)
t_d = sol.t / 24
X, S, Tb, Ta, H, cco2, cch4 = sol.y
dXdt = np.array([rhs(ti, yi, P)[0] for ti, yi in zip(sol.t, sol.y.T)])
qco2 = P["a_co2"] * dXdt + P["m_co2"] * X + P["m_mic"] * S * f_T(Tb, P)

OUT = Path(__file__).resolve().parent.parent
RES = OUT / "resultados"; DAT = OUT / "datos"
RES.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
axs[0, 0].plot(t_d, X, label="Biomasa larvaria")
axs[0, 0].plot(t_d, S * 500, label="Sustrato x500 g")
axs[0, 0].set_ylabel("[g]"); axs[0, 0].legend(); axs[0, 0].set_title("Crecimiento")
axs[0, 1].plot(t_d, Tb, label="T cama"); axs[0, 1].plot(t_d, Ta, label="T aire")
axs[0, 1].axhline(P["T_amb"], color="k", ls=":", label="T ambiente")
axs[0, 1].set_ylabel("[C]"); axs[0, 1].legend(); axs[0, 1].set_title("Autocalentamiento")
axs[1, 0].plot(t_d, cco2, label="CO2 cabecera [%]")
axs[1, 0].plot(t_d, cch4 * 100, label="CH4 x100 [%]")
axs[1, 0].plot(t_d, qco2 / max(qco2) * max(cco2), ":", label="Tasa CO2 (esc.)")
axs[1, 0].legend(); axs[1, 0].set_title("GEI en cabecera (lo que miden los sensores)")
axs[1, 1].plot(t_d, H * 100, color="tab:brown")
axs[1, 1].set_ylabel("[% base seca]"); axs[1, 1].set_title("Humedad de la cama")
for ax in axs.flat:
    ax.set_xlabel("t [dias]"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RES / "gemelo_0d_series.png", dpi=150)

# --- CSV sintetico con ruido de sensor (para probar la calibracion) ---
rng = np.random.default_rng(42)
datos = np.column_stack([
    sol.t, Tb, Ta, cco2 * 1e4, cch4 * 1e4,
    Tb + rng.normal(0, 0.3, len(t_d)),
    Ta + rng.normal(0, 0.2, len(t_d)),
    cco2 * 1e4 * (1 + rng.normal(0, 0.03, len(t_d))),
    cch4 * 1e4 * (1 + rng.normal(0, 0.05, len(t_d)))])
np.savetxt(DAT / "sintetico_base.csv", datos, delimiter=",",
           header="t_h,T_cama_C,T_aire_C,CO2_ppm,CH4_ppm,"
                  "T_cama_sensor,T_aire_sensor,CO2_sensor_ppm,CH4_sensor_ppm",
           comments="")
print(f"Picos: T_cama {Tb.max():.1f} C | CO2 {cco2.max():.2f} % "
      f"({cco2.max()*1e4:.0f} ppm) | CH4 {cch4.max()*1e4:.0f} ppm")
print(f"Biomasa final: {X[-1]:.0f} g | Humedad final: {H[-1]*100:.0f} %")
print(f"Figura: {RES/'gemelo_0d_series.png'}")
print(f"CSV:    {DAT/'sintetico_base.csv'}")

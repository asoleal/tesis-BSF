"""CO2 y CH4 en seccion 2D de la caja, CON ventilacion forzada.
Ventilacion: dilucion del aire de la cabecera (sobre la cama) a tasa k.
k se parametriza en ACH (recambios/hora del volumen total de la caja).
Dentro de la cama el transporte sigue siendo por difusion (aire atrapado).
Unidades: cm, s, concentracion en % (1 % = 10 000 ppm)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pde import PDEBase, ScalarField, FieldCollection, UnitGrid

# --- Parametros ---
Nx, Ny = 60, 40          # seccion 60 x 40 cm, celda de 1 cm
H_CAMA = 15.0            # altura del lecho [cm]
D_CO2, D_CH4 = 0.16, 0.22          # difusividades en aire [cm^2/s]
C_CO2_AMB, C_CH4_AMB = 0.042, 0.0002   # ambiente [%] (420 y ~2 ppm)
G_CO2 = 2.78e-3          # generacion CO2 en la cama [%/s] (~3.6 L/h)
G_CH4 = G_CO2 * 0.005    # CH4 ~0.5 % del CO2 en volumen (placeholder)
T_HORAS = 6

grid = UnitGrid([Nx, Ny])
ycoord = grid.cell_coords[..., 1]
f_co2 = ScalarField(grid, np.where(ycoord < H_CAMA, G_CO2, 0.0))
f_ch4 = ScalarField(grid, np.where(ycoord < H_CAMA, G_CH4, 0.0))
m_cab = ScalarField(grid, np.where(ycoord >= H_CAMA, 1.0, 0.0))  # 1 = aire ventilado
FRAC_CAB = (Ny - H_CAMA) / Ny      # fraccion de volumen ventilado

class CamaGEI(PDEBase):
    def __init__(self, ach):
        super().__init__()
        self.k = (ach / 3600.0) / FRAC_CAB   # dilucion en cabecera ~ ACH del volumen total
        self.bc = ["neumann", "neumann"]     # caja sellada; el recambio es el termino fuente
    def evolution_rate(self, state, t=0):
        co2, ch4 = state[0], state[1]
        dco2 = D_CO2 * co2.laplace(bc=self.bc) + f_co2 \
               - self.k * m_cab * (co2 - C_CO2_AMB)
        dch4 = D_CH4 * ch4.laplace(bc=self.bc) + f_ch4 \
               - self.k * m_cab * (ch4 - C_CH4_AMB)
        return FieldCollection([dco2, dch4])

def simular(ach):
    est = FieldCollection([
        ScalarField(grid, C_CO2_AMB, label="CO2"),
        ScalarField(grid, C_CH4_AMB, label="CH4")])
    eq = CamaGEI(ach)
    for _ in range(T_HORAS * 2):                       # bloques de 30 min
        est = eq.solve(est, t_range=1800, dt=1.0, solver="euler", tracker=[])
    return est

OUT = Path(__file__).resolve().parent.parent / "resultados"
OUT.mkdir(exist_ok=True)

ACH_LISTA = [0, 1, 2, 4, 8]
print(f"{'ACH':>4} | {'CO2 max cama':>12} | {'CO2 cab. 35cm':>13} | {'CH4 max cama':>12}")
print("-" * 52)
resultados = {}
for ach in ACH_LISTA:
    sol = simular(ach)
    resultados[ach] = sol
    print(f"{ach:>4} | {sol[0].data.max():>11.2f} % | {sol[0].data[30,35]:>12.2f} % "
          f"| {sol[1].data.max():>10.4f} %", flush=True)

# --- Figura 1: mapas CO2 y CH4 con ACH = 4 ---
sol = resultados[4]
ref = sol[0].plot(title=f"CO2 [%] a las {T_HORAS} h, ventilacion 4 ACH")
ref.ax.figure.savefig(OUT / "gei_mapa_co2_4ach.png", dpi=150, bbox_inches="tight")
ref = sol[1].plot(title=f"CH4 [%] a las {T_HORAS} h, ventilacion 4 ACH")
ref.ax.figure.savefig(OUT / "gei_mapa_ch4_4ach.png", dpi=150, bbox_inches="tight")

# --- Figura 2: curva de diseno CO2 max vs ACH ---
fig, ax = plt.subplots()
co2max = [resultados[a][0].data.max() for a in ACH_LISTA]
ax.plot(ACH_LISTA, co2max, "o-", label="CO2 max en la cama")
ax.axhline(1.0, color="r", ls="--",
           label="1 % (10 000 ppm): limite sensor MH-Z19 / seguridad larvas")
ax.set_xlabel("Ventilacion [recambios/hora, ACH]")
ax.set_ylabel("CO2 maximo en la cama [%]")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig(OUT / "gei_curva_diseno_ventilacion.png", dpi=150, bbox_inches="tight")
print(f"\nFiguras en: {OUT}")

"""CO2 en seccion 2D de la caja de bioconversion (BSFL).
Unidades: cm, s, concentracion en % (420 ppm = 0.042 %)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pde import PDEBase, ScalarField, UnitGrid

# --- Parametros (ajustar con tus medidas reales) ---
Nx, Ny = 60, 40         # celdas: 60 x 40 cm (1 cm por celda)
H_CAMA = 15.0           # altura del lecho sustrato+larvas [cm]
D_CO2  = 0.16           # difusividad CO2 en aire [cm^2/s]
C_AMB  = 0.042          # CO2 ambiente [%]
G_CAMA = 2.78e-3        # generacion en la cama [%/s]  (~3.6 L/h en caja 60x40x15)

grid = UnitGrid([Nx, Ny])   # celda = 1 cm; NO periodico por defecto
ycoord = grid.cell_coords[..., 1]
fuente = ScalarField(grid, np.where(ycoord < H_CAMA, G_CAMA, 0.0))

class CamaCO2(PDEBase):
    def __init__(self, fuente):
        super().__init__()
        self.fuente = fuente
        # BC por eje: x sellado; y: piso sellado, techo abierto al ambiente
        self.bc = ["neumann", ("neumann", {"dirichlet": C_AMB})]
    def evolution_rate(self, state, t=0):
        return D_CO2 * state.laplace(bc=self.bc) + self.fuente

estado = ScalarField(grid, C_AMB)
eq = CamaCO2(fuente)

# --- Resolver por bloques de 30 min, guardando copias ---
snapshots = []
T_BLOQUE = 1800
for k in range(12):                                    # 12 x 30 min = 6 h
    estado = eq.solve(estado, t_range=T_BLOQUE, dt=1.0,
                      solver="euler", tracker=[])
    snapshots.append((estado.copy(), (k + 1) * T_BLOQUE))
    print(f"  t = {(k+1)*0.5:.1f} h | CO2 max = {estado.data.max():.2f} %", flush=True)

sol = snapshots[-1][0]

# --- Figura 1: mapa a las 6 h ---
ref = sol.plot(title="CO2 [%] tras 6 h (cama abajo, techo abierto)")
ref.ax.figure.savefig("co2_mapa_6h.png", dpi=150, bbox_inches="tight")

# --- Figura 2: perfiles verticales en el centro ---
fig, ax2 = plt.subplots()
for f, tt in snapshots:
    ax2.plot(grid.cell_coords[0, :, 1], f.data[30, :], label=f"{tt/3600:.1f} h")
ax2.set_xlabel("altura y [cm]"); ax2.set_ylabel("CO2 [%]")
ax2.legend(); ax2.grid(alpha=0.3)
fig.savefig("co2_perfiles.png", dpi=150, bbox_inches="tight")

# --- Numeros para ubicar sensores ---
print(f"CO2 maximo (dentro de la cama): {sol.data.max():.2f} %")
print(f"Sensor a 5 cm  del fondo, centro: {sol.data[30, 5]:.2f} %")
print(f"Sensor a 20 cm del fondo, centro: {sol.data[30, 20]:.2f} %")
print(f"Sensor a 35 cm del fondo, centro: {sol.data[30, 35]:.2f} %")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. CONFIGURACIÓN Y PREPARACIÓN DE DATOS
# ==========================================

file_path = 'datos_entrenamiento_PINN.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: No encuentro '{file_path}'. Ejecuta primero procesar_todo.py")
    exit()

# Pivotamos la tabla para tener CO2 y CH4 como columnas
# (Si hay duplicados, tomamos el promedio)
df_pivot = df.pivot_table(index=['dia', 'dieta'], columns='gas_tipo', values='tasa_neta', aggfunc='mean').reset_index()
df_pivot.columns = ['dia', 'dieta', 'co2_rate', 'ch4_rate']

print("Datos organizados:")
print(df_pivot)

# Preparar Arrays Numpy
X_data = df_pivot[['dia', 'dieta']].values.astype(np.float32)
y_data = df_pivot[['co2_rate', 'ch4_rate']].values.astype(np.float32)

# --- NORMALIZACIÓN ---
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_norm = scaler_x.fit_transform(X_data)
y_norm = scaler_y.fit_transform(y_data)

# Tensores PyTorch (Datos Reales)
inputs = torch.tensor(X_norm, dtype=torch.float32, requires_grad=True)
targets = torch.tensor(y_norm, dtype=torch.float32)

# ==========================================
# 2. ARQUITECTURA PINN
# ==========================================

class BioPINN(nn.Module):
    def __init__(self):
        super(BioPINN, self).__init__()
        # 2 Entradas (Dia, Dieta) -> Capas Ocultas -> 2 Salidas (CO2, CH4)
        self.hidden1 = nn.Linear(2, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.hidden3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 2)
        self.activation = nn.Tanh() # Tanh es ideal para física continua

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.output(x)
        return x

model = BioPINN()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 3. ENTRENAMIENTO CON FÍSICA (PHYSICS LOSS)
# ==========================================

epochs = 10000
loss_history = []

print("\nIniciando entrenamiento híbrido (Datos + Física)...")

for epoch in range(epochs):
    optimizer.zero_grad()

    # --- A. PÉRDIDA DE DATOS (Supervisada) ---
    # La IA debe pasar por los puntos reales medidos
    outputs_data = model(inputs)
    loss_data = nn.MSELoss()(outputs_data, targets)

    # --- B. PÉRDIDA FÍSICA (No Supervisada / Collocation Points) ---
    # Generamos "días fantasma" aleatorios para vigilar los huecos
    # Queremos asegurar que la predicción NUNCA sea negativa en todo el rango

    # 100 puntos aleatorios entre día 9 y 18
    random_days = (torch.rand(100, 1) * 9) + 9

    # Normalizamos estos días aleatorios usando el mismo scaler
    # Nota: Creamos una columna dummy de dieta para que el scaler funcione, luego extraemos solo día
    dummy_input = np.column_stack((random_days.numpy(), np.zeros(100)))
    random_days_norm = torch.tensor(scaler_x.transform(dummy_input), dtype=torch.float32)[:, 0].view(-1, 1)

    # Dietas aleatorias (0 o 1)
    random_diets = torch.randint(0, 2, (100, 1)).float()

    # Input físico completo
    physics_inputs = torch.cat((random_days_norm, random_diets), dim=1)

    # Predicción en los días fantasma
    outputs_physics = model(physics_inputs)

    # REGLA DE ORO: No Negatividad (ReLU de -x)
    # Si x es positivo, error es 0. Si x es negativo (-5), error es 5^2 = 25.
    loss_negativity = torch.mean(torch.relu(-outputs_physics)**2) * 10 # Peso x10 para forzar cumplimiento

    loss_physics = loss_negativity

    # --- C. SUMA TOTAL ---
    loss_total = loss_data + loss_physics

    loss_total.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Data Loss {loss_data.item():.6f} + Physics Loss {loss_physics.item():.6f}")

# ==========================================
# 4. VISUALIZACIÓN FINAL (PANELES SEPARADOS)
# ==========================================

# Generar rango suave para graficar líneas
days_range = np.linspace(9, 18, 100)

# --- Predicciones D1 (Rápido) ---
input_D1 = np.column_stack((days_range, np.zeros_like(days_range)))
tensor_D1 = torch.tensor(scaler_x.transform(input_D1), dtype=torch.float32)
pred_D1_real = scaler_y.inverse_transform(model(tensor_D1).detach().numpy())

# --- Predicciones D4 (Lento) ---
input_D4 = np.column_stack((days_range, np.ones_like(days_range)))
tensor_D4 = torch.tensor(scaler_x.transform(input_D4), dtype=torch.float32)
pred_D4_real = scaler_y.inverse_transform(model(tensor_D4).detach().numpy())

# --- CONFIGURACIÓN GRÁFICA ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# === GRÁFICA 1: CO2 (Respiración) ===
# Datos Reales
ax1.scatter(df_pivot[df_pivot['dieta']==0]['dia'], df_pivot[df_pivot['dieta']==0]['co2_rate'],
            color='blue', s=80, zorder=5, label='Real D1 (Rápido)')
ax1.scatter(df_pivot[df_pivot['dieta']==1]['dia'], df_pivot[df_pivot['dieta']==1]['co2_rate'],
            color='orange', s=80, zorder=5, label='Real D4 (Lento)')

# Predicciones IA
ax1.plot(days_range, pred_D1_real[:, 0], 'b-', linewidth=2.5, label='IA D1')
ax1.plot(days_range, pred_D4_real[:, 0], color='orange', linestyle='-', linewidth=2.5, label='IA D4')

ax1.set_title("Respiración Aerobia (CO2)")
ax1.set_xlabel("Días")
ax1.set_ylabel("Tasa CO2 (ppm/min)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# === GRÁFICA 2: CH4 (Fermentación) ===
# Datos Reales
ax2.scatter(df_pivot[df_pivot['dieta']==0]['dia'], df_pivot[df_pivot['dieta']==0]['ch4_rate'],
            color='blue', marker='x', s=100, linewidth=2, zorder=5, label='Real D1 (Rápido)')
ax2.scatter(df_pivot[df_pivot['dieta']==1]['dia'], df_pivot[df_pivot['dieta']==1]['ch4_rate'],
            color='orange', marker='x', s=100, linewidth=2, zorder=5, label='Real D4 (Lento)')

# Predicciones IA
ax2.plot(days_range, pred_D1_real[:, 1], 'b--', linewidth=2.5, label='IA D1')
ax2.plot(days_range, pred_D4_real[:, 1], color='orange', linestyle='--', linewidth=2.5, label='IA D4')

ax2.set_title("Fermentación Anaerobia (CH4)")
ax2.set_xlabel("Días")
ax2.set_ylabel("Tasa CH4 (ppm/min)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("¡Proceso completado! Gráfica generada.")

from scipy.integrate import simpson

# --- CÁLCULO DE TOTALES ACUMULADOS (INTEGRAL) ---
print("\n" + "="*40)
print("PREDICCIONES DE PRODUCCIÓN TOTAL ACUMULADA")
print("="*40)

# Función para integrar: Área bajo la curva de Tasa (ppm/min) vs Tiempo (Días)
# Factor de conversión: 1 día = 1440 minutos
# Fórmula: Integral = Suma(Tasa * dt) * 1440

def calcular_integral(tasa_predicha, dias):
    # tasa_predicha es un array de valores de y
    # dias es el eje x
    area = simpson(tasa_predicha, x=dias)
    return area * 1440 # Convertir días a minutos

# 1. CO2 Totales
total_co2_d1 = calcular_integral(pred_D1_real[:, 0], days_range)
total_co2_d4 = calcular_integral(pred_D4_real[:, 0], days_range)

# 2. CH4 Totales
total_ch4_d1 = calcular_integral(pred_D1_real[:, 1], days_range)
total_ch4_d4 = calcular_integral(pred_D4_real[:, 1], days_range)

print(f"CO2 ACUMULADO D1 (Rápido): {total_co2_d1:.0f} ppm acumulados")
print(f"CO2 ACUMULADO D4 (Lento):  {total_co2_d4:.0f} ppm acumulados")
print("-" * 30)
print(f"CH4 ACUMULADO D1 (Rápido): {total_ch4_d1:.0f} ppm acumulados")
print(f"CH4 ACUMULADO D4 (Lento):  {total_ch4_d4:.0f} ppm acumulados")

# ==========================================
# 6. SENSOR VIRTUAL: ESTIMACIÓN DE BIOMASA (B)
# ==========================================
# Parámetros aproximados basados en literatura (Eriksen/Luedeking-Piret)
# Nota: Estos valores son teóricos. Para precisión exacta se requiere calibración.
Y_B = 1.5   # Costo de crecimiento (Cuanto CO2 cuesta hacer 1g de larva)
m_coeff = 0.1 # Costo de mantenimiento (Cuanto respira 1g de larva quieta)

def sensor_virtual_biomasa(tasa_co2_predicha, dias, Y_B, m):
    """
    Resuelve la EDO: dB/dt = (r_CO2 - m*B) / Y_B
    Usamos el método de Euler simple para integrar paso a paso.
    """
    dt = dias[1] - dias[0] # Paso de tiempo
    biomasa = [0.1] # Biomasa inicial (asumimos larvas pequeñas 0.1g o unidad arbitraria)

    for i in range(len(tasa_co2_predicha) - 1):
        r_co2 = tasa_co2_predicha[i]
        B_actual = biomasa[-1]

        # Ecuación Inversa de Eriksen
        dB_dt = (r_co2 - (m * B_actual)) / Y_B

        # Evitar crecimiento negativo (biológicamente la larva no se encoge drásticamente comiendo)
        if dB_dt < 0: dB_dt = 0

        B_siguiente = B_actual + dB_dt * dt
        biomasa.append(B_siguiente)

    return np.array(biomasa)

# Calcular Biomasa Virtual para D1 y D4
# Nota: Normalizamos la tasa dividiendo por 100 para que los números de biomasa sean manejables
# (Esto asume que la tasa está en ppm y queremos una unidad arbitraria de masa)
factor_escala = 100.0
biomasa_est_D1 = sensor_virtual_biomasa(pred_D1_real[:, 0]/factor_escala, days_range, Y_B, m_coeff)
biomasa_est_D4 = sensor_virtual_biomasa(pred_D4_real[:, 0]/factor_escala, days_range, Y_B, m_coeff)

# --- GRAFICAR BIOMASA ESTIMADA ---
plt.figure(figsize=(10, 6))

plt.plot(days_range, biomasa_est_D1, 'b-', linewidth=3, label='Biomasa Estimada D1 (Rápido)')
plt.plot(days_range, biomasa_est_D4, color='orange', linestyle='-', linewidth=3, label='Biomasa Estimada D4 (Lento)')

plt.title("Sensor Virtual: Estimación de Crecimiento Larval basada en CO2")
plt.xlabel("Días")
plt.ylabel("Biomasa Acumulada (Unidades Arbitrarias)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Sensor Virtual ejecutado. Revisa la gráfica de crecimiento estimado.")

# ==========================================
# 7. VALIDACIÓN: SENSOR VIRTUAL VS TEORÍA ERIKSEN
# ==========================================

# Parámetros Biológicos (Aproximados de Literatura)
Y_B = 1.5   # Costo de Crecimiento (mol CO2 / C-mol biomasa)
m_coeff = 0.15 # Mantenimiento (mol CO2 / C-mol biomasa * día)

def sensor_virtual_biomasa(tasa_co2, dias, Y_B, m):
    """
    Integra la EDO inversa: dB/dt = (r_CO2 - m*B) / Y_B
    """
    dt = dias[1] - dias[0]
    # Normalizamos la tasa para obtener unidades de biomasa relativas (0 a 1 aprox)
    # Esto es necesario porque no tenemos la conversión exacta a moles reales,
    # pero la FORMA de la curva es lo que importa.
    tasa_norm = tasa_co2 / np.max(tasa_co2) * 2.5

    biomasa = [0.05] # Biomasa inicial (larva pequeña)

    for i in range(len(tasa_norm) - 1):
        r_co2 = tasa_norm[i]
        B_actual = biomasa[-1]

        # Ecuación fundamental de Eriksen despejada para Crecimiento
        tasa_crecimiento = (r_co2 - (m * B_actual)) / Y_B

        # Restricción biológica: La larva no se encoge (salvo inanición extrema)
        if tasa_crecimiento < 0: tasa_crecimiento = 0

        B_siguiente = B_actual + tasa_crecimiento * dt
        biomasa.append(B_siguiente)

    return np.array(biomasa)

# 1. Calcular Biomasa Inferida por la IA
biomasa_D1 = sensor_virtual_biomasa(pred_D1_real[:, 0], days_range, Y_B, m_coeff)
biomasa_D4 = sensor_virtual_biomasa(pred_D4_real[:, 0], days_range, Y_B, m_coeff)

# 2. Generar Curva Teórica Ideal (Sigmoidea de Eriksen)
# B(t) = B_max / (1 + exp(-k * (t - t_mid)))
B_max_ref = np.max(biomasa_D1) # Usamos D1 como referencia de "éxito"
k_eriksen = 0.9  # Tasa de crecimiento típica en literatura
t_mid = 11.5     # Día de máxima velocidad de crecimiento

curva_teorica = B_max_ref / (1 + np.exp(-k_eriksen * (days_range - t_mid)))

# --- GRÁFICA FINAL ---
plt.figure(figsize=(12, 7))

# Curva Teórica (Lo que dice el libro)
plt.plot(days_range, curva_teorica, 'k--', linewidth=2, alpha=0.6, label='Teoría Eriksen (Modelo Ideal)')

# Curvas Reales Inferidas (Lo que dice tu IA)
plt.plot(days_range, biomasa_D1, 'b-', linewidth=4, alpha=0.9, label='IA: Dieta D1 (Rápida)')
plt.plot(days_range, biomasa_D4, 'orange', linewidth=4, alpha=0.9, label='IA: Dieta D4 (Lenta)')

# Decoración
plt.fill_between(days_range, biomasa_D1, alpha=0.1, color='blue')
plt.title("Sensor Virtual: Estimación de Biomasa basada en Respiración (PINN)", fontsize=14)
plt.xlabel("Días de Desarrollo", fontsize=12)
plt.ylabel("Biomasa Acumulada (Unidades Relativas)", fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Anotaciones
idx_max = np.argmax(biomasa_D1)
plt.annotate('Fase Prepupa\n(Crecimiento Detenido)',
             xy=(days_range[idx_max], biomasa_D1[idx_max]),
             xytext=(14, biomasa_D1[idx_max]*0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()
print("¡Gráfica generada! Analicemos las curvas...")

import numpy as np
import pandas as pd

# ==========================================
# 7. GENERACIÓN DE DATOS: SENSOR VIRTUAL (BIOMASA)
# ==========================================

print("\n" + "="*50)
print("EJECUTANDO SENSOR VIRTUAL (ESTIMACIÓN DE BIOMASA)")
print("="*50)

# --- 1. Definición de Parámetros Biológicos (Aproximación) ---
# Y_B: Rendimiento (Cuánto CO2 cuesta crear estructura).
# m: Mantenimiento (Cuánto CO2 cuesta mantener la estructura).
Y_B = 1.5
m_coeff = 0.15

# --- 2. Función del Sensor Virtual (Inversa de Eriksen) ---
def sensor_virtual_biomasa_numerico(tasa_co2, dias, Y_B, m):
    dt = dias[1] - dias[0]

    # Normalización para unidades relativas (0 a 100% del tamaño final esperado)
    # Esto es crucial porque la tasa_co2 está en ppm, no en moles exactos.
    # Usamos un factor de escala basado en el máximo de D1 para tener una referencia.
    factor_escala = np.max(tasa_co2) / 2.5
    tasa_norm = tasa_co2 / factor_escala

    biomasa = [0.05] # Iniciamos con una "semilla" de biomasa (larva pequeña)

    for i in range(len(tasa_norm) - 1):
        r_co2 = tasa_norm[i]
        B_actual = biomasa[-1]

        # Ecuación diferencial inversa: Crecimiento = (RespiraciónTotal - Mantenimiento) / CostoCrecimiento
        tasa_crecimiento = (r_co2 - (m * B_actual)) / Y_B

        # Restricción física: No crecimiento negativo (a menos que sea inanición extrema)
        if tasa_crecimiento < 0: tasa_crecimiento = 0

        B_siguiente = B_actual + tasa_crecimiento * dt
        biomasa.append(B_siguiente)

    return np.array(biomasa)

# --- 3. Calcular las Curvas ---
# Usamos las predicciones de CO2 que ya tienes en memoria (pred_D1_real y pred_D4_real)
bio_est_D1 = sensor_virtual_biomasa_numerico(pred_D1_real[:, 0], days_range, Y_B, m_coeff)
bio_est_D4 = sensor_virtual_biomasa_numerico(pred_D4_real[:, 0], days_range, Y_B, m_coeff)

# Calcular curva teórica Eriksen (Sigmoidea Ideal) para comparar
# Parámetros ajustados al máximo de D1 para comparar la FORMA
B_max_ref = np.max(bio_est_D1)
k_eriksen = 0.9
t_mid = 11.5
bio_teorica = B_max_ref / (1 + np.exp(-k_eriksen * (days_range - t_mid)))

# --- 4. IMPRIMIR TABLA DE DATOS PARA ANÁLISIS ---
# Seleccionamos índices representativos (uno por día aprox) para no imprimir 1000 lineas
indices = np.linspace(0, len(days_range)-1, 10, dtype=int)

print(f"{'DÍA':<6} | {'D1 ESTIMADA':<12} | {'D4 ESTIMADA':<12} | {'TEORÍA ERIKSEN':<15} | {'ESTADO D1'}")
print("-" * 75)

for i in indices:
    dia = days_range[i]
    d1_val = bio_est_D1[i]
    d4_val = bio_est_D4[i]
    teo_val = bio_teorica[i]

    # Determinar estado simple
    estado = "Creciendo"
    if i > 0:
        # Si creció menos del 1% respecto al punto anterior del loop (aprox)
        if (d1_val - bio_est_D1[indices[indices.tolist().index(i)-1]]) < 0.05:
            estado = "Meseta (Prepupa)"

    if dia < 10: estado = "Inicio Exp."

    print(f"{dia:.1f}   | {d1_val:.4f}       | {d4_val:.4f}       | {teo_val:.4f}          | {estado}")

print("-" * 75)
print("Copia esta tabla y pégala en el chat para analizarla.")

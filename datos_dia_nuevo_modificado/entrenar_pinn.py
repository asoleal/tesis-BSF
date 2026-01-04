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

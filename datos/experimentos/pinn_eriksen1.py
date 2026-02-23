import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
print("--- 🧠 INICIANDO PINN DE ERIKSEN (2022) - VERSIÓN CORREGIDA ---")
print("Cargando datos limpios...")

df = pd.read_csv("datos_PINN_netos.csv")

df_d1 = df[(df["Etiqueta"] == "D1_Tratamiento") & (df["Gas"] == "co2")].copy()
df_d4 = df[(df["Etiqueta"] == "D4_Tratamiento") & (df["Gas"] == "co2")].copy()

def preparar_datos(dataframe):
    if dataframe.empty:
        print("⚠️ Advertencia: Dataframe vacío.")
        return None, None, None, None, None
        
    t = dataframe["Dia_Experimento"].values.reshape(-1, 1).astype(np.float32)
    co2 = dataframe["Tasa_Neta"].values.reshape(-1, 1).astype(np.float32)
    
    scaler_t = MinMaxScaler()
    scaler_co2 = MinMaxScaler()
    
    t_norm = scaler_t.fit_transform(t)
    co2_norm = scaler_co2.fit_transform(co2)
    
    return (
        torch.tensor(t_norm, requires_grad=True), 
        torch.tensor(co2_norm), 
        scaler_t, 
        scaler_co2,
        t 
    )

t_d1, co2_d1, s_t_d1, s_co2_d1, t_raw_d1 = preparar_datos(df_d1)
t_d4, co2_d4, s_t_d4, s_co2_d4, t_raw_d4 = preparar_datos(df_d4)

# ==========================================
# 2. DEFINICIÓN DE LA RED NEURONAL (PINN)
# ==========================================
class LarvaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(), 
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2) 
        )
        # Y_g, Y_l, m
        self.raw_params = nn.Parameter(torch.tensor([0.5, 0.3, 0.1])) 

    def forward(self, t):
        states = self.net(t)
        B = torch.abs(states[:, 0:1]) 
        L = torch.abs(states[:, 1:2])
        return B, L

    def get_physics_params(self):
        return torch.nn.functional.softplus(self.raw_params)

# ==========================================
# 3. LOSS FUNCTION (FÍSICA DE ERIKSEN)
# ==========================================
def physics_loss(model, t, co2_medido):
    B, L = model(t)
    
    dB_dt = torch.autograd.grad(B, t, grad_outputs=torch.ones_like(B), create_graph=True)[0]
    dL_dt = torch.autograd.grad(L, t, grad_outputs=torch.ones_like(L), create_graph=True)[0]
    
    Y_g, Y_l, m = model.get_physics_params()
    
    # Ecuación Eriksen
    co2_pred = (Y_g * dB_dt) + (Y_l * dL_dt) + (m * B)
    
    loss = torch.mean((co2_pred - co2_medido) ** 2)
    loss_reg = torch.mean(torch.relu(-dB_dt)) * 0.1 
    
    return loss + loss_reg, co2_pred

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
def entrenar_modelo(t_train, co2_train, epochs=5000, lr=0.005, nombre="Modelo"):
    if t_train is None: return None, []
    
    model = LarvaNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    historial = []
    
    print(f"\n🚀 Entrenando {nombre}...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _ = physics_loss(model, t_train, co2_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"   Epoch {epoch}: Loss {loss.item():.6f}")
            
    return model, historial

model_d1, hist_d1 = entrenar_modelo(t_d1, co2_d1, nombre="Tratamiento D1")
model_d4, hist_d4 = entrenar_modelo(t_d4, co2_d4, nombre="Tratamiento D4")

# ==========================================
# 5. RESULTADOS Y VISUALIZACIÓN (CORREGIDO)
# ==========================================
def visualizar_resultados(model, t_tensor, t_raw, co2_medido, scaler_co2, titulo):
    if model is None: return

    # --- CORRECCIÓN AQUÍ: NO USAMOS torch.no_grad() ---
    # La física necesita calcular derivadas, así que mantenemos el grafo encendido
    # para el cálculo, y luego hacemos .detach() para graficar.
    
    B_norm, L_norm = model(t_tensor)
    _, co2_pred_norm = physics_loss(model, t_tensor, co2_medido)
    
    # Desconectar del grafo (detach) y convertir a numpy
    co2_pred_real = scaler_co2.inverse_transform(co2_pred_norm.detach().numpy())
    co2_real = scaler_co2.inverse_transform(co2_medido.detach().numpy())
    
    B_plot = B_norm.detach().numpy()
    L_plot = L_norm.detach().numpy()
    
    # GRÁFICA
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Eje Izquierdo: CO2
    ax1.set_xlabel('Día Experimental')
    ax1.set_ylabel('Tasa CO2 (ppm/min)', color='black')
    ax1.scatter(t_raw, co2_real, color='black', alpha=0.6, label='Datos Reales') 
    ax1.plot(t_raw, co2_pred_real, 'k--', linewidth=2, label='Predicción PINN') 
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Eje Derecho: Variables Latentes
    ax2 = ax1.twinx()
    ax2.set_ylabel('Variables Latentes (Escala Relativa)', color='tab:blue')
    
    # Líneas de variables latentes
    ax2.plot(t_raw, B_plot, color='tab:blue', linewidth=3, linestyle='-', label='Biomasa Estructural (B)') 
    ax2.plot(t_raw, L_plot, color='tab:red', linewidth=3, linestyle='-', label='Reservas de Grasa (L)') 
    
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f"{titulo}: Inferencia de Variables Ocultas (Eriksen 2022)")
    
    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

print("\n--- 📊 GENERANDO GRÁFICAS BIOLÓGICAS ---")
visualizar_resultados(model_d1, t_d1, t_raw_d1, co2_d1, s_co2_d1, "Tratamiento D1")
visualizar_resultados(model_d4, t_d4, t_raw_d4, co2_d4, s_co2_d4, "Tratamiento D4")
def extraer_parametros_biologicos(model, nombre_tratamiento):
    # Obtener los parámetros del modelo (se pasan por softplus para ser positivos)
    params_tensor = model.get_physics_params()
    
    # Convertir a números normales de Python
    params = params_tensor.detach().numpy()
    Y_g = params[0] # Yield Growth (Costo de crear estructura)
    Y_l = params[1] # Yield Lipids (Costo de crear/quemar grasa)
    m   = params[2] # Maintenance (Costo de mantenimiento basal)
    
    print(f"--- 🧬 PARÁMETROS DESCUBIERTOS: {nombre_tratamiento} ---")
    print(f"  [Y_g] Costo Energético de Crecer:      {Y_g:.5f}")
    print(f"  [Y_l] Costo Energético de Grasas:      {Y_l:.5f}")
    print(f"  [m]   Tasa de Mantenimiento Basal:     {m:.5f}")
    print("-" * 50)

# Ejecutar para ambos modelos
extraer_parametros_biologicos(model_d1, "TRATAMIENTO D1 (Voraz/Estrés)")
extraer_parametros_biologicos(model_d4, "TRATAMIENTO D4 (Crecimiento)")

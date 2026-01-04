import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# --- 1. Reutilizamos la función de limpieza que ya funcionó ---
def limpiar_y_recortar(ruta_archivo, umbral_salto=200):
    try:
        df = pd.read_csv(ruta_archivo, sep='\t', header=None, engine='python')
    except:
        df = pd.read_csv(ruta_archivo, sep='\s+', header=None, engine='python')

    df.columns = ['hora', 'valor']
    df['hora_dt'] = pd.to_datetime(df['hora'], format='%H:%M:%S')
    df['diferencia'] = df['valor'].diff()

    indices_salto = df[df['diferencia'] > umbral_salto].index
    if len(indices_salto) > 0:
        idx_inicio = indices_salto[0]
    else:
        idx_inicio = 0 # Si no hay salto, asumimos desde el principio (común en controles planos)

    df_limpio = df.iloc[idx_inicio:].copy()
    hora_cero = df_limpio['hora_dt'].iloc[0]
    # Creamos columna de minutos (X) y valor (Y)
    df_limpio['minutos'] = (df_limpio['hora_dt'] - hora_cero).dt.total_seconds() / 60.0
    return df_limpio

# --- 2. Función nueva para calcular pendiente (La Velocidad) ---
def obtener_pendiente(df):
    # Usamos regresión lineal: y = mx + b
    # m (slope) es la tasa de producción (ppm/min)
    slope, intercept, r_value, p_value, std_err = linregress(df['minutos'], df['valor'])
    return slope, intercept, df['minutos'], slope * df['minutos'] + intercept

# --- ZONA DE CONFIGURACIÓN ---
# Pon aquí las rutas de UN par para probar (Ejemplo: Día 9, D1)
archivo_experimento = '9/experimento_D1/experimento_co2.csv'  # Larvas
archivo_control =     '9/experimento_D1_alimento/experimento_co2.csv'  # Solo Alimento

try:
    # A. Procesar Experimento
    print(f"1. Procesando Experimento: {archivo_experimento}")
    df_exp = limpiar_y_recortar(archivo_experimento, umbral_salto=200)
    m_exp, b_exp, x_exp, y_pred_exp = obtener_pendiente(df_exp)

    # B. Procesar Control (A veces el control no tiene saltos bruscos, usa umbral bajo)
    print(f"2. Procesando Control: {archivo_control}")
    df_ctrl = limpiar_y_recortar(archivo_control, umbral_salto=50)
    m_ctrl, b_ctrl, x_ctrl, y_pred_ctrl = obtener_pendiente(df_ctrl)

    # C. Resultado Final
    tasa_neta = m_exp - m_ctrl

    print("-" * 40)
    print(f"Velocidad Larvas (Total): {m_exp:.2f} ppm/min")
    print(f"Velocidad Alimento (Ruido): {m_ctrl:.2f} ppm/min")
    print(f"==> TASA METABÓLICA REAL: {tasa_neta:.2f} ppm/min")
    print("-" * 40)

    # --- GRAFICAR PARA VER LA RESTA ---
    plt.figure(figsize=(10, 6))

    # Graficar Experimento
    plt.scatter(df_exp['minutos'], df_exp['valor'], color='blue', alpha=0.3, label='Datos Larvas')
    plt.plot(x_exp, y_pred_exp, color='blue', linewidth=2, label=f'Tendencia Larvas ({m_exp:.1f})')

    # Graficar Control
    plt.scatter(df_ctrl['minutos'], df_ctrl['valor'], color='orange', alpha=0.3, label='Datos Control')
    plt.plot(x_ctrl, y_pred_ctrl, color='orange', linewidth=2, label=f'Tendencia Control ({m_ctrl:.1f})')

    plt.title(f"Cálculo: {m_exp:.1f} - {m_ctrl:.1f} = {tasa_neta:.1f} ppm/min")
    plt.xlabel("Tiempo (minutos)")
    plt.ylabel("CO2 (ppm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except FileNotFoundError:
    print("Error: No encontré los archivos. Verifica las rutas en la 'ZONA DE CONFIGURACIÓN'.")
except Exception as e:
    print(f"Ocurrió un error: {e}")

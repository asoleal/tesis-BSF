import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import csv

# ======= CONFIGURACIÓN - CAMBIA AQUÍ EL ARCHIVO =======
#archivo_entrada = 'experimento1co2.csv'  # Cambia este nombre por el archivo que quieres procesar
archivo_entrada = 'experimento_ch4.csv'
# ======================================================

# Generar nombres automáticamente basados en el archivo de entrada
base_name = archivo_entrada.replace('.csv', '').replace('.ods', '')
archivo_grafica = f'{base_name}_grafica.png'
numero_experimento = base_name.replace('experimento', '').replace('co2', '').replace('_alimento', '')

print(f"Archivo a procesar: {archivo_entrada}")
print(f"Gráfica se guardará como: {archivo_grafica}")
print("-" * 50)

# Leer el archivo CSV
try:
    df = pd.read_csv(archivo_entrada, sep='\t', header=None)
    print("Archivo CSV leído correctamente (separador: tabulación)")
except:
    try:
        # Intentar con separador de coma
        df = pd.read_csv(archivo_entrada, sep=',', header=None)
        print("Archivo CSV leído correctamente (separador: coma)")
    except:
        try:
            # Intentar detectar separador automáticamente
            df = pd.read_csv(archivo_entrada, header=None)
            print("Archivo CSV leído correctamente (separador automático)")
        except Exception as e:
            print(f"Error leyendo el archivo CSV: {e}")
            df = None

# Verificar si se pudo leer el archivo
if df is None:
    print("No se pudo leer el archivo. Script terminado.")
    exit()

# Mostrar información del archivo para debug
print(f"Forma del DataFrame: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")
print("Primeras 5 filas:")
print(df.head())
print()

# Verificar que tengamos al menos 2 columnas
if df.shape[1] < 2:
    print("Error: El archivo no tiene suficientes columnas")
    print("Se esperan 2 columnas: tiempo y CO2")
    exit()

# Extraer las columnas (asumiendo que la primera es tiempo y la segunda es CO2)
time_col = df.iloc[:, 0].astype(str)
co2_col = df.iloc[:, 1].astype(float)

# Convertir tiempo a minutos transcurridos
time_minutes = []
co2_values = []
base_time = None

for i, time_str in enumerate(time_col):
    try:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
        if base_time is None:
            base_time = time_obj

        # Calcular minutos transcurridos desde el primer punto
        minutes_elapsed = ((time_obj.hour - base_time.hour) * 60 +
                          (time_obj.minute - base_time.minute) +
                          (time_obj.second - base_time.second) / 60)

        time_minutes.append(minutes_elapsed)
        co2_values.append(co2_col.iloc[i])
    except:
        continue

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(time_minutes, co2_values, 'b-', linewidth=2, marker='o', markersize=4)

# Configurar ejes y títulos
plt.xlabel('t (min)', fontsize=12)
##############################################################
#plt.ylabel('CO2 (ppm)', fontsize=12)
plt.ylabel('CH4 (ppm)', fontsize=12)
###############################################################


# Título dinámico basado en el nombre del archivo
if 'alimento' in base_name:
    titulo = f'Experimento {numero_experimento} - Concentración de CO2 (con alimento)'
else:
    #####################################################################################
    #titulo = f'Experimento {numero_experimento} - Concentración de CO2 generada por larvas y dieta D1'
    titulo = f'Concentración de CH4 generada por larvas'
    #######################################################################################
plt.title(titulo, fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Configurar exactamente 4 marcas en el eje X
if time_minutes:
    x_min, x_max = min(time_minutes), max(time_minutes)
    x_ticks = np.linspace(x_min, x_max, 4)
    plt.xticks(x_ticks, [f'{tick:.1f}' for tick in x_ticks])

# Guardar y mostrar la gráfica
plt.savefig(archivo_grafica, dpi=300, bbox_inches='tight')
plt.show()

print(f"Gráfica generada con {len(time_minutes)} puntos de datos")
print(f"Archivo guardado: {archivo_grafica}")
print(f"Título usado: {titulo}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def limpiar_y_recortar(ruta_archivo, tipo_gas='co2', umbral_salto=200):
    """
    1. Lee el archivo.
    2. Detecta el salto brusco (inicio de reacción).
    3. Recorta el 'ruido' anterior.
    4. Reinicia el tiempo a Minuto 0.
    """

    # 1. Cargar datos (Asumiendo que no tiene encabezados, y separado por tabulación o espacio)
    # Ajusta 'sep' si tus archivos usan comas (',') o punto y coma (';')
    try:
        df = pd.read_csv(ruta_archivo, sep='\t', header=None, engine='python')
    except:
        df = pd.read_csv(ruta_archivo, sep='\s+', header=None, engine='python')

    df.columns = ['hora', 'valor']

    # Convertir la columna de hora a objetos datetime para poder restar
    # Asumimos formato HH:MM:SS, ajustamos a una fecha ficticia de hoy para calcular deltas
    df['hora_dt'] = pd.to_datetime(df['hora'], format='%H:%M:%S')

    # 2. Detectar el inicio (El Salto)
    # Calculamos la diferencia entre cada punto y el anterior
    df['diferencia'] = df['valor'].diff()

    # Buscamos el primer índice donde la diferencia supera el umbral
    # (Ej: si sube más de 200 ppm de golpe)
    indices_salto = df[df['diferencia'] > umbral_salto].index

    if len(indices_salto) > 0:
        idx_inicio = indices_salto[0]
        print(f"--> Salto detectado en la fila {idx_inicio} (Hora: {df['hora'].iloc[idx_inicio]})")
    else:
        print("--> No se detectó un salto brusco automático. Se usará el inicio del archivo.")
        idx_inicio = 0

    # 3. Recortar (Sección 1)
    df_limpio = df.iloc[idx_inicio:].copy()

    # 4. Reiniciar Cronómetro (t=0)
    hora_cero = df_limpio['hora_dt'].iloc[0]
    # Calculamos minutos transcurridos desde el salto
    df_limpio['minutos'] = (df_limpio['hora_dt'] - hora_cero).dt.total_seconds() / 60.0

    return df, df_limpio, idx_inicio

# --- ZONA DE PRUEBA ---
# Cambia esta ruta por uno de tus archivos reales para probar
archivo_prueba = '9/experimento_D1/experimento_co2.csv'

# Solo ejecutamos esto si tienes el archivo
try:
    # CO2 suele saltar > 150-200. CH4 suele saltar > 500-1000. Ajusta 'umbral_salto'.
    df_original, df_final, punto_corte = limpiar_y_recortar(archivo_prueba, umbral_salto=300)

    # --- VISUALIZACIÓN PARA TU CONTROL ---
    plt.figure(figsize=(10, 5))

    # Graficar datos originales (en gris)
    plt.plot(df_original.index, df_original['valor'], color='lightgray', label='Datos Crudos (Ruido)', marker='.')

    # Graficar datos limpios (en color)
    # Notar que los graficamos en su posición original de índice para ver la superposición
    plt.plot(df_final.index, df_final['valor'], color='green', label='Datos Limpios (Reacción)', linewidth=2)

    # Línea de corte
    plt.axvline(x=punto_corte, color='red', linestyle='--', label='Corte Automático (t=0)')

    plt.title(f"Limpieza de Datos: {archivo_prueba}")
    plt.xlabel("Número de Muestra (Índice)")
    plt.ylabel("Concentración (ppm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Primeras 5 filas de tus datos LIMPIOS (Listos para la IA):")
    print(df_final[['minutos', 'valor']].head())

except FileNotFoundError:
    print("Por favor, edita la variable 'archivo_prueba' con la ruta de uno de tus archivos .csv o .txt")
